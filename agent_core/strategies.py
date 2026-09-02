"""
Estratégias: quem transforma evidências numa decisão.

Contrato (usado pelo AgentLoop)::

    decision = await strategy.decide(ctx)      # ctx: FailureContext
    decision.action in {patch, observe_again, run_tests, rollback, finish}

Implementações:

* ``FixStrategy`` (base): mantém o método clássico ``propose(ctx) -> FixProposal``
  e o adapta a ``decide`` (proposta -> ação ``patch``; ``None`` -> ``finish``).
* ``HeuristicFixStrategy``: regras determinísticas, offline.
* ``ModelFixStrategy``: pede ao ``ModelProvider`` um patch em JSON, enviando
  texto e imagens do ``MultimodalContext``.
* ``ClaudeFixStrategy``: ``ModelFixStrategy`` já ligado ao ``AnthropicProvider``
  (mantido por compatibilidade com a API anterior).
* ``AutoStrategy``: combina evidências (traceback, logs, testes, visão, memória)
  num ``Diagnosis`` e percorre uma lista extensível de ``Planner`` para decidir
  dinamicamente a próxima ação.
* ``CompositeFixStrategy``: encadeia estratégias clássicas.
"""

from __future__ import annotations

import builtins
import difflib
import json
import logging
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from .code_manager import FilePatch, ModuleAnalysis, Replacement
from .config import AgentConfig
from .memory import AgentMemory, patch_signature
from .observations import MultimodalContext, ObservationKind
from .providers import (
    ContentPart,
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ProviderError,
    ProviderInterrupted,
)
from .sandbox import ExecutionResult
from .tools import PATCH_SCHEMA, ProjectToolbox, summarize_tool_input

logger = logging.getLogger("agent_core.strategies")


# --------------------------------------------------------------------- modelos
@dataclass
class AttemptSummary:
    """Resumo de uma tentativa anterior, para o modelo não repetir o mesmo erro."""

    attempt: int
    error_signature: str
    rationale: str
    patched_files: list[str]
    outcome: str  # "fixed" | "new_error" | "same_error" | "rolled_back" | "patch_failed"


@dataclass
class FailureContext:
    """Tudo que uma estratégia precisa saber sobre a falha atual."""

    script: str                                  # caminho relativo do script alvo
    result: ExecutionResult                      # resultado bruto do sandbox
    failing_file: str | None                     # arquivo onde a exceção estourou
    failing_source: str | None                   # conteúdo desse arquivo
    failing_analysis: ModuleAnalysis | None      # análise AST desse arquivo
    project_outline: dict[str, ModuleAnalysis]   # esqueleto de todo o projeto
    attempt: int                                 # número da iteração atual (1-based)
    history: list[AttemptSummary] = field(default_factory=list)
    attachments: list[Any] = field(default_factory=list)  # gancho multimodal genérico
    multimodal: MultimodalContext | None = None  # observações (código, logs, testes, visão...)
    memory: AgentMemory | None = None            # memória de execução
    tests: ExecutionResult | None = None         # último resultado de testes, se houver
    vision_available: bool = False               # o loop consegue observar visualmente?
    code_manager: Any = None                     # acesso confinado ao projeto (ferramentas do modelo)
    effort: str | None = None                    # esforço sugerido para o modelo nesta falha

    @property
    def error_line(self) -> int | None:
        tb = self.result.traceback
        return tb.location.line if tb and tb.location else None


@dataclass
class FixProposal:
    patches: list[FilePatch]
    rationale: str
    confidence: float = 0.5
    strategy: str = ""

    @property
    def signature(self) -> str:
        return patch_signature(self.patches)


class ActionKind(str, Enum):
    PATCH = "patch"
    OBSERVE_AGAIN = "observe_again"
    RUN_TESTS = "run_tests"
    ROLLBACK = "rollback"
    FINISH = "finish"


@dataclass
class Decision:
    """Resultado de ``decide``: o que o loop deve fazer a seguir."""

    action: ActionKind
    proposal: FixProposal | None = None
    reason: str = ""
    diagnosis: str = ""
    strategy: str = ""

    @classmethod
    def patch(cls, proposal: FixProposal, *, diagnosis: str = "", strategy: str = "") -> "Decision":
        return cls(ActionKind.PATCH, proposal, proposal.rationale, diagnosis, strategy or proposal.strategy)

    @classmethod
    def finish(cls, reason: str, *, diagnosis: str = "", strategy: str = "") -> "Decision":
        return cls(ActionKind.FINISH, None, reason, diagnosis, strategy)


# ------------------------------------------------------------------ interface
class FixStrategy(ABC):
    """Contrato de uma estratégia de correção."""

    name: str = "base"

    @abstractmethod
    async def propose(self, ctx: FailureContext) -> FixProposal | None:
        """Devolve uma proposta de patch ou ``None`` se não souber corrigir."""

    async def decide(self, ctx: FailureContext) -> Decision:
        """Adapta ``propose`` ao contrato de decisão do loop."""
        proposal = await self.propose(ctx)
        if proposal is None:
            return Decision.finish("no_fix", strategy=self.name)
        return Decision.patch(proposal, strategy=proposal.strategy or self.name)


# ----------------------------------------------------------------- heurística
_NAME_ERROR_RE = re.compile(r"name '(?P<name>[A-Za-z_]\w*)' is not defined")


class HeuristicFixStrategy(FixStrategy):
    """Correções mecânicas sem LLM.

    Regras (em ordem):
        1. ``NameError`` para módulo da stdlib      -> ``import <mod>``
        2. ``NameError`` para símbolo de outro módulo do projeto
                                                    -> ``from <mod> import <sym>``
        3. ``NameError`` por erro de digitação (um único nome muito parecido
           definido/importado/builtin)              -> corrige o nome na linha do erro
        4. ``TabError``/``IndentationError`` com tabs -> converte tabs em 4 espaços
    """

    name = "heuristic"
    STDLIB = set(getattr(sys, "stdlib_module_names", ()))

    async def propose(self, ctx: FailureContext) -> FixProposal | None:
        tb = ctx.result.traceback
        if tb is None or ctx.failing_file is None or ctx.failing_source is None:
            return None

        if tb.exc_type == "NameError":
            return self._fix_name_error(ctx, tb.message)
        if tb.exc_type in ("TabError", "IndentationError") and "\t" in ctx.failing_source:
            return self._fix_tabs(ctx)
        return None

    # -- regra 1 e 2
    def _fix_name_error(self, ctx: FailureContext, message: str) -> FixProposal | None:
        m = _NAME_ERROR_RE.search(message)
        if not m:
            return None
        name = m.group("name")
        analysis = ctx.failing_analysis
        if analysis and name in (analysis.defined_names | analysis.imported_names):
            # O nome existe mas é usado antes de ser definido: não é um import.
            return None

        import_line: str | None = None
        if name in self.STDLIB:
            import_line = f"import {name}"
        else:
            # Procura o símbolo em outro módulo do projeto (mesmo diretório do
            # arquivo que falhou, para o import absoluto funcionar via sys.path[0]).
            failing_dir = ctx.failing_file.rsplit("/", 1)[0] if "/" in ctx.failing_file else ""
            for path, other in ctx.project_outline.items():
                if path == ctx.failing_file or not other.is_valid:
                    continue
                other_dir = path.rsplit("/", 1)[0] if "/" in path else ""
                if other_dir != failing_dir:
                    continue
                if name in other.defined_names:
                    module = path.rsplit("/", 1)[-1].removesuffix(".py")
                    import_line = f"from {module} import {name}"
                    break
        if import_line is None:
            return self._fix_typo(ctx, name)

        new_source = insert_import(ctx.failing_source, import_line)
        return FixProposal(
            patches=[FilePatch(path=ctx.failing_file, content=new_source, reason=f"heuristic: {import_line}")],
            rationale=f"NameError para '{name}': adicionado '{import_line}' no topo do módulo.",
            confidence=0.9 if name in self.STDLIB else 0.75,
            strategy=self.name,
        )

    # -- regra 3
    def _fix_typo(self, ctx: FailureContext, name: str) -> FixProposal | None:
        analysis = ctx.failing_analysis
        line_no = ctx.error_line
        if analysis is None or line_no is None or ctx.failing_source is None:
            return None
        candidates = set(analysis.defined_names) | set(analysis.imported_names) | {b for b in dir(builtins) if not b.startswith("_")}
        candidates.discard(name)
        matches = difflib.get_close_matches(name, sorted(candidates), n=2, cutoff=0.8)
        if len(matches) != 1:
            return None  # ambíguo ou sem candidato: não arrisca
        lines = ctx.failing_source.splitlines(keepends=True)
        if line_no > len(lines):
            return None
        pattern = re.compile(rf"\b{re.escape(name)}\b")
        fixed_line, count = pattern.subn(matches[0], lines[line_no - 1])
        if count == 0:
            return None
        lines[line_no - 1] = fixed_line
        return FixProposal(
            patches=[FilePatch(path=ctx.failing_file, content="".join(lines), reason=f"heuristic: typo {name}->{matches[0]}")],
            rationale=f"NameError para '{name}': provável erro de digitação de '{matches[0]}' na linha {line_no}.",
            confidence=0.7,
            strategy=self.name,
        )

    # -- regra 4
    def _fix_tabs(self, ctx: FailureContext) -> FixProposal | None:
        # O interpretador expande tabs para colunas múltiplas de 8; tenta essa
        # leitura primeiro e só depois a convenção de 4 espaços, sempre
        # verificando se o resultado compila.
        new_source: str | None = None
        for width in (8, 4):
            candidate = "\n".join(line.expandtabs(width) for line in ctx.failing_source.splitlines())
            if ctx.failing_source.endswith("\n"):
                candidate += "\n"
            try:
                compile(candidate, ctx.failing_file or "<tabs>", "exec")
            except SyntaxError:
                continue
            new_source = candidate
            break
        if new_source is None:
            return None
        return FixProposal(
            patches=[FilePatch(path=ctx.failing_file, content=new_source, reason="heuristic: tabs->spaces")],
            rationale="Indentação mista (tabs/espaços) normalizada para 4 espaços.",
            confidence=0.7,
            strategy=self.name,
        )


def insert_import(source: str, import_line: str) -> str:
    """Insere ``import_line`` após docstring/``__future__``/imports existentes.

    Mantém o arquivo idiomático: o novo import fica junto do bloco de imports,
    não na primeira linha.
    """
    lines = source.splitlines(keepends=True)
    insert_at = 0
    i = 0
    # Pula shebang, encoding e docstring de módulo.
    if lines and lines[0].startswith("#!"):
        i += 1
    while i < len(lines) and (lines[i].strip() == "" or lines[i].lstrip().startswith("#")):
        i += 1
    if i < len(lines) and lines[i].lstrip().startswith(('"""', "'''")):
        quote = lines[i].lstrip()[:3]
        # docstring de uma linha?
        if lines[i].strip().count(quote) >= 2 and len(lines[i].strip()) > 3:
            i += 1
        else:
            i += 1
            while i < len(lines) and quote not in lines[i]:
                i += 1
            i += 1
    insert_at = i
    # Avança até o fim do bloco contíguo de imports.
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith(("import ", "from ")) or stripped == "":
            if stripped:
                insert_at = i + 1
            i += 1
            continue
        break
    lines.insert(insert_at, import_line + "\n")
    return "".join(lines)


# ------------------------------------------------------------- via provider
_SYSTEM_PROMPT = """\
Você é o módulo de auto-correção de um agente autônomo que reescreve o próprio \
código-fonte Python. Receberá evidências (traceback, arquivo que falhou com \
números de linha, esqueleto do projeto, logs, resultado de testes, observações \
visuais e imagens quando existirem) e o histórico de tentativas anteriores.

Sua tarefa: propor a menor alteração que corrija a causa raiz do erro, sem \
mudar o comportamento pretendido do programa. Nunca "corrija" silenciando a \
exceção com try/except genérico. Não repita uma tentativa já feita.

Responda SOMENTE com um objeto JSON (sem texto antes ou depois, sem cercas de \
código) no formato:
{
  "rationale": "explicação curta da causa raiz e da correção",
  "confidence": 0.0 a 1.0,
  "patches": [
    {"path": "caminho/relativo.py", "mode": "search_replace",
     "replacements": [{"search": "trecho exato existente", "replace": "trecho novo"}]},
    {"path": "outro/arquivo.py", "mode": "replace_full", "content": "arquivo inteiro"}
  ]
}
Prefira "search_replace" com trechos únicos e exatos (respeitando indentação). \
Use "replace_full" apenas quando a mudança for extensa. Se não houver correção \
segura possível, devolva {"rationale": "...", "confidence": 0, "patches": []}.
"""


class ModelFixStrategy(FixStrategy):
    """Formula correções perguntando a um ``ModelProvider`` (qualquer backend)."""

    name = "model"

    def __init__(self, provider: ModelProvider, *, system_prompt: str = _SYSTEM_PROMPT, include_images: bool = True, structured_output: bool = True) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.include_images = include_images
        self.structured_output = structured_output  # output_config.format com o JSON Schema do patch
        self.last_response: Any = None

    # -- prompt
    @staticmethod
    def _numbered(source: str) -> str:
        return "\n".join(f"{i:4d} | {line}" for i, line in enumerate(source.splitlines(), 1))

    def build_prompt(self, ctx: FailureContext, diagnosis: str = "") -> str:
        parts = [f"## Script executado\n{ctx.script}"]
        if diagnosis:
            parts.append(f"## Diagnóstico preliminar\n{diagnosis}")
        parts.append(f"## Resultado\n{ctx.result.summary()}")
        if ctx.result.stderr.strip():
            parts.append(f"## stderr (traceback)\n```\n{ctx.result.stderr.strip()[-6000:]}\n```")
        if ctx.result.stdout.strip():
            parts.append(f"## stdout (últimas linhas)\n```\n{ctx.result.stdout.strip()[-2000:]}\n```")
        if ctx.failing_file and ctx.failing_source is not None:
            parts.append(f"## Arquivo que falhou: {ctx.failing_file}\n```python\n{self._numbered(ctx.failing_source)}\n```")
        if ctx.project_outline:
            outline = "\n\n".join(a.outline() for a in ctx.project_outline.values())
            parts.append(f"## Esqueleto do projeto\n```\n{outline[:8000]}\n```")
        if ctx.tests is not None:
            parts.append(f"## Testes\n{ctx.tests.summary()}\n```\n{ctx.tests.stderr.strip()[-3000:]}\n```")
        if ctx.multimodal is not None:
            extra = [o for o in ctx.multimodal.observations if o.kind in (ObservationKind.LOG, ObservationKind.VISION, ObservationKind.TEST)]
            if extra:
                parts.append("## Outras evidências\n" + "\n\n".join(o.to_prompt_text(2000) for o in extra[-8:]))
        if ctx.memory is not None and len(ctx.memory):
            parts.append(f"## Memória do agente (não repita)\n{ctx.memory.to_prompt_text()}")
        elif ctx.history:
            hist = "\n".join(
                f"- tentativa {h.attempt}: {h.rationale} -> {h.outcome} (arquivos: {', '.join(h.patched_files)})" for h in ctx.history
            )
            parts.append(f"## Tentativas anteriores (não repita)\n{hist}")
        parts.append(f"## Iteração atual\n{ctx.attempt}")
        return "\n\n".join(parts)

    def _image_parts(self, ctx: FailureContext) -> list[ContentPart]:
        parts: list[ContentPart] = []
        if self.include_images and ctx.multimodal is not None and self.provider.supports_images:
            for obs in reversed(ctx.multimodal.images()):
                parts.append(ContentPart.from_text(f"[imagem: {obs.source}] {obs.summary}"))
                parts.append(ContentPart.from_image(obs.image.data, obs.image.media_type))
        return parts

    def build_request(self, ctx: FailureContext, diagnosis: str = "") -> ModelRequest:
        parts: list[ContentPart] = [ContentPart.from_text(self.build_prompt(ctx, diagnosis))] + self._image_parts(ctx)
        return ModelRequest(
            messages=[ModelMessage("user", parts)],
            system=self.system_prompt,
            effort=ctx.effort,
            output_schema=PATCH_SCHEMA if self.structured_output else None,
        )

    # -- chamada
    async def propose(self, ctx: FailureContext, diagnosis: str = "") -> FixProposal | None:
        request = self.build_request(ctx, diagnosis)
        try:
            response = await self.provider.complete(request)
        except ProviderInterrupted:
            raise
        except ProviderError as exc:
            logger.error("provider %s falhou: %s", self.provider.name, exc)
            return None
        self.last_response = response
        if response.truncated:
            logger.warning("resposta truncada por max_tokens; aumente llm_max_tokens.")
        proposal = self.parse_response(response.text)
        if proposal is not None:
            proposal.strategy = self.name
        return proposal

    # -- parse
    def parse_response(self, text: str) -> FixProposal | None:
        """Converte o JSON devolvido pelo modelo num ``FixProposal``."""
        data = _extract_json(text)
        if not isinstance(data, dict):
            logger.error("Resposta do modelo não é JSON válido: %r", text[:300])
            return None
        patches: list[FilePatch] = []
        for raw in data.get("patches", []) or []:
            if not isinstance(raw, dict):
                continue
            path = raw.get("path")
            if not path:
                continue
            if raw.get("mode") == "replace_full" or "content" in raw:
                patches.append(FilePatch(path=path, content=raw.get("content", ""), reason="model:replace_full"))
            else:
                reps = [
                    Replacement(search=r["search"], replace=r.get("replace", ""), count=int(r.get("count", 1)))
                    for r in raw.get("replacements", [])
                    if isinstance(r, dict) and r.get("search")
                ]
                if reps:
                    patches.append(FilePatch(path=path, replacements=reps, reason="model:search_replace"))
        if not patches:
            logger.info("Modelo não propôs patches: %s", data.get("rationale"))
            return None
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return FixProposal(
            patches=patches,
            rationale=str(data.get("rationale", "")),
            confidence=max(0.0, min(1.0, confidence)),
            strategy=self.name,
        )


class ClaudeFixStrategy(ModelFixStrategy):
    """``ModelFixStrategy`` já ligado ao SDK oficial (via ``AnthropicProvider``).

    Mantido por compatibilidade. ``client`` permite injetar um cliente falso.
    """

    name = "claude"

    def __init__(self, config: AgentConfig, client: Any | None = None, provider: ModelProvider | None = None) -> None:
        if provider is None:
            from .providers import AnthropicProvider, build_provider

            if client is not None:
                provider = AnthropicProvider(
                    model=config.llm_model,
                    max_tokens=config.llm_max_tokens,
                    effort=config.llm_effort,
                    server_fallbacks=config.llm_enable_fallbacks,
                    timeout=config.llm_timeout,
                    client=client,
                )
            else:
                provider = build_provider(config)
        super().__init__(provider)
        self.config = config


_TOOL_SYSTEM_PROMPT = """\
Você é o módulo de auto-correção de um agente autônomo que reescreve o próprio \
código-fonte Python. Receberá um diagnóstico inicial (traceback, arquivo que \
falhou, evidências de logs/testes/tela) e ferramentas de leitura do projeto.

Método: (1) leia o que for necessário com read_file/search/outline/list_files, \
sem pedir arquivos irrelevantes; (2) identifique a causa raiz; (3) chame \
propose_patch UMA vez com a menor alteração que corrija a causa, sem mudar o \
comportamento pretendido e sem silenciar exceções com try/except genérico. \
Não repita tentativas listadas na memória. Prefira search_replace com trechos \
exatos e únicos (respeite a indentação); use replace_full só para mudanças \
extensas. Se não houver correção segura, chame propose_patch com patches vazio \
e explique no rationale.
"""


class ToolFixStrategy(ModelFixStrategy):
    """Correção com ferramentas: o modelo lê o projeto sob demanda.

    Loop manual de ``tool_use``: cada rodada envia o histórico, executa as
    chamadas pedidas (somente leitura, via ``ProjectToolbox``) e devolve os
    resultados até o modelo chamar ``propose_patch`` ou esgotar
    ``max_rounds``. Se o modelo responder em texto JSON em vez de usar a
    ferramenta, o texto ainda é aceito.
    """

    name = "tools"

    def __init__(self, provider: ModelProvider, *, max_rounds: int = 8, system_prompt: str = _TOOL_SYSTEM_PROMPT, include_images: bool = True) -> None:
        super().__init__(provider, system_prompt=system_prompt, include_images=include_images, structured_output=True)
        self.max_rounds = max_rounds
        self.last_rounds = 0
        self.last_tool_calls: list[str] = []

    def build_prompt(self, ctx: FailureContext, diagnosis: str = "") -> str:
        """Prompt compacto: sem esqueleto do projeto (o modelo pede via ferramentas)."""
        parts = [f"## Script executado\n{ctx.script}"]
        if diagnosis:
            parts.append(f"## Diagnóstico preliminar\n{diagnosis}")
        parts.append(f"## Resultado\n{ctx.result.summary()}")
        if ctx.result.stderr.strip():
            parts.append(f"## stderr (traceback)\n```\n{ctx.result.stderr.strip()[-4000:]}\n```")
        if ctx.failing_file and ctx.failing_source is not None:
            parts.append(f"## Arquivo que falhou: {ctx.failing_file}\n```python\n{self._numbered(ctx.failing_source)}\n```")
        if ctx.project_outline:
            parts.append("## Arquivos do projeto\n" + "\n".join(sorted(ctx.project_outline)))
        if ctx.tests is not None:
            parts.append(f"## Testes\n{ctx.tests.summary()}\n```\n{ctx.tests.stderr.strip()[-2000:]}\n```")
        if ctx.multimodal is not None:
            extra = [o for o in ctx.multimodal.observations if o.kind in (ObservationKind.LOG, ObservationKind.VISION, ObservationKind.TEST)]
            if extra:
                parts.append("## Outras evidências\n" + "\n\n".join(o.to_prompt_text(1500) for o in extra[-6:]))
        if ctx.memory is not None and len(ctx.memory):
            parts.append(f"## Memória do agente (não repita)\n{ctx.memory.to_prompt_text()}")
        parts.append(f"## Iteração atual\n{ctx.attempt}")
        return "\n\n".join(parts)

    async def propose(self, ctx: FailureContext, diagnosis: str = "") -> FixProposal | None:
        if ctx.code_manager is None or not self.provider.supports_tools:
            # Sem acesso ao projeto: cai no modo de prompt único.
            return await ModelFixStrategy.propose(self, ctx, diagnosis)
        toolbox = ProjectToolbox(ctx.code_manager)
        messages = [ModelMessage("user", [ContentPart.from_text(self.build_prompt(ctx, diagnosis))] + self._image_parts(ctx))]
        request = ModelRequest(messages=messages, system=self.system_prompt, tools=toolbox.specs(), effort=ctx.effort)
        self.last_rounds = 0
        self.last_tool_calls = []
        final_text = ""
        for _ in range(self.max_rounds):
            self.last_rounds += 1
            try:
                response = await self.provider.complete(request)
            except ProviderInterrupted:
                raise
            except ProviderError as exc:
                logger.error("provider %s falhou: %s", self.provider.name, exc)
                return None
            self.last_response = response
            if response.truncated:
                logger.warning("resposta truncada por max_tokens; aumente llm_max_tokens.")
            final_text = response.text
            if not response.wants_tools:
                break
            messages.append(ModelMessage.assistant_from(response))
            results: list[ContentPart] = []
            for call in response.tool_calls:
                result = await toolbox.execute(call)
                self.last_tool_calls.append(summarize_tool_input(call))
                logger.info("ferramenta %s -> %s", summarize_tool_input(call), "erro" if result.is_error else f"{len(result.content)} chars")
                results.append(ContentPart.from_tool_result(call.id, result.content, is_error=result.is_error))
            messages.append(ModelMessage.tool_results(results))
            if toolbox.proposal is not None:
                break
        else:
            logger.warning("limite de %d rodadas de ferramentas atingido", self.max_rounds)
        payload = toolbox.proposal_json() or final_text
        proposal = self.parse_response(payload) if payload else None
        if proposal is not None:
            proposal.strategy = self.name
        return proposal


def _extract_json(text: str) -> Any:
    """Extrai o primeiro objeto JSON do texto (tolera cercas ``` e prosa em volta)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    candidates.append(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


# ------------------------------------------------------------------ composta
class CompositeFixStrategy(FixStrategy):
    """Tenta cada estratégia em ordem; devolve a primeira proposta não nula."""

    name = "composite"

    def __init__(self, strategies: list[FixStrategy]) -> None:
        if not strategies:
            raise ValueError("CompositeFixStrategy precisa de ao menos uma estratégia")
        self.strategies = strategies

    async def propose(self, ctx: FailureContext) -> FixProposal | None:
        for strategy in self.strategies:
            try:
                proposal = await strategy.propose(ctx)
            except ProviderInterrupted:
                raise
            except Exception as exc:
                logger.error("Estratégia %s falhou: %s", strategy.name, exc)
                continue
            if proposal is not None:
                logger.info("Proposta obtida via '%s' (confiança %.2f)", strategy.name, proposal.confidence)
                return proposal
        return None


# ================================================================= AUTO ======
@dataclass
class Finding:
    """Uma evidência interpretada por um analisador."""

    source: str          # traceback | logs | tests | vision | memory | ...
    summary: str
    severity: float = 0.5  # 0..1
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Diagnosis:
    findings: list[Finding] = field(default_factory=list)
    primary_cause: str = ""
    needs: set[str] = field(default_factory=set)   # "code" | "vision" | "tests" | "logs" | "observe_again"

    @property
    def sources(self) -> set[str]:
        return {f.source for f in self.findings}

    def to_text(self) -> str:
        lines = [f"causa provável: {self.primary_cause or 'indeterminada'}"]
        for f in sorted(self.findings, key=lambda f: -f.severity):
            lines.append(f"- [{f.source} sev={f.severity:.1f}] {f.summary}")
        if self.needs:
            lines.append(f"necessidades: {', '.join(sorted(self.needs))}")
        return "\n".join(lines)


@runtime_checkable
class EvidenceAnalyzer(Protocol):
    name: str

    def analyze(self, ctx: FailureContext) -> list[Finding]: ...


@runtime_checkable
class Planner(Protocol):
    name: str

    async def plan(self, ctx: FailureContext, diagnosis: Diagnosis) -> Decision | None: ...


# -- analisadores
class TracebackAnalyzer:
    name = "traceback"

    def analyze(self, ctx: FailureContext) -> list[Finding]:
        r = ctx.result
        if r.timed_out:
            return [Finding("runtime", f"execução excedeu o tempo limite ({r.duration:.0f}s): possível loop infinito ou bloqueio", 0.9)]
        tb = r.traceback
        if tb is None:
            if not r.success:
                return [Finding("runtime", f"processo terminou com código {r.returncode} sem traceback", 0.6, {"stderr": r.stderr[-500:]})]
            return []
        loc = tb.location
        where = f"{loc.file}:{loc.line} ({loc.function})" if loc else "?"
        sev = 0.95 if tb.exc_type in ("SyntaxError", "IndentationError", "ImportError", "ModuleNotFoundError", "NameError") else 0.8
        return [Finding("traceback", f"{tb.exc_type}: {tb.message} em {where}", sev, {"exception": tb.exc_type, "line": loc.line if loc else None})]


class LogAnalyzer:
    name = "logs"
    KEYWORDS = ("error", "exception", "traceback", "critical", "fatal")

    def analyze(self, ctx: FailureContext) -> list[Finding]:
        findings: list[Finding] = []
        if ctx.multimodal is None:
            return findings
        for obs in ctx.multimodal.by_kind(ObservationKind.LOG):
            lines = obs.extracted.get("relevant_lines") or []
            hits = [l for l in lines if any(k in l.lower() for k in self.KEYWORDS)]
            if hits:
                findings.append(Finding("logs", f"{len(hits)} linhas de erro em {obs.source}: {hits[-1][:160]}", 0.5, {"lines": hits[-5:]}))
        # stdout/stderr do próprio run também são "logs"
        for line in ctx.result.stdout.splitlines()[-50:]:
            if any(k in line.lower() for k in self.KEYWORDS):
                findings.append(Finding("logs", f"stdout menciona erro: {line[:160]}", 0.3))
                break
        return findings


class TestAnalyzer:
    name = "tests"

    def analyze(self, ctx: FailureContext) -> list[Finding]:
        tests = ctx.tests
        if tests is None and ctx.multimodal is not None:
            obs = ctx.multimodal.latest(ObservationKind.TEST)
            if obs is not None and not obs.extracted.get("passed", True):
                failed = obs.extracted.get("failed_tests") or []
                return [Finding("tests", f"testes falhando: {', '.join(failed[:3]) or 'ver saída'}", 0.85, {"failed": failed})]
            return []
        if tests is not None and not tests.success:
            failed = [l for l in tests.stderr.splitlines() if l.startswith(("FAIL:", "ERROR:"))]
            return [Finding("tests", f"testes falhando: {', '.join(failed[:3]) or tests.summary()}", 0.85, {"failed": failed})]
        return []


class VisionAnalyzer:
    name = "vision"
    ERROR_WORDS = ("error", "erro", "exception", "traceback", "failed", "falha")

    def analyze(self, ctx: FailureContext) -> list[Finding]:
        findings: list[Finding] = []
        if ctx.multimodal is None:
            return findings
        visions = ctx.multimodal.by_kind(ObservationKind.VISION)
        if not visions:
            return findings
        latest = visions[-1]
        if latest.metadata.get("invalid"):
            findings.append(Finding("vision", "último frame inválido", 0.2))
            return findings
        text = latest.extracted.get("text")
        if text and any(w in text.lower() for w in self.ERROR_WORDS):
            findings.append(Finding("vision", f"a tela mostra um erro: {text[:160]!r}", 0.7, {"text": text}))
        change = latest.extracted.get("change") or {}
        if len(visions) >= 2 and change.get("changed"):
            findings.append(Finding("vision", f"a interface mudou {change.get('score', 0) * 100:.1f}% desde a observação anterior ({len(change.get('regions', []))} regiões)", 0.4, change))
        elif len(visions) >= 2 and not change.get("changed"):
            findings.append(Finding("vision", "sem mudança visual entre observações", 0.2))
        findings.append(Finding("vision", latest.summary, 0.3, {"resolution": latest.extracted.get("resolution")}))
        return findings


class MemoryAnalyzer:
    name = "memory"

    def analyze(self, ctx: FailureContext) -> list[Finding]:
        if ctx.memory is None:
            return []
        failed = ctx.memory.failed_attempts(ctx.result.signature)
        if not failed:
            return []
        files = sorted({f for e in failed for f in e.patch_files})
        return [Finding("memory", f"{len(failed)} tentativas anteriores falharam para este erro (arquivos: {', '.join(files) or '-'})", 0.6, {"attempts": len(failed)})]


# -- planejadores
class ObservationPlanner:
    """Pede uma nova observação quando as evidências são insuficientes."""

    name = "observe"

    def __init__(self, max_observations: int = 1) -> None:
        self.max_observations = max_observations

    async def plan(self, ctx: FailureContext, diagnosis: Diagnosis) -> Decision | None:
        if "observe_again" not in diagnosis.needs:
            return None
        already = ctx.memory.count_action("observe_again", ctx.result.signature) if ctx.memory else 0
        if already >= self.max_observations:
            return None
        return Decision(ActionKind.OBSERVE_AGAIN, reason="evidências insuficientes: nova observação", diagnosis=diagnosis.to_text(), strategy=self.name)


class HeuristicPlanner:
    """Usa a ``HeuristicFixStrategy`` quando o diagnóstico é mecânico."""

    name = "heuristic"

    def __init__(self, heuristic: HeuristicFixStrategy | None = None) -> None:
        self.heuristic = heuristic or HeuristicFixStrategy()

    async def plan(self, ctx: FailureContext, diagnosis: Diagnosis) -> Decision | None:
        proposal = await self.heuristic.propose(ctx)
        if proposal is None:
            return None
        if ctx.memory is not None and ctx.memory.has_tried(proposal.signature):
            return None  # já falhou antes; deixa outro planner tentar
        return Decision.patch(proposal, diagnosis=diagnosis.to_text(), strategy=self.name)


class ModelPlanner:
    """Consulta o provider com o contexto multimodal completo.

    Com ``use_tools`` (padrão) usa ``ToolFixStrategy``: o modelo recebe um
    prompt compacto e lê arquivos sob demanda; caso contrário usa
    ``ModelFixStrategy`` com o esqueleto do projeto no prompt.
    """

    name = "model"

    def __init__(self, provider: ModelProvider | None, *, max_attempts: int = 3, use_tools: bool = True, max_tool_rounds: int = 8) -> None:
        if provider is None:
            self.strategy = None
        elif use_tools:
            self.strategy = ToolFixStrategy(provider, max_rounds=max_tool_rounds)
        else:
            self.strategy = ModelFixStrategy(provider)
        self.max_attempts = max_attempts

    async def plan(self, ctx: FailureContext, diagnosis: Diagnosis) -> Decision | None:
        if self.strategy is None:
            return None
        for _ in range(2):  # uma segunda chance se a primeira proposta já foi tentada
            proposal = await self.strategy.propose(ctx, diagnosis.to_text())
            if proposal is None:
                return None
            if ctx.memory is not None and ctx.memory.has_tried(proposal.signature):
                logger.info("modelo repetiu um patch já fracassado; pedindo alternativa")
                continue
            return Decision.patch(proposal, diagnosis=diagnosis.to_text(), strategy=self.name)
        return None


class RollbackPlanner:
    """Se o erro atual surgiu após um patch que "progrediu" mas o loop está
    acumulando falhas, prefere reverter ao estado anterior."""

    name = "rollback"

    def __init__(self, after_failures: int = 3) -> None:
        self.after_failures = after_failures

    async def plan(self, ctx: FailureContext, diagnosis: Diagnosis) -> Decision | None:
        if ctx.memory is None:
            return None
        last = ctx.memory.last()
        failed = len(ctx.memory.failed_attempts())
        if last is not None and last.outcome == "new_error" and failed >= self.after_failures:
            return Decision(ActionKind.ROLLBACK, reason="muitas falhas após um patch parcial: revertendo", diagnosis=diagnosis.to_text(), strategy=self.name)
        return None


class AutoStrategy(FixStrategy):
    """Estratégia orientada a evidências.

    ``decide`` = analisar (todos os ``analyzers``) -> diagnosticar -> percorrer
    ``planners`` em ordem até um devolver uma ``Decision``. Ambos são listas
    públicas: adicione/remova/reordene para estender o comportamento.
    """

    name = "auto"

    def __init__(
        self,
        provider: ModelProvider | None = None,
        *,
        analyzers: list[EvidenceAnalyzer] | None = None,
        planners: list[Planner] | None = None,
        use_heuristics: bool = True,
        use_tools: bool = True,
        max_tool_rounds: int = 8,
        effort_by_error: dict[str, str] | None = None,
        diagnose_hook: Callable[[Diagnosis], Awaitable[None] | None] | None = None,
    ) -> None:
        self.provider = provider
        self.effort_by_error = effort_by_error
        self.analyzers: list[EvidenceAnalyzer] = analyzers if analyzers is not None else [
            TracebackAnalyzer(), TestAnalyzer(), LogAnalyzer(), VisionAnalyzer(), MemoryAnalyzer()
        ]
        if planners is not None:
            self.planners: list[Planner] = planners
        else:
            self.planners = [ObservationPlanner(), RollbackPlanner()]
            if use_heuristics:
                self.planners.append(HeuristicPlanner())
            self.planners.append(ModelPlanner(provider, use_tools=use_tools, max_tool_rounds=max_tool_rounds))
        self._diagnose_hook = diagnose_hook
        self.last_diagnosis: Diagnosis | None = None

    # -- diagnóstico
    def diagnose(self, ctx: FailureContext) -> Diagnosis:
        findings: list[Finding] = []
        for analyzer in self.analyzers:
            try:
                findings.extend(analyzer.analyze(ctx))
            except Exception as exc:  # um analisador quebrado não invalida os outros
                findings.append(Finding(getattr(analyzer, "name", "analyzer"), f"analisador falhou: {exc}", 0.1))
        diagnosis = Diagnosis(findings=findings)
        if findings:
            top = max(findings, key=lambda f: f.severity)
            diagnosis.primary_cause = top.summary
        else:
            diagnosis.primary_cause = "falha sem evidências estruturadas"
        # Necessidades: o que ainda falta para decidir bem.
        has_tb = any(f.source == "traceback" for f in findings)
        has_vision = any(f.source == "vision" for f in findings)
        if not has_tb and ctx.vision_available and not has_vision:
            diagnosis.needs.add("observe_again")  # sem traceback: só a tela pode explicar
        if ctx.failing_file is not None:
            diagnosis.needs.add("code")
        if ctx.tests is not None and not ctx.tests.success:
            diagnosis.needs.add("tests")
        return diagnosis

    def effort_for(self, ctx: FailureContext) -> str | None:
        """Esforço do modelo para esta falha (mapa próprio, senão o do contexto)."""
        if self.effort_by_error:
            key = ctx.result.signature.split("@", 1)[0].split(":", 1)[0]
            return self.effort_by_error.get(key, self.effort_by_error.get("default"))
        return ctx.effort

    async def decide(self, ctx: FailureContext) -> Decision:
        ctx.effort = self.effort_for(ctx)
        diagnosis = self.diagnose(ctx)
        self.last_diagnosis = diagnosis
        if self._diagnose_hook is not None:
            maybe = self._diagnose_hook(diagnosis)
            if maybe is not None:
                await maybe
        for planner in self.planners:
            try:
                decision = await planner.plan(ctx, diagnosis)
            except ProviderInterrupted:
                raise
            except Exception as exc:
                logger.error("planner %s falhou: %s", getattr(planner, "name", planner), exc)
                continue
            if decision is not None:
                if not decision.diagnosis:
                    decision.diagnosis = diagnosis.to_text()
                return decision
        return Decision.finish("no_fix", diagnosis=diagnosis.to_text(), strategy=self.name)

    async def propose(self, ctx: FailureContext) -> FixProposal | None:
        """Compatibilidade com o contrato clássico: devolve só o patch, se houver."""
        decision = await self.decide(ctx)
        return decision.proposal
