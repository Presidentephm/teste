"""
Estratégias de correção: quem "pensa" o patch a partir de uma falha.

O loop do agente é agnóstico a *como* a correção é formulada. Ele entrega um
``FailureContext`` (traceback parseado, fonte do arquivo, análise AST,
histórico de tentativas) e espera de volta um ``FixProposal`` (lista de
``FilePatch`` + justificativa + confiança) ou ``None`` ("não sei corrigir").

Estratégias incluídas:

* ``HeuristicFixStrategy`` - regras determinísticas, 100% offline, para as
  falhas mais mecânicas (import faltando, tab/espaço, nome definido em outro
  módulo do projeto). Rápida, barata e previsível.
* ``ClaudeFixStrategy`` - usa o SDK oficial ``anthropic`` para pedir ao modelo
  uma correção estruturada em JSON. É o "cérebro" de propósito geral.
* ``CompositeFixStrategy`` - encadeia estratégias: tenta a heurística
  primeiro e cai para o LLM só quando necessário.

A interface é assíncrona e o contexto é um dataclass simples, então é trivial
plugar outras fontes (outro LLM, um humano no loop, entradas multimodais como
screenshots de erro adicionadas ao ``FailureContext.attachments``).
"""

from __future__ import annotations

import json
import logging
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .code_manager import FilePatch, ModuleAnalysis, Replacement
from .config import AgentConfig
from .sandbox import ExecutionResult

logger = logging.getLogger("agent_core.strategies")


# --------------------------------------------------------------------- modelos
@dataclass
class AttemptSummary:
    """Resumo de uma tentativa anterior, para o modelo não repetir o mesmo erro."""

    attempt: int
    error_signature: str
    rationale: str
    patched_files: list[str]
    outcome: str  # "fixed" | "new_error" | "same_error" | "rolled_back"


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
    attachments: list[Any] = field(default_factory=list)  # gancho multimodal (imagens, logs extras)

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


# ------------------------------------------------------------------ interface
class FixStrategy(ABC):
    """Contrato de uma estratégia de correção."""

    name: str = "base"

    @abstractmethod
    async def propose(self, ctx: FailureContext) -> FixProposal | None:
        """Devolve uma proposta de patch ou ``None`` se não souber corrigir."""


# ----------------------------------------------------------------- heurística
_NAME_ERROR_RE = re.compile(r"name '(?P<name>[A-Za-z_]\w*)' is not defined")


class HeuristicFixStrategy(FixStrategy):
    """Correções mecânicas sem LLM.

    Regras (em ordem):
        1. ``NameError`` para módulo da stdlib      -> ``import <mod>``
        2. ``NameError`` para símbolo de outro módulo do projeto
                                                    -> ``from <mod> import <sym>``
        3. ``TabError``/``IndentationError`` com tabs -> converte tabs em 4 espaços
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
            return None

        new_source = insert_import(ctx.failing_source, import_line)
        return FixProposal(
            patches=[FilePatch(path=ctx.failing_file, content=new_source, reason=f"heuristic: {import_line}")],
            rationale=f"NameError para '{name}': adicionado '{import_line}' no topo do módulo.",
            confidence=0.9 if name in self.STDLIB else 0.75,
            strategy=self.name,
        )

    # -- regra 3
    def _fix_tabs(self, ctx: FailureContext) -> FixProposal:
        new_source = "\n".join(line.expandtabs(4) for line in ctx.failing_source.splitlines())
        if ctx.failing_source.endswith("\n"):
            new_source += "\n"
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


# -------------------------------------------------------------------- Claude
_SYSTEM_PROMPT = """\
Você é o módulo de auto-correção de um agente autônomo que reescreve o próprio \
código-fonte Python. Receberá um traceback, o arquivo que falhou (com números de \
linha), o esqueleto do projeto e o histórico de tentativas anteriores.

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


class ClaudeFixStrategy(FixStrategy):
    """Formula correções usando o SDK oficial ``anthropic``.

    Requer ``pip install anthropic`` e credenciais no ambiente
    (``ANTHROPIC_API_KEY`` ou perfil de ``ant auth login``). O import é
    preguiçoso para que o núcleo funcione sem o SDK instalado quando apenas a
    estratégia heurística é usada.
    """

    name = "claude"

    def __init__(self, config: AgentConfig, client: Any | None = None) -> None:
        self.config = config
        self._client = client  # permite injetar um cliente falso em testes

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic  # import tardio proposital
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "ClaudeFixStrategy requer o pacote 'anthropic' (pip install anthropic)."
                ) from exc
            self._client = anthropic.AsyncAnthropic()
        return self._client

    # -- prompt
    @staticmethod
    def _numbered(source: str) -> str:
        return "\n".join(f"{i:4d} | {line}" for i, line in enumerate(source.splitlines(), 1))

    def build_prompt(self, ctx: FailureContext) -> str:
        parts = [f"## Script executado\n{ctx.script}"]
        parts.append(f"## Resultado\n{ctx.result.summary()}")
        if ctx.result.stderr.strip():
            parts.append(f"## stderr (traceback)\n```\n{ctx.result.stderr.strip()[-6000:]}\n```")
        if ctx.result.stdout.strip():
            parts.append(f"## stdout (últimas linhas)\n```\n{ctx.result.stdout.strip()[-2000:]}\n```")
        if ctx.failing_file and ctx.failing_source is not None:
            parts.append(
                f"## Arquivo que falhou: {ctx.failing_file}\n```python\n{self._numbered(ctx.failing_source)}\n```"
            )
        if ctx.project_outline:
            outline = "\n\n".join(a.outline() for a in ctx.project_outline.values())
            parts.append(f"## Esqueleto do projeto\n```\n{outline[:8000]}\n```")
        if ctx.history:
            hist = "\n".join(
                f"- tentativa {h.attempt}: {h.rationale} -> {h.outcome} (arquivos: {', '.join(h.patched_files)})"
                for h in ctx.history
            )
            parts.append(f"## Tentativas anteriores (não repita)\n{hist}")
        parts.append(f"## Iteração atual\n{ctx.attempt}")
        return "\n\n".join(parts)

    # -- chamada
    async def propose(self, ctx: FailureContext) -> FixProposal | None:
        client = self._get_client()
        prompt = self.build_prompt(ctx)
        kwargs: dict[str, Any] = dict(
            model=self.config.llm_model,
            max_tokens=self.config.llm_max_tokens,
            system=_SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config={"effort": self.config.llm_effort},
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            # Streaming evita timeouts em respostas longas; get_final_message()
            # devolve a mensagem completa quando não precisamos dos eventos.
            if self.config.llm_enable_fallbacks:
                # Fallback server-side: se o modelo recusar por política, a API
                # reexecuta o mesmo pedido num modelo alternativo na mesma chamada.
                async with client.beta.messages.stream(
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    **kwargs,
                ) as stream:
                    message = await stream.get_final_message()
            else:
                async with client.messages.stream(**kwargs) as stream:
                    message = await stream.get_final_message()
        except Exception as exc:  # erros de rede/API não podem derrubar o loop
            logger.error("Falha ao consultar o modelo: %s", exc)
            return None

        if message.stop_reason == "refusal":
            logger.warning("O modelo recusou a solicitação de correção.")
            return None
        if message.stop_reason == "max_tokens":
            logger.warning("Resposta truncada por max_tokens; aumente llm_max_tokens.")

        text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
        return self.parse_response(text)

    # -- parse
    def parse_response(self, text: str) -> FixProposal | None:
        """Converte o JSON devolvido pelo modelo num ``FixProposal``."""
        data = _extract_json(text)
        if not isinstance(data, dict):
            logger.error("Resposta do modelo não é JSON válido: %r", text[:300])
            return None
        patches: list[FilePatch] = []
        for raw in data.get("patches", []) or []:
            path = raw.get("path")
            if not path:
                continue
            if raw.get("mode") == "replace_full" or "content" in raw:
                patches.append(FilePatch(path=path, content=raw.get("content", ""), reason="claude:replace_full"))
            else:
                reps = [
                    Replacement(search=r["search"], replace=r.get("replace", ""), count=int(r.get("count", 1)))
                    for r in raw.get("replacements", [])
                    if isinstance(r, dict) and r.get("search")
                ]
                if reps:
                    patches.append(FilePatch(path=path, replacements=reps, reason="claude:search_replace"))
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
            except Exception as exc:
                logger.error("Estratégia %s falhou: %s", strategy.name, exc)
                continue
            if proposal is not None:
                logger.info("Proposta obtida via '%s' (confiança %.2f)", strategy.name, proposal.confidence)
                return proposal
        return None
