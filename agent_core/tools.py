"""
Caixa de ferramentas do projeto para o modelo (somente leitura).

Em vez de receber o esqueleto inteiro do projeto no prompt, o modelo recebe
um resumo compacto e ferramentas para pedir o que precisar:

    read_file(path, start_line?, end_line?)   -> linhas numeradas
    list_files(pattern)                        -> caminhos relativos
    search(query, glob?)                       -> "caminho:linha: texto"
    outline(path)                              -> esqueleto AST de um módulo
    propose_patch(rationale, confidence, patches) -> entrega a correção

Todas as leituras passam pelo ``CodeManager`` (confinamento à raiz do
projeto). Nenhuma ferramenta escreve; a escrita continua sendo feita pelo
loop com checkpoint, guarda e validação.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .code_manager import CodeManager, PathOutsideProjectError
from .providers import ToolCall, ToolSpec
from .safety import redact

PATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string", "description": "causa raiz e correção, em uma ou duas frases"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "patches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "mode": {"type": "string", "enum": ["search_replace", "replace_full"]},
                    "content": {"type": "string", "description": "arquivo inteiro (replace_full)"},
                    "replacements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"search": {"type": "string"}, "replace": {"type": "string"}},
                            "required": ["search", "replace"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["path", "mode"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["rationale", "confidence", "patches"],
    "additionalProperties": False,
}


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


class ProjectToolbox:
    """Executa as ferramentas de leitura sobre o projeto."""

    MAX_LINES = 400
    MAX_FILES = 200
    MAX_MATCHES = 60

    def __init__(self, code: CodeManager) -> None:
        self.code = code
        self.calls: list[ToolCall] = []
        self.proposal: dict[str, Any] | None = None

    # -- especificações
    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                "read_file",
                "Lê um arquivo do projeto com números de linha. Use start_line/end_line para trechos.",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "caminho relativo à raiz do projeto"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
            ToolSpec(
                "list_files",
                "Lista arquivos do projeto que casam com um glob (padrão: todos os .py).",
                {"type": "object", "properties": {"pattern": {"type": "string"}}, "additionalProperties": False},
            ),
            ToolSpec(
                "search",
                "Procura um texto literal nos arquivos do projeto e devolve caminho:linha: conteúdo.",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "glob": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            ToolSpec(
                "outline",
                "Esqueleto (imports, funções, classes) de um módulo Python.",
                {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False},
            ),
            ToolSpec(
                "propose_patch",
                "Entrega a correção final. Chame exatamente uma vez, quando tiver certeza da causa raiz.",
                PATCH_SCHEMA,
            ),
        ]

    # -- execução
    async def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        handler = getattr(self, f"_tool_{call.name}", None)
        if handler is None:
            return ToolResult(f"ferramenta desconhecida: {call.name}", is_error=True)
        try:
            return await handler(**call.input)
        except PathOutsideProjectError as exc:
            return ToolResult(f"acesso negado: {exc}", is_error=True)
        except FileNotFoundError as exc:
            return ToolResult(f"arquivo não encontrado: {exc}", is_error=True)
        except TypeError as exc:  # argumentos inesperados
            return ToolResult(f"argumentos inválidos: {exc}", is_error=True)
        except Exception as exc:
            return ToolResult(f"erro ao executar {call.name}: {type(exc).__name__}: {exc}", is_error=True)

    async def _tool_read_file(self, path: str, start_line: int | None = None, end_line: int | None = None) -> ToolResult:
        source = await self.code.read(path)
        lines = source.splitlines()
        start = max(1, int(start_line or 1))
        end = min(len(lines), int(end_line or len(lines)))
        capped = False
        if end - start + 1 > self.MAX_LINES:
            end = start + self.MAX_LINES - 1
            capped = True
        chunk = "\n".join(f"{i:4d} | {lines[i - 1]}" for i in range(start, end + 1))
        note = f"\n… ({len(lines) - end} linhas restantes; peça outro trecho)" if capped else ""
        return ToolResult(redact(chunk + note) if chunk else "(arquivo vazio)")

    async def _tool_list_files(self, pattern: str = "**/*.py") -> ToolResult:
        root = self.code.root
        found = []
        for p in sorted(root.glob(pattern)):
            if not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            if "__pycache__" in rel or any(rel == x or rel.startswith(x.rstrip("/") + "/") for x in self.code.config.all_protected_paths):
                continue
            found.append(rel)
            if len(found) >= self.MAX_FILES:
                found.append("… (lista truncada)")
                break
        return ToolResult("\n".join(found) or "(nenhum arquivo)")

    async def _tool_search(self, query: str, glob: str = "**/*.py") -> ToolResult:
        if not query:
            return ToolResult("query vazia", is_error=True)
        root = self.code.root
        hits: list[str] = []
        for p in sorted(root.glob(glob)):
            if not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            if "__pycache__" in rel or any(rel == x or rel.startswith(x.rstrip("/") + "/") for x in self.code.config.all_protected_paths):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if query in line:
                    hits.append(f"{rel}:{i}: {line.strip()[:160]}")
                    if len(hits) >= self.MAX_MATCHES:
                        hits.append("… (resultados truncados)")
                        return ToolResult(redact("\n".join(hits)))
        return ToolResult(redact("\n".join(hits)) or "(sem resultados)")

    async def _tool_outline(self, path: str) -> ToolResult:
        analysis = await self.code.analyze(path)
        return ToolResult(analysis.outline())

    async def _tool_propose_patch(self, rationale: str = "", confidence: float = 0.5, patches: list | None = None) -> ToolResult:
        self.proposal = {"rationale": rationale, "confidence": confidence, "patches": patches or []}
        return ToolResult("proposta registrada" if patches else "proposta sem patches registrada")

    def proposal_json(self) -> str | None:
        return json.dumps(self.proposal, ensure_ascii=False) if self.proposal is not None else None


def summarize_tool_input(call: ToolCall) -> str:
    """Resumo de uma chamada para logs (sem despejar conteúdos grandes)."""
    compact = {k: (v if not isinstance(v, str) or len(v) < 60 else v[:57] + "…") for k, v in call.input.items() if k != "patches"}
    if "patches" in call.input:
        compact["patches"] = [p.get("path") for p in call.input["patches"] if isinstance(p, dict)]
    payload = re.sub(r"\s+", " ", json.dumps(compact, ensure_ascii=False))
    return f"{call.name}({payload})"
