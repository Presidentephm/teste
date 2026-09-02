"""
Memória de execução do agente.

Cada ciclo do loop grava um ``MemoryEntry`` com o que foi observado, o
diagnóstico, a estratégia, a ação, o patch (assinatura + arquivos), os testes,
o resultado, se houve rollback e os erros. Com isso a estratégia consegue
responder "essa abordagem já foi tentada e falhou?" e evitar repetição.

A memória tem limite configurável (FIFO) e pode ser persistida em JSON dentro
da pasta de backups, sobrevivendo entre execuções.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

from .safety import redact


def patch_signature(patches: Iterable[Any]) -> str:
    """Hash estável do conteúdo de um conjunto de patches."""
    h = hashlib.sha256()
    for p in sorted(patches, key=lambda p: p.path):
        h.update(p.path.encode())
        if p.content is not None:
            h.update(b"\x00full\x00" + p.content.encode())
        for r in p.replacements:
            h.update(b"\x00sr\x00" + r.search.encode() + b"\x00" + r.replace.encode())
    return h.hexdigest()[:16]


@dataclass
class MemoryEntry:
    iteration: int
    timestamp: float
    observation: str            # resumo do contexto observado
    diagnosis: str              # causa provável identificada
    strategy: str               # nome da estratégia/planner que decidiu
    action: str                 # patch | observe_again | run_tests | rollback | finish | ...
    error_signature: str        # assinatura do erro enfrentado
    patch_signature: str = ""   # hash dos patches (vazio se não houve patch)
    patch_files: list[str] = field(default_factory=list)
    tests: str = ""             # resumo dos testes (ou vazio)
    result: str = ""            # resumo do resultado após a ação
    outcome: str = ""           # fixed | new_error | rolled_back | ...
    rollback: bool = False
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.observation = redact(self.observation)
        self.result = redact(self.result)
        self.errors = [redact(e) for e in self.errors]

    @property
    def failed(self) -> bool:
        return self.outcome not in ("fixed", "observed", "tested") or self.rollback


class AgentMemory:
    """Memória FIFO com limite e persistência opcional."""

    def __init__(self, limit: int = 100, path: Path | None = None) -> None:
        if limit < 1:
            raise ValueError("limit deve ser >= 1")
        self.limit = limit
        self.path = Path(path) if path else None
        self._entries: list[MemoryEntry] = []
        if self.path and self.path.exists():
            self.load()

    # -- coleção
    def add(self, entry: MemoryEntry) -> MemoryEntry:
        self._entries.append(entry)
        if len(self._entries) > self.limit:
            del self._entries[: len(self._entries) - self.limit]
        if self.path:
            self.save()
        return entry

    @property
    def entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        if self.path:
            self.save()

    # -- consultas
    def has_tried(self, patch_sig: str, *, failed_only: bool = True) -> bool:
        """Já houve um patch com esta assinatura (que falhou, por padrão)?"""
        return any(e.patch_signature == patch_sig and (e.failed or not failed_only) for e in self._entries if e.patch_signature)

    def failed_attempts(self, error_signature: str | None = None) -> list[MemoryEntry]:
        return [e for e in self._entries if e.failed and e.action == "patch" and (error_signature is None or e.error_signature == error_signature)]

    def count_action(self, action: str, error_signature: str | None = None) -> int:
        return sum(1 for e in self._entries if e.action == action and (error_signature is None or e.error_signature == error_signature))

    def last(self) -> MemoryEntry | None:
        return self._entries[-1] if self._entries else None

    # -- prompt
    def to_prompt_text(self, max_entries: int = 8) -> str:
        if not self._entries:
            return "sem tentativas anteriores"
        lines = []
        for e in self._entries[-max_entries:]:
            files = ", ".join(e.patch_files) or "-"
            diag = " ".join(e.diagnosis.split())[:120]
            lines.append(
                f"- it{e.iteration} [{e.strategy}/{e.action}] diag: {diag} | arquivos: {files} | "
                f"resultado: {e.outcome or e.result[:80]}{' (rollback)' if e.rollback else ''}"
            )
        return "\n".join(lines)

    # -- persistência
    def to_list(self) -> list[dict[str, Any]]:
        return [asdict(e) for e in self._entries]

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_list(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = []
        entries = []
        for d in raw:
            try:
                entries.append(MemoryEntry(**d))
            except TypeError:
                continue  # entrada de versão antiga: ignora
        self._entries = entries[-self.limit :]


def new_entry(iteration: int, **kwargs: Any) -> MemoryEntry:
    """Atalho: cria uma entrada com timestamp atual."""
    defaults = dict(observation="", diagnosis="", strategy="", action="", error_signature="")
    defaults.update(kwargs)
    return MemoryEntry(iteration=iteration, timestamp=time.time(), **defaults)
