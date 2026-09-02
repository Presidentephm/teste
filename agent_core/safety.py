"""
Proteções transversais: redação de credenciais e guarda de patches.

* ``redact()`` remove chaves/tokens de qualquer texto que vá para logs,
  memória ou prompts.
* ``PatchGuard`` rejeita conjuntos de patches perigosos antes que cheguem ao
  ``CodeManager``: muitos arquivos de uma vez, esvaziamento de arquivos,
  remoção massiva de linhas, caminhos suspeitos.
* ``RedactingFormatter`` aplica ``redact`` em toda linha de log.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable

_SECRET_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),                     # chaves Anthropic
    re.compile(r"sk-[A-Za-z0-9]{20,}"),                            # chaves genéricas sk-
    re.compile(r"AKIA[0-9A-Z]{16}"),                               # AWS access key
    re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]{16,}=*"),         # tokens bearer
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password|passwd)\b(\s*[=:]\s*)(['\"]?)([^\s'\"]{6,})"),
]


def redact(text: str) -> str:
    """Substitui credenciais reconhecíveis por ``[REDACTED]``."""
    if not text:
        return text
    out = text
    out = _SECRET_PATTERNS[0].sub("[REDACTED]", out)
    out = _SECRET_PATTERNS[1].sub("[REDACTED]", out)
    out = _SECRET_PATTERNS[2].sub("[REDACTED]", out)
    out = _SECRET_PATTERNS[3].sub("Bearer [REDACTED]", out)
    out = _SECRET_PATTERNS[4].sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}[REDACTED]", out)
    return out


class RedactingFormatter(logging.Formatter):
    """Formatter de logging que redige credenciais da mensagem final."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


class UnsafePatchError(ValueError):
    """Conjunto de patches rejeitado pela guarda de segurança."""


@dataclass
class PatchGuard:
    """Regras estáticas aplicadas a um conjunto de patches.

    Attributes:
        max_files: máximo de arquivos alterados por decisão.
        max_removed_ratio: fração máxima de linhas que um patch de substituição
            total pode remover de um arquivo existente (protege contra o modelo
            "corrigir" apagando o código).
        min_remaining_lines: patches que deixem um arquivo com menos linhas
            que isto (quando o original era maior) são rejeitados.
    """

    max_files: int = 8
    max_removed_ratio: float = 0.6
    min_remaining_lines: int = 1

    def check(self, patches: Iterable, current_sources: dict[str, str]) -> None:
        """Valida patches contra os conteúdos atuais (``{path: source}``).

        Raises:
            UnsafePatchError: descrevendo a primeira regra violada.
        """
        patches = list(patches)
        if not patches:
            raise UnsafePatchError("conjunto de patches vazio")
        if len(patches) > self.max_files:
            raise UnsafePatchError(f"{len(patches)} arquivos num único patch (máximo {self.max_files})")
        seen: set[str] = set()
        for patch in patches:
            if patch.path in seen:
                raise UnsafePatchError(f"arquivo repetido no mesmo conjunto: {patch.path}")
            seen.add(patch.path)
            if patch.content is None:
                continue  # busca/substituição é validada pelo CodeManager
            original = current_sources.get(patch.path)
            if original is None:
                continue  # arquivo novo
            old_lines = original.count("\n") + 1
            new_lines = patch.content.count("\n") + 1 if patch.content else 0
            if not patch.content.strip() and original.strip():
                raise UnsafePatchError(f"patch esvazia {patch.path}")
            if old_lines >= 5 and new_lines < old_lines * (1 - self.max_removed_ratio):
                raise UnsafePatchError(
                    f"patch remove {old_lines - new_lines} de {old_lines} linhas de {patch.path} "
                    f"(limite {int(self.max_removed_ratio * 100)}%)"
                )
            if old_lines > self.min_remaining_lines and new_lines < self.min_remaining_lines:
                raise UnsafePatchError(f"patch deixa {patch.path} praticamente vazio")
