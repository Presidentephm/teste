"""
Contexto multimodal: o contrato CAPTURE -> PROCESS -> CONTEXT -> AGENT.

* ``Observation``: unidade atômica de evidência (código, runtime, teste, log,
  visão, resposta do modelo...). Pode carregar uma imagem (``ImageData``) e
  informação já extraída/estruturada (``extracted``) com uma confiança.
* ``MultimodalContext``: coleção de observações com limites (quantidade,
  imagens, caracteres), serializável, capaz de se converter em partes de
  mensagem para o provider (``to_parts``) sem depender do SDK.
* ``Observer``: contrato de quem produz observações. Implementações para
  código, runtime, testes e logs vivem aqui; a visual vive em ``vision``.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from .providers import ContentPart
from .safety import redact


class ObservationKind(str, Enum):
    CODE = "code"
    RUNTIME = "runtime"
    TEST = "test"
    LOG = "log"
    VISION = "vision"
    MODEL = "model"
    MEMORY = "memory"
    OTHER = "other"


@dataclass
class ImageData:
    """Imagem codificada (JPEG/PNG) pronta para ser enviada ao modelo."""

    data: bytes
    media_type: str = "image/jpeg"
    width: int = 0
    height: int = 0

    def to_base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    @property
    def size_bytes(self) -> int:
        return len(self.data)


@dataclass
class Observation:
    """Uma evidência observada pelo agente."""

    kind: ObservationKind
    source: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    extracted: dict[str, Any] = field(default_factory=dict)
    image: ImageData | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        self.kind = ObservationKind(self.kind)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.summary = redact(self.summary)

    @property
    def has_image(self) -> bool:
        return self.image is not None and self.image.size_bytes > 0

    def to_dict(self, *, include_image: bool = False) -> dict[str, Any]:
        d = {
            "id": self.id,
            "kind": self.kind.value,
            "source": self.source,
            "summary": self.summary,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "data": _jsonable(self.data),
            "extracted": _jsonable(self.extracted),
            "metadata": _jsonable(self.metadata),
        }
        if self.image is not None:
            d["image"] = {
                "media_type": self.image.media_type,
                "width": self.image.width,
                "height": self.image.height,
                "size_bytes": self.image.size_bytes,
            }
            if include_image:
                d["image"]["base64"] = self.image.to_base64()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Observation":
        image = None
        if d.get("image") and d["image"].get("base64"):
            im = d["image"]
            image = ImageData(base64.b64decode(im["base64"]), im.get("media_type", "image/jpeg"), im.get("width", 0), im.get("height", 0))
        return cls(
            kind=ObservationKind(d["kind"]),
            source=d["source"],
            summary=d["summary"],
            data=d.get("data", {}),
            extracted=d.get("extracted", {}),
            image=image,
            confidence=d.get("confidence", 1.0),
            metadata=d.get("metadata", {}),
            timestamp=d.get("timestamp", time.time()),
            id=d.get("id", uuid.uuid4().hex[:12]),
        )

    def to_prompt_text(self, max_chars: int = 4000) -> str:
        """Representação textual compacta para prompts."""
        head = f"[{self.kind.value} | {self.source} | conf={self.confidence:.2f}] {self.summary}"
        body_parts: list[str] = []
        for key, value in self.extracted.items():
            text = value if isinstance(value, str) else json.dumps(_jsonable(value), ensure_ascii=False)
            body_parts.append(f"{key}: {text}")
        body = "\n".join(body_parts)
        text = head + ("\n" + body if body else "")
        return redact(text[:max_chars] + ("…" if len(text) > max_chars else ""))


def _jsonable(value: Any) -> Any:
    """Torna um valor serializável (dataclasses, Paths, bytes, sets...)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return {k: _jsonable(getattr(value, k)) for k in value.__dataclass_fields__}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# --------------------------------------------------------------------- contexto
@dataclass
class ContextLimits:
    max_observations: int = 40
    max_images: int = 4
    max_text_chars: int = 40000
    max_image_bytes: int = 4 * 1024 * 1024


class MultimodalContext:
    """Coleção ordenada de observações com limites e conversão para o provider."""

    def __init__(self, limits: ContextLimits | None = None, observations: Iterable[Observation] = ()):
        self.limits = limits or ContextLimits()
        self._observations: list[Observation] = []
        for obs in observations:
            self.add(obs)

    # -- coleção
    def add(self, observation: Observation) -> Observation:
        if observation.image is not None and observation.image.size_bytes > self.limits.max_image_bytes:
            observation.metadata["image_dropped"] = "excede max_image_bytes"
            observation.image = None
        self._observations.append(observation)
        overflow = len(self._observations) - self.limits.max_observations
        if overflow > 0:
            # Descarta as mais antigas, preservando a ordem cronológica.
            del self._observations[:overflow]
        return observation

    def extend(self, observations: Iterable[Observation]) -> None:
        for obs in observations:
            self.add(obs)

    @property
    def observations(self) -> list[Observation]:
        return list(self._observations)

    def __len__(self) -> int:
        return len(self._observations)

    def by_kind(self, kind: ObservationKind | str) -> list[Observation]:
        kind = ObservationKind(kind)
        return [o for o in self._observations if o.kind == kind]

    def latest(self, kind: ObservationKind | str) -> Observation | None:
        items = self.by_kind(kind)
        return items[-1] if items else None

    def images(self) -> list[Observation]:
        """Observações com imagem, as mais recentes primeiro, limitadas."""
        with_img = [o for o in reversed(self._observations) if o.has_image]
        return with_img[: self.limits.max_images]

    # -- serialização
    def to_dict(self, *, include_images: bool = False) -> dict[str, Any]:
        return {
            "limits": self.limits.__dict__,
            "observations": [o.to_dict(include_image=include_images) for o in self._observations],
        }

    def to_json(self, *, include_images: bool = False, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(include_images=include_images), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MultimodalContext":
        limits = ContextLimits(**d.get("limits", {}))
        return cls(limits, (Observation.from_dict(o) for o in d.get("observations", [])))

    # -- saída para o modelo
    def to_text(self) -> str:
        """Todas as observações como texto, respeitando ``max_text_chars``."""
        budget = self.limits.max_text_chars
        chunks: list[str] = []
        # As mais recentes têm prioridade: percorremos de trás para frente.
        for obs in reversed(self._observations):
            text = obs.to_prompt_text()
            if len(text) + 2 > budget:
                if budget > 200:
                    chunks.append(text[: budget - 2] + "…")
                break
            chunks.append(text)
            budget -= len(text) + 2
        return "\n\n".join(reversed(chunks))

    def to_parts(self) -> list[ContentPart]:
        """Partes de mensagem (texto + imagens) para ``ModelRequest``."""
        parts: list[ContentPart] = [ContentPart.from_text(self.to_text())]
        for obs in reversed(self.images()):  # ordem cronológica
            parts.append(ContentPart.from_text(f"[imagem: {obs.source} @ {obs.timestamp:.0f}] {obs.summary}"))
            parts.append(ContentPart.from_image(obs.image.data, obs.image.media_type))
        return parts

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for obs in self._observations:
            counts[obs.kind.value] = counts.get(obs.kind.value, 0) + 1
        imgs = sum(1 for o in self._observations if o.has_image)
        return ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) + f", imagens={imgs}"


# -------------------------------------------------------------------- observers
class Observer(ABC):
    """Produz observações a partir de uma fonte."""

    name: str = "observer"
    kind: ObservationKind = ObservationKind.OTHER

    @abstractmethod
    async def observe(self, **scope: Any) -> list[Observation]:
        """Devolve zero ou mais observações. Nunca deve levantar por falha da fonte."""


class RuntimeObserver(Observer):
    """Converte um ``ExecutionResult`` do sandbox numa observação."""

    name = "runtime"
    kind = ObservationKind.RUNTIME

    async def observe(self, *, result: Any = None, label: str = "script", **_: Any) -> list[Observation]:
        if result is None:
            return []
        tb = getattr(result, "traceback", None)
        extracted: dict[str, Any] = {"returncode": result.returncode, "timed_out": result.timed_out}
        if tb is not None:
            loc = tb.location
            extracted.update(
                {
                    "exception": tb.exc_type,
                    "message": tb.message,
                    "file": loc.file if loc else None,
                    "line": loc.line if loc else None,
                    "function": loc.function if loc else None,
                }
            )
        if result.stderr.strip():
            extracted["stderr_tail"] = redact(result.stderr.strip()[-3000:])
        if result.stdout.strip():
            extracted["stdout_tail"] = redact(result.stdout.strip()[-1500:])
        return [
            Observation(
                kind=self.kind,
                source=label,
                summary=result.summary(),
                data={"signature": result.signature, "duration": result.duration},
                extracted=extracted,
                confidence=1.0,
            )
        ]


class TestObserver(Observer):
    """Executa (ou recebe) um resultado de testes e o converte em observação."""

    name = "tests"
    kind = ObservationKind.TEST

    def __init__(self, sandbox: Any, command: tuple[str, ...] | None) -> None:
        self.sandbox = sandbox
        self.command = tuple(command) if command else None

    async def run(self) -> Any | None:
        if not self.command:
            return None
        cmd = [self.sandbox.config.python_executable, *self.command]
        return await self.sandbox.run_command(cmd, timeout=self.sandbox.config.sandbox_timeout * 4)

    async def observe(self, *, result: Any = None, **_: Any) -> list[Observation]:
        if result is None:
            result = await self.run()
        if result is None:
            return []
        stderr = result.stderr.strip()
        failed = [l for l in stderr.splitlines() if l.startswith(("FAIL:", "ERROR:"))]
        extracted: dict[str, Any] = {"passed": result.success, "failed_tests": failed[:20]}
        if stderr:
            extracted["output_tail"] = redact(stderr[-3000:])
        summary = "testes OK" if result.success else f"testes falharam ({len(failed)} casos identificados)"
        return [
            Observation(kind=self.kind, source="tests", summary=summary, data={"returncode": result.returncode}, extracted=extracted)
        ]


class LogObserver(Observer):
    """Lê o final de arquivos de log do projeto."""

    name = "logs"
    kind = ObservationKind.LOG

    def __init__(self, root: Path, patterns: tuple[str, ...] = ("*.log", "logs/*.log"), tail_chars: int = 3000) -> None:
        self.root = Path(root)
        self.patterns = patterns
        self.tail_chars = tail_chars

    async def observe(self, **_: Any) -> list[Observation]:
        observations: list[Observation] = []
        for pattern in self.patterns:
            for path in sorted(self.root.glob(pattern)):
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                tail = text[-self.tail_chars :]
                interesting = [l for l in tail.splitlines() if any(k in l.lower() for k in ("error", "exception", "traceback", "critical", "warning"))]
                observations.append(
                    Observation(
                        kind=self.kind,
                        source=path.relative_to(self.root).as_posix(),
                        summary=f"{len(interesting)} linhas relevantes em {path.name}",
                        extracted={"relevant_lines": [redact(l) for l in interesting[-30:]], "tail": redact(tail[-1500:])},
                        confidence=0.8,
                    )
                )
        return observations


class CodeObserver(Observer):
    """Esqueleto do projeto e fonte do arquivo que falhou."""

    name = "code"
    kind = ObservationKind.CODE

    def __init__(self, code_manager: Any) -> None:
        self.code = code_manager

    async def observe(self, *, failing_file: str | None = None, **_: Any) -> list[Observation]:
        observations: list[Observation] = []
        try:
            outline = await self.code.analyze_project()
        except Exception as exc:  # leitura do projeto não pode derrubar o ciclo
            return [Observation(kind=self.kind, source="project", summary=f"falha ao analisar projeto: {exc}", confidence=0.1)]
        observations.append(
            Observation(
                kind=self.kind,
                source="project",
                summary=f"{len(outline)} módulos analisados",
                extracted={"outline": "\n\n".join(a.outline() for a in outline.values())[:8000]},
            )
        )
        if failing_file:
            try:
                source = await self.code.read(failing_file)
                numbered = "\n".join(f"{i:4d} | {l}" for i, l in enumerate(source.splitlines(), 1))
                observations.append(
                    Observation(kind=self.kind, source=failing_file, summary="arquivo onde a falha ocorreu", extracted={"source": numbered})
                )
            except Exception as exc:
                observations.append(Observation(kind=self.kind, source=failing_file, summary=f"não foi possível ler: {exc}", confidence=0.1))
        return observations
