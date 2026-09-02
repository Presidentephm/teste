"""
Captura contínua desacoplada do AgentLoop.

``VisionCapture`` lê frames de um ``VisualSource`` numa task asyncio (a
leitura bloqueante roda em thread), passa cada frame pelo ``VisualPipeline``
e publica observações quando há mudança relevante ou quando o intervalo de
observação expira. Frames relevantes podem ser gravados em disco.

Garantias:
    * ``start()`` nunca levanta por dispositivo indisponível: o erro fica em
      ``status()['error']`` e ``is_running`` é ``False``.
    * ``stop()`` encerra a task e fecha a fonte, idempotente.
    * ``snapshot()`` faz uma captura única sob demanda (mesmo sem ``start``).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..observations import Observation, ObservationKind, Observer
from .frames import Frame, VisionUnavailableError, is_available, require_cv2
from .processing import OCREngine, VisualAnalyzer, VisualPipeline
from .sources import VisualSource, open_source

logger = logging.getLogger("agent_core.vision")

ObservationHook = Callable[[Observation], Awaitable[None] | None]


class VisionCapture:
    def __init__(
        self,
        source: VisualSource,
        pipeline: VisualPipeline | None = None,
        *,
        fps: float = 2.0,
        observation_interval: float = 5.0,
        store_dir: Path | None = None,
        max_stored: int = 50,
        history: int = 20,
        on_observation: ObservationHook | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps deve ser > 0")
        self.source = source
        self.pipeline = pipeline or VisualPipeline()
        self.fps = fps
        self.observation_interval = observation_interval
        self.store_dir = Path(store_dir) if store_dir else None
        self.max_stored = max_stored
        self._on_observation = on_observation
        self._history: deque[Observation] = deque(maxlen=history)
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()
        self._first_ready = asyncio.Event()
        self.error: str | None = None
        self.frames_read = 0
        self.frames_failed = 0
        self.observations_emitted = 0
        self.stored = 0
        self._last_emit = 0.0

    # -- estado
    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def latest(self) -> Observation | None:
        return self._history[-1] if self._history else None

    async def wait_latest(self, timeout: float = 3.0) -> Observation | None:
        """Última observação; se ainda não houver, espera a primeira até ``timeout``."""
        if not self._history and self.is_running:
            try:
                await asyncio.wait_for(self._first_ready.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return None
        return self.latest()

    def history(self) -> list[Observation]:
        return list(self._history)

    def status(self) -> dict[str, Any]:
        return {
            "source": self.source.name,
            "running": self.is_running,
            "error": self.error,
            "frames_read": self.frames_read,
            "frames_failed": self.frames_failed,
            "observations": self.observations_emitted,
            "stored": self.stored,
            "source_errors": self.source.errors,
        }

    # -- ciclo
    async def start(self) -> bool:
        """Inicia a captura em background. Devolve ``False`` se a fonte não abriu."""
        if self.is_running:
            return True
        if not is_available():
            self.error = "OpenCV indisponível"
            logger.warning("visão desativada: %s", self.error)
            return False
        try:
            await asyncio.to_thread(self.source.open)
        except VisionUnavailableError as exc:
            self.error = str(exc)
            logger.warning("visão desativada: %s", exc)
            return False
        except Exception as exc:  # qualquer outro erro do driver
            self.error = f"{type(exc).__name__}: {exc}"
            logger.warning("visão desativada: %s", self.error)
            return False
        self.error = None
        self._stop = asyncio.Event()
        self._first_ready = asyncio.Event()
        self.pipeline.reset()
        self._task = asyncio.create_task(self._run(), name="vision-capture")
        return True

    async def _run(self) -> None:
        period = 1.0 / self.fps
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                frame = await asyncio.to_thread(self.source.read)
                if frame is None:
                    self.frames_failed += 1
                    if self.source.errors and self.source.errors % 25 == 0:
                        logger.warning("%s: %d erros consecutivos de leitura", self.source.name, self.source.errors)
                else:
                    self.frames_read += 1
                    await self._handle(frame)
                elapsed = time.monotonic() - started
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=max(0.0, period - elapsed))
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # nunca deixa a task morrer silenciosamente
            self.error = f"{type(exc).__name__}: {exc}"
            logger.error("captura visual encerrada por erro: %s", self.error)

    async def _handle(self, frame: Frame) -> None:
        observation = await asyncio.to_thread(self.pipeline.process, frame)
        now = time.monotonic()
        due = (now - self._last_emit) >= self.observation_interval
        if observation.metadata.get("changed") or due or not self._history:
            await self._emit(observation, frame)

    async def _emit(self, observation: Observation, frame: Frame | None) -> None:
        async with self._lock:
            self._history.append(observation)
            self.observations_emitted += 1
            self._last_emit = time.monotonic()
            self._first_ready.set()
            if frame is not None and self.store_dir and observation.metadata.get("changed") and self.stored < self.max_stored:
                path = await asyncio.to_thread(self._store, frame)
                if path:
                    observation.metadata["stored_path"] = str(path)
        if self._on_observation is not None:
            try:
                result = self._on_observation(observation)
                if result is not None:
                    await result
            except Exception as exc:
                logger.warning("hook de observação falhou: %s", exc)

    def _store(self, frame: Frame) -> Path | None:
        try:
            cv = require_cv2()
            self.store_dir.mkdir(parents=True, exist_ok=True)
            path = self.store_dir / f"{frame.source.replace(':', '_')}_{int(frame.timestamp * 1000)}_{frame.index}.jpg"
            if cv.imwrite(str(path), frame.image):
                self.stored += 1
                return path
        except Exception as exc:
            logger.warning("não foi possível gravar frame: %s", exc)
        return None

    async def snapshot(self) -> Observation | None:
        """Captura e processa um único frame agora (ou ``None`` se indisponível)."""
        if not is_available():
            self.error = "OpenCV indisponível"
            return None
        try:
            frame = await asyncio.to_thread(self.source.read)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return None
        if frame is None:
            self.frames_failed += 1
            if self.error is None and self.source.errors:
                self.error = "fonte visual sem frames"
            return None
        self.frames_read += 1
        observation = await asyncio.to_thread(self.pipeline.process, frame)
        await self._emit(observation, frame)
        return observation

    async def stop(self) -> None:
        self._stop.set()
        task = self._task
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            self._task = None
        await asyncio.to_thread(self.source.close)

    async def __aenter__(self) -> "VisionCapture":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()


class VisionObserver(Observer):
    """Adapta ``VisionCapture`` ao contrato ``Observer`` do contexto multimodal."""

    name = "vision"
    kind = ObservationKind.VISION

    def __init__(self, capture: VisionCapture, *, fresh: bool = True, wait_timeout: float = 3.0) -> None:
        self.capture = capture
        self.fresh = fresh
        self.wait_timeout = wait_timeout

    async def observe(self, **_: Any) -> list[Observation]:
        try:
            if self.fresh or not self.capture.is_running:
                obs = await self.capture.snapshot()
            else:
                obs = await self.capture.wait_latest(self.wait_timeout)
        except Exception as exc:  # jamais derruba o loop
            logger.warning("observação visual falhou: %s", exc)
            return []
        return [obs] if obs is not None else []


def build_vision_capture(config: Any, *, on_observation: ObservationHook | None = None) -> VisionCapture:
    """Constrói ``VisionCapture`` a partir de ``AgentConfig``. Pode levantar
    ``VisionUnavailableError``/``ValueError`` em configuração inválida; erros
    de dispositivo só aparecem em ``start()``/``snapshot()``."""
    source = open_source(
        config.vision_source,
        camera_index=config.vision_camera_index,
        monitor=config.vision_monitor,
        images=list(config.vision_images),
    )
    store = (config.backup_dir / "frames") if config.vision_store_frames else None
    ocr = OCREngine() if getattr(config, "vision_ocr", True) else OCREngine.disabled()
    return VisionCapture(
        source,
        VisualPipeline(analyzer=VisualAnalyzer(ocr=ocr)),
        fps=config.vision_fps,
        observation_interval=config.observation_interval,
        store_dir=store,
        on_observation=on_observation,
    )
