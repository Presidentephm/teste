"""
Fontes visuais.

    VisualSource
    ├── CameraSource  (cv2.VideoCapture)
    ├── ScreenSource  (mss, quando disponível)
    └── ImageSource   (arquivos de imagem ou arrays em memória; ideal p/ testes)

Contrato: ``open()`` levanta ``VisionUnavailableError`` se a fonte não puder
ser usada; ``read()`` devolve um ``Frame`` ou ``None`` (fim/erro transitório) e
nunca levanta; ``close()`` é idempotente. Todas suportam ``with``.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Sequence

from .frames import Frame, VisionUnavailableError, np, require_cv2

logger = logging.getLogger("agent_core.vision")


class VisualSource(ABC):
    name: str = "source"

    def __init__(self) -> None:
        self._opened = False
        self._index = 0
        self.errors = 0

    @property
    def is_open(self) -> bool:
        return self._opened

    @abstractmethod
    def _open(self) -> None: ...

    @abstractmethod
    def _read(self) -> Any | None:
        """Devolve uma imagem BGR (ndarray) ou None."""

    def _close(self) -> None: ...

    def open(self) -> "VisualSource":
        if not self._opened:
            self._open()
            self._opened = True
        return self

    def read(self) -> Frame | None:
        if not self._opened:
            try:
                self.open()
            except VisionUnavailableError as exc:
                self.errors += 1
                logger.warning("%s indisponível: %s", self.name, exc)
                return None
        try:
            image = self._read()
        except Exception as exc:  # erro transitório do dispositivo
            self.errors += 1
            logger.warning("%s: erro de leitura: %s", self.name, exc)
            return None
        if image is None:
            return None
        frame = Frame(image=image, source=self.name, index=self._index)
        self._index += 1
        if not frame.is_valid():
            self.errors += 1
            return None
        return frame

    def close(self) -> None:
        if self._opened:
            try:
                self._close()
            finally:
                self._opened = False

    def __enter__(self) -> "VisualSource":
        return self.open()

    def __exit__(self, *_: Any) -> None:
        self.close()


class CameraSource(VisualSource):
    """Webcam/câmera via ``cv2.VideoCapture``."""

    def __init__(self, index: int = 0, width: int | None = None, height: int | None = None) -> None:
        super().__init__()
        self.index = index
        self.width = width
        self.height = height
        self.name = f"camera:{index}"
        self._cap: Any = None

    def _open(self) -> None:
        cv = require_cv2()
        cap = cv.VideoCapture(self.index)
        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            raise VisionUnavailableError(f"câmera {self.index} não pôde ser aberta")
        if self.width:
            cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap

    def _read(self) -> Any | None:
        ok, image = self._cap.read()
        return image if ok else None

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ScreenSource(VisualSource):
    """Captura de tela via ``mss`` (opcional). Falha limpa sem display."""

    def __init__(self, monitor: int = 1, region: dict[str, int] | None = None) -> None:
        super().__init__()
        self.monitor = monitor
        self.region = region
        self.name = f"screen:{monitor}"
        self._sct: Any = None
        self._area: dict[str, int] | None = None

    def _open(self) -> None:
        require_cv2()
        try:
            import mss  # type: ignore
        except ImportError as exc:
            raise VisionUnavailableError("captura de tela requer 'mss' (pip install mss)") from exc
        try:
            # mss.MSS a partir da 10.x; mss.mss() nas versões anteriores.
            factory = getattr(mss, "MSS", None) or mss.mss
            self._sct = factory()
            monitors = self._sct.monitors
            if self.monitor >= len(monitors):
                raise VisionUnavailableError(f"monitor {self.monitor} inexistente ({len(monitors) - 1} disponíveis)")
            self._area = self.region or monitors[self.monitor]
        except VisionUnavailableError:
            raise
        except Exception as exc:  # mss.ScreenShotError e afins (sem DISPLAY, etc.)
            raise VisionUnavailableError(f"captura de tela indisponível: {exc}") from exc

    def _read(self) -> Any | None:
        cv = require_cv2()
        shot = self._sct.grab(self._area)
        image = np.asarray(shot)  # BGRA
        return cv.cvtColor(image, cv.COLOR_BGRA2BGR)

    def _close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            finally:
                self._sct = None


class ImageSource(VisualSource):
    """Sequência de imagens (caminhos ou arrays). Com ``loop`` repete indefinidamente."""

    def __init__(self, items: Sequence[str | Path | Any], *, loop: bool = False, name: str = "image") -> None:
        super().__init__()
        if not items:
            raise ValueError("ImageSource precisa de ao menos uma imagem")
        self.items = list(items)
        self.loop = loop
        self.name = name
        self._pos = 0

    def _open(self) -> None:
        require_cv2()
        for item in self.items:
            if isinstance(item, (str, Path)) and not Path(item).is_file():
                raise VisionUnavailableError(f"imagem não encontrada: {item}")

    def _read(self) -> Any | None:
        if self._pos >= len(self.items):
            if not self.loop:
                return None
            self._pos = 0
        item = self.items[self._pos]
        self._pos += 1
        if isinstance(item, (str, Path)):
            cv = require_cv2()
            image = cv.imread(str(item), cv.IMREAD_COLOR)
            if image is None:
                raise VisionUnavailableError(f"não foi possível decodificar {item}")
            return image
        return item

    @classmethod
    def from_directory(cls, directory: str | Path, patterns: Iterable[str] = ("*.png", "*.jpg", "*.jpeg"), **kw: Any) -> "ImageSource":
        paths: list[Path] = []
        for pattern in patterns:
            paths.extend(sorted(Path(directory).glob(pattern)))
        return cls(paths, **kw)


def open_source(kind: str, **options: Any) -> VisualSource:
    """Fábrica por nome: ``camera``, ``screen`` ou ``image``."""
    if kind == "camera":
        return CameraSource(index=int(options.get("camera_index", 0)))
    if kind == "screen":
        return ScreenSource(monitor=int(options.get("monitor", 1)))
    if kind == "image":
        paths = options.get("images") or []
        if not paths:
            raise VisionUnavailableError("fonte 'image' requer ao menos um caminho (--image)")
        return ImageSource(paths, loop=bool(options.get("loop", True)))
    raise ValueError(f"fonte visual desconhecida: {kind}")


def wait_until(deadline: float) -> None:
    """Pequeno utilitário síncrono usado por fontes com FPS limitado."""
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)
