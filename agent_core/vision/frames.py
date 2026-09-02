"""Tipos básicos do subsistema visual."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..observations import ImageData

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - ambiente sem OpenCV
    cv2 = None  # type: ignore
    np = None  # type: ignore


class VisionUnavailableError(RuntimeError):
    """OpenCV ausente, dispositivo inexistente ou captura impossível."""


def is_available() -> bool:
    """OpenCV + numpy estão importáveis?"""
    return cv2 is not None and np is not None


def require_cv2() -> Any:
    if cv2 is None:
        raise VisionUnavailableError("OpenCV (cv2) não está instalado: pip install opencv-python-headless")
    return cv2


@dataclass
class Frame:
    """Um frame BGR (convenção do OpenCV) com metadados de origem."""

    image: Any                       # numpy.ndarray HxWx3 (uint8, BGR)
    source: str
    index: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height

    def is_valid(self) -> bool:
        """Frame com dados coerentes (2D/3D, não vazio, uint8)."""
        if np is None or self.image is None or not isinstance(self.image, np.ndarray):
            return False
        if self.image.ndim not in (2, 3) or self.image.size == 0:
            return False
        if self.image.ndim == 3 and self.image.shape[2] not in (1, 3, 4):
            return False
        return self.image.dtype == np.uint8


def encode_image(frame: Frame, *, fmt: str = "jpeg", quality: int = 80) -> ImageData:
    """Codifica um frame em JPEG/PNG para transporte ao modelo."""
    cv = require_cv2()
    if fmt == "png":
        ok, buf = cv.imencode(".png", frame.image)
        media = "image/png"
    else:
        ok, buf = cv.imencode(".jpg", frame.image, [int(cv.IMWRITE_JPEG_QUALITY), int(quality)])
        media = "image/jpeg"
    if not ok:
        raise VisionUnavailableError("falha ao codificar o frame")
    return ImageData(data=buf.tobytes(), media_type=media, width=frame.width, height=frame.height)
