"""
Pipeline visual: Frame -> pré-processamento -> análise -> Observation.

Nada aqui "inventa" percepção: as saídas são estatísticas e estruturas reais
calculadas com OpenCV (resolução, brilho, densidade de bordas, cores
dominantes, regiões que mudaram, contornos salientes) e, quando o Tesseract
está instalado junto com ``pytesseract``, texto reconhecido por OCR. Sem OCR
disponível o campo ``text`` é ``None`` e ``ocr="unavailable"``.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from typing import Any

from ..observations import Observation, ObservationKind
from .frames import Frame, encode_image, np, require_cv2

logger = logging.getLogger("agent_core.vision")


# --------------------------------------------------------------- pré-processo
class FramePreprocessor:
    """Redimensiona (mantendo proporção) e opcionalmente converte para cinza."""

    def __init__(self, max_width: int = 1024, grayscale: bool = False) -> None:
        self.max_width = max_width
        self.grayscale = grayscale

    def process(self, frame: Frame) -> Frame:
        cv = require_cv2()
        image = frame.image
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
        elif image.ndim == 2:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        scale = 1.0
        if self.max_width and image.shape[1] > self.max_width:
            scale = self.max_width / image.shape[1]
            image = cv.resize(image, (self.max_width, max(1, int(image.shape[0] * scale))), interpolation=cv.INTER_AREA)
        if self.grayscale:
            image = cv.cvtColor(cv.cvtColor(image, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)
        meta = dict(frame.metadata, scale=scale, original_resolution=frame.resolution)
        return Frame(image=image, source=frame.source, index=frame.index, timestamp=frame.timestamp, metadata=meta)


# ------------------------------------------------------------------- mudança
@dataclass
class ChangeResult:
    changed: bool
    score: float                      # fração de pixels alterados (0..1)
    regions: list[dict[str, int]] = field(default_factory=list)  # caixas x,y,w,h
    first: bool = False               # sem frame de referência (primeira observação)

    def to_dict(self) -> dict[str, Any]:
        return {"changed": self.changed, "score": round(self.score, 4), "regions": self.regions, "first": self.first}


class ChangeDetector:
    """Compara o frame atual com o anterior (diferença absoluta em cinza)."""

    def __init__(self, threshold: float = 0.02, pixel_delta: int = 25, min_area: int = 64, max_regions: int = 10) -> None:
        self.threshold = threshold
        self.pixel_delta = pixel_delta
        self.min_area = min_area
        self.max_regions = max_regions
        self._previous: Any | None = None

    def reset(self) -> None:
        self._previous = None

    def _gray(self, frame: Frame) -> Any:
        cv = require_cv2()
        gray = cv.cvtColor(frame.image, cv.COLOR_BGR2GRAY)
        return cv.GaussianBlur(gray, (5, 5), 0)

    def detect(self, frame: Frame) -> ChangeResult:
        cv = require_cv2()
        gray = self._gray(frame)
        prev = self._previous
        self._previous = gray
        if prev is None:
            return ChangeResult(changed=True, score=1.0, regions=[], first=True)  # primeiro frame: tudo é novo
        if prev.shape != gray.shape:
            return ChangeResult(changed=True, score=1.0, regions=[], first=True)  # resolução mudou: sem referência
        diff = cv.absdiff(prev, gray)
        _, mask = cv.threshold(diff, self.pixel_delta, 255, cv.THRESH_BINARY)
        mask = cv.dilate(mask, None, iterations=2)
        score = float(np.count_nonzero(mask)) / float(mask.size)
        regions: list[dict[str, int]] = []
        if score > 0:
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            boxes = [cv.boundingRect(c) for c in contours if cv.contourArea(c) >= self.min_area]
            boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
            regions = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)} for x, y, w, h in boxes[: self.max_regions]]
        return ChangeResult(changed=score >= self.threshold, score=score, regions=regions)


# ----------------------------------------------------------------------- OCR
class OCREngine:
    """OCR real via Tesseract (``pytesseract`` + binário). Opcional."""

    def __init__(self, lang: str = "eng") -> None:
        self.lang = lang
        self._pt: Any | None = None
        self.available = False
        try:
            import pytesseract  # type: ignore

            if shutil.which("tesseract"):
                self._pt = pytesseract
                self.available = True
        except ImportError:
            pass

    def read(self, frame: Frame) -> str | None:
        if not self.available or self._pt is None:
            return None
        cv = require_cv2()
        gray = cv.cvtColor(frame.image, cv.COLOR_BGR2GRAY)
        try:
            text = self._pt.image_to_string(gray, lang=self.lang)
        except Exception as exc:  # binário falhou: registra e segue
            logger.warning("OCR falhou: %s", exc)
            return None
        text = text.strip()
        return text or None


# ------------------------------------------------------------------- análise
class VisualAnalyzer:
    """Extrai estrutura de um frame: estatísticas, regiões salientes, texto."""

    def __init__(self, ocr: OCREngine | None = None, max_elements: int = 12) -> None:
        self.ocr = ocr if ocr is not None else OCREngine()
        self.max_elements = max_elements

    def analyze(self, frame: Frame, change: ChangeResult | None = None) -> dict[str, Any]:
        cv = require_cv2()
        img = frame.image
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        h, w = gray.shape
        brightness = float(gray.mean()) / 255.0
        contrast = float(gray.std()) / 255.0
        edges = cv.Canny(gray, 100, 200)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)

        # Elementos visuais: contornos salientes (caixas grandes o bastante).
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        boxes = [cv.boundingRect(c) for c in contours]
        boxes = [b for b in boxes if b[2] * b[3] >= max(64, (w * h) // 400)]
        boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
        elements = [{"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)} for x, y, bw, bh in boxes[: self.max_elements]]

        # Cores dominantes (k-means pequeno em amostra reduzida).
        small = cv.resize(img, (32, 32), interpolation=cv.INTER_AREA).reshape(-1, 3).astype(np.float32)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 3
        _, labels, centers = cv.kmeans(small, k, None, criteria, 1, cv.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.flatten(), minlength=k)
        order = np.argsort(-counts)
        dominant = [
            {"bgr": [int(c) for c in centers[i]], "share": round(float(counts[i]) / float(len(labels)), 3)} for i in order
        ]

        text = self.ocr.read(frame)
        description = self._describe(w, h, brightness, contrast, edge_density, len(elements), change, text)
        return {
            "resolution": {"width": w, "height": h},
            "brightness": round(brightness, 3),
            "contrast": round(contrast, 3),
            "edge_density": round(edge_density, 4),
            "elements": elements,
            "dominant_colors": dominant,
            "change": change.to_dict() if change else None,
            "text": text,
            "ocr": "tesseract" if self.ocr.available else "unavailable",
            "description": description,
        }

    @staticmethod
    def _describe(w: int, h: int, brightness: float, contrast: float, edge_density: float, n_elements: int, change: ChangeResult | None, text: str | None) -> str:
        tone = "escura" if brightness < 0.25 else "clara" if brightness > 0.75 else "média"
        detail = "muito detalhada" if edge_density > 0.12 else "com poucos detalhes" if edge_density < 0.02 else "com detalhe moderado"
        parts = [f"imagem {w}x{h}, luminosidade {tone}, {detail}, {n_elements} elementos salientes"]
        if change is not None:
            if change.first:
                parts.append("primeira observação (sem frame de referência)")
            elif change.changed:
                parts.append(f"mudança de {change.score * 100:.1f}% em {len(change.regions)} regiões")
            else:
                parts.append("sem mudança relevante em relação ao frame anterior")
        if text:
            parts.append(f"texto detectado: {text[:120]!r}")
        return "; ".join(parts)


# ------------------------------------------------------------------ pipeline
class VisualPipeline:
    """Compõe pré-processamento, detecção de mudança e análise numa ``Observation``."""

    def __init__(
        self,
        preprocessor: FramePreprocessor | None = None,
        change_detector: ChangeDetector | None = None,
        analyzer: VisualAnalyzer | None = None,
        *,
        attach_image: bool = True,
        jpeg_quality: int = 75,
    ) -> None:
        self.preprocessor = preprocessor or FramePreprocessor()
        self.change_detector = change_detector or ChangeDetector()
        self.analyzer = analyzer or VisualAnalyzer()
        self.attach_image = attach_image
        self.jpeg_quality = jpeg_quality

    def reset(self) -> None:
        self.change_detector.reset()

    def process(self, frame: Frame) -> Observation:
        if not frame.is_valid():
            return Observation(
                kind=ObservationKind.VISION,
                source=frame.source,
                summary="frame inválido descartado",
                confidence=0.0,
                metadata={"invalid": True, "index": frame.index},
            )
        processed = self.preprocessor.process(frame)
        change = self.change_detector.detect(processed)
        analysis = self.analyzer.analyze(processed, change)
        image = encode_image(processed, quality=self.jpeg_quality) if self.attach_image else None
        # Confiança: alta quando houve OCR; média para estatísticas puras.
        confidence = 0.85 if analysis.get("text") else 0.6
        return Observation(
            kind=ObservationKind.VISION,
            source=frame.source,
            summary=analysis["description"],
            data={"index": frame.index, "frame_timestamp": frame.timestamp, "scale": processed.metadata.get("scale", 1.0)},
            extracted=analysis,
            image=image,
            confidence=confidence,
            metadata={"changed": change.changed, "change_score": change.score},
            timestamp=frame.timestamp,
        )
