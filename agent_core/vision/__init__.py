"""
Os "olhos" do agente.

    VisualSource (câmera / tela / imagem)
        -> Frame
        -> VisualPipeline (pré-processamento -> detecção de mudança -> análise)
        -> Observation(kind=VISION)
        -> MultimodalContext

``VisionCapture`` roda a captura em background (FPS e intervalo configuráveis)
e nunca propaga erros de dispositivo para o AgentLoop: falhas ficam em
``status()`` e o agente segue sem visão.
"""

from .frames import Frame, VisionUnavailableError, encode_image, is_available as vision_available
from .sources import VisualSource, CameraSource, ScreenSource, ImageSource
from .processing import FramePreprocessor, ChangeDetector, ChangeResult, VisualAnalyzer, VisualPipeline, OCREngine
from .capture import VisionCapture, VisionObserver, build_vision_capture

__all__ = [
    "Frame",
    "VisionUnavailableError",
    "encode_image",
    "vision_available",
    "VisualSource",
    "CameraSource",
    "ScreenSource",
    "ImageSource",
    "FramePreprocessor",
    "ChangeDetector",
    "ChangeResult",
    "VisualAnalyzer",
    "VisualPipeline",
    "OCREngine",
    "VisionCapture",
    "VisionObserver",
    "build_vision_capture",
]
