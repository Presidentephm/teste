"""
Exemplo 3 - processamento de frame.

Gera dois frames sintéticos, roda o pipeline (pré-processamento -> detecção de
mudança -> análise) e imprime a observação estruturada resultante.

    python examples/frame_processing.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_core.vision import ChangeDetector, Frame, FramePreprocessor, VisualPipeline, vision_available


def main() -> int:
    if not vision_available():
        print("OpenCV não instalado: pip install opencv-python-headless numpy")
        return 1
    import cv2
    import numpy as np

    base = np.full((480, 640, 3), 235, np.uint8)
    cv2.putText(base, "Painel OK", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 2)
    changed = base.copy()
    cv2.rectangle(changed, (380, 200), (600, 320), (0, 0, 200), -1)
    cv2.putText(changed, "ERROR 500", (390, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    pipeline = VisualPipeline(preprocessor=FramePreprocessor(max_width=320), change_detector=ChangeDetector(threshold=0.01))
    for i, image in enumerate((base, changed)):
        obs = pipeline.process(Frame(image, source="demo", index=i))
        print(f"--- frame {i}: {obs.summary}")
        info = {k: obs.extracted[k] for k in ("resolution", "brightness", "edge_density", "ocr", "text")}
        info["change"] = obs.extracted["change"]
        info["elements"] = obs.extracted["elements"][:3]
        info["image_bytes"] = obs.image.size_bytes
        print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
