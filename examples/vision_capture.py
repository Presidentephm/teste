"""
Exemplo 2 - captura visual contínua.

Tenta, nesta ordem, câmera -> tela -> imagens sintéticas geradas em memória,
para que o exemplo funcione também em ambientes sem dispositivos.

    python examples/vision_capture.py --seconds 2
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_core.vision import CameraSource, ImageSource, ScreenSource, VisionCapture, VisionUnavailableError, vision_available


def synthetic_frames():
    import cv2
    import numpy as np

    frames = []
    for i in range(3):
        img = np.full((240, 320, 3), 220, np.uint8)
        cv2.rectangle(img, (30 + 60 * i, 40), (120 + 60 * i, 140), (0, 0, 255), -1)
        cv2.putText(img, f"frame {i}", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        frames.append(img)
    return frames


def pick_source():
    for factory, label in ((lambda: CameraSource(0), "câmera"), (lambda: ScreenSource(1), "tela")):
        src = factory()
        try:
            src.open()
            print(f"fonte: {label} ({src.name})")
            return src
        except VisionUnavailableError as exc:
            print(f"{label} indisponível: {exc}")
    print("fonte: imagens sintéticas em memória")
    return ImageSource(synthetic_frames(), loop=True, name="synthetic")


async def main() -> int:
    if not vision_available():
        print("OpenCV não instalado: pip install opencv-python-headless numpy")
        return 1
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--store", default=None, help="pasta para gravar frames com mudança")
    ns = ap.parse_args()

    async def on_observation(obs):
        print(f"  observação #{obs.data['index']}: {obs.summary}")

    capture = VisionCapture(pick_source(), fps=ns.fps, observation_interval=0.5, store_dir=Path(ns.store) if ns.store else None, on_observation=on_observation)
    if not await capture.start():
        print("captura não iniciou:", capture.error)
        return 1
    await asyncio.sleep(ns.seconds)
    await capture.stop()
    print("status final:", capture.status())
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
