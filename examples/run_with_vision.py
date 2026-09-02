"""
Exemplo 7 - ciclo completo com visão.

Usa uma fonte de imagens sintéticas (funciona sem câmera/tela) para que as
observações visuais entrem no contexto enviado à estratégia. Equivale a:

    python -m agent_core run examples/broken_script.py --strategy auto --vision \
        --vision-source image --image tela1.png --image tela2.png

    python examples/run_with_vision.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_core import AgentConfig, AutoStrategy, SelfImprovementAgent
from agent_core.observations import ObservationKind
from agent_core.vision import ImageSource, VisionCapture, VisualPipeline, vision_available

HERE = Path(__file__).resolve().parent


def screens():
    import cv2
    import numpy as np

    a = np.full((240, 320, 3), 245, np.uint8)
    cv2.putText(a, "app: starting", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
    b = a.copy()
    cv2.rectangle(b, (20, 150), (300, 220), (0, 0, 200), -1)
    cv2.putText(b, "NameError", (40, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return [a, b]


async def main() -> int:
    if not vision_available():
        print("OpenCV não instalado: pip install opencv-python-headless numpy")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("broken_script.py", "report_utils.py"):
            shutil.copy(HERE / name, root / name)
        config = AgentConfig(project_root=root, max_iterations=5, log_level="WARNING", memory_persist=False, observation_interval=0.2)
        capture = VisionCapture(ImageSource(screens(), loop=True), VisualPipeline(), fps=30, observation_interval=0.05)
        # Sem provider: a AutoStrategy usa heurística + evidências; com build_provider(config) usaria o modelo.
        agent = SelfImprovementAgent(config, AutoStrategy(provider=None), vision=capture)
        report = await agent.run("broken_script.py")
        print(report.summary())
        visions = agent.context.by_kind(ObservationKind.VISION)
        print(f"\nobservações visuais no contexto: {len(visions)}")
        for obs in visions[-2:]:
            print("  -", obs.summary)
        return 0 if report.success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
