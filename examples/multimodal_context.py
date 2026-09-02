"""
Exemplo 4 - contexto multimodal.

Combina código, log, resultado de execução e uma observação visual num
``MultimodalContext``, mostra a serialização e as partes que iriam ao modelo.

    python examples/multimodal_context.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_core import AgentConfig, CodeManager, ContextLimits, MultimodalContext, Sandbox
from agent_core.observations import CodeObserver, LogObserver, RuntimeObserver
from agent_core.vision import Frame, VisualPipeline, vision_available


async def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "app.py").write_text("import json\nprint(json.loads('{'))\n")
        (root / "app.log").write_text("INFO start\nERROR payload inválido\n")
        config = AgentConfig(project_root=root, log_level="WARNING")

        ctx = MultimodalContext(ContextLimits(max_images=2))
        result = await Sandbox(config).run_script("app.py")
        ctx.extend(await RuntimeObserver().observe(result=result))
        ctx.extend(await LogObserver(root).observe())
        ctx.extend(await CodeObserver(CodeManager(config)).observe(failing_file="app.py"))
        if vision_available():
            import cv2
            import numpy as np

            img = np.full((120, 160, 3), 240, np.uint8)
            cv2.putText(img, "ERR", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ctx.add(VisualPipeline().process(Frame(img, "screen")))

        print("resumo:", ctx.summary())
        print("\n--- texto para o modelo (trecho)")
        print(ctx.to_text()[:1200])
        parts = ctx.to_parts()
        print("\n--- partes:", [p.type for p in parts])
        print("--- JSON (bytes):", len(ctx.to_json(include_images=True)))
        restored = MultimodalContext.from_dict(__import__("json").loads(ctx.to_json(include_images=True)))
        print("--- roundtrip ok:", len(restored) == len(ctx))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
