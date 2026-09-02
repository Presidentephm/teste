"""
Exemplo 5 - AgentLoop programático (offline, estratégia heurística).

Copia ``broken_script.py`` e ``report_utils.py`` para uma pasta temporária,
roda o ciclo e mostra o relatório e a memória.

    python examples/agent_loop_demo.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_core import AgentConfig, HeuristicFixStrategy, SelfImprovementAgent

HERE = Path(__file__).resolve().parent


async def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("broken_script.py", "report_utils.py"):
            shutil.copy(HERE / name, root / name)
        config = AgentConfig(project_root=root, max_iterations=5, log_level="WARNING", memory_persist=False)

        async def on_event(event: str, data: dict) -> None:
            if event in ("decision", "iteration.end", "rollback"):
                print(f"  {event}: {data}")

        agent = SelfImprovementAgent(config, HeuristicFixStrategy(), on_event=on_event)
        report = await agent.run("broken_script.py")
        print(report.summary())
        print("\nmemória:\n" + agent.memory.to_prompt_text())
        return 0 if report.success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
