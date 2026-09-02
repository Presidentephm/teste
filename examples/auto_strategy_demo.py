"""
Exemplo 6 - Strategy Auto com um bug que a heurística não resolve.

Sem credenciais usa um ``FakeProvider`` com a resposta que o modelo daria;
com ``ANTHROPIC_API_KEY`` definido usa o SDK real (``--real``).

    python examples/auto_strategy_demo.py
    ANTHROPIC_API_KEY=... python examples/auto_strategy_demo.py --real
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_core import AgentConfig, AutoStrategy, FakeProvider, SelfImprovementAgent, build_provider

BUGGY = '''\
"""Calcula a média de leituras vindas de um sensor."""


def average(values):
    return sum(values) / len(values)


if __name__ == "__main__":
    print("média:", average([]))
'''

FAKE_ANSWER = json.dumps(
    {
        "rationale": "ZeroDivisionError: average() divide por len(values) quando a lista está vazia; devolve 0.0 nesse caso.",
        "confidence": 0.9,
        "patches": [
            {
                "path": "sensor.py",
                "mode": "search_replace",
                "replacements": [{"search": "    return sum(values) / len(values)", "replace": "    if not values:\n        return 0.0\n    return sum(values) / len(values)"}],
            }
        ],
    }
)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="usa o SDK real (requer credenciais)")
    ns = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "sensor.py").write_text(BUGGY)
        config = AgentConfig(project_root=root, max_iterations=4, log_level="WARNING", memory_persist=False, llm_effort="medium")
        provider = build_provider(config) if ns.real else FakeProvider([FAKE_ANSWER])
        strategy = AutoStrategy(provider)

        async def show_diagnosis(diag):
            print("diagnóstico:\n  " + diag.to_text().replace("\n", "\n  "))

        strategy._diagnose_hook = show_diagnosis
        report = await SelfImprovementAgent(config, strategy).run("sensor.py")
        print(report.summary())
        print("\n--- sensor.py corrigido:\n" + (root / "sensor.py").read_text())
        await provider.aclose()
        return 0 if report.success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
