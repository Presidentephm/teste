"""
Exemplo 8 - estratégia por ferramentas.

O modelo recebe um prompt compacto e ferramentas de leitura (read_file,
search, outline, list_files) e entrega a correção via propose_patch. Sem
credenciais, um FakeProvider reproduz a sequência de chamadas que o modelo
faria; com ``--real`` usa o SDK.

    python examples/tool_strategy_demo.py
    ANTHROPIC_API_KEY=... python examples/tool_strategy_demo.py --real
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_core import AgentConfig, AutoStrategy, FakeProvider, SelfImprovementAgent, build_provider

APP = "from pricing import total\n\nprint(total([10, 20], discount='10'))\n"
PRICING = "def total(values, discount=0):\n    return sum(values) - discount\n"

SCRIPTED = [
    FakeProvider.tool_response("search", {"query": "def total"}, text="Vou localizar a função."),
    FakeProvider.tool_response("read_file", {"path": "pricing.py"}),
    FakeProvider.tool_response(
        "propose_patch",
        {
            "rationale": "discount chega como str; converter para número antes de subtrair.",
            "confidence": 0.85,
            "patches": [{"path": "pricing.py", "mode": "search_replace", "replacements": [{"search": "    return sum(values) - discount", "replace": "    return sum(values) - float(discount)"}]}],
        },
    ),
]


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ns = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "app.py").write_text(APP)
        (root / "pricing.py").write_text(PRICING)
        config = AgentConfig(project_root=root, max_iterations=3, log_level="INFO", memory_persist=False)
        provider = build_provider(config) if ns.real else FakeProvider(SCRIPTED)
        strategy = AutoStrategy(provider, use_tools=True, effort_by_error=config.effort_by_error)
        report = await SelfImprovementAgent(config, strategy).run("app.py")
        print(report.summary())
        planner = strategy.planners[-1].strategy
        print("\nchamadas de ferramenta:", planner.last_tool_calls)
        print("pricing.py:\n" + (root / "pricing.py").read_text())
        await provider.aclose()
        return 0 if report.success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
