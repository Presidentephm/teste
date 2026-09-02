"""
Harness de benchmark: mede taxa de correção, iterações, tempo, tokens e custo.

Cada ``BenchCase`` é um mini-projeto (arquivos + script + testes opcionais)
criado num diretório temporário. O agente roda sobre ele e o resultado é
agregado num ``BenchmarkReport``.

Modos:
    * offline: ``FakeProvider`` com as respostas canônicas de cada caso
      (valida o harness e o loop sem custo);
    * real: um ``ModelProvider`` compartilhado (``build_provider(config)``),
      que acumula tokens/custo entre os casos.

    python -m agent_core bench --offline
    ANTHROPIC_API_KEY=... python -m agent_core bench --strategy auto --model claude-opus-5
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .agent_loop import SelfImprovementAgent
from .config import AgentConfig
from .providers import FakeProvider, ModelProvider
from .strategies import AutoStrategy, FixStrategy, HeuristicFixStrategy


@dataclass
class BenchCase:
    name: str
    files: dict[str, str]
    script: str
    description: str = ""
    test_command: tuple[str, ...] | None = None
    fake_answers: list[str] = field(default_factory=list)  # respostas do FakeProvider (modo offline)
    needs_model: bool = False                               # a heurística sozinha não resolve
    expected_stdout: str | None = None                      # validação extra do resultado


def _patch(path: str, search: str, replace: str, rationale: str) -> str:
    return json.dumps({"rationale": rationale, "confidence": 0.9, "patches": [{"path": path, "mode": "search_replace", "replacements": [{"search": search, "replace": replace}]}]})


DEFAULT_CASES: list[BenchCase] = [
    BenchCase("name_error_stdlib", {"app.py": 'print(json.dumps({"a": 1}))\n'}, "app.py", "import da stdlib ausente", expected_stdout='{"a": 1}'),
    BenchCase("name_error_sibling", {"app.py": "print(helper())\n", "util.py": "def helper():\n    return 'ok'\n"}, "app.py", "símbolo de módulo irmão sem import", expected_stdout="ok"),
    BenchCase("typo", {"app.py": "import json\nprint(jsn.dumps([1]))\n"}, "app.py", "erro de digitação de nome", expected_stdout="[1]"),
    BenchCase("tabs", {"app.py": "def f():\n    if True:\n\treturn 7\nprint(f())\n"}, "app.py", "mistura de tabs e espaços", expected_stdout="7"),
    BenchCase(
        "zero_division",
        {"app.py": "def avg(values):\n    return sum(values) / len(values)\n\nprint(avg([]))\n"},
        "app.py",
        "divisão por zero em lista vazia",
        fake_answers=[_patch("app.py", "    return sum(values) / len(values)", "    if not values:\n        return 0.0\n    return sum(values) / len(values)", "guarda para lista vazia")],
        needs_model=True,
        expected_stdout="0.0",
    ),
    BenchCase(
        "type_error",
        {"app.py": 'total = "5" + 5\nprint(total)\n'},
        "app.py",
        "concatenação de str com int",
        fake_answers=[_patch("app.py", 'total = "5" + 5', 'total = int("5") + 5', "converter antes de somar")],
        needs_model=True,
        expected_stdout="10",
    ),
    BenchCase(
        "failing_test",
        {
            "app.py": "from lib import add\nprint(add(1, 2))\n",
            "lib.py": "def add(a, b):\n    return a - b\n",
            "tests/test_lib.py": "import unittest\nfrom lib import add\n\nclass T(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(1, 2), 3)\n",
        },
        "app.py",
        "script roda, mas o teste falha",
        test_command=("-m", "unittest", "discover", "-s", "tests"),
        fake_answers=[_patch("lib.py", "    return a - b", "    return a + b", "add subtraía em vez de somar")],
        needs_model=True,
        expected_stdout="3",
    ),
]


@dataclass
class CaseResult:
    name: str
    status: str
    fixed: bool
    iterations: int
    outcomes: list[str]
    duration: float
    usage: dict[str, Any] | None
    stdout_ok: bool | None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class BenchmarkReport:
    strategy: str
    model: str
    results: list[CaseResult] = field(default_factory=list)
    duration: float = 0.0

    @property
    def fix_rate(self) -> float:
        return sum(1 for r in self.results if r.fixed) / len(self.results) if self.results else 0.0

    @property
    def totals(self) -> dict[str, Any]:
        keys = ("calls", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "cost_usd")
        total = {k: 0 for k in keys}
        for r in self.results:
            for k in keys:
                total[k] += (r.usage or {}).get(k, 0)
        total["cost_usd"] = round(total["cost_usd"], 6)
        fixed = [r for r in self.results if r.fixed]
        total["cost_per_fix_usd"] = round(total["cost_usd"] / len(fixed), 6) if fixed else None
        total["avg_iterations"] = round(sum(r.iterations for r in self.results) / len(self.results), 2) if self.results else 0
        return total

    def to_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "model": self.model, "fix_rate": self.fix_rate, "duration": round(self.duration, 2), "totals": self.totals, "results": [r.to_dict() for r in self.results]}

    def table(self) -> str:
        rows = [f"{'caso':<20} {'status':<18} {'it':>3} {'tempo':>7} {'tokens in/out':>15} {'custo':>9} stdout"]
        for r in self.results:
            u = r.usage or {}
            cost = f"{u.get('cost_usd', 0):.4f}" if u else "-"
            tok = f"{u.get('input_tokens', 0)}/{u.get('output_tokens', 0)}" if u else "-"
            ok = "-" if r.stdout_ok is None else ("ok" if r.stdout_ok else "DIFERENTE")
            rows.append(f"{r.name:<20} {r.status:<18} {r.iterations:>3} {r.duration:>6.1f}s {tok:>15} {cost:>9} {ok}")
        t = self.totals
        rows.append(f"\nfix rate: {self.fix_rate * 100:.0f}%  iterações médias: {t['avg_iterations']}  chamadas: {t['calls']}  custo total: US${t['cost_usd']:.4f}  custo/correção: {t['cost_per_fix_usd']}")
        return "\n".join(rows)


StrategyFactory = Callable[[AgentConfig, BenchCase], FixStrategy]


def offline_factory(config: AgentConfig, case: BenchCase) -> FixStrategy:
    """AutoStrategy com FakeProvider programado com as respostas do caso."""
    return AutoStrategy(FakeProvider(list(case.fake_answers)), effort_by_error=config.effort_by_error, use_tools=config.llm_use_tools)


def heuristic_factory(config: AgentConfig, case: BenchCase) -> FixStrategy:
    return AutoStrategy(None, effort_by_error=config.effort_by_error)


def provider_factory(provider: ModelProvider) -> StrategyFactory:
    """Compartilha um provider real entre os casos (acumula uso/custo)."""

    def factory(config: AgentConfig, case: BenchCase) -> FixStrategy:
        return AutoStrategy(provider, effort_by_error=config.effort_by_error, use_tools=config.llm_use_tools, max_tool_rounds=config.llm_max_tool_rounds)

    return factory


async def run_case(case: BenchCase, factory: StrategyFactory, *, config_overrides: dict[str, Any] | None = None) -> CaseResult:
    tmp = Path(tempfile.mkdtemp(prefix=f"bench_{case.name}_"))
    started = time.perf_counter()
    try:
        for rel, content in case.files.items():
            path = tmp / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        overrides = dict(max_iterations=5, log_level="WARNING", memory_persist=False, sandbox_timeout=20)
        overrides.update(config_overrides or {})
        overrides["test_command"] = case.test_command
        config = AgentConfig(project_root=tmp, **overrides)
        agent = SelfImprovementAgent(config, factory(config, case))
        report = await agent.run(case.script)
        stdout_ok = None
        if case.expected_stdout is not None and report.final_result is not None:
            stdout_ok = report.final_result.stdout.strip() == case.expected_stdout
        return CaseResult(
            name=case.name,
            status=report.status,
            fixed=report.success and (stdout_ok is not False),
            iterations=len(report.iterations),
            outcomes=[it.outcome for it in report.iterations],
            duration=time.perf_counter() - started,
            usage=report.usage,
            stdout_ok=stdout_ok,
        )
    except Exception as exc:  # um caso quebrado não derruba o benchmark
        return CaseResult(case.name, "error", False, 0, [], time.perf_counter() - started, None, None, f"{type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def run_benchmark(
    cases: Sequence[BenchCase],
    factory: StrategyFactory,
    *,
    strategy_name: str = "auto",
    model: str = "-",
    config_overrides: dict[str, Any] | None = None,
) -> BenchmarkReport:
    report = BenchmarkReport(strategy=strategy_name, model=model)
    started = time.perf_counter()
    for case in cases:
        report.results.append(await run_case(case, factory, config_overrides=config_overrides))
    report.duration = time.perf_counter() - started
    return report


def select_cases(names: Sequence[str] | None, *, heuristic_only: bool = False) -> list[BenchCase]:
    cases = [c for c in DEFAULT_CASES if not (heuristic_only and c.needs_model)]
    if names:
        wanted = set(names)
        unknown = wanted - {c.name for c in DEFAULT_CASES}
        if unknown:
            raise ValueError(f"casos desconhecidos: {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c.name in wanted]
    return cases


def main_sync(cases: Sequence[BenchCase], factory: StrategyFactory, **kw: Any) -> BenchmarkReport:
    return asyncio.run(run_benchmark(cases, factory, **kw))
