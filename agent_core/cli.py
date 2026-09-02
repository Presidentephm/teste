"""
Interface de linha de comando.

    python -m agent_core run examples/broken_script.py [--strategy auto|heuristic|claude]
    python -m agent_core analyze examples/broken_script.py
    python -m agent_core backups [arquivo]
    python -m agent_core rollback examples/broken_script.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .agent_loop import SelfImprovementAgent
from .code_manager import CodeManager
from .config import AgentConfig
from .strategies import ClaudeFixStrategy, CompositeFixStrategy, FixStrategy, HeuristicFixStrategy


def build_strategy(name: str, config: AgentConfig) -> FixStrategy:
    if name == "heuristic":
        return HeuristicFixStrategy()
    if name == "claude":
        return ClaudeFixStrategy(config)
    return CompositeFixStrategy([HeuristicFixStrategy(), ClaudeFixStrategy(config)])


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent_core", description="Núcleo do agente auto-modificável")
    p.add_argument("--root", default=".", help="raiz do projeto (padrão: diretório atual)")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="executa o loop de auto-correção sobre um script")
    run.add_argument("script")
    run.add_argument("args", nargs="*", help="argumentos passados ao script")
    run.add_argument("--strategy", choices=["auto", "heuristic", "claude"], default="auto")
    run.add_argument("--max-iter", type=int, default=5)
    run.add_argument("--timeout", type=float, default=30.0)
    run.add_argument("--model", default="claude-opus-5")
    run.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    run.add_argument("--no-fallbacks", action="store_true", help="desativa o fallback server-side do modelo")
    run.add_argument("--no-self-modify", action="store_true", help="proíbe alterar o pacote agent_core")
    run.add_argument("--json", action="store_true", help="imprime o relatório em JSON")

    an = sub.add_parser("analyze", help="mostra a estrutura (AST) de um arquivo")
    an.add_argument("file")
    an.add_argument("--json", action="store_true")

    bk = sub.add_parser("backups", help="lista backups")
    bk.add_argument("file", nargs="?")

    rb = sub.add_parser("rollback", help="restaura o backup mais recente de um arquivo")
    rb.add_argument("file")
    return p


async def _main(argv: list[str]) -> int:
    ns = make_parser().parse_args(argv)
    kwargs = dict(project_root=Path(ns.root), log_level=ns.log_level)
    if ns.command == "run":
        kwargs.update(
            max_iterations=ns.max_iter,
            sandbox_timeout=ns.timeout,
            llm_model=ns.model,
            llm_effort=ns.effort,
            llm_enable_fallbacks=not ns.no_fallbacks,
            allow_self_modification=not ns.no_self_modify,
        )
    config = AgentConfig(**kwargs)

    if ns.command == "run":
        agent = SelfImprovementAgent(config, build_strategy(ns.strategy, config))
        report = await agent.run(ns.script, ns.args)
        if ns.json:
            print(json.dumps(_report_to_dict(report), indent=2, ensure_ascii=False))
        else:
            print(report.summary())
        return 0 if report.status in ("fixed", "already_ok") else 1

    code = CodeManager(config)
    if ns.command == "analyze":
        analysis = await code.analyze(ns.file)
        print(json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False) if ns.json else analysis.outline())
        return 0 if analysis.is_valid else 1
    if ns.command == "backups":
        target = code.resolve(ns.file) if ns.file else None
        for rec in code.backups.list_backups(target):
            print(f"{rec.timestamp}  {rec.original}  ->  {rec.backup_path}  ({rec.reason or '-'})")
        return 0
    if ns.command == "rollback":
        rec = await code.rollback(ns.file)
        print(f"restaurado {rec.original} a partir de {rec.backup_path} ({rec.timestamp})")
        return 0
    return 2


def _report_to_dict(report) -> dict:
    return {
        "script": report.script,
        "status": report.status,
        "duration": report.duration,
        "touched_files": report.touched_files,
        "iterations": [
            {
                "attempt": it.attempt,
                "before": it.before.summary(),
                "proposal": None if it.proposal is None else {
                    "strategy": it.proposal.strategy,
                    "rationale": it.proposal.rationale,
                    "confidence": it.proposal.confidence,
                    "files": [p.path for p in it.proposal.patches],
                },
                "after": None if it.after is None else it.after.summary(),
                "outcome": it.outcome,
                "note": it.note,
            }
            for it in report.iterations
        ],
    }


def main(argv: list[str] | None = None) -> None:
    sys.exit(asyncio.run(_main(sys.argv[1:] if argv is None else argv)))
