"""
Interface de linha de comando.

    python -m agent_core run examples/broken_script.py --strategy auto --vision --max-iterations 10
    python -m agent_core analyze examples/broken_script.py
    python -m agent_core backups [arquivo]
    python -m agent_core rollback examples/broken_script.py
    python -m agent_core observe --vision-source image --image foto.png
    python -m agent_core ask "Explique este traceback: ..." [--fake]
    python -m agent_core memory [--clear]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
from pathlib import Path

from .agent_loop import SelfImprovementAgent
from .code_manager import CodeManager
from .config import AgentConfig
from .memory import AgentMemory
from .providers import FakeProvider, ModelMessage, ModelProvider, ModelRequest, ProviderError, build_provider
from .strategies import AutoStrategy, ClaudeFixStrategy, FixStrategy, HeuristicFixStrategy


def build_strategy(name: str, config: AgentConfig, provider: ModelProvider | None = None) -> FixStrategy:
    if name == "heuristic":
        return HeuristicFixStrategy()
    if name == "claude":
        return ClaudeFixStrategy(config, provider=provider)
    return AutoStrategy(provider if provider is not None else build_provider(config))


def _add_vision_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("visão")
    g.add_argument("--vision", dest="vision", action="store_true", default=False, help="liga a observação visual")
    g.add_argument("--no-vision", dest="vision", action="store_false", help="desliga a observação visual (padrão)")
    g.add_argument("--vision-source", choices=["camera", "screen", "image"], default="camera")
    g.add_argument("--camera-index", type=int, default=0)
    g.add_argument("--monitor", type=int, default=1)
    g.add_argument("--image", action="append", default=[], help="imagem para --vision-source image (repetível)")
    g.add_argument("--fps", type=float, default=2.0, help="frames por segundo da captura")
    g.add_argument("--observation-interval", type=float, default=5.0, help="segundos entre observações visuais")
    g.add_argument("--store-frames", action="store_true", help="grava frames relevantes em .agent_backups/frames")


def _add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("modelo")
    g.add_argument("--model", default="claude-opus-5")
    g.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    g.add_argument("--max-tokens", type=int, default=16000)
    g.add_argument("--fallback", dest="fallback", action="store_true", default=True, help="ativa retries/fallback (padrão)")
    g.add_argument("--no-fallback", "--no-fallbacks", dest="fallback", action="store_false", help="desativa retries e fallback")
    g.add_argument("--fallback-model", action="append", default=[], help="modelo alternativo (repetível)")
    g.add_argument("--llm-timeout", type=float, default=600.0)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent_core", description="Núcleo do agente autônomo multimodal")
    p.add_argument("--root", default=".", help="raiz do projeto (padrão: diretório atual)")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="executa o ciclo autônomo sobre um script")
    run.add_argument("script")
    run.add_argument("args", nargs="*", help="argumentos passados ao script")
    run.add_argument("--strategy", choices=["auto", "heuristic", "claude"], default="auto")
    run.add_argument("--max-iterations", "--max-iter", dest="max_iterations", type=int, default=5)
    run.add_argument("--max-retries", type=int, default=3, help="patches fracassados tolerados antes de desistir")
    run.add_argument("--timeout", type=float, default=30.0, help="timeout (s) de cada execução no sandbox")
    run.add_argument("--total-timeout", type=float, default=None, help="tempo máximo (s) do ciclo inteiro")
    run.add_argument("--tests", default=None, help='comando de testes após "python", ex.: "-m unittest discover -s tests"')
    run.add_argument("--memory-limit", type=int, default=100)
    run.add_argument("--reset-memory", action="store_true", help="apaga a memória persistida antes de começar")
    run.add_argument("--no-self-modify", action="store_true", help="proíbe alterar o pacote agent_core")
    run.add_argument("--json", action="store_true", help="imprime o relatório em JSON")
    _add_model_args(run)
    _add_vision_args(run)

    an = sub.add_parser("analyze", help="mostra a estrutura (AST) de um arquivo")
    an.add_argument("file")
    an.add_argument("--json", action="store_true")

    bk = sub.add_parser("backups", help="lista backups")
    bk.add_argument("file", nargs="?")

    rb = sub.add_parser("rollback", help="restaura o backup mais recente de um arquivo")
    rb.add_argument("file")

    ob = sub.add_parser("observe", help="faz uma observação visual única e imprime o resultado")
    ob.add_argument("--json", action="store_true")
    ob.add_argument("--save", default=None, help="grava a imagem observada (JPEG) neste caminho")
    _add_vision_args(ob)

    ask = sub.add_parser("ask", help="envia um prompt ao provider e imprime a resposta")
    ask.add_argument("prompt")
    ask.add_argument("--image", action="append", default=[], help="imagem anexada ao prompt (repetível)")
    ask.add_argument("--fake", action="store_true", help="usa um provider falso (offline)")
    _add_model_args(ask)

    mem = sub.add_parser("memory", help="mostra ou limpa a memória persistida")
    mem.add_argument("--clear", action="store_true")
    return p


def config_from_args(ns: argparse.Namespace) -> AgentConfig:
    kwargs: dict = dict(project_root=Path(ns.root), log_level=ns.log_level)
    if getattr(ns, "model", None):
        kwargs.update(
            llm_model=ns.model,
            llm_effort=ns.effort,
            llm_max_tokens=ns.max_tokens,
            llm_enable_fallbacks=ns.fallback,
            llm_fallback_models=tuple(ns.fallback_model),
            llm_timeout=ns.llm_timeout,
        )
    if hasattr(ns, "vision"):
        kwargs.update(
            vision_enabled=ns.vision,
            vision_source=ns.vision_source,
            vision_camera_index=ns.camera_index,
            vision_monitor=ns.monitor,
            vision_images=tuple(ns.image),
            vision_fps=ns.fps,
            vision_store_frames=ns.store_frames,
            observation_interval=ns.observation_interval,
        )
    if ns.command == "run":
        kwargs.update(
            max_iterations=ns.max_iterations,
            max_retries=ns.max_retries,
            sandbox_timeout=ns.timeout,
            total_timeout=ns.total_timeout,
            test_command=tuple(shlex.split(ns.tests)) if ns.tests else None,
            memory_limit=ns.memory_limit,
            allow_self_modification=not ns.no_self_modify,
        )
    return AgentConfig(**kwargs)


async def _cmd_run(ns: argparse.Namespace) -> int:
    config = config_from_args(ns)
    if ns.reset_memory and config.memory_path and config.memory_path.exists():
        config.memory_path.unlink()
    agent = SelfImprovementAgent(config, build_strategy(ns.strategy, config))
    report = await agent.run(ns.script, ns.args)
    print(json.dumps(report_to_dict(report), indent=2, ensure_ascii=False) if ns.json else report.summary())
    return 0 if report.success else 1


async def _cmd_observe(ns: argparse.Namespace) -> int:
    from .vision.capture import build_vision_capture

    config = config_from_args(ns)
    capture = build_vision_capture(config)
    observation = await capture.snapshot()
    await capture.stop()
    if observation is None:
        print(f"observação indisponível: {capture.error or 'sem frame'}", file=sys.stderr)
        return 1
    if ns.save and observation.image is not None:
        Path(ns.save).write_bytes(observation.image.data)
    print(json.dumps(observation.to_dict(), indent=2, ensure_ascii=False) if ns.json else observation.to_prompt_text())
    return 0


async def _cmd_ask(ns: argparse.Namespace) -> int:
    from .providers import ContentPart

    config = config_from_args(ns)
    provider: ModelProvider = FakeProvider([f"(fake) recebi {len(ns.prompt)} caracteres e {len(ns.image)} imagens"]) if ns.fake else build_provider(config)
    parts = [ContentPart.from_text(ns.prompt)]
    for path in ns.image:
        data = Path(path).read_bytes()
        media = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        parts.append(ContentPart.from_image(data, media))
    try:
        response = await provider.complete(ModelRequest(messages=[ModelMessage("user", parts)]))
    except ProviderError as exc:
        print(f"erro do provider [{exc.code}]: {exc}", file=sys.stderr)
        return 1
    finally:
        await provider.aclose()
    print(response.text)
    if response.usage:
        print(f"-- modelo={response.model} tokens={response.usage} fallback={response.fallback_used}", file=sys.stderr)
    return 0


async def _cmd_memory(ns: argparse.Namespace) -> int:
    config = AgentConfig(project_root=Path(ns.root), log_level=ns.log_level)
    memory = AgentMemory(config.memory_limit, config.memory_path)
    if ns.clear:
        memory.clear()
        print("memória limpa")
        return 0
    print(memory.to_prompt_text(max_entries=50) if len(memory) else "memória vazia")
    return 0


async def _main(argv: list[str]) -> int:
    ns = make_parser().parse_args(argv)
    if ns.command == "run":
        return await _cmd_run(ns)
    if ns.command == "observe":
        return await _cmd_observe(ns)
    if ns.command == "ask":
        return await _cmd_ask(ns)
    if ns.command == "memory":
        return await _cmd_memory(ns)

    config = AgentConfig(project_root=Path(ns.root), log_level=ns.log_level)
    code = CodeManager(config)
    if ns.command == "analyze":
        analysis = await code.analyze(ns.file)
        print(json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False) if ns.json else analysis.outline())
        return 0 if analysis.is_valid else 1
    if ns.command == "backups":
        target = code.resolve(ns.file) if ns.file else None
        for rec in code.backups.list_backups(target):
            flag = " (restaurado)" if rec.restored_at else ""
            print(f"{rec.timestamp}  {rec.original}  ->  {rec.backup_path}  ({rec.reason or '-'}){flag}")
        return 0
    if ns.command == "rollback":
        rec = await code.rollback(ns.file)
        print(f"restaurado {rec.original} a partir de {rec.backup_path} ({rec.timestamp})")
        return 0
    return 2


def report_to_dict(report) -> dict:
    return {
        "script": report.script,
        "status": report.status,
        "duration": report.duration,
        "touched_files": report.touched_files,
        "vision": report.vision_status,
        "context": report.context_summary,
        "iterations": [
            {
                "attempt": it.attempt,
                "before": it.before.summary(),
                "decision": None if it.decision is None else {"action": it.decision.action.value, "strategy": it.decision.strategy, "reason": it.decision.reason},
                "proposal": None if it.proposal is None else {
                    "strategy": it.proposal.strategy,
                    "rationale": it.proposal.rationale,
                    "confidence": it.proposal.confidence,
                    "files": [p.path for p in it.proposal.patches],
                },
                "after": None if it.after is None else it.after.summary(),
                "tests": None if it.tests is None else it.tests.summary(),
                "outcome": it.outcome,
                "note": it.note,
            }
            for it in report.iterations
        ],
    }


def main(argv: list[str] | None = None) -> None:
    try:
        code = asyncio.run(_main(sys.argv[1:] if argv is None else argv))
    except KeyboardInterrupt:
        print("interrompido", file=sys.stderr)
        code = 130
    sys.exit(code)
