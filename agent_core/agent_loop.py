"""
Loop autônomo de auto-aprimoramento.

Ciclo por iteração::

    OBSERVAR   runtime (sandbox) + testes + logs + visão  -> MultimodalContext
    CONTEXTO   FailureContext (traceback, fonte, AST, contexto, memória)
    DECIDIR    strategy.decide(ctx) -> Decision
    AGIR       patch | observe_again | run_tests | rollback | finish
    VALIDAR    reexecuta script + testes + nova observação
    MEMÓRIA    grava MemoryEntry (o que foi tentado e o resultado)
    RESULTADO  fixed | new_error (mantém) | rolled_back (restaura checkpoint)

Proteções: checkpoint antes de cada patch, ``PatchGuard`` (arquivos demais,
esvaziamento, remoção massiva), rejeição de patch já fracassado (memória),
limites de iterações/retries/tempo, detecção de estagnação e rollback
automático em regressão crítica (sintaxe, import, timeout, testes que
passavam e quebraram).

A visão é opcional e desacoplada: falhas de câmera/tela apenas desligam a
observação visual; o ciclo continua com as demais evidências.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from .backup import BackupManager, BackupRecord, Checkpoint
from .code_manager import CodeManager, InvalidSourceError, PathOutsideProjectError
from .config import AgentConfig, setup_logging
from .memory import AgentMemory, new_entry
from .observations import (
    CodeObserver,
    LogObserver,
    MultimodalContext,
    Observation,
    ObservationKind,
    RuntimeObserver,
    TestObserver,
)
from .providers import ModelProvider, ProviderInterrupted
from .safety import PatchGuard, UnsafePatchError, redact
from .sandbox import ExecutionResult, Sandbox
from .strategies import (
    ActionKind,
    AttemptSummary,
    AutoStrategy,
    CompositeFixStrategy,
    Decision,
    FailureContext,
    FixProposal,
    FixStrategy,
    HeuristicFixStrategy,
)

EventHook = Callable[[str, dict], Awaitable[None] | None]

CRITICAL_EXCEPTIONS = {"SyntaxError", "IndentationError", "TabError", "ImportError", "ModuleNotFoundError"}


@dataclass
class IterationRecord:
    attempt: int
    before: ExecutionResult
    proposal: FixProposal | None
    backups: list[BackupRecord]
    after: ExecutionResult | None
    outcome: str  # fixed | new_error | rolled_back | no_fix | stagnated | patch_failed | patch_rejected | repeated_patch | observed | tested
    note: str = ""
    decision: Decision | None = None
    tests: ExecutionResult | None = None
    checkpoint_id: str = ""


@dataclass
class AgentRunReport:
    script: str
    status: str  # already_ok | fixed | no_fix | stagnated | exhausted | retries_exhausted | timeout | interrupted | aborted
    iterations: list[IterationRecord] = field(default_factory=list)
    final_result: ExecutionResult | None = None
    duration: float = 0.0
    vision_status: dict[str, Any] | None = None
    context_summary: str = ""
    usage: dict[str, Any] | None = None  # tokens/custo do provider durante o run

    @property
    def all_backups(self) -> list[BackupRecord]:
        return [b for it in self.iterations for b in it.backups]

    @property
    def touched_files(self) -> list[str]:
        return sorted({b.original for b in self.all_backups})

    @property
    def success(self) -> bool:
        return self.status in ("fixed", "already_ok")

    def summary(self) -> str:
        lines = [f"script={self.script} status={self.status} iterações={len(self.iterations)} tempo={self.duration:.1f}s"]
        for it in self.iterations:
            what = it.proposal.rationale if it.proposal else (it.decision.reason if it.decision else "-")
            lines.append(f"  [{it.attempt}] {it.before.summary()} -> {it.outcome}: {what}")
        if self.final_result is not None:
            lines.append(f"  resultado final: {self.final_result.summary()}")
        if self.touched_files:
            lines.append(f"  arquivos alterados: {', '.join(self.touched_files)}")
        if self.vision_status:
            lines.append(f"  visão: {self.vision_status}")
        if self.context_summary:
            lines.append(f"  contexto: {self.context_summary}")
        if self.usage and self.usage.get("calls"):
            u = self.usage
            cost = f" custo≈US${u['cost_usd']:.4f}" if u.get("priced") else ""
            lines.append(f"  modelo: {u['calls']} chamadas, {u['input_tokens']} in / {u['output_tokens']} out tokens, cache {u['cache_read_tokens']} lidos{cost}")
        return redact("\n".join(lines))


@dataclass
class _Snapshot:
    """Resultado de uma rodada de execução (script + testes)."""

    result: ExecutionResult
    tests: ExecutionResult | None = None

    @property
    def success(self) -> bool:
        return self.result.success and (self.tests is None or self.tests.success)

    @property
    def signature(self) -> str:
        sig = self.result.signature
        if self.result.success and self.tests is not None and not self.tests.success:
            sig = f"TESTS:{self.tests.returncode}"
        return sig


class SelfImprovementAgent:
    """Orquestra observadores + CodeManager + Sandbox + BackupManager + estratégia + memória."""

    def __init__(
        self,
        config: AgentConfig,
        strategy: FixStrategy | None = None,
        *,
        code_manager: CodeManager | None = None,
        sandbox: Sandbox | None = None,
        on_event: EventHook | None = None,
        vision: Any | None = None,
        memory: AgentMemory | None = None,
        provider: ModelProvider | None = None,
    ) -> None:
        self.config = config
        self.log = setup_logging(config.log_level)
        self.code = code_manager or CodeManager(config)
        self.backups: BackupManager = self.code.backups
        self.sandbox = sandbox or Sandbox(config)
        if strategy is None:
            # Com provider -> estratégia auto (multimodal); sem -> heurística offline.
            strategy = AutoStrategy(provider) if provider is not None else CompositeFixStrategy([HeuristicFixStrategy()])
        self.strategy = strategy
        self._on_event = on_event
        self.memory = memory if memory is not None else AgentMemory(config.memory_limit, config.memory_path)
        self.guard = PatchGuard(max_files=config.max_patch_files, max_removed_ratio=config.max_removed_ratio)
        self.context = MultimodalContext()
        self.vision = vision if vision is not None else self._build_vision()
        self._vision_observer = None
        if self.vision is not None:
            from .vision.capture import VisionObserver

            self._vision_observer = VisionObserver(self.vision, fresh=False)
        self._runtime_observer = RuntimeObserver()
        self._test_observer = TestObserver(self.sandbox, config.test_command)
        self._log_observer = LogObserver(config.project_root, config.log_patterns)
        self._code_observer = CodeObserver(self.code)
        self._surviving: list[Checkpoint] = []

    # ------------------------------------------------------------ construção
    def _build_vision(self) -> Any | None:
        if not self.config.vision_enabled:
            return None
        try:
            from .vision.capture import build_vision_capture

            return build_vision_capture(self.config)
        except Exception as exc:  # configuração/dispositivo: segue sem visão
            self.log.warning("visão indisponível (%s); prosseguindo sem ela", exc)
            return None

    @property
    def vision_available(self) -> bool:
        return self.vision is not None and (self.vision.is_running or self.vision.error is None)

    # --------------------------------------------------------------- eventos
    async def _emit(self, event: str, **data: Any) -> None:
        printable = {k: (v if isinstance(v, (str, int, float, bool)) else "...") for k, v in data.items()}
        self.log.info("%s %s", event, redact(str(printable)))
        if self._on_event is not None:
            maybe = self._on_event(event, data)
            if maybe is not None:
                await maybe

    # ------------------------------------------------------------- observar
    async def _execute(self, script: str, args: Sequence[str]) -> _Snapshot:
        result = await self.sandbox.run_script(script, args)
        tests = await self._test_observer.run() if self.config.test_command else None
        return _Snapshot(result, tests)

    async def _observe(self, snap: _Snapshot, *, label: str = "script") -> list[Observation]:
        """Coleta observações de todas as fontes e as adiciona ao contexto."""
        observations: list[Observation] = []
        observations += await self._runtime_observer.observe(result=snap.result, label=label)
        if snap.tests is not None:
            observations += await self._test_observer.observe(result=snap.tests)
        observations += await self._log_observer.observe()
        if self._vision_observer is not None:
            observations += await self._vision_observer.observe()
        self.context.extend(observations)
        return observations

    async def _build_context(self, script: str, snap: _Snapshot, attempt: int, history: list[AttemptSummary]) -> FailureContext:
        """Monta o FailureContext localizando o arquivo do projeto onde a exceção estourou."""
        result = snap.result
        failing_file: str | None = None
        failing_source: str | None = None
        failing_analysis = None
        tb = result.traceback
        if tb:
            for frame in reversed(tb.frames):  # frame mais interno que pertence ao projeto
                try:
                    rel = self.code.relative(frame.file)
                except (PathOutsideProjectError, ValueError, OSError):
                    continue
                if (self.config.project_root / rel).is_file():
                    failing_file = rel
                    break
        if failing_file is None and tb is None and not result.timed_out and not result.success:
            failing_file = script  # falha sem traceback: assume o próprio script
        if failing_file:
            try:
                failing_source = await self.code.read(failing_file)
                failing_analysis = self.code.analyze_source(failing_source, failing_file)
            except (OSError, PathOutsideProjectError):
                failing_file = None
        outline = await self.code.analyze_project()
        # Observação de código entra no contexto (substituindo a anterior).
        code_obs = await self._code_observer.observe(failing_file=failing_file)
        self.context.extend(code_obs)
        return FailureContext(
            script=script,
            result=result,
            failing_file=failing_file,
            failing_source=failing_source,
            failing_analysis=failing_analysis,
            project_outline=outline,
            attempt=attempt,
            history=list(history),
            multimodal=self.context,
            memory=self.memory,
            tests=snap.tests,
            vision_available=self.vision_available,
            code_manager=self.code,
            effort=self.config.effort_for(snap.signature),
        )

    # ------------------------------------------------------------------ loop
    async def run(self, script: str | Path, args: Sequence[str] = ()) -> AgentRunReport:
        """Executa o ciclo autônomo sobre ``script`` até sucesso ou limite."""
        started = time.perf_counter()
        script_rel = self.code.relative(script)
        report = AgentRunReport(script=script_rel, status="aborted")
        self.context = MultimodalContext()
        self._surviving = []
        usage_before = self._usage_snapshot()
        if self.vision is not None:
            ok = await self.vision.start()
            if not ok:
                await self._emit("vision.unavailable", error=self.vision.error or "?")
        try:
            await self._loop(script_rel, args, report, started)
        except (asyncio.CancelledError, KeyboardInterrupt, ProviderInterrupted):
            report.status = "interrupted"
            await self._emit("run.interrupted")
        finally:
            report.duration = time.perf_counter() - started
            report.context_summary = self.context.summary()
            report.usage = self._usage_delta(usage_before)
            if self.vision is not None:
                report.vision_status = self.vision.status()
                await self.vision.stop()
            await self._emit("run.end", status=report.status, iterations=len(report.iterations))
        return report

    @property
    def provider(self) -> ModelProvider | None:
        """Provider usado pela estratégia, se ela expuser um."""
        return getattr(self.strategy, "provider", None)

    def _usage_snapshot(self):
        provider = self.provider
        return copy.copy(provider.usage) if provider is not None else None

    def _usage_delta(self, before) -> dict[str, Any] | None:
        provider = self.provider
        if provider is None or before is None:
            return None
        after = provider.usage
        delta = {k: getattr(after, k) - getattr(before, k) for k in ("calls", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "latency_total", "cost_usd")}
        delta["cost_usd"] = round(delta["cost_usd"], 6)
        delta["latency_total"] = round(delta["latency_total"], 3)
        delta["priced"] = after.priced
        return delta

    def _out_of_time(self, started: float) -> bool:
        return self.config.total_timeout is not None and (time.perf_counter() - started) > self.config.total_timeout

    async def _loop(self, script: str, args: Sequence[str], report: AgentRunReport, started: float) -> None:
        history: list[AttemptSummary] = []
        seen: dict[str, int] = {}
        retries = 0

        await self._emit("run.start", script=script, max_iterations=self.config.max_iterations, strategy=self.strategy.name)
        current = await self._execute(script, args)
        await self._observe(current)
        if current.success:
            report.status, report.final_result = "already_ok", current.result
            return

        for attempt in range(1, self.config.max_iterations + 1):
            if self._out_of_time(started):
                report.status = "timeout"
                break
            if retries > self.config.max_retries:
                report.status = "retries_exhausted"
                break
            sig = current.signature
            seen[sig] = seen.get(sig, 0) + 1
            await self._emit("iteration.start", attempt=attempt, error=current.result.summary(), tests=current.tests.summary() if current.tests else "-")
            if seen[sig] > self.config.stagnation_limit:
                report.iterations.append(IterationRecord(attempt, current.result, None, [], None, "stagnated", sig))
                report.status = "stagnated"
                break

            ctx = await self._build_context(script, current, attempt, history)
            decision = await self.strategy.decide(ctx)
            await self._emit("decision", attempt=attempt, action=decision.action.value, strategy=decision.strategy, reason=decision.reason)

            if decision.action == ActionKind.FINISH:
                report.iterations.append(IterationRecord(attempt, current.result, None, [], None, "no_fix", decision.reason, decision))
                self._remember(attempt, ctx, decision, outcome="no_fix")
                report.status = "no_fix"
                break

            if decision.action == ActionKind.OBSERVE_AGAIN:
                if self.vision is not None and self.vision.is_running and self.config.observation_interval:
                    await asyncio.sleep(self.config.observation_interval)
                current = await self._execute(script, args)
                await self._observe(current, label="re-observation")
                report.iterations.append(IterationRecord(attempt, ctx.result, None, [], current.result, "observed", decision.reason, decision, current.tests))
                self._remember(attempt, ctx, decision, outcome="observed", result=current.result.summary())
                if current.success:
                    report.status = "fixed"
                    break
                continue

            if decision.action == ActionKind.RUN_TESTS:
                tests = await self._test_observer.run()
                current = _Snapshot(current.result, tests)
                if tests is not None:
                    self.context.extend(await self._test_observer.observe(result=tests))
                report.iterations.append(IterationRecord(attempt, ctx.result, None, [], None, "tested", decision.reason, decision, tests))
                self._remember(attempt, ctx, decision, outcome="tested", tests=tests.summary() if tests else "sem comando de testes")
                continue

            if decision.action == ActionKind.ROLLBACK:
                restored = await self._rollback_last()
                current = await self._execute(script, args)
                await self._observe(current, label="after-rollback")
                report.iterations.append(IterationRecord(attempt, ctx.result, None, [], current.result, "rolled_back", f"{restored} arquivos restaurados", decision, current.tests))
                self._remember(attempt, ctx, decision, outcome="rolled_back", rollback=True, result=current.result.summary())
                retries += 1
                continue

            # ---- PATCH
            outcome, after = await self._apply_and_validate(script, args, attempt, ctx, decision, current, report, history)
            if outcome == "fixed":
                report.status, current = "fixed", after
                break
            if outcome == "new_error":
                current = after  # progresso: segue a partir do novo erro
            else:
                retries += 1  # rolled_back / patch_failed / patch_rejected / repeated_patch
        else:
            report.status = "exhausted"

        report.final_result = current.result

    # ----------------------------------------------------------------- patch
    async def _apply_and_validate(
        self,
        script: str,
        args: Sequence[str],
        attempt: int,
        ctx: FailureContext,
        decision: Decision,
        current: _Snapshot,
        report: AgentRunReport,
        history: list[AttemptSummary],
    ) -> tuple[str, _Snapshot]:
        proposal = decision.proposal
        assert proposal is not None
        sig = current.signature
        files = [p.path for p in proposal.patches]
        psig = proposal.signature

        def _record(outcome: str, note: str, backups: list[BackupRecord] | None = None, after: _Snapshot | None = None, cp: Checkpoint | None = None) -> None:
            report.iterations.append(
                IterationRecord(attempt, ctx.result, proposal, backups or [], after.result if after else None, outcome, note, decision, after.tests if after else None, cp.id if cp else "")
            )
            history.append(AttemptSummary(attempt, sig, proposal.rationale, files, outcome))
            self._remember(
                attempt, ctx, decision, outcome=outcome, patch_sig=psig, files=files,
                result=after.result.summary() if after else note,
                tests=after.tests.summary() if after and after.tests else "",
                rollback=outcome == "rolled_back", errors=[note] if outcome in ("patch_failed", "patch_rejected", "repeated_patch") else [],
            )

        # 1. memória: não repete um patch que já fracassou
        if self.memory.has_tried(psig):
            note = "patch idêntico a uma tentativa anterior fracassada"
            await self._emit("patch.repeated", attempt=attempt, files=", ".join(files))
            _record("repeated_patch", note)
            return "repeated_patch", current

        # 2. guarda de segurança
        try:
            sources = await self.code.current_sources(files)
            self.guard.check(proposal.patches, sources)
            targets = [self.code.resolve(f, for_write=True) for f in files]
        except (UnsafePatchError, PathOutsideProjectError) as exc:
            note = f"patch rejeitado: {exc}"
            await self._emit("patch.rejected", attempt=attempt, error=str(exc))
            _record("patch_rejected", note)
            return "patch_rejected", current

        # 3. checkpoint + aplicação
        checkpoint = await self.backups.checkpoint(targets, label=f"iter{attempt}")
        try:
            await self.code.apply_patches(proposal.patches, backup=False)
        except (InvalidSourceError, ValueError, PathOutsideProjectError, OSError) as exc:
            await self.backups.restore(checkpoint)
            note = f"patch inválido: {exc}"
            await self._emit("patch.failed", attempt=attempt, error=str(exc))
            _record("patch_failed", note, checkpoint.records, None, checkpoint)
            return "patch_failed", current
        await self._emit("patch.applied", attempt=attempt, files=", ".join(files), strategy=decision.strategy)

        # 4. validação: reexecuta script + testes + observa de novo
        after = await self._execute(script, args)
        await self._observe(after, label="validation")
        outcome = self._judge(current, after, sig)
        note = after.result.summary() if after.tests is None else f"{after.result.summary()} | testes: {after.tests.summary()}"
        if outcome == "rolled_back":
            await self.backups.restore(checkpoint)
            await self._emit("rollback", attempt=attempt, reason=note)
        else:
            self._surviving.append(checkpoint)
        _record(outcome, note, checkpoint.records, after, checkpoint)
        await self._emit("iteration.end", attempt=attempt, outcome=outcome, result=note)
        return outcome, after

    def _judge(self, before: _Snapshot, after: _Snapshot, before_sig: str) -> str:
        """Classifica o efeito de um patch."""
        if after.success:
            return "fixed"
        if self._is_critical_regression(after.result):
            return "rolled_back"
        if after.signature == before_sig:
            return "rolled_back"  # patch inócuo
        if before.tests is not None and before.tests.success and after.tests is not None and not after.tests.success:
            return "rolled_back"  # quebrou testes que passavam
        return "new_error"

    @staticmethod
    def _is_critical_regression(result: ExecutionResult) -> bool:
        """Sintaxe quebrada, import quebrado ou travamento = reverter imediatamente."""
        if result.timed_out:
            return True
        tb = result.traceback
        return bool(tb and tb.exc_type in CRITICAL_EXCEPTIONS)

    async def _rollback_last(self) -> int:
        """Restaura o checkpoint sobrevivente mais recente (ação ROLLBACK)."""
        if not self._surviving:
            return 0
        checkpoint = self._surviving.pop()
        restored = await self.backups.restore(checkpoint)
        return len(restored)

    # ---------------------------------------------------------------- memória
    def _remember(self, attempt: int, ctx: FailureContext, decision: Decision, *, outcome: str, patch_sig: str = "", files: list[str] | None = None, result: str = "", tests: str = "", rollback: bool = False, errors: list[str] | None = None) -> None:
        self.memory.add(
            new_entry(
                attempt,
                observation=self.context.summary() + " | " + ctx.result.summary(),
                diagnosis=decision.diagnosis[:500] or decision.reason,
                strategy=decision.strategy or self.strategy.name,
                action=decision.action.value,
                error_signature=ctx.result.signature,
                patch_signature=patch_sig,
                patch_files=list(files or []),
                tests=tests,
                result=result,
                outcome=outcome,
                rollback=rollback,
                errors=list(errors or []),
            )
        )

    # --------------------------------------------------------------- rollback
    async def rollback_run(self, report: AgentRunReport) -> list[BackupRecord]:
        """Desfaz todas as alterações que sobreviveram numa execução (``run``)."""
        surviving = [b for it in report.iterations if it.outcome in ("fixed", "new_error") for b in it.backups]
        restored = await self.backups.rollback_many(surviving)
        self._surviving = []
        await self._emit("run.rollback", files=len(restored))
        return restored
