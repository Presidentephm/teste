"""
Loop de auto-aprimoramento (Agent Loop).

Ciclo por iteração::

    executar no sandbox
        └─ sucesso? -> encerra com status "fixed"/"already_ok"
        └─ falha:
             1. parseia o traceback e monta o FailureContext
             2. detecta estagnação (mesmo erro N vezes) -> encerra "stagnated"
             3. pede uma FixProposal à estratégia -> None? encerra "no_fix"
             4. aplica os patches (backup automático de cada arquivo)
             5. reexecuta
                  - sucesso                 -> "fixed"
                  - erro de sintaxe/timeout -> rollback dos patches (regressão crítica)
                  - mesmo erro              -> rollback (patch inócuo)
                  - erro diferente          -> mantém (progresso) e continua

Ao final, ``AgentRunReport`` traz o histórico completo e a lista de backups
criados, permitindo ``agent.rollback_run(report)`` para desfazer tudo.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Sequence

from .backup import BackupManager, BackupRecord
from .code_manager import CodeManager, InvalidSourceError, PathOutsideProjectError
from .config import AgentConfig, setup_logging
from .sandbox import ExecutionResult, Sandbox
from .strategies import (
    AttemptSummary,
    CompositeFixStrategy,
    FailureContext,
    FixProposal,
    FixStrategy,
    HeuristicFixStrategy,
)

EventHook = Callable[[str, dict], Awaitable[None] | None]


@dataclass
class IterationRecord:
    attempt: int
    before: ExecutionResult
    proposal: FixProposal | None
    backups: list[BackupRecord]
    after: ExecutionResult | None
    outcome: str  # fixed | new_error | same_error | rolled_back | no_fix | stagnated | patch_failed
    note: str = ""


@dataclass
class AgentRunReport:
    script: str
    status: str  # already_ok | fixed | no_fix | stagnated | exhausted | aborted
    iterations: list[IterationRecord] = field(default_factory=list)
    final_result: ExecutionResult | None = None
    duration: float = 0.0

    @property
    def all_backups(self) -> list[BackupRecord]:
        return [b for it in self.iterations for b in it.backups]

    @property
    def touched_files(self) -> list[str]:
        return sorted({b.original for b in self.all_backups})

    def summary(self) -> str:
        lines = [f"script={self.script} status={self.status} iterações={len(self.iterations)} tempo={self.duration:.1f}s"]
        for it in self.iterations:
            rationale = it.proposal.rationale if it.proposal else "-"
            lines.append(f"  [{it.attempt}] {it.before.summary()} -> {it.outcome}: {rationale}")
        if self.final_result is not None:
            lines.append(f"  resultado final: {self.final_result.summary()}")
        if self.touched_files:
            lines.append(f"  arquivos alterados: {', '.join(self.touched_files)}")
        return "\n".join(lines)


class SelfImprovementAgent:
    """Orquestra CodeManager + Sandbox + BackupManager + FixStrategy."""

    def __init__(
        self,
        config: AgentConfig,
        strategy: FixStrategy | None = None,
        *,
        code_manager: CodeManager | None = None,
        sandbox: Sandbox | None = None,
        on_event: EventHook | None = None,
    ) -> None:
        self.config = config
        self.log = setup_logging(config.log_level)
        self.backups: BackupManager = BackupManager(config)
        self.code = code_manager or CodeManager(config, self.backups)
        self.backups = self.code.backups  # garante um único BackupManager
        self.sandbox = sandbox or Sandbox(config)
        # Sem estratégia explícita, usa só a heurística (funciona offline).
        self.strategy = strategy or CompositeFixStrategy([HeuristicFixStrategy()])
        self._on_event = on_event

    # --------------------------------------------------------------- eventos
    async def _emit(self, event: str, **data) -> None:
        self.log.info("%s %s", event, {k: (v if isinstance(v, (str, int, float)) else "...") for k, v in data.items()})
        if self._on_event is not None:
            maybe = self._on_event(event, data)
            if maybe is not None:
                await maybe

    # --------------------------------------------------------------- contexto
    async def _build_context(
        self, script: str, result: ExecutionResult, attempt: int, history: list[AttemptSummary]
    ) -> FailureContext:
        """Monta o FailureContext localizando o arquivo do projeto onde a exceção estourou."""
        failing_file: str | None = None
        failing_source: str | None = None
        failing_analysis = None
        tb = result.traceback
        if tb:
            # O frame mais interno que pertence ao projeto (ignora stdlib/site-packages).
            for frame in reversed(tb.frames):
                try:
                    rel = self.code.relative(frame.file)
                except (PathOutsideProjectError, ValueError, OSError):
                    continue
                if (self.config.project_root / rel).is_file():
                    failing_file = rel
                    break
        if failing_file is None and tb is None and not result.timed_out:
            failing_file = script  # falha sem traceback: assume o próprio script
        if failing_file:
            try:
                failing_source = await self.code.read(failing_file)
                failing_analysis = self.code.analyze_source(failing_source, failing_file)
            except (OSError, PathOutsideProjectError):
                failing_file = None
        outline = await self.code.analyze_project()
        return FailureContext(
            script=script,
            result=result,
            failing_file=failing_file,
            failing_source=failing_source,
            failing_analysis=failing_analysis,
            project_outline=outline,
            attempt=attempt,
            history=list(history),
        )

    # ------------------------------------------------------------------ loop
    async def run(self, script: str | Path, args: Sequence[str] = ()) -> AgentRunReport:
        """Executa o ciclo de auto-correção sobre ``script`` até sucesso ou limite."""
        started = time.perf_counter()
        script_rel = self.code.relative(script)
        report = AgentRunReport(script=script_rel, status="aborted")
        history: list[AttemptSummary] = []
        seen: dict[str, int] = {}

        await self._emit("run.start", script=script_rel, max_iterations=self.config.max_iterations)
        current = await self.sandbox.run_script(script_rel, args)
        if current.success:
            report.status, report.final_result = "already_ok", current
            report.duration = time.perf_counter() - started
            await self._emit("run.end", status=report.status)
            return report

        for attempt in range(1, self.config.max_iterations + 1):
            sig = current.signature
            seen[sig] = seen.get(sig, 0) + 1
            await self._emit("iteration.start", attempt=attempt, error=current.summary())

            # Estagnação: o mesmo erro reapareceu mais vezes do que o permitido.
            if seen[sig] > self.config.stagnation_limit:
                report.iterations.append(IterationRecord(attempt, current, None, [], None, "stagnated", sig))
                report.status = "stagnated"
                break

            ctx = await self._build_context(script_rel, current, attempt, history)
            proposal = await self.strategy.propose(ctx)
            if proposal is None:
                report.iterations.append(IterationRecord(attempt, current, None, [], None, "no_fix"))
                report.status = "no_fix"
                break
            await self._emit("proposal", attempt=attempt, strategy=proposal.strategy, rationale=proposal.rationale)

            # Aplica os patches (cada arquivo recebe backup antes de ser tocado).
            try:
                backups = await self.code.apply_patches(proposal.patches)
            except (InvalidSourceError, ValueError, PathOutsideProjectError, OSError) as exc:
                # Patch inválido: nada foi alterado (apply_patches já reverteu). Registra e tenta de novo.
                note = f"patch rejeitado: {exc}"
                await self._emit("patch.rejected", attempt=attempt, error=str(exc))
                history.append(AttemptSummary(attempt, sig, proposal.rationale, [p.path for p in proposal.patches], "patch_failed"))
                report.iterations.append(IterationRecord(attempt, current, proposal, [], None, "patch_failed", note))
                continue

            after = await self.sandbox.run_script(script_rel, args)
            patched = [p.path for p in proposal.patches]

            if after.success:
                outcome = "fixed"
            elif self._is_critical_regression(after):
                outcome = "rolled_back"
            elif after.signature == sig:
                outcome = "rolled_back"
            else:
                outcome = "new_error"

            if outcome == "rolled_back":
                await self.backups.rollback_many(backups)
                await self._emit("rollback", attempt=attempt, reason=after.summary())
                history_outcome = "rolled_back"
            else:
                history_outcome = outcome

            history.append(AttemptSummary(attempt, sig, proposal.rationale, patched, history_outcome))
            report.iterations.append(IterationRecord(attempt, current, proposal, backups, after, outcome, after.summary()))
            await self._emit("iteration.end", attempt=attempt, outcome=outcome, result=after.summary())

            if outcome == "fixed":
                report.status, current = "fixed", after
                break
            if outcome == "new_error":
                current = after  # progresso: segue a partir do novo erro
            # rolled_back: mantém `current` (o erro original) e tenta outra abordagem
        else:
            report.status = "exhausted"

        report.final_result = current
        report.duration = time.perf_counter() - started
        await self._emit("run.end", status=report.status, iterations=len(report.iterations))
        return report

    @staticmethod
    def _is_critical_regression(result: ExecutionResult) -> bool:
        """Sintaxe quebrada, import quebrado ou travamento = reverter imediatamente."""
        if result.timed_out:
            return True
        tb = result.traceback
        return bool(tb and tb.exc_type in {"SyntaxError", "IndentationError", "TabError", "ImportError", "ModuleNotFoundError"})

    # --------------------------------------------------------------- rollback
    async def rollback_run(self, report: AgentRunReport) -> list[BackupRecord]:
        """Desfaz todas as alterações que sobreviveram numa execução (``run``)."""
        surviving = [b for it in report.iterations if it.outcome in ("fixed", "new_error") for b in it.backups]
        restored = await self.backups.rollback_many(surviving)
        await self._emit("run.rollback", files=len(restored))
        return restored
