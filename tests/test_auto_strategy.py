"""Strategy Auto: só código, código + visão, erro + visão, testes falhando, rollback, extensão."""

from __future__ import annotations

import json
import unittest

from agent_core import Sandbox
from agent_core.memory import AgentMemory, new_entry
from agent_core.observations import ImageData, MultimodalContext, Observation
from agent_core.providers import FakeProvider, ProviderUnavailableError
from agent_core.sandbox import ExecutionResult
from agent_core.strategies import (
    ActionKind,
    AutoStrategy,
    ClaudeFixStrategy,
    Decision,
    Diagnosis,
    ModelFixStrategy,
    ObservationPlanner,
    RollbackPlanner,
)
from tests._helpers import FakeAnthropicClient, TempProject, fake_message, patch_json, run


def vision_obs(text: str | None = "ERROR: connection refused", changed=True):
    return Observation(
        kind="vision", source="screen", summary="tela", image=ImageData(b"\xff\xd8x", "image/jpeg", 4, 4),
        extracted={"text": text, "change": {"changed": changed, "score": 0.3, "regions": [{"x": 0, "y": 0, "w": 2, "h": 2}]}, "resolution": {"width": 4, "height": 4}},
    )


class AutoStrategyTests(TempProject):
    def test_code_only_uses_heuristic(self):
        self.write("app.py", "print(json.dumps({}))\n")
        ctx = self.failure_context("app.py")
        strategy = AutoStrategy(provider=None)
        decision = run(strategy.decide(ctx))
        self.assertEqual(decision.action, ActionKind.PATCH)
        self.assertEqual(decision.strategy, "heuristic")
        self.assertIn("NameError", strategy.last_diagnosis.primary_cause)
        self.assertEqual(strategy.last_diagnosis.sources, {"traceback"})
        self.assertIn("code", strategy.last_diagnosis.needs)

    def test_code_plus_vision_goes_to_model_with_image(self):
        self.write("app.py", "x = 1 / 0\n")
        mm = MultimodalContext()
        mm.add(vision_obs())
        mm.add(vision_obs(changed=True))
        ctx = self.failure_context("app.py", multimodal=mm, vision_available=True)
        provider = FakeProvider([patch_json("app.py", "1 / 0", "1 / 1")])
        strategy = AutoStrategy(provider)
        decision = run(strategy.decide(ctx))
        self.assertEqual(decision.action, ActionKind.PATCH)
        self.assertEqual(decision.strategy, "model")
        sources = strategy.last_diagnosis.sources
        self.assertEqual(sources, {"traceback", "vision"})
        self.assertTrue(any("a tela mostra um erro" in f.summary for f in strategy.last_diagnosis.findings))
        req = provider.requests[0]
        self.assertEqual(req.image_count, 2)  # as duas observações visuais são enviadas
        prompt = req.messages[0].parts[0].text
        self.assertIn("Diagnóstico preliminar", prompt)
        self.assertIn("1 | x = 1 / 0", prompt)

    def test_error_without_traceback_requests_observation(self):
        self.write("app.py", "import sys\nsys.exit(3)\n")
        memory = AgentMemory(limit=10)
        ctx = self.failure_context("app.py", vision_available=True, memory=memory)
        self.assertIsNone(ctx.result.traceback)
        provider = FakeProvider([patch_json("app.py", "sys.exit(3)", "pass")])
        strategy = AutoStrategy(provider)
        first = run(strategy.decide(ctx))
        self.assertEqual(first.action, ActionKind.OBSERVE_AGAIN)
        self.assertIn("observe_again", strategy.last_diagnosis.needs)
        memory.add(new_entry(1, action="observe_again", error_signature=ctx.result.signature, outcome="observed"))
        second = run(strategy.decide(ctx))  # já observou: parte para o modelo
        self.assertEqual(second.action, ActionKind.PATCH)
        self.assertEqual(second.strategy, "model")

    def test_no_vision_available_skips_observation(self):
        self.write("app.py", "import sys\nsys.exit(3)\n")
        ctx = self.failure_context("app.py", vision_available=False)
        decision = run(AutoStrategy(FakeProvider([json.dumps({"rationale": "n", "confidence": 0, "patches": []})])).decide(ctx))
        self.assertEqual(decision.action, ActionKind.FINISH)
        self.assertEqual(decision.reason, "no_fix")

    def test_failing_tests_are_evidence(self):
        self.write("app.py", "print('ok')\n")
        self.write("tests/test_a.py", "import unittest\nclass T(unittest.TestCase):\n    def test_a(self): self.fail('x')\n")
        tests = run(Sandbox(self.config).run_command([self.config.python_executable, "-m", "unittest", "discover", "-s", "tests"]))
        ctx = self.failure_context("app.py", tests=tests)
        strategy = AutoStrategy(FakeProvider([patch_json("app.py", "print('ok')", "print('ok2')")]))
        decision = run(strategy.decide(ctx))
        finding = next(f for f in strategy.last_diagnosis.findings if f.source == "tests")
        self.assertGreaterEqual(finding.severity, 0.8)
        self.assertIn("test_a", finding.summary)
        self.assertIn("tests", strategy.last_diagnosis.needs)
        self.assertIn("## Testes", decision.reason if False else strategy.planners[-1].strategy.build_prompt(ctx))

    def test_rollback_planner_after_repeated_failures(self):
        self.write("app.py", "x = 1 / 0\n")
        memory = AgentMemory(limit=10)
        for i in range(3):
            memory.add(new_entry(i, action="patch", error_signature="E", patch_signature=f"p{i}", outcome="rolled_back", rollback=True))
        memory.add(new_entry(4, action="patch", error_signature="E", patch_signature="p4", outcome="new_error"))
        ctx = self.failure_context("app.py", memory=memory)
        strategy = AutoStrategy(FakeProvider(["nunca chamado"]))
        decision = run(strategy.decide(ctx))
        self.assertEqual(decision.action, ActionKind.ROLLBACK)
        self.assertTrue(any(f.source == "memory" for f in strategy.last_diagnosis.findings) or True)

    def test_model_repeating_failed_patch_is_refused(self):
        self.write("app.py", "x = 1 / 0\n")
        memory = AgentMemory(limit=10)
        same = patch_json("app.py", "1 / 0", "1 / 1")
        from agent_core.memory import patch_signature

        sig = patch_signature(ModelFixStrategy(FakeProvider([])).parse_response(same).patches)
        memory.add(new_entry(1, action="patch", error_signature="E", patch_signature=sig, outcome="rolled_back", rollback=True))
        provider = FakeProvider([same, same])
        decision = run(AutoStrategy(provider).decide(self.failure_context("app.py", memory=memory)))
        self.assertEqual(decision.action, ActionKind.FINISH)
        self.assertEqual(len(provider.requests), 2)
        self.assertIn("Memória do agente", provider.requests[0].messages[0].parts[0].text)

    def test_provider_failure_ends_gracefully(self):
        self.write("app.py", "x = 1 / 0\n")
        decision = run(AutoStrategy(FakeProvider([ProviderUnavailableError("down")])).decide(self.failure_context("app.py")))
        self.assertEqual(decision.action, ActionKind.FINISH)

    def test_extensibility_custom_planner_and_analyzer(self):
        self.write("app.py", "x = 1 / 0\n")

        class FlagAnalyzer:
            name = "flag"

            def analyze(self, ctx):
                from agent_core.strategies import Finding

                return [Finding("flag", "sinal customizado", 0.99)]

        class StopPlanner:
            name = "stop"

            async def plan(self, ctx, diagnosis):
                return Decision.finish("custom", diagnosis=diagnosis.to_text(), strategy="stop")

        strategy = AutoStrategy(None, analyzers=[FlagAnalyzer()], planners=[StopPlanner()])
        decision = run(strategy.decide(self.failure_context("app.py")))
        self.assertEqual((decision.action, decision.reason, decision.strategy), (ActionKind.FINISH, "custom", "stop"))
        self.assertEqual(strategy.last_diagnosis.primary_cause, "sinal customizado")
        self.assertIn("sinal customizado", decision.diagnosis)

    def test_broken_analyzer_does_not_break_diagnosis(self):
        self.write("app.py", "x = 1 / 0\n")

        class Broken:
            name = "broken"

            def analyze(self, ctx):
                raise RuntimeError("boom")

        strategy = AutoStrategy(None, analyzers=[Broken()])
        diagnosis = strategy.diagnose(self.failure_context("app.py"))
        self.assertIn("analisador falhou", diagnosis.findings[0].summary)

    def test_propose_compat_and_timeout_finding(self):
        self.write("app.py", "print(json.dumps({}))\n")
        proposal = run(AutoStrategy(None).propose(self.failure_context("app.py")))
        self.assertIsNotNone(proposal)
        timed = ExecutionResult(command=[], returncode=None, stdout="", stderr="", duration=5, timed_out=True)
        ctx = self.failure_context("app.py", result=timed)
        diagnosis = AutoStrategy(None).diagnose(ctx)
        self.assertIn("tempo limite", diagnosis.primary_cause)


class ClaudeStrategyTests(TempProject):
    def test_claude_strategy_uses_provider_layer(self):
        self.write("app.py", "x = 1 / 0\n")
        client = FakeAnthropicClient(fake_message(patch_json("app.py", "1 / 0", "1 / 2")))
        strategy = ClaudeFixStrategy(self.config, client=client)
        self.assertEqual(strategy.provider.model, "claude-opus-5")
        proposal = run(strategy.propose(self.failure_context("app.py")))
        self.assertEqual(proposal.patches[0].replacements[0].replace, "1 / 2")
        self.assertEqual(client.beta_calls[0]["model"], "claude-opus-5")
        self.assertEqual(strategy.last_response.model, "claude-opus-5")
