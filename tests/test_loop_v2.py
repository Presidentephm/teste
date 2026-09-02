"""AgentLoop: ciclo completo, sucesso, retry, falha, rollback, limites, ações, visão, memória."""

from __future__ import annotations

import json
import unittest

from agent_core import AgentConfig, SelfImprovementAgent
from agent_core.memory import AgentMemory
from agent_core.observations import ObservationKind
from agent_core.providers import FakeProvider, ProviderInterrupted
from agent_core.strategies import ActionKind, AutoStrategy, Decision, FixStrategy
from agent_core.vision import vision_available
from tests._helpers import TempProject, full_json, patch_json, run

DIV = "def f(x):\n    return 10 / x\n\nprint(f(0))\n"


class Scripted(FixStrategy):
    """Estratégia que devolve decisões pré-programadas (para testar o loop)."""

    name = "scripted"

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.contexts = []

    async def propose(self, ctx):
        return None

    async def decide(self, ctx):
        self.contexts.append(ctx)
        if not self.decisions:
            return Decision.finish("no_fix", strategy=self.name)
        item = self.decisions.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class FullCycleTests(TempProject):
    def test_success_with_auto_strategy_and_model(self):
        self.write("app.py", DIV)
        provider = FakeProvider([patch_json("app.py", "f(0)", "f(2)")])
        agent = SelfImprovementAgent(self.config, AutoStrategy(provider))
        report = run(agent.run("app.py"))
        self.assertEqual(report.status, "fixed")
        self.assertTrue(report.success)
        self.assertEqual([it.outcome for it in report.iterations], ["fixed"])
        self.assertEqual(report.final_result.stdout.strip(), "5.0")
        self.assertIn("f(2)", (self.root / "app.py").read_text())
        self.assertEqual(len(agent.memory), 1)
        entry = agent.memory.last()
        self.assertEqual((entry.action, entry.outcome, entry.strategy), ("patch", "fixed", "model"))
        self.assertIn("1 | def f(x):", provider.requests[0].messages[0].parts[0].text)
        self.assertIn("runtime=", report.context_summary)
        self.assertEqual(report.iterations[0].decision.action, ActionKind.PATCH)
        self.assertTrue(report.iterations[0].checkpoint_id)

    def test_retry_after_invalid_patch(self):
        self.write("app.py", DIV)
        provider = FakeProvider([patch_json("app.py", "nao existe", "x"), patch_json("app.py", "f(0)", "f(5)")])
        report = run(SelfImprovementAgent(self.config, AutoStrategy(provider)).run("app.py"))
        self.assertEqual([it.outcome for it in report.iterations], ["patch_failed", "fixed"])
        self.assertIn("patch inválido", report.iterations[0].note)
        self.assertEqual(report.status, "fixed")

    def test_failure_when_model_has_no_fix(self):
        self.write("app.py", DIV)
        provider = FakeProvider([json.dumps({"rationale": "sem ideia", "confidence": 0, "patches": []})])
        report = run(SelfImprovementAgent(self.config, AutoStrategy(provider)).run("app.py"))
        self.assertEqual(report.status, "no_fix")
        self.assertEqual(report.iterations[0].outcome, "no_fix")

    def test_rollback_when_tests_regress(self):
        self.write("app.py", "from lib import double\nprint(double(2) / 0)\n")
        self.write("lib.py", "def double(x):\n    return x * 2\n")
        self.write("tests/test_lib.py", "import unittest\nfrom lib import double\nclass T(unittest.TestCase):\n    def test_d(self): self.assertEqual(double(2), 4)\n")
        self.config.test_command = ("-m", "unittest", "discover", "-s", "tests")
        # 1º patch: "corrige" o script quebrando a lib (testes regridem) -> rollback.
        bad = json.dumps({"rationale": "bad", "confidence": 0.5, "patches": [
            {"path": "lib.py", "mode": "replace_full", "content": "def double(x):\n    return 0\n"},
            {"path": "app.py", "mode": "search_replace", "replacements": [{"search": "/ 0", "replace": "/ 1"}]},
        ]})
        good = patch_json("app.py", "/ 0", "/ 1")
        agent = SelfImprovementAgent(self.config, AutoStrategy(FakeProvider([bad, good])))
        report = run(agent.run("app.py"))
        self.assertEqual([it.outcome for it in report.iterations], ["rolled_back", "fixed"])
        self.assertIn("return x * 2", (self.root / "lib.py").read_text())
        self.assertTrue(report.iterations[1].tests.success)
        self.assertTrue(agent.memory.entries[0].rollback)
        self.assertEqual(report.status, "fixed")

    def test_iteration_limit(self):
        self.write("app.py", DIV)
        self.config.max_iterations = 1
        provider = FakeProvider([patch_json("app.py", "f(0)", "g(0)")])  # gera NameError (progresso, não corrige)
        report = run(SelfImprovementAgent(self.config, AutoStrategy(provider, use_heuristics=False)).run("app.py"))
        self.assertEqual(report.status, "exhausted")
        self.assertEqual(report.iterations[0].outcome, "new_error")

    def test_retry_limit(self):
        self.write("app.py", DIV)
        self.config.max_retries = 0
        self.config.max_iterations = 5
        provider = FakeProvider([patch_json("app.py", "f(0)", "f(0 + 0)"), patch_json("app.py", "f(0)", "f(3)")])
        report = run(SelfImprovementAgent(self.config, AutoStrategy(provider)).run("app.py"))
        self.assertEqual(report.iterations[0].outcome, "rolled_back")  # mesmo erro = patch inócuo
        self.assertEqual(report.status, "retries_exhausted")

    def test_total_timeout(self):
        self.write("app.py", DIV)
        self.config.total_timeout = 0.0
        report = run(SelfImprovementAgent(self.config, AutoStrategy(FakeProvider(["x"]))).run("app.py"))
        self.assertEqual(report.status, "timeout")
        self.assertEqual(report.iterations, [])

    def test_guard_rejects_destructive_patch(self):
        self.write("app.py", DIV + "\n" * 10)
        provider = FakeProvider([full_json("app.py", ""), patch_json("app.py", "f(0)", "f(1)")])
        report = run(SelfImprovementAgent(self.config, AutoStrategy(provider)).run("app.py"))
        self.assertEqual([it.outcome for it in report.iterations], ["patch_rejected", "fixed"])
        self.assertIn("esvazia", report.iterations[0].note)

    def test_repeated_failed_patch_is_skipped(self):
        self.write("app.py", DIV)
        same = patch_json("app.py", "f(0)", "f(0 + 0)")
        self.config.max_iterations = 3
        # 1) Estratégia sem memória própria: o loop bloqueia a repetição.
        from agent_core.code_manager import FilePatch, Replacement
        from agent_core.strategies import FixProposal

        class Stubborn(FixStrategy):
            name = "stubborn"

            async def propose(self, ctx):
                return FixProposal([FilePatch("app.py", replacements=[Replacement("f(0)", "f(0 + 0)")])], "sempre igual", strategy="stubborn")

        report = run(SelfImprovementAgent(self.config, Stubborn()).run("app.py"))
        outcomes = [it.outcome for it in report.iterations]
        self.assertEqual(outcomes[0], "rolled_back")
        self.assertEqual(outcomes, ["rolled_back", "repeated_patch", "stagnated"])  # bloqueia e depois desiste
        self.assertEqual((self.root / "app.py").read_text(), DIV)
        # 2) AutoStrategy consulta a memória antes: pede alternativa ao modelo e desiste.
        provider = FakeProvider([same, same, same])
        report = run(SelfImprovementAgent(self.config, AutoStrategy(provider, use_heuristics=False)).run("app.py"))
        self.assertEqual([it.outcome for it in report.iterations], ["rolled_back", "no_fix"])
        self.assertEqual(len(provider.requests), 3)  # 1 aplicado + 2 recusados por repetição

    def test_interruption(self):
        self.write("app.py", DIV)
        report = run(SelfImprovementAgent(self.config, Scripted([ProviderInterrupted("ctrl-c")])).run("app.py"))
        self.assertEqual(report.status, "interrupted")


class ActionTests(TempProject):
    def test_observe_again_then_finish(self):
        self.write("app.py", DIV)
        strategy = Scripted([Decision(ActionKind.OBSERVE_AGAIN, reason="olhar de novo", strategy="s")])
        report = run(SelfImprovementAgent(self.config, strategy).run("app.py"))
        self.assertEqual([it.outcome for it in report.iterations], ["observed", "no_fix"])
        self.assertEqual(len(strategy.contexts[1].multimodal.by_kind(ObservationKind.RUNTIME)), 2)
        self.assertEqual(strategy.contexts[1].memory.count_action("observe_again"), 1)

    def test_run_tests_action(self):
        self.write("app.py", DIV)
        self.write("tests/test_ok.py", "import unittest\nclass T(unittest.TestCase):\n    def test_a(self): pass\n")
        self.config.test_command = ("-m", "unittest", "discover", "-s", "tests")
        strategy = Scripted([Decision(ActionKind.RUN_TESTS, reason="testar", strategy="s")])
        report = run(SelfImprovementAgent(self.config, strategy).run("app.py"))
        self.assertEqual(report.iterations[0].outcome, "tested")
        self.assertTrue(report.iterations[0].tests.success)
        self.assertEqual(strategy.contexts[1].multimodal.latest(ObservationKind.TEST).extracted["passed"], True)

    def test_rollback_action_restores_previous_patch(self):
        self.write("app.py", DIV)
        from agent_core.code_manager import FilePatch, Replacement
        from agent_core.strategies import FixProposal

        progress = Decision.patch(FixProposal([FilePatch("app.py", replacements=[Replacement("f(0)", "g(0)")])], "progresso", strategy="s"))
        strategy = Scripted([progress, Decision(ActionKind.ROLLBACK, reason="volta", strategy="s")])
        report = run(SelfImprovementAgent(self.config, strategy).run("app.py"))
        self.assertEqual([it.outcome for it in report.iterations], ["new_error", "rolled_back", "no_fix"])
        self.assertEqual((self.root / "app.py").read_text(), DIV)
        self.assertEqual(report.iterations[1].note, "1 arquivos restaurados")


class MemoryAndVisionTests(TempProject):
    def test_memory_persists_between_runs(self):
        self.config.memory_persist = True
        self.write("app.py", DIV)
        provider = FakeProvider([patch_json("app.py", "f(0)", "f(0 + 0)")])
        agent = SelfImprovementAgent(self.config, AutoStrategy(provider, use_heuristics=False))
        run(agent.run("app.py"))
        self.assertTrue(self.config.memory_path.exists())
        provider2 = FakeProvider([patch_json("app.py", "f(0)", "f(0 + 0)")])
        second = SelfImprovementAgent(self.config, AutoStrategy(provider2, use_heuristics=False))
        self.assertEqual(len(second.memory), 2)  # rolled_back + no_fix carregados do disco
        report = run(second.run("app.py"))
        self.assertEqual(report.iterations[0].outcome, "no_fix")  # lembrou: não reaplica o patch fracassado
        self.assertEqual(report.touched_files, [])
        self.assertIn("Memória do agente", provider2.requests[0].messages[0].parts[0].text)

    def test_memory_limit(self):
        self.write("app.py", DIV)
        mem = AgentMemory(limit=2)
        strategy = Scripted([Decision(ActionKind.OBSERVE_AGAIN, reason="1", strategy="s")] * 4)
        self.config.max_iterations = 5
        run(SelfImprovementAgent(self.config, strategy, memory=mem).run("app.py"))
        self.assertEqual(len(mem), 2)

    @unittest.skipUnless(vision_available(), "OpenCV não instalado")
    def test_vision_feeds_context_and_never_blocks(self):
        import numpy as np

        from agent_core.vision import CameraSource, ImageSource, VisionCapture

        self.write("app.py", DIV)
        frames = [np.full((60, 80, 3), c, np.uint8) for c in (10, 200)]
        capture = VisionCapture(ImageSource(frames, loop=True), fps=30, observation_interval=0.01)
        provider = FakeProvider([patch_json("app.py", "f(0)", "f(2)")])
        agent = SelfImprovementAgent(self.config, AutoStrategy(provider), vision=capture)
        report = run(agent.run("app.py"))
        self.assertEqual(report.status, "fixed")
        self.assertGreater(report.vision_status["frames_read"], 0)
        self.assertFalse(capture.is_running)
        self.assertTrue(agent.context.by_kind(ObservationKind.VISION))
        self.assertGreaterEqual(provider.requests[0].image_count, 1)

        broken = VisionCapture(CameraSource(index=99))
        events = []
        agent = SelfImprovementAgent(self.config, AutoStrategy(FakeProvider([patch_json("app.py", "f(0)", "f(4)")])), vision=broken, on_event=lambda e, d: events.append(e))
        (self.root / "app.py").write_text(DIV)
        report = run(agent.run("app.py"))
        self.assertEqual(report.status, "fixed")
        self.assertIn("vision.unavailable", events)
        self.assertIn("câmera", report.vision_status["error"])

    def test_vision_from_config_with_bad_device_is_ignored(self):
        cfg = AgentConfig(project_root=self.root, vision_enabled=True, vision_source="camera", vision_camera_index=99, memory_persist=False, log_level="WARNING")
        self.write("ok.py", "print('ok')\n")
        report = run(SelfImprovementAgent(cfg).run("ok.py"))
        self.assertEqual(report.status, "already_ok")
