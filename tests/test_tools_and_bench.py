"""Ferramentas do modelo, estratégia por ferramentas, esforço por erro, typo, testes isolados, OCR, uso/custo, benchmark."""

from __future__ import annotations

import json
import unittest

from agent_core import AgentConfig, CodeManager, FakeProvider, FallbackProvider, SelfImprovementAgent, UsageTracker
from agent_core.bench import DEFAULT_CASES, heuristic_factory, offline_factory, run_benchmark, select_cases
from agent_core.providers import ContentPart, ModelMessage, ModelRequest, ModelResponse, ToolCall, ToolSpec
from agent_core.strategies import ActionKind, AutoStrategy, HeuristicFixStrategy, ToolFixStrategy
from agent_core.tools import PATCH_SCHEMA, ProjectToolbox
from tests._helpers import FakeAnthropicClient, TempProject, fake_message, patch_json, run

DIV = "def f(x):\n    return 10 / x\n\nprint(f(0))\n"


class ToolboxTests(TempProject):
    def setUp(self):
        super().setUp()
        self.write("app.py", DIV)
        self.write("pkg/util.py", "import os\n\ndef helper():\n    return os.sep\n")
        self.write("notes.log", "ERROR x\n")
        self.box = ProjectToolbox(CodeManager(self.config))

    def call(self, name, **kw):
        return run(self.box.execute(ToolCall(id="1", name=name, input=kw)))

    def test_specs_and_schema(self):
        names = [s.name for s in self.box.specs()]
        self.assertEqual(names, ["read_file", "list_files", "search", "outline", "propose_patch"])
        self.assertEqual(self.box.specs()[-1].input_schema, PATCH_SCHEMA)
        self.assertEqual(PATCH_SCHEMA["required"], ["rationale", "confidence", "patches"])

    def test_read_list_search_outline(self):
        r = self.call("read_file", path="app.py", start_line=2, end_line=2)
        self.assertEqual(r.content.strip(), "2 |     return 10 / x")
        self.assertFalse(r.is_error)
        self.assertIn("app.py", self.call("list_files").content)
        self.assertNotIn("notes.log", self.call("list_files").content)
        self.assertIn("notes.log", self.call("list_files", pattern="*.log").content)
        hits = self.call("search", query="helper").content
        self.assertIn("pkg/util.py:3:", hits)
        self.assertIn("def helper()", self.call("outline", path="pkg/util.py").content)
        self.assertEqual(self.call("search", query="").is_error, True)

    def test_errors_and_confinement(self):
        self.assertTrue(self.call("read_file", path="../etc/passwd").is_error)
        self.assertTrue(self.call("read_file", path="nao.py").is_error)
        self.assertTrue(self.call("read_file", path="app.py", foo=1).is_error)
        self.assertTrue(self.call("nao_existe").is_error)
        self.assertTrue(self.call("read_file", path=".agent_backups/x").is_error)

    def test_propose_patch_records(self):
        r = self.call("propose_patch", rationale="r", confidence=0.7, patches=[{"path": "app.py", "mode": "search_replace", "replacements": [{"search": "f(0)", "replace": "f(1)"}]}])
        self.assertFalse(r.is_error)
        self.assertEqual(json.loads(self.box.proposal_json())["confidence"], 0.7)
        self.assertEqual(len(self.box.calls), 1)

    def test_read_truncation(self):
        self.write("big.py", "\n".join(f"x{i} = {i}" for i in range(600)) + "\n")
        r = self.call("read_file", path="big.py")
        self.assertIn("linhas restantes", r.content)
        self.assertEqual(r.content.count("\n"), 400)


class ToolStrategyTests(TempProject):
    def ctx(self):
        self.write("app.py", DIV)
        return self.failure_context("app.py", code_manager=CodeManager(self.config), effort="medium")

    def test_tool_loop_reads_then_proposes(self):
        provider = FakeProvider([
            FakeProvider.tool_response("read_file", {"path": "app.py"}, call_id="c1", text="vou olhar"),
            FakeProvider.tool_response("propose_patch", {"rationale": "x é 0", "confidence": 0.8, "patches": [{"path": "app.py", "mode": "search_replace", "replacements": [{"search": "f(0)", "replace": "f(5)"}]}]}, call_id="c2"),
        ])
        strategy = ToolFixStrategy(provider, max_rounds=4)
        proposal = run(strategy.propose(self.ctx(), "diag"))
        self.assertEqual(proposal.patches[0].replacements[0].replace, "f(5)")
        self.assertEqual(proposal.strategy, "tools")
        self.assertEqual(strategy.last_rounds, 2)
        self.assertEqual(strategy.last_tool_calls[0], 'read_file({"path": "app.py"})')
        first, second = provider.requests
        self.assertEqual([t.name for t in first.tools][0], "read_file")
        self.assertEqual(first.effort, "medium")
        self.assertNotIn("Esqueleto do projeto", first.messages[0].parts[0].text)
        self.assertIn("## Arquivos do projeto", first.messages[0].parts[0].text)
        self.assertEqual([m.role for m in second.messages], ["user", "assistant", "user"])
        result_part = second.messages[2].parts[0]
        self.assertEqual((result_part.type, result_part.tool_use_id), ("tool_result", "c1"))
        self.assertIn("1 | def f(x):", result_part.text)

    def test_text_json_fallback_and_round_limit(self):
        provider = FakeProvider([patch_json("app.py", "f(0)", "f(2)")])
        proposal = run(ToolFixStrategy(provider).propose(self.ctx()))
        self.assertEqual(proposal.patches[0].replacements[0].replace, "f(2)")
        looping = FakeProvider([FakeProvider.tool_response("list_files", {}) for _ in range(5)])
        strategy = ToolFixStrategy(looping, max_rounds=3)
        self.assertIsNone(run(strategy.propose(self.ctx())))
        self.assertEqual(strategy.last_rounds, 3)

    def test_tool_error_is_reported_to_model(self):
        provider = FakeProvider([
            FakeProvider.tool_response("read_file", {"path": "../x"}, call_id="c1"),
            json.dumps({"rationale": "n", "confidence": 0, "patches": []}),
        ])
        self.assertIsNone(run(ToolFixStrategy(provider).propose(self.ctx())))
        part = provider.requests[1].messages[2].parts[0]
        self.assertTrue(part.is_error)
        self.assertIn("acesso negado", part.text)

    def test_without_code_manager_uses_single_prompt(self):
        self.write("app.py", DIV)
        provider = FakeProvider([patch_json("app.py", "f(0)", "f(2)")])
        ctx = self.failure_context("app.py")
        self.assertIsNotNone(run(ToolFixStrategy(provider).propose(ctx)))
        self.assertEqual(provider.requests[0].tools, [])
        self.assertEqual(provider.requests[0].output_schema, PATCH_SCHEMA)

    def test_auto_strategy_uses_tools_and_effort(self):
        provider = FakeProvider([patch_json("app.py", "f(0)", "f(2)")])
        strategy = AutoStrategy(provider, use_heuristics=False, effort_by_error={"ZeroDivisionError": "low", "default": "max"})
        decision = run(strategy.decide(self.ctx()))
        self.assertEqual(decision.action, ActionKind.PATCH)
        self.assertEqual(provider.requests[0].effort, "low")
        self.assertTrue(provider.requests[0].tools)
        no_tools = AutoStrategy(FakeProvider([patch_json("app.py", "f(0)", "f(2)")]), use_heuristics=False, use_tools=False)
        run(no_tools.decide(self.ctx()))
        self.assertEqual(no_tools.planners[-1].strategy.__class__.__name__, "ModelFixStrategy")

    def test_anthropic_provider_tool_translation(self):
        from agent_core.providers import AnthropicProvider
        from types import SimpleNamespace

        msg = fake_message("", stop_reason="tool_use")
        msg.content = [SimpleNamespace(type="tool_use", id="t1", name="read_file", input={"path": "a.py"})]
        client = FakeAnthropicClient(msg)
        provider = AnthropicProvider(client=client, server_fallbacks=False)
        req = ModelRequest(messages=[ModelMessage.user("oi")], tools=[ToolSpec("read_file", "lê", {"type": "object"})], output_schema=None)
        resp = run(provider.complete(req))
        self.assertTrue(resp.wants_tools)
        self.assertEqual(resp.tool_calls[0].input, {"path": "a.py"})
        self.assertEqual(client.calls[0]["tools"][0]["name"], "read_file")
        # reenvio do turno do assistente usa o conteúdo bruto; tool_result vira bloco
        req2 = ModelRequest(messages=[ModelMessage.user("oi"), ModelMessage.assistant_from(resp), ModelMessage.tool_results([ContentPart.from_tool_result("t1", "conteúdo", is_error=True)])], output_schema=PATCH_SCHEMA)
        kw = provider._build_kwargs(req2)
        self.assertIs(kw["messages"][1]["content"], msg.content)
        self.assertEqual(kw["messages"][2]["content"][0], {"type": "tool_result", "tool_use_id": "t1", "content": "conteúdo", "is_error": True})
        self.assertEqual(kw["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(provider.usage.calls, 1)


class HeuristicTypoAndEffortTests(TempProject):
    def test_typo_fix(self):
        self.write("app.py", "import json\nprint(jsn.dumps([1]))\n")
        proposal = run(HeuristicFixStrategy().propose(self.failure_context("app.py")))
        self.assertIn("typo", proposal.patches[0].reason)
        self.assertIn("print(json.dumps([1]))", proposal.patches[0].content)

    def test_typo_ambiguous_is_skipped(self):
        self.write("app.py", "value1 = 1\nvalue2 = 2\nprint(value)\n")
        self.assertIsNone(run(HeuristicFixStrategy().propose(self.failure_context("app.py"))))

    def test_config_effort_for(self):
        cfg = AgentConfig(project_root=self.root)
        self.assertEqual(cfg.effort_for("NameError@a.py:1:x"), "low")
        self.assertEqual(cfg.effort_for("TIMEOUT"), "high")
        self.assertEqual(cfg.effort_for("TESTS:1"), "high")
        self.assertEqual(cfg.effort_for("ZeroDivisionError@x"), "high")
        self.assertEqual(cfg.effort_for(None), "high")
        with self.assertRaises(ValueError):
            AgentConfig(project_root=self.root, effort_by_error={"NameError": "turbo"})


class IsolationAndOCRTests(TempProject):
    def test_tests_run_in_isolated_copy(self):
        self.write("app.py", "print('ok')\n")
        self.write("tests/test_side.py", "import unittest, pathlib\nclass T(unittest.TestCase):\n    def test_a(self):\n        pathlib.Path('side_effect.txt').write_text('x')\n")
        self.config.test_command = ("-m", "unittest", "discover", "-s", "tests")
        report = run(SelfImprovementAgent(self.config).run("app.py"))
        self.assertEqual(report.status, "already_ok")
        self.assertFalse((self.root / "side_effect.txt").exists())
        self.config.tests_isolated = False
        run(SelfImprovementAgent(self.config).run("app.py"))
        self.assertTrue((self.root / "side_effect.txt").exists())

    def test_ocr_reads_rendered_text(self):
        from agent_core.vision import Frame, OCREngine, VisualAnalyzer, vision_available

        if not vision_available() or not OCREngine().available:
            self.skipTest("Tesseract não instalado")
        import cv2
        import numpy as np

        img = np.full((120, 420, 3), 255, np.uint8)
        cv2.putText(img, "ERROR 500", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
        analysis = VisualAnalyzer().analyze(Frame(img, "t"))
        self.assertEqual(analysis["ocr"], "tesseract")
        self.assertIn("ERROR", analysis["text"])
        off = VisualAnalyzer(ocr=OCREngine.disabled()).analyze(Frame(img, "t"))
        self.assertEqual((off["ocr"], off["text"]), ("unavailable", None))


class UsageTests(TempProject):
    def test_tracker_pricing_and_merge(self):
        t = UsageTracker()
        t.record("claude-opus-5", {"input_tokens": 1_000_000, "output_tokens": 100_000, "cache_read_input_tokens": 1_000_000, "cache_creation_input_tokens": 0}, 1.5)
        self.assertAlmostEqual(t.cost_usd, 5 + 2.5 + 0.5)
        self.assertTrue(t.priced)
        t.record("modelo-desconhecido", {"input_tokens": 10, "output_tokens": 1})
        self.assertFalse(t.priced)
        merged = t.merge(UsageTracker(calls=1, input_tokens=5))
        self.assertEqual((merged.calls, merged.input_tokens), (3, 1_000_015))
        self.assertIn("cost_usd", t.to_dict())

    def test_fallback_usage_aggregates_and_report_delta(self):
        a, b = FakeProvider(["x"]), FakeProvider(["y"])
        fb = FallbackProvider([a, b])
        run(fb.complete(ModelRequest(messages=[ModelMessage.user("q")])))
        run(b.complete(ModelRequest(messages=[ModelMessage.user("q")])))
        self.assertEqual(fb.usage.calls, 2)
        self.assertEqual(fb.describe()["usage"]["calls"], 2)
        self.write("app.py", DIV)
        provider = FakeProvider([patch_json("app.py", "f(0)", "f(2)")])
        run(provider.complete(ModelRequest(messages=[ModelMessage.user("antes")])))  # uso anterior ao run
        provider._responses.append(patch_json("app.py", "f(0)", "f(2)"))
        report = run(SelfImprovementAgent(self.config, AutoStrategy(provider, use_heuristics=False)).run("app.py"))
        self.assertEqual(report.status, "fixed")
        self.assertEqual(report.usage["calls"], 1)
        self.assertIn("modelo: 1 chamadas", report.summary())


class BenchTests(unittest.TestCase):
    def test_offline_benchmark_fixes_everything(self):
        report = run(run_benchmark(DEFAULT_CASES, offline_factory, strategy_name="auto", model="fake"))
        failed = [(r.name, r.status, r.outcomes, r.error) for r in report.results if not r.fixed]
        self.assertEqual(failed, [])
        self.assertEqual(report.fix_rate, 1.0)
        self.assertTrue(all(r.stdout_ok for r in report.results))
        self.assertIn("fix rate: 100%", report.table())
        self.assertEqual(report.to_dict()["totals"]["calls"], sum(1 for c in DEFAULT_CASES if c.needs_model))

    def test_heuristic_subset_and_selection(self):
        cases = select_cases(None, heuristic_only=True)
        self.assertEqual([c.name for c in cases], ["name_error_stdlib", "name_error_sibling", "typo", "tabs"])
        report = run(run_benchmark(cases, heuristic_factory, strategy_name="heuristic"))
        self.assertEqual(report.fix_rate, 1.0)
        self.assertEqual([c.name for c in select_cases(["typo"])], ["typo"])
        with self.assertRaises(ValueError):
            select_cases(["nao_existe"])
