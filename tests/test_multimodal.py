"""Contexto multimodal: criação, combinação código + logs + visão, limites, serialização, observers."""

from __future__ import annotations

import unittest

from agent_core import CodeManager, Sandbox
from agent_core.observations import (
    CodeObserver,
    ContextLimits,
    ImageData,
    LogObserver,
    MultimodalContext,
    Observation,
    ObservationKind,
    RuntimeObserver,
    TestObserver,
)
from tests._helpers import TempProject, run


def obs(kind="log", source="s", summary="x", image: bytes | None = None, **kw):
    return Observation(kind=kind, source=source, summary=summary, image=ImageData(image, "image/png", 2, 2) if image else None, **kw)


class ObservationTests(unittest.TestCase):
    def test_normalization(self):
        o = Observation(kind="vision", source="cam", summary="token=abcdefghij123", confidence=7)
        self.assertEqual(o.kind, ObservationKind.VISION)
        self.assertEqual(o.confidence, 1.0)
        self.assertIn("[REDACTED]", o.summary)
        self.assertFalse(o.has_image)

    def test_roundtrip_with_image(self):
        o = obs(image=b"\x89PNG\x00", extracted={"text": "ERROR 500", "n": 3}, data={"idx": 1})
        d = o.to_dict(include_image=True)
        self.assertEqual(d["image"]["size_bytes"], 5)
        back = Observation.from_dict(d)
        self.assertEqual(back.image.data, b"\x89PNG\x00")
        self.assertEqual(back.extracted["text"], "ERROR 500")
        self.assertEqual(back.id, o.id)
        self.assertNotIn("base64", o.to_dict()["image"])
        self.assertIn("text: ERROR 500", o.to_prompt_text())
        self.assertTrue(o.to_prompt_text(max_chars=20).endswith("…"))


class ContextTests(unittest.TestCase):
    def test_limits_fifo_and_images(self):
        ctx = MultimodalContext(ContextLimits(max_observations=3, max_images=1, max_image_bytes=10))
        for i in range(5):
            ctx.add(obs(summary=f"o{i}", image=b"12345" if i % 2 else None))
        self.assertEqual(len(ctx), 3)
        self.assertEqual([o.summary for o in ctx.observations], ["o2", "o3", "o4"])
        self.assertEqual(len(ctx.images()), 1)
        self.assertEqual(ctx.images()[0].summary, "o3")
        big = ctx.add(obs(summary="big", image=b"x" * 11))
        self.assertIsNone(big.image)
        self.assertIn("image_dropped", big.metadata)

    def test_combination_code_logs_vision(self):
        ctx = MultimodalContext()
        ctx.add(obs(kind="code", source="app.py", summary="fonte", extracted={"source": "1 | x = 1"}))
        ctx.add(obs(kind="log", source="app.log", summary="2 erros", extracted={"relevant_lines": ["ERROR boom"]}))
        ctx.add(obs(kind="vision", source="screen", summary="tela com erro", image=b"\xff\xd8jpg", extracted={"text": "Traceback"}))
        self.assertEqual(ctx.summary(), "code=1, log=1, vision=1, imagens=1")
        self.assertEqual(ctx.latest("vision").source, "screen")
        self.assertEqual(len(ctx.by_kind(ObservationKind.LOG)), 1)
        text = ctx.to_text()
        self.assertIn("ERROR boom", text)
        self.assertIn("1 | x = 1", text)
        parts = ctx.to_parts()
        self.assertEqual([p.type for p in parts], ["text", "text", "image"])
        self.assertEqual(parts[2].data, b"\xff\xd8jpg")

    def test_text_budget_prefers_recent(self):
        ctx = MultimodalContext(ContextLimits(max_text_chars=300))
        for i in range(10):
            ctx.add(obs(summary=f"obs{i} " + "x" * 80))
        text = ctx.to_text()
        self.assertLessEqual(len(text), 320)
        self.assertIn("obs9", text)
        self.assertNotIn("obs0", text)

    def test_serialization_roundtrip(self):
        ctx = MultimodalContext(ContextLimits(max_observations=5))
        ctx.add(obs(summary="a", image=b"img", data={"nested": {"k": [1, 2]}}))
        ctx.add(obs(kind="runtime", summary="b"))
        back = MultimodalContext.from_dict(__import__("json").loads(ctx.to_json(include_images=True)))
        self.assertEqual(len(back), 2)
        self.assertEqual(back.limits.max_observations, 5)
        self.assertEqual(back.observations[0].image.data, b"img")
        self.assertEqual(back.observations[0].data["nested"]["k"], [1, 2])
        self.assertIsNone(MultimodalContext.from_dict(__import__("json").loads(ctx.to_json())).observations[0].image)


class ObserverTests(TempProject):
    def test_runtime_observer(self):
        self.write("s.py", "raise KeyError('k')\n")
        result = run(Sandbox(self.config).run_script("s.py"))
        (o,) = run(RuntimeObserver().observe(result=result))
        self.assertEqual(o.kind, ObservationKind.RUNTIME)
        self.assertEqual(o.extracted["exception"], "KeyError")
        self.assertEqual(o.extracted["line"], 1)
        self.assertEqual(run(RuntimeObserver().observe()), [])

    def test_test_observer(self):
        self.write("tests/test_x.py", "import unittest\nclass T(unittest.TestCase):\n    def test_a(self): self.assertEqual(1, 2)\n")
        self.config.test_command = ("-m", "unittest", "discover", "-s", "tests")
        observer = TestObserver(Sandbox(self.config), self.config.test_command)
        (o,) = run(observer.observe())
        self.assertEqual(o.kind, ObservationKind.TEST)
        self.assertFalse(o.extracted["passed"])
        self.assertTrue(any("test_a" in f for f in o.extracted["failed_tests"]))
        self.assertEqual(run(TestObserver(Sandbox(self.config), None).observe()), [])

    def test_log_observer(self):
        self.write("app.log", "INFO ok\nERROR disco cheio\nWARNING x\n")
        (o,) = run(LogObserver(self.root).observe())
        self.assertEqual(o.source, "app.log")
        self.assertEqual(len(o.extracted["relevant_lines"]), 2)
        self.assertIn("ERROR disco cheio", o.extracted["tail"])

    def test_code_observer(self):
        self.write("m.py", "def f():\n    return 1\n")
        observations = run(CodeObserver(CodeManager(self.config)).observe(failing_file="m.py"))
        self.assertEqual([o.source for o in observations], ["project", "m.py"])
        self.assertIn("def f()", observations[0].extracted["outline"])
        self.assertIn("1 | def f():", observations[1].extracted["source"])
        missing = run(CodeObserver(CodeManager(self.config)).observe(failing_file="nao.py"))
        self.assertEqual(missing[1].confidence, 0.1)
