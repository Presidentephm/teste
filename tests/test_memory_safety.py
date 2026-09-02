"""Memória, redação de credenciais, guarda de patches e checkpoints."""

from __future__ import annotations

import logging
import unittest

from agent_core import BackupManager, FilePatch, Replacement
from agent_core.memory import AgentMemory, new_entry, patch_signature
from agent_core.safety import PatchGuard, RedactingFormatter, UnsafePatchError, redact
from tests._helpers import TempProject, run


class RedactTests(unittest.TestCase):
    def test_patterns(self):
        cases = [
            "key sk-ant-api03-abcdefghijklmnopqrstuvwxyz",
            "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123",
            "AWS AKIAABCDEFGHIJKLMNOP",
            "api_key=supersecretvalue",
            'password: "hunter2hunter2"',
        ]
        for text in cases:
            with self.subTest(text=text):
                out = redact(text)
                self.assertIn("[REDACTED]", out)
                self.assertNotIn("supersecret", out)
        self.assertEqual(redact("nada aqui"), "nada aqui")
        self.assertEqual(redact(""), "")

    def test_formatter(self):
        fmt = RedactingFormatter("%(message)s")
        rec = logging.LogRecord("x", logging.INFO, "", 0, "token=abcdefgh1234", None, None)
        self.assertEqual(fmt.format(rec), "token=[REDACTED]")


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.guard = PatchGuard(max_files=2, max_removed_ratio=0.5)
        self.src = {"a.py": "\n".join(f"l{i}" for i in range(10)) + "\n"}

    def test_rules(self):
        with self.assertRaisesRegex(UnsafePatchError, "vazio"):
            self.guard.check([], self.src)
        with self.assertRaisesRegex(UnsafePatchError, "arquivos"):
            self.guard.check([FilePatch("a.py", content="x"), FilePatch("b.py", content="x"), FilePatch("c.py", content="x")], self.src)
        with self.assertRaisesRegex(UnsafePatchError, "repetido"):
            self.guard.check([FilePatch("a.py", content="x\n" * 9), FilePatch("a.py", content="y\n" * 9)], self.src)
        with self.assertRaisesRegex(UnsafePatchError, "esvazia"):
            self.guard.check([FilePatch("a.py", content="  \n")], self.src)
        with self.assertRaisesRegex(UnsafePatchError, "remove"):
            self.guard.check([FilePatch("a.py", content="l0\nl1\n")], self.src)
        # OK: novo arquivo, busca/substituição, redução moderada
        self.guard.check([FilePatch("novo.py", content="x = 1\n")], self.src)
        self.guard.check([FilePatch("a.py", replacements=[Replacement("l0", "z")])], self.src)
        self.guard.check([FilePatch("a.py", content="\n".join(f"l{i}" for i in range(7)) + "\n")], self.src)


class MemoryTests(TempProject):
    def test_signature_and_queries(self):
        p1 = [FilePatch("a.py", replacements=[Replacement("x", "y")])]
        p2 = [FilePatch("a.py", replacements=[Replacement("x", "z")])]
        self.assertEqual(patch_signature(p1), patch_signature(list(p1)))
        self.assertNotEqual(patch_signature(p1), patch_signature(p2))
        mem = AgentMemory(limit=3)
        mem.add(new_entry(1, action="patch", error_signature="E1", patch_signature="s1", outcome="rolled_back", rollback=True, patch_files=["a.py"]))
        mem.add(new_entry(2, action="patch", error_signature="E1", patch_signature="s2", outcome="fixed"))
        mem.add(new_entry(3, action="observe_again", error_signature="E2", outcome="observed"))
        self.assertTrue(mem.has_tried("s1"))
        self.assertFalse(mem.has_tried("s2"))
        self.assertTrue(mem.has_tried("s2", failed_only=False))
        self.assertEqual(len(mem.failed_attempts("E1")), 1)
        self.assertEqual(mem.count_action("observe_again", "E2"), 1)
        mem.add(new_entry(4, action="patch", error_signature="E3", outcome="new_error"))
        self.assertEqual(len(mem), 3)  # limite FIFO
        self.assertEqual(mem.entries[0].iteration, 2)
        self.assertIn("it4", mem.to_prompt_text())
        with self.assertRaises(ValueError):
            AgentMemory(limit=0)

    def test_persistence_and_redaction(self):
        path = self.root / ".agent_backups" / "memory.json"
        mem = AgentMemory(limit=5, path=path)
        mem.add(new_entry(1, action="patch", error_signature="E", result="falhou com api_key=abcdefgh1234", errors=["token=zzzzzzzzzz"]))
        self.assertTrue(path.exists())
        again = AgentMemory(limit=5, path=path)
        self.assertEqual(len(again), 1)
        self.assertIn("[REDACTED]", again.entries[0].result)
        self.assertIn("[REDACTED]", again.entries[0].errors[0])
        path.write_text("{corrompido")
        self.assertEqual(len(AgentMemory(limit=5, path=path)), 0)
        again.clear()
        self.assertEqual(AgentMemory(limit=5, path=path).entries, [])


class CheckpointTests(TempProject):
    def test_checkpoint_restore(self):
        a = self.write("a.py", "a1\n")
        b = self.write("b.py", "b1\n")
        bm = BackupManager(self.config)
        cp = run(bm.checkpoint([a, b], label="t"))
        self.assertEqual(cp.files, ["a.py", "b.py"])
        self.assertEqual(cp.label, "t")
        a.write_text("a2\n")
        b.write_text("b2\n")
        restored = run(bm.restore(cp))
        self.assertEqual(len(restored), 2)
        self.assertEqual((a.read_text(), b.read_text()), ("a1\n", "b1\n"))
        self.assertTrue(all(r.reason.startswith("checkpoint:t") for r in bm.list_backups()))
