"""Testes do núcleo (unittest, sem dependências externas).

Rodar: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path

from agent_core import (
    AgentConfig,
    BackupManager,
    CodeManager,
    FilePatch,
    HeuristicFixStrategy,
    Replacement,
    Sandbox,
    SelfImprovementAgent,
)
from agent_core.code_manager import InvalidSourceError, PathOutsideProjectError
from agent_core.sandbox import parse_traceback
from agent_core.strategies import ClaudeFixStrategy, insert_import


def run(coro):
    return asyncio.run(coro)


class TempProject(unittest.TestCase):
    """Cria um projeto temporário limpo para cada teste."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="agent_test_"))
        self.config = AgentConfig(project_root=self.root, sandbox_timeout=10, log_level="WARNING")

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def write(self, rel: str, content: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(content), encoding="utf-8")
        return p


class BackupTests(TempProject):
    def test_backup_and_rollback_restores_content(self):
        f = self.write("a.py", "x = 1\n")
        bm = BackupManager(self.config)
        rec = run(bm.backup(f, reason="teste"))
        self.assertTrue(rec.existed)
        f.write_text("x = 2\n")
        run(bm.rollback(f))
        self.assertEqual(f.read_text(), "x = 1\n")
        self.assertEqual(len(bm.list_backups(f)), 1)
        self.assertEqual(bm.list_backups(f)[0].reason, "teste")

    def test_rollback_of_new_file_deletes_it(self):
        bm = BackupManager(self.config)
        f = self.root / "novo.py"
        rec = run(bm.backup(f))
        self.assertFalse(rec.existed)
        f.write_text("print(1)\n")
        run(bm.rollback(f))
        self.assertFalse(f.exists())

    def test_prune_keeps_only_latest(self):
        self.config.max_backups_per_file = 2
        f = self.write("a.py", "x = 1\n")
        bm = BackupManager(self.config)
        for i in range(4):
            f.write_text(f"x = {i}\n")
            run(bm.backup(f))
        self.assertEqual(len(bm.list_backups(f)), 2)
        # O backup mais recente guarda o penúltimo conteúdo escrito.
        latest = bm.latest(f)
        self.assertEqual((bm.backup_dir / latest.backup_path).read_text(), "x = 3\n")

    def test_successive_rollbacks_walk_back_in_history(self):
        f = self.write("a.py", "v1\n")
        bm = BackupManager(self.config)
        run(bm.backup(f)); f.write_text("v2\n")
        run(bm.backup(f)); f.write_text("v3\n")
        rec = run(bm.rollback(f))
        self.assertIsNotNone(rec.restored_at)
        self.assertEqual(f.read_text(), "v2\n")
        run(bm.rollback(f))
        self.assertEqual(f.read_text(), "v1\n")
        self.assertIsNone(bm.latest(f))
        self.assertIsNotNone(bm.latest(f, include_restored=True))
        with self.assertRaises(FileNotFoundError):
            run(bm.rollback(f))

    def test_rollback_without_backup_raises(self):
        bm = BackupManager(self.config)
        with self.assertRaises(FileNotFoundError):
            run(bm.rollback(self.root / "inexistente.py"))


class CodeManagerTests(TempProject):
    def test_analyze_extracts_structure(self):
        self.write(
            "mod.py",
            '''
            """Doc do módulo."""
            import os
            from json import dumps as d, loads

            X = 1

            def f(a, b=2, *args, **kw):
                """doc f"""

            async def g():
                pass

            class C(Base):
                def m(self): ...
            ''',
        )
        cm = CodeManager(self.config)
        a = run(cm.analyze("mod.py"))
        self.assertTrue(a.is_valid)
        self.assertEqual(a.docstring, "Doc do módulo.")
        self.assertEqual([i.module for i in a.imports], ["os", "json"])
        self.assertEqual(a.imports[1].names, ["d", "loads"])
        self.assertEqual({fn.name for fn in a.functions}, {"f", "g"})
        self.assertEqual(a.functions[0].args, ["a", "b", "*args", "**kw"])
        self.assertTrue(a.functions[1].is_async)
        self.assertEqual(a.classes[0].bases, ["Base"])
        self.assertEqual(a.classes[0].methods[0].name, "m")
        self.assertEqual(a.defined_names, {"X", "f", "g", "C"})
        self.assertEqual(a.imported_names, {"os", "d", "loads"})
        self.assertIn("def f(a, b, *args, **kw)", a.outline())

    def test_analyze_reports_syntax_error(self):
        self.write("bad.py", "def f(:\n    pass\n")
        a = run(CodeManager(self.config).analyze("bad.py"))
        self.assertFalse(a.is_valid)
        self.assertEqual(a.syntax_issue.lineno, 1)

    def test_write_creates_backup_and_rejects_invalid_syntax(self):
        f = self.write("a.py", "x = 1\n")
        cm = CodeManager(self.config)
        with self.assertRaises(InvalidSourceError):
            run(cm.write("a.py", "def (:\n"))
        self.assertEqual(f.read_text(), "x = 1\n")  # nada foi tocado
        self.assertEqual(cm.backups.list_backups(f), [])  # nem backup foi criado
        rec = run(cm.write("a.py", "x = 2\n", reason="ok"))
        self.assertEqual(f.read_text(), "x = 2\n")
        self.assertEqual(rec.reason, "ok")
        run(cm.rollback("a.py"))
        self.assertEqual(f.read_text(), "x = 1\n")

    def test_path_confinement(self):
        cm = CodeManager(self.config)
        with self.assertRaises(PathOutsideProjectError):
            cm.resolve("../fora.py")
        with self.assertRaises(PathOutsideProjectError):
            cm.resolve(Path("/etc/passwd"))
        with self.assertRaises(PathOutsideProjectError):
            cm.resolve(f"{self.config.backup_dir_name}/x.bak")
        with self.assertRaises(PathOutsideProjectError):
            cm.resolve(".git/config")

    def test_self_modification_can_be_disabled(self):
        # Projeto cujo root contém o próprio pacote agent_core.
        core_root = Path(__file__).resolve().parent.parent
        cfg = AgentConfig(project_root=core_root, allow_self_modification=False, log_level="WARNING")
        cm = CodeManager(cfg)
        cm.resolve("agent_core/config.py")  # leitura permitida
        with self.assertRaises(PathOutsideProjectError):
            cm.resolve("agent_core/config.py", for_write=True)

    def test_search_replace_patch_and_missing_search(self):
        f = self.write("a.py", "x = 1\ny = x + 1\n")
        cm = CodeManager(self.config)
        run(cm.apply_patch(FilePatch("a.py", replacements=[Replacement("x = 1", "x = 10")])))
        self.assertEqual(f.read_text(), "x = 10\ny = x + 1\n")
        with self.assertRaises(ValueError):
            run(cm.apply_patch(FilePatch("a.py", replacements=[Replacement("nao existe", "z")])))

    def test_apply_patches_rolls_back_on_failure(self):
        a = self.write("a.py", "x = 1\n")
        self.write("b.py", "y = 1\n")
        cm = CodeManager(self.config)
        patches = [FilePatch("a.py", content="x = 2\n"), FilePatch("b.py", content="def (:\n")]
        with self.assertRaises(InvalidSourceError):
            run(cm.apply_patches(patches))
        self.assertEqual(a.read_text(), "x = 1\n")  # o primeiro patch foi revertido


class SandboxTests(TempProject):
    def test_captures_stdout_and_traceback(self):
        self.write("s.py", "print('ola')\nraise ValueError('boom')\n")
        r = run(Sandbox(self.config).run_script("s.py"))
        self.assertFalse(r.success)
        self.assertEqual(r.returncode, 1)
        self.assertEqual(r.stdout.strip(), "ola")
        self.assertEqual(r.traceback.exc_type, "ValueError")
        self.assertEqual(r.traceback.message, "boom")
        self.assertEqual(r.traceback.location.line, 2)
        # Caminho mapeado da cópia temporária de volta para o projeto real.
        self.assertEqual(r.traceback.location.file, str(self.root / "s.py"))

    def test_timeout_kills_process(self):
        self.config.sandbox_timeout = 1
        self.write("loop.py", "while True: pass\n")
        r = run(Sandbox(self.config).run_script("loop.py"))
        self.assertTrue(r.timed_out)
        self.assertFalse(r.success)
        self.assertEqual(r.signature, "TIMEOUT")

    def test_isolated_copy_protects_project(self):
        self.write("w.py", "open('efeito.txt', 'w').write('x')\n")
        r = run(Sandbox(self.config).run_script("w.py"))
        self.assertTrue(r.success)
        self.assertFalse((self.root / "efeito.txt").exists())

    def test_syntax_error_is_parsed(self):
        self.write("bad.py", "def f(:\n    pass\n")
        r = run(Sandbox(self.config).run_script("bad.py"))
        self.assertEqual(r.traceback.exc_type, "SyntaxError")
        self.assertEqual(r.traceback.location.line, 1)

    def test_parse_traceback_chained(self):
        stderr = textwrap.dedent(
            """
            Traceback (most recent call last):
              File "/p/a.py", line 3, in <module>
                inner()
              File "/p/a.py", line 2, in inner
                1/0
            ZeroDivisionError: division by zero
            """
        )
        tb = parse_traceback(stderr, {"/p": "/real"})
        self.assertEqual(tb.exc_type, "ZeroDivisionError")
        self.assertEqual(len(tb.frames), 2)
        self.assertEqual(tb.location.function, "inner")
        self.assertEqual(tb.location.file, "/real/a.py")
        self.assertEqual(tb.frames[1].code, "1/0")


class HeuristicTests(unittest.TestCase):
    def test_insert_import_after_docstring_and_imports(self):
        src = '"""doc"""\nimport os\n\nx = 1\n'
        self.assertEqual(insert_import(src, "import json"), '"""doc"""\nimport os\nimport json\n\nx = 1\n')
        self.assertEqual(insert_import("x = 1\n", "import json"), "import json\nx = 1\n")
        multi = '"""\ndoc\nlonga\n"""\n\nx = 1\n'
        self.assertEqual(insert_import(multi, "import re"), '"""\ndoc\nlonga\n"""\nimport re\n\nx = 1\n')

    def test_claude_response_parsing(self):
        strat = ClaudeFixStrategy.__new__(ClaudeFixStrategy)
        text = 'Segue:\n```json\n{"rationale": "r", "confidence": 0.8, "patches": [{"path": "a.py", "mode": "search_replace", "replacements": [{"search": "x", "replace": "y"}]}]}\n```'
        p = strat.parse_response(text)
        self.assertEqual(p.rationale, "r")
        self.assertEqual(p.patches[0].replacements[0].replace, "y")
        self.assertIsNone(strat.parse_response('{"rationale": "nada", "confidence": 0, "patches": []}'))
        self.assertIsNone(strat.parse_response("sem json"))


class AgentLoopTests(TempProject):
    def test_fixes_missing_imports_across_two_iterations(self):
        self.write("utils.py", "def helper():\n    return 42\n")
        self.write("app.py", "print(json.dumps({'v': helper()}))\n")
        agent = SelfImprovementAgent(self.config, HeuristicFixStrategy())
        report = run(agent.run("app.py"))
        self.assertEqual(report.status, "fixed")
        self.assertEqual(len(report.iterations), 2)
        self.assertEqual([it.outcome for it in report.iterations], ["new_error", "fixed"])
        src = (self.root / "app.py").read_text()
        self.assertIn("import json", src)
        self.assertIn("from utils import helper", src)
        self.assertEqual(report.final_result.stdout.strip(), '{"v": 42}')
        # rollback_run desfaz tudo o que sobreviveu.
        run(agent.rollback_run(report))
        self.assertEqual((self.root / "app.py").read_text(), "print(json.dumps({'v': helper()}))\n")

    def test_already_ok(self):
        self.write("ok.py", "print('ok')\n")
        report = run(SelfImprovementAgent(self.config).run("ok.py"))
        self.assertEqual(report.status, "already_ok")
        self.assertEqual(report.iterations, [])

    def test_no_fix_when_strategy_gives_up(self):
        self.write("zero.py", "1/0\n")
        report = run(SelfImprovementAgent(self.config).run("zero.py"))
        self.assertEqual(report.status, "no_fix")
        self.assertEqual(report.iterations[0].outcome, "no_fix")

    def test_bad_patch_is_rolled_back(self):
        """Uma estratégia que quebra a sintaxe deve ter o patch revertido."""
        from agent_core.strategies import FixProposal, FixStrategy

        class Vandal(FixStrategy):
            name = "vandal"

            async def propose(self, ctx):
                # Compila, mas quebra o import na execução -> regressão crítica.
                return FixProposal([FilePatch(ctx.failing_file, content="import modulo_inexistente\n1/0\n")], "ruim")

        self.config.max_iterations = 2
        f = self.write("zero.py", "1/0\n")
        report = run(SelfImprovementAgent(self.config, Vandal()).run("zero.py"))
        self.assertEqual([it.outcome for it in report.iterations][:1], ["rolled_back"])
        self.assertEqual(f.read_text(), "1/0\n")
        self.assertIn(report.status, ("exhausted", "stagnated"))


if __name__ == "__main__":
    unittest.main()
