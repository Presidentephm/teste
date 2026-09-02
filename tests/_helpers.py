"""Utilitários compartilhados pelas suítes novas (sem dependências externas)."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent_core import AgentConfig, Sandbox
from agent_core.strategies import FailureContext


def run(coro):
    return asyncio.run(coro)


class TempProject(unittest.TestCase):
    """Projeto temporário limpo por teste, com memória desligada por padrão."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="agent_v2_"))
        self.config = AgentConfig(project_root=self.root, sandbox_timeout=10, log_level="WARNING", memory_persist=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def write(self, rel: str, content: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(content), encoding="utf-8")
        return p

    def failure_context(self, script: str, **extra) -> FailureContext:
        """Executa ``script`` no sandbox e monta um FailureContext mínimo."""
        from agent_core import CodeManager

        result = run(Sandbox(self.config).run_script(script))
        code = CodeManager(self.config)
        source = (self.root / script).read_text()
        kwargs = dict(
            script=script,
            result=result,
            failing_file=script,
            failing_source=source,
            failing_analysis=code.analyze_source(source, script),
            project_outline=run(code.analyze_project()),
            attempt=1,
        )
        kwargs.update(extra)
        return FailureContext(**kwargs)


def fake_message(text: str = "ok", *, stop_reason: str = "end_turn", model: str = "claude-opus-5", fallback: bool = False):
    """Objeto com a forma mínima de ``anthropic.types.Message``."""
    content = [SimpleNamespace(type="text", text=text)] if text is not None else []
    if fallback:
        content.insert(0, SimpleNamespace(type="fallback"))
    usage = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=None)
    return SimpleNamespace(content=content, stop_reason=stop_reason, model=model, usage=usage, stop_details=None)


class FakeStream:
    """Imita o gerenciador de contexto assíncrono de ``messages.stream``."""

    def __init__(self, message=None, error: BaseException | None = None):
        self.message = message
        self.error = error

    async def __aenter__(self):
        if self.error is not None:
            raise self.error
        return self

    async def __aexit__(self, *_):
        return False

    async def get_final_message(self):
        return self.message


class FakeAnthropicClient:
    """Cliente falso que registra kwargs e devolve mensagens programadas."""

    def __init__(self, message=None, error: BaseException | None = None):
        self.calls: list[dict] = []
        self.beta_calls: list[dict] = []
        outer = self

        class _Messages:
            def __init__(self, beta: bool):
                self.beta = beta

            def stream(self, **kwargs):
                (outer.beta_calls if self.beta else outer.calls).append(kwargs)
                return FakeStream(message, error)

        self.messages = _Messages(beta=False)
        self.beta = SimpleNamespace(messages=_Messages(beta=True))
        self.closed = False

    async def close(self):
        self.closed = True


def patch_json(path: str, search: str, replace: str, rationale: str = "fix", confidence: float = 0.9) -> str:
    return json.dumps(
        {"rationale": rationale, "confidence": confidence, "patches": [{"path": path, "mode": "search_replace", "replacements": [{"search": search, "replace": replace}]}]}
    )


def full_json(path: str, content: str, rationale: str = "rewrite") -> str:
    return json.dumps({"rationale": rationale, "confidence": 0.8, "patches": [{"path": path, "mode": "replace_full", "content": content}]})
