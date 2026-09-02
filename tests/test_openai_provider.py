"""Provider compatível com a API OpenAI (NVIDIA NIM e afins)."""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import openai

from agent_core import AgentConfig, OpenAICompatProvider
from agent_core.openai_provider import OpenAICompatProvider as OAP
from agent_core.providers import (
    PROVIDER_PRESETS,
    ContentPart,
    FallbackProvider,
    ModelMessage,
    ModelRequest,
    ProviderAuthError,
    ProviderInvalidResponseError,
    ProviderRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ToolSpec,
    build_provider,
)
from agent_core.tools import PATCH_SCHEMA
from tests._helpers import TempProject, run

import httpx2 as httpx


def _msg(content="ok", *, tool_calls=None, reasoning=None):
    m = SimpleNamespace(content=content, tool_calls=tool_calls, role="assistant")
    if reasoning is not None:
        m.reasoning_content = reasoning
    return m


def _completion(message=None, *, finish="stop", usage=True, model="moonshotai/kimi-k3"):
    u = SimpleNamespace(prompt_tokens=120, completion_tokens=30, prompt_tokens_details=SimpleNamespace(cached_tokens=20)) if usage else None
    return SimpleNamespace(choices=[SimpleNamespace(message=message or _msg(), finish_reason=finish)], usage=u, model=model)


class FakeOpenAIClient:
    """Cliente falso com a forma de ``AsyncOpenAI``."""

    def __init__(self, result=None, error: BaseException | None = None):
        self.calls: list[dict] = []
        outer = self
        queue = list(result) if isinstance(result, list) else [result]

        class _Completions:
            async def create(self, **kwargs):
                outer.calls.append(kwargs)
                if error is not None:
                    raise error
                return queue.pop(0) if len(queue) > 1 else queue[0]

        self.chat = SimpleNamespace(completions=_Completions())
        self.closed = False

    async def close(self):
        self.closed = True


def _http_error(cls, status: int):
    request = httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return cls("erro simulado", response=response, body=None)


def _req(**kw) -> ModelRequest:
    kw.setdefault("messages", [ModelMessage("user", [ContentPart.from_text("oi"), ContentPart.from_image(b"PNG", "image/png")])])
    kw.setdefault("system", "sistema")
    return ModelRequest(**kw)


class TranslationTests(unittest.TestCase):
    def setUp(self):
        self.p = OAP("moonshotai/kimi-k3", base_url="https://integrate.api.nvidia.com/v1")

    def test_messages_text_image_and_system(self):
        kw = self.p._build_kwargs(_req())
        self.assertEqual(kw["model"], "moonshotai/kimi-k3")
        self.assertEqual(kw["messages"][0], {"role": "system", "content": "sistema"})
        content = kw["messages"][1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "oi"})
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_text_only_message_is_plain_string(self):
        kw = self.p._build_kwargs(ModelRequest(messages=[ModelMessage.user("só texto")]))
        self.assertEqual(kw["messages"][0]["content"], "só texto")

    def test_tools_and_structured_output_are_exclusive(self):
        with_tools = self.p._build_kwargs(_req(tools=[ToolSpec("read_file", "lê", {"type": "object"})], output_schema=PATCH_SCHEMA))
        self.assertEqual(with_tools["tools"][0]["type"], "function")
        self.assertEqual(with_tools["tools"][0]["function"]["name"], "read_file")
        self.assertNotIn("response_format", with_tools)  # não envia os dois juntos
        schema_only = self.p._build_kwargs(_req(output_schema=PATCH_SCHEMA))
        self.assertEqual(schema_only["response_format"]["json_schema"]["schema"], PATCH_SCHEMA)

    def test_tool_results_become_tool_role_messages(self):
        assistant_raw = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}], "reasoning_content": "pensei"}
        request = ModelRequest(messages=[
            ModelMessage.user("q"),
            ModelMessage("assistant", [], raw=assistant_raw),
            ModelMessage.tool_results([ContentPart.from_tool_result("c1", "conteúdo do arquivo")]),
        ])
        msgs = self.p._build_kwargs(request)["messages"]
        self.assertIs(msgs[1], assistant_raw)  # raciocínio preservado no reenvio
        self.assertEqual(msgs[2], {"role": "tool", "tool_call_id": "c1", "content": "conteúdo do arquivo"})
        self.assertEqual(len(msgs), 3)  # nenhuma mensagem de usuário vazia


class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.p = OAP("moonshotai/kimi-k3", client=None)

    def test_text_usage_and_cache(self):
        p = OAP("moonshotai/kimi-k3", client=FakeOpenAIClient(_completion(_msg('{"ok": 1}'))))
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "x"}):
            resp = run(p.complete(_req()))
        self.assertEqual(resp.text, '{"ok": 1}')
        self.assertEqual(resp.usage, {"input_tokens": 100, "output_tokens": 30, "cache_read_input_tokens": 20})
        self.assertFalse(resp.truncated)
        self.assertEqual(p.usage.calls, 1)
        self.assertFalse(p.usage.priced)  # preço da NVIDIA não é o da Moonshot

    def test_tool_calls_are_parsed(self):
        call = SimpleNamespace(id="c1", function=SimpleNamespace(name="read_file", arguments='{"path": "a.py"}'))
        p = OAP("m", client=FakeOpenAIClient(_completion(_msg(None, tool_calls=[call], reasoning="vou ler"), finish="tool_calls")))
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "x"}):
            resp = run(p.complete(_req()))
        self.assertTrue(resp.wants_tools)
        self.assertEqual(resp.tool_calls[0].input, {"path": "a.py"})
        self.assertEqual(resp.raw_content["reasoning_content"], "vou ler")  # histórico de raciocínio
        self.assertEqual(resp.raw_content["role"], "assistant")

    def test_malformed_tool_arguments_do_not_crash(self):
        call = SimpleNamespace(id="c1", function=SimpleNamespace(name="x", arguments="{nao é json"))
        p = OAP("m", client=FakeOpenAIClient(_completion(_msg(None, tool_calls=[call]), finish="tool_calls")))
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "x"}):
            resp = run(p.complete(_req()))
        self.assertEqual(resp.tool_calls[0].input["_raw"], "{nao é json")

    def test_truncation_empty_and_filtered(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "x"}):
            p = OAP("m", client=FakeOpenAIClient(_completion(_msg("parcial"), finish="length")))
            self.assertTrue(run(p.complete(_req())).truncated)
            p = OAP("m", client=FakeOpenAIClient(_completion(_msg(None), finish="stop")))
            with self.assertRaises(ProviderInvalidResponseError):
                run(p.complete(_req()))
            p = OAP("m", client=FakeOpenAIClient(SimpleNamespace(choices=[], usage=None, model="m")))
            with self.assertRaises(ProviderInvalidResponseError):
                run(p.complete(_req()))
            p = OAP("m", client=FakeOpenAIClient(_completion(_msg("x"), finish="content_filter")))
            with self.assertRaises(ProviderInvalidResponseError):
                run(p.complete(_req()))


class CredentialTests(unittest.TestCase):
    def test_prefix_is_added_when_missing(self):
        p = OAP("m", api_key_env="NVIDIA_API_KEY", key_prefix="nvapi-")
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "abc123"}):
            self.assertEqual(p._credential(), "nvapi-abc123")
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-abc123"}):
            self.assertEqual(p._credential(), "nvapi-abc123")  # não duplica

    def test_missing_credential(self):
        p = OAP("m", api_key_env="NAO_DEFINIDA_XYZ")
        with self.assertRaises(ProviderAuthError) as cm:
            p._credential()
        self.assertIn("NAO_DEFINIDA_XYZ", str(cm.exception))

    def test_error_mapping(self):
        p = OAP("moonshotai/kimi-k3", api_key_env="NVIDIA_API_KEY", key_prefix="nvapi-")
        p._load_sdk()
        cases = [
            (_http_error(openai.AuthenticationError, 401), ProviderAuthError),
            (_http_error(openai.PermissionDeniedError, 403), ProviderAuthError),
            (_http_error(openai.NotFoundError, 404), ProviderRequestError),
            (_http_error(openai.BadRequestError, 400), ProviderRequestError),
            (_http_error(openai.InternalServerError, 503), ProviderUnavailableError),
            (openai.APITimeoutError(request=httpx.Request("POST", "https://x")), ProviderTimeoutError),
            (openai.APIConnectionError(request=httpx.Request("POST", "https://x")), ProviderUnavailableError),
        ]
        for exc, expected in cases:
            with self.subTest(exc=type(exc).__name__):
                self.assertIsInstance(p._map_exception(exc), expected)
        self.assertIn("nvapi-", str(p._map_exception(cases[1][0])))       # dica do prefixo
        self.assertIn("ajuste --model", str(p._map_exception(cases[2][0])))
        gone = p._map_exception(_http_error(openai.APIStatusError, 410))
        self.assertIn("aposentado", str(gone))


class PresetTests(TempProject):
    def test_nvidia_preset(self):
        preset = PROVIDER_PRESETS["nvidia"]
        self.assertEqual(preset.base_url, "https://integrate.api.nvidia.com/v1")
        self.assertEqual((preset.api_key_env, preset.key_prefix, preset.kind), ("NVIDIA_API_KEY", "nvapi-", "openai"))
        self.assertEqual(preset.default_model, "moonshotai/kimi-k3")

    def test_build_provider_uses_openai_layer(self):
        cfg = AgentConfig(project_root=self.root, llm_provider="nvidia", llm_model="")
        p = build_provider(cfg)
        self.assertIsInstance(p, FallbackProvider)
        head = p.providers[0]
        self.assertIsInstance(head, OpenAICompatProvider)
        self.assertEqual(head.model, "moonshotai/kimi-k3")
        self.assertEqual(head.key_prefix, "nvapi-")
        self.assertEqual(head.describe()["endpoint"], "https://integrate.api.nvidia.com/v1")

    def test_generic_openai_preset_requires_model(self):
        with self.assertRaises(ValueError):
            build_provider(AgentConfig(project_root=self.root, llm_provider="openai-compat", llm_model=""))
        p = build_provider(AgentConfig(project_root=self.root, llm_provider="openai-compat", llm_model="meu", llm_base_url="https://x.local/v1", llm_enable_fallbacks=False))
        self.assertIsInstance(p, OpenAICompatProvider)
        self.assertEqual(p.base_url, "https://x.local/v1")


class LoopIntegrationTests(TempProject):
    """O loop e as estratégias funcionam sem saber qual provider está por baixo."""

    def test_agent_fixes_bug_through_openai_provider(self):
        from agent_core import SelfImprovementAgent
        from agent_core.strategies import AutoStrategy

        self.write("app.py", "def f(x):\n    return 10 / x\n\nprint(f(0))\n")
        answer = json.dumps({"rationale": "x é zero", "confidence": 0.9, "patches": [{"path": "app.py", "mode": "search_replace", "replacements": [{"search": "f(0)", "replace": "f(5)"}]}]})
        client = FakeOpenAIClient(_completion(_msg(answer)))
        provider = OAP("moonshotai/kimi-k3", client=client)
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "x"}):
            report = run(SelfImprovementAgent(self.config, AutoStrategy(provider, use_heuristics=False)).run("app.py"))
        self.assertEqual(report.status, "fixed")
        self.assertEqual(report.final_result.stdout.strip(), "2.0")
        self.assertEqual(report.usage["calls"], 1)
        self.assertIn("Diagnóstico preliminar", client.calls[0]["messages"][1]["content"])
