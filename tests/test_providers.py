"""Provider: configuração, tradução de tipos, chamada (mock), erros, timeout, fallback."""

from __future__ import annotations

import asyncio
import base64
import unittest

import httpx2

from agent_core import AgentConfig
from agent_core.providers import (
    AnthropicProvider,
    ContentPart,
    FakeProvider,
    FallbackProvider,
    ModelMessage,
    ModelRequest,
    ProviderAuthError,
    ProviderError,
    ProviderInterrupted,
    ProviderInvalidResponseError,
    ProviderRateLimitError,
    ProviderRefusalError,
    ProviderRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    build_provider,
)
from tests._helpers import FakeAnthropicClient, fake_message, run

import anthropic


def _req(text: str = "olá", image: bytes | None = None) -> ModelRequest:
    parts = [ContentPart.from_text(text)]
    if image is not None:
        parts.append(ContentPart.from_image(image, "image/png"))
    return ModelRequest(messages=[ModelMessage("user", parts)], system="sys")


def _http_error(cls, status: int, headers: dict | None = None):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request, headers=headers or {})
    return cls("erro simulado", response=response, body={"error": {"type": "x"}})


class TranslationTests(unittest.TestCase):
    def test_blocks_text_and_image(self):
        blocks = AnthropicProvider._to_blocks([ContentPart.from_text("a"), ContentPart.from_image(b"\x89PNG", "image/png")])
        self.assertEqual(blocks[0], {"type": "text", "text": "a"})
        self.assertEqual(blocks[1]["type"], "image")
        self.assertEqual(blocks[1]["source"]["media_type"], "image/png")
        self.assertEqual(base64.b64decode(blocks[1]["source"]["data"]), b"\x89PNG")

    def test_empty_parts_become_placeholder(self):
        self.assertEqual(AnthropicProvider._to_blocks([]), [{"type": "text", "text": "(vazio)"}])

    def test_request_kwargs(self):
        p = AnthropicProvider(model="claude-opus-5", max_tokens=123, effort="low")
        kw = p._build_kwargs(_req(image=b"img"))
        self.assertEqual(kw["model"], "claude-opus-5")
        self.assertEqual(kw["max_tokens"], 123)
        self.assertEqual(kw["thinking"], {"type": "adaptive"})
        self.assertEqual(kw["output_config"], {"effort": "low"})
        self.assertEqual(kw["system"], [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}])
        self.assertEqual(AnthropicProvider(cache_prompts=False)._build_kwargs(_req())["system"], "sys")
        self.assertEqual(kw["messages"][0]["content"][1]["type"], "image")
        self.assertEqual(ModelRequest(messages=[ModelMessage("user", [ContentPart.from_image(b"x")])]).image_count, 1)


class AnthropicProviderCallTests(unittest.TestCase):
    def test_success_with_server_fallbacks(self):
        client = FakeAnthropicClient(fake_message('{"ok": 1}'))
        p = AnthropicProvider(client=client)
        resp = run(p.complete(_req()))
        self.assertEqual(resp.text, '{"ok": 1}')
        self.assertEqual(resp.usage, {"input_tokens": 10, "output_tokens": 5})
        self.assertFalse(resp.truncated)
        self.assertFalse(resp.fallback_used)
        self.assertEqual(len(client.beta_calls), 1)
        self.assertEqual(client.beta_calls[0]["betas"], ["server-side-fallback-2026-07-01"])
        self.assertEqual(client.beta_calls[0]["fallbacks"], "default")
        run(p.aclose())
        self.assertTrue(client.closed)

    def test_success_without_server_fallbacks(self):
        client = FakeAnthropicClient(fake_message("x", stop_reason="max_tokens", fallback=True))
        p = AnthropicProvider(client=client, server_fallbacks=False)
        resp = run(p.complete(_req()))
        self.assertEqual(len(client.calls), 1)
        self.assertNotIn("betas", client.calls[0])
        self.assertTrue(resp.truncated)
        self.assertTrue(resp.fallback_used)

    def test_refusal_and_empty_response(self):
        p = AnthropicProvider(client=FakeAnthropicClient(fake_message("", stop_reason="refusal")))
        with self.assertRaises(ProviderRefusalError):
            run(p.complete(_req()))
        p = AnthropicProvider(client=FakeAnthropicClient(fake_message("   ")))
        with self.assertRaises(ProviderInvalidResponseError):
            run(p.complete(_req()))

    def test_timeout_is_mapped(self):
        err = anthropic.APITimeoutError(request=httpx2.Request("POST", "https://x"))
        p = AnthropicProvider(client=FakeAnthropicClient(error=err))
        with self.assertRaises(ProviderTimeoutError) as cm:
            run(p.complete(_req()))
        self.assertTrue(cm.exception.retryable)

    def test_cancellation_is_interruption(self):
        p = AnthropicProvider(client=FakeAnthropicClient(error=asyncio.CancelledError()))
        with self.assertRaises(ProviderInterrupted):
            run(p.complete(_req()))

    def test_missing_credentials(self):
        p = AnthropicProvider(client=FakeAnthropicClient(error=TypeError("Could not resolve authentication method")))
        with self.assertRaises(ProviderAuthError):
            run(p.complete(_req()))

    def test_error_mapping(self):
        p = AnthropicProvider()
        cases = [
            (_http_error(anthropic.AuthenticationError, 401), ProviderAuthError),
            (_http_error(anthropic.PermissionDeniedError, 403), ProviderAuthError),
            (_http_error(anthropic.RateLimitError, 429, {"retry-after": "3"}), ProviderRateLimitError),
            (_http_error(anthropic.NotFoundError, 404), ProviderRequestError),
            (_http_error(anthropic.BadRequestError, 400), ProviderRequestError),
            (_http_error(anthropic.InternalServerError, 500), ProviderUnavailableError),
            (_http_error(anthropic.OverloadedError, 529), ProviderUnavailableError),
            (anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x")), ProviderUnavailableError),
        ]
        for exc, expected in cases:
            with self.subTest(exc=type(exc).__name__):
                mapped = p._map_exception(exc)
                self.assertIsInstance(mapped, expected)
                self.assertIs(mapped.cause, exc)
        rl = p._map_exception(cases[2][0])
        self.assertEqual(rl.retry_after, 3.0)
        self.assertIn("ajuste --model", str(p._map_exception(cases[3][0])))

    def test_sdk_error_via_complete(self):
        err = _http_error(anthropic.InternalServerError, 503)
        p = AnthropicProvider(client=FakeAnthropicClient(error=err))
        with self.assertRaises(ProviderUnavailableError):
            run(p.complete(_req()))

    def test_secret_redacted_in_error(self):
        err = ProviderError("falhou com api_key=sk-ant-abcdefghijklmnop123456")
        self.assertNotIn("sk-ant-abcdef", str(err))


class FallbackProviderTests(unittest.TestCase):
    def setUp(self):
        self.sleeps: list[float] = []

        async def fake_sleep(d):
            self.sleeps.append(d)

        self.sleep = fake_sleep

    def test_retries_retryable_then_succeeds(self):
        primary = FakeProvider([ProviderRateLimitError("429", retry_after=1.5), "ok"])
        fb = FallbackProvider([primary], max_retries=2, sleep=self.sleep)
        resp = run(fb.complete(_req()))
        self.assertEqual(resp.text, "ok")
        self.assertFalse(resp.fallback_used)
        self.assertEqual(self.sleeps, [1.5])

    def test_non_retryable_goes_to_next_provider(self):
        primary = FakeProvider([ProviderAuthError("401")], model="a")
        secondary = FakeProvider(["do segundo"], model="b")
        fb = FallbackProvider([primary, secondary], max_retries=3, sleep=self.sleep)
        resp = run(fb.complete(_req()))
        self.assertEqual(resp.text, "do segundo")
        self.assertTrue(resp.fallback_used)
        self.assertEqual(len(primary.requests), 1)  # sem retry para erro de auth
        self.assertEqual(self.sleeps, [])

    def test_all_fail_retryable(self):
        p1 = FakeProvider([ProviderTimeoutError("t")] * 3)
        p2 = FakeProvider([ProviderUnavailableError("u")] * 3)
        fb = FallbackProvider([p1, p2], max_retries=2, sleep=self.sleep)
        with self.assertRaises(ProviderUnavailableError) as cm:
            run(fb.complete(_req()))
        self.assertIn("6 erros", str(cm.exception))
        self.assertEqual(len(self.sleeps), 4)

    def test_all_fail_non_retryable_keeps_type(self):
        fb = FallbackProvider([FakeProvider([ProviderRequestError("bad")])], sleep=self.sleep)
        with self.assertRaises(ProviderRequestError):
            run(fb.complete(_req()))

    def test_interruption_propagates(self):
        fb = FallbackProvider([FakeProvider([ProviderInterrupted("x")]), FakeProvider(["never"])], sleep=self.sleep)
        with self.assertRaises(ProviderInterrupted):
            run(fb.complete(_req()))

    def test_requires_provider(self):
        with self.assertRaises(ValueError):
            FallbackProvider([])


class FactoryTests(unittest.TestCase):
    def test_build_provider_with_fallbacks(self):
        cfg = AgentConfig(project_root=".", llm_model="claude-opus-5", llm_fallback_models=("claude-sonnet-5",), llm_max_retries=1)
        p = build_provider(cfg)
        self.assertIsInstance(p, FallbackProvider)
        self.assertEqual([x.model for x in p.providers], ["claude-opus-5", "claude-sonnet-5"])
        self.assertEqual(p.max_retries, 1)
        self.assertEqual(p.describe()["chain"][0]["model"], "claude-opus-5")

    def test_build_provider_without_fallbacks(self):
        cfg = AgentConfig(project_root=".", llm_model="claude-x", llm_enable_fallbacks=False)
        p = build_provider(cfg)
        self.assertIsInstance(p, AnthropicProvider)
        self.assertEqual(p.model, "claude-x")
        self.assertFalse(p.server_fallbacks)

    def test_fake_provider_records_and_exhausts(self):
        fp = FakeProvider([lambda r: f"echo:{r.messages[0].parts[0].text}"])
        self.assertEqual(run(fp.complete(_req("x"))).text, "echo:x")
        self.assertEqual(len(fp.requests), 1)
        with self.assertRaises(ProviderInvalidResponseError):
            run(fp.complete(_req()))
