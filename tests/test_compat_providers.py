"""Provedores compatíveis com a API Messages (ex.: Moonshot/Kimi)."""

from __future__ import annotations

import os
import unittest
from unittest import mock

from agent_core import AgentConfig, FallbackProvider
from agent_core.providers import (
    PROVIDER_PRESETS,
    AnthropicProvider,
    ContentPart,
    ModelMessage,
    ModelRequest,
    ProviderAuthError,
    ToolSpec,
    build_provider,
)
from agent_core.strategies import ModelFixStrategy, ToolFixStrategy
from agent_core.tools import PATCH_SCHEMA
from tests._helpers import FakeAnthropicClient, TempProject, fake_message, run


def _req(**kw) -> ModelRequest:
    kw.setdefault("messages", [ModelMessage("user", [ContentPart.from_text("oi"), ContentPart.from_image(b"img", "image/png")])])
    kw.setdefault("system", "sys")
    return ModelRequest(**kw)


class PresetTests(unittest.TestCase):
    def test_kimi_preset_shape(self):
        preset = PROVIDER_PRESETS["kimi"]
        self.assertEqual(preset.base_url, "https://api.moonshot.ai/anthropic")
        self.assertEqual(preset.api_key_env, "MOONSHOT_API_KEY")
        self.assertTrue(preset.compat)
        self.assertEqual(PROVIDER_PRESETS["kimi-cn"].base_url, "https://api.moonshot.cn/anthropic")
        self.assertFalse(PROVIDER_PRESETS["anthropic"].compat)
        self.assertIsNone(PROVIDER_PRESETS["anthropic"].base_url)

    def test_from_preset_disables_anthropic_only_features(self):
        p = AnthropicProvider.from_preset("kimi")
        self.assertEqual(p.base_url, "https://api.moonshot.ai/anthropic")
        self.assertTrue(p.compat)
        self.assertFalse(p.server_fallbacks)      # betas/fallbacks não existem no endpoint
        self.assertFalse(p.cache_prompts)         # cache_control idem
        self.assertFalse(p.supports_structured_output)
        self.assertTrue(p.supports_tools and p.supports_images)
        self.assertEqual(p.describe()["endpoint"], "https://api.moonshot.ai/anthropic")

    def test_from_preset_overrides_and_generic(self):
        p = AnthropicProvider.from_preset("kimi", model="kimi-k2.5")
        self.assertEqual(p.model, "kimi-k2.5")
        with self.assertRaises(ValueError):
            AnthropicProvider.from_preset("compat")  # sem modelo padrão
        generic = AnthropicProvider.from_preset("compat", model="meu-modelo", base_url="https://exemplo.local/v1")
        self.assertEqual((generic.model, generic.base_url, generic.api_key_env), ("meu-modelo", "https://exemplo.local/v1", "LLM_API_KEY"))


class CompatRequestTests(unittest.TestCase):
    def test_compat_omits_anthropic_only_fields(self):
        p = AnthropicProvider.from_preset("kimi")
        kw = p._build_kwargs(_req(effort="low", output_schema=PATCH_SCHEMA, tools=[ToolSpec("read_file", "lê", {"type": "object"})]))
        for forbidden in ("thinking", "output_config", "betas", "fallbacks"):
            self.assertNotIn(forbidden, kw)
        self.assertEqual(kw["system"], "sys")  # string simples, sem cache_control
        self.assertEqual(kw["model"], "kimi-k2-turbo-preview")
        self.assertEqual(kw["tools"][0]["name"], "read_file")          # ferramentas continuam
        self.assertEqual(kw["messages"][0]["content"][1]["type"], "image")  # imagens continuam

    def test_anthropic_mode_keeps_fields(self):
        kw = AnthropicProvider()._build_kwargs(_req(effort="low", output_schema=PATCH_SCHEMA))
        self.assertEqual(kw["thinking"], {"type": "adaptive"})
        self.assertEqual(kw["output_config"]["effort"], "low")
        self.assertEqual(kw["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(kw["system"][0]["cache_control"], {"type": "ephemeral"})

    def test_compat_call_uses_plain_endpoint(self):
        client = FakeAnthropicClient(fake_message('{"rationale": "r", "confidence": 1, "patches": []}'))
        p = AnthropicProvider.from_preset("kimi", client=client)
        resp = run(p.complete(_req()))
        self.assertEqual(len(client.beta_calls), 0)  # nunca usa o caminho beta
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(resp.model, "claude-opus-5")  # o que o endpoint devolver
        self.assertEqual(p.usage.calls, 1)

    def test_missing_credential_is_reported_clearly(self):
        p = AnthropicProvider.from_preset("kimi")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MOONSHOT_API_KEY", None)
            with self.assertRaises(ProviderAuthError) as cm:
                p._get_client()
        self.assertIn("MOONSHOT_API_KEY", str(cm.exception))

    def test_credential_comes_from_env_and_is_not_logged(self):
        p = AnthropicProvider.from_preset("kimi")
        captured = {}

        class _SDK:
            class AnthropicError(Exception):
                pass

            @staticmethod
            def AsyncAnthropic(**kwargs):
                captured.update(kwargs)
                return object()

        p._sdk = _SDK
        with mock.patch.dict(os.environ, {"MOONSHOT_API_KEY": "sk-moonshot-secreta-123456"}):
            p._get_client()
        self.assertEqual(captured["api_key"], "sk-moonshot-secreta-123456")
        self.assertEqual(captured["base_url"], "https://api.moonshot.ai/anthropic")
        self.assertNotIn("secreta", str(p.describe()))  # a chave nunca aparece no describe


class FactoryTests(TempProject):
    def test_build_provider_from_config(self):
        cfg = AgentConfig(project_root=self.root, llm_provider="kimi", llm_model="")
        p = build_provider(cfg)
        self.assertIsInstance(p, FallbackProvider)
        head = p.providers[0]
        self.assertEqual(head.model, "kimi-k2-turbo-preview")  # modelo padrão do preset
        self.assertTrue(head.compat)
        self.assertFalse(p.supports_structured_output)

    def test_config_overrides_and_validation(self):
        cfg = AgentConfig(project_root=self.root, llm_provider="kimi", llm_model="kimi-k2.5", llm_base_url="https://proxy.local/anthropic", llm_api_key_env="MINHA_CHAVE", llm_enable_fallbacks=False)
        p = build_provider(cfg)
        self.assertEqual((p.model, p.base_url, p.api_key_env), ("kimi-k2.5", "https://proxy.local/anthropic", "MINHA_CHAVE"))
        with self.assertRaises(ValueError):
            build_provider(AgentConfig(project_root=self.root, llm_provider="inexistente"))
        with self.assertRaises(ValueError):
            build_provider(AgentConfig(project_root=self.root, llm_provider="compat", llm_model=""))
        with self.assertRaises(ValueError):
            AgentConfig(project_root=self.root, llm_base_url="ftp://x")


class StrategyCompatTests(TempProject):
    """Sem saída estruturada, a estratégia cai no JSON em texto."""

    def test_schema_is_skipped_for_compat_provider(self):
        self.write("app.py", "print(json.dumps({}))\n")
        client = FakeAnthropicClient(fake_message('{"rationale": "faltou import", "confidence": 0.9, "patches": [{"path": "app.py", "mode": "search_replace", "replacements": [{"search": "print(", "replace": "import json\\nprint("}]}]}'))
        provider = AnthropicProvider.from_preset("kimi", client=client)
        ctx = self.failure_context("app.py")
        proposal = run(ModelFixStrategy(provider).propose(ctx))
        self.assertIsNotNone(proposal)
        self.assertNotIn("output_config", client.calls[0])
        self.assertEqual(proposal.patches[0].path, "app.py")

    def test_tool_strategy_never_sends_schema(self):
        self.write("app.py", "print(json.dumps({}))\n")
        provider = AnthropicProvider.from_preset("kimi", client=FakeAnthropicClient(fake_message('{"rationale": "x", "confidence": 0, "patches": []}')))
        strategy = ToolFixStrategy(provider)
        request = strategy.build_request(self.failure_context("app.py"))
        self.assertIsNone(request.output_schema)


class EnvFileTests(TempProject):
    """O .env é conveniência local: não sobrescreve o ambiente nem vaza valores."""

    def test_load_env_file(self):
        from agent_core.config import load_env_file

        path = self.root / ".env"
        path.write_text('# comentário\n\nMOONSHOT_API_KEY=sk-do-arquivo\nexport OUTRA="com aspas"\nJA_DEFINIDA=do-arquivo\nlinha-invalida\n')
        with mock.patch.dict(os.environ, {"JA_DEFINIDA": "do-ambiente"}, clear=False):
            loaded = load_env_file(path)
            self.assertEqual(sorted(loaded), ["MOONSHOT_API_KEY", "OUTRA"])
            self.assertEqual(os.environ["MOONSHOT_API_KEY"], "sk-do-arquivo")
            self.assertEqual(os.environ["OUTRA"], "com aspas")
            self.assertEqual(os.environ["JA_DEFINIDA"], "do-ambiente")  # ambiente vence
            self.assertEqual(load_env_file(path, override=True).count("JA_DEFINIDA"), 1)
            self.assertEqual(os.environ["JA_DEFINIDA"], "do-arquivo")
        os.environ.pop("MOONSHOT_API_KEY", None)
        os.environ.pop("OUTRA", None)
        os.environ.pop("JA_DEFINIDA", None)

    def test_missing_file_is_noop(self):
        from agent_core.config import load_env_file

        self.assertEqual(load_env_file(self.root / "nao-existe.env"), [])
