"""
Exemplo 1 - chamada ao provider.

Com credenciais (ANTHROPIC_API_KEY ou perfil `ant auth login`) usa o SDK real
via ``build_provider``; sem credenciais cai num ``FakeProvider`` e avisa.

    python examples/provider_call.py
    python examples/provider_call.py --model claude-sonnet-5 --no-fallback
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_core import AgentConfig, FakeProvider, ModelMessage, ModelRequest, ProviderError, build_provider


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--fake", action="store_true", help="força o provider falso")
    ns = ap.parse_args()

    config = AgentConfig(project_root=Path(__file__).resolve().parent.parent, llm_model=ns.model, llm_enable_fallbacks=not ns.no_fallback, llm_effort="low")
    has_credentials = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if ns.fake or not has_credentials:
        print("[aviso] sem credenciais no ambiente: usando FakeProvider (defina ANTHROPIC_API_KEY para o SDK real)")
        provider = FakeProvider(['{"rationale": "exemplo", "confidence": 1.0, "patches": []}'])
    else:
        provider = build_provider(config)
    print("provider:", provider.describe())

    request = ModelRequest(
        system="Responda em JSON com as chaves rationale, confidence e patches.",
        messages=[ModelMessage.user("Qual é a causa provável de `NameError: name 'json' is not defined`?")],
        effort="low",
    )
    try:
        response = await provider.complete(request)
    except ProviderError as exc:
        print(f"erro [{exc.code}] retryable={exc.retryable}: {exc}")
        return 1
    finally:
        await provider.aclose()
    print(f"modelo={response.model} stop={response.stop_reason} fallback={response.fallback_used} tokens={response.usage}")
    print(response.text)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
