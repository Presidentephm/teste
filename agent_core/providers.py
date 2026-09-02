"""
Camada de provider de modelo (desacoplada do SDK).

    AgentLoop -> Strategy -> ModelProvider -> Anthropic SDK -> modelo

Somente este módulo importa ``anthropic``. O restante do sistema fala com
``ModelProvider`` usando tipos próprios (``ModelRequest``/``ModelResponse``,
``ContentPart``, ``ToolSpec``/``ToolCall``) e recebe erros normalizados
(``ProviderError`` e subclasses), o que permite trocar o SDK, injetar um
``FakeProvider`` em testes ou encadear providers com ``FallbackProvider`` sem
tocar nas estratégias.

Compatibilidade verificada com ``anthropic`` 1.3.0:
    * ``AsyncAnthropic().messages.stream(...)`` + ``get_final_message()``;
    * ``thinking={"type": "adaptive"}`` e ``output_config={"effort": ...}``;
    * ferramentas (``tools=[{name, description, input_schema}]``) com loop
      manual: blocos ``tool_use`` na resposta e ``tool_result`` na volta;
    * saída estruturada ``output_config={"format": {"type": "json_schema", ...}}``;
    * cache de prompt: ``cache_control`` no bloco de sistema;
    * fallback server-side por recusa via ``client.beta.messages.stream(
      betas=["server-side-fallback-2026-07-01"], fallbacks="default")``;
    * blocos de imagem ``{"type": "image", "source": {"type": "base64", ...}}``.

O identificador de modelo NÃO é validado localmente: a API responde com
``NotFoundError`` (-> ``ProviderRequestError``) se o ID não existir na conta.
``claude-opus-5`` é o padrão e consta na tipagem do SDK instalado; qualquer
outro ID pode ser configurado via ``AgentConfig.llm_model`` ou ``--model``.

Credenciais: nunca são passadas em código. O SDK lê ``ANTHROPIC_API_KEY`` /
``ANTHROPIC_AUTH_TOKEN`` ou um perfil de ``ant auth login``; com um preset de
outro provedor, a variável de ambiente correspondente (ex.: ``MOONSHOT_API_KEY``).

Provedores compatíveis com a API Messages
-----------------------------------------
Alguns provedores expõem um endpoint no formato Anthropic (ex.: Moonshot/Kimi
em ``https://api.moonshot.ai/anthropic``). ``PROVIDER_PRESETS`` guarda base
URL, variável de credencial e modelo padrão de cada um; o modo ``compat``
envia apenas o subconjunto universal da API Messages (model, max_tokens,
messages, system, tools) e omite o que é exclusivo da Anthropic — thinking
adaptativo, ``output_config`` (effort e saída estruturada), ``cache_control``
e o fallback server-side. Todo o resto do sistema (loop de ferramentas,
imagens, normalização de erros, contabilidade de uso) é reaproveitado.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Sequence

from .safety import redact

logger = logging.getLogger("agent_core.providers")

# ------------------------------------------------------------------ mensagens
PartType = Literal["text", "image", "tool_use", "tool_result"]


@dataclass
class ToolSpec:
    """Definição de uma ferramenta que o modelo pode chamar."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    """Pedido do modelo para executar uma ferramenta."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ContentPart:
    """Parte de uma mensagem: texto, imagem, chamada ou resultado de ferramenta."""

    type: PartType
    text: str = ""
    data: bytes = b""
    media_type: str = "image/jpeg"
    tool_call: ToolCall | None = None
    tool_use_id: str = ""
    is_error: bool = False

    @classmethod
    def from_text(cls, text: str) -> "ContentPart":
        return cls(type="text", text=text)

    @classmethod
    def from_image(cls, data: bytes, media_type: str = "image/jpeg") -> "ContentPart":
        return cls(type="image", data=data, media_type=media_type)

    @classmethod
    def from_tool_use(cls, call: ToolCall) -> "ContentPart":
        return cls(type="tool_use", tool_call=call)

    @classmethod
    def from_tool_result(cls, tool_use_id: str, content: str, *, is_error: bool = False) -> "ContentPart":
        return cls(type="tool_result", text=content, tool_use_id=tool_use_id, is_error=is_error)


@dataclass
class ModelMessage:
    role: Literal["user", "assistant"]
    parts: list[ContentPart]
    raw: Any = None  # payload opaco do provider para reenviar um turno do assistente sem perdas

    @classmethod
    def user(cls, *parts: ContentPart | str) -> "ModelMessage":
        return cls("user", [ContentPart.from_text(p) if isinstance(p, str) else p for p in parts])

    @classmethod
    def assistant_from(cls, response: "ModelResponse") -> "ModelMessage":
        return cls("assistant", list(response.parts), raw=response.raw_content)

    @classmethod
    def tool_results(cls, results: Sequence[ContentPart]) -> "ModelMessage":
        return cls("user", list(results))


@dataclass
class ModelRequest:
    messages: list[ModelMessage]
    system: str = ""
    max_tokens: int | None = None
    effort: str | None = None
    tools: list[ToolSpec] = field(default_factory=list)
    output_schema: dict[str, Any] | None = None  # saída estruturada (JSON Schema)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def image_count(self) -> int:
        return sum(1 for m in self.messages for p in m.parts if p.type == "image")


@dataclass
class ModelResponse:
    text: str
    model: str
    stop_reason: str | None
    usage: dict[str, int] = field(default_factory=dict)
    truncated: bool = False
    fallback_used: bool = False
    latency: float = 0.0
    parts: list[ContentPart] = field(default_factory=list)
    raw: Any = None
    raw_content: Any = None

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [p.tool_call for p in self.parts if p.type == "tool_use" and p.tool_call is not None]

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# --------------------------------------------------------------------- custo
# USD por milhão de tokens (entrada, saída). Cache: leitura 10 %, escrita 125 % da entrada.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # Moonshot/Kimi (confira valores atuais no console do provedor).
    "kimi-k2": (0.6, 2.5),
    "kimi-latest": (0.6, 2.5),
    "moonshot-v1": (0.6, 2.5),
}


@dataclass(frozen=True)
class ProviderPreset:
    """Configuração de um endpoint compatível com a API Messages.

    Attributes:
        base_url: endereço do endpoint (``None`` = API da Anthropic).
        api_key_env: variável de ambiente que guarda a credencial.
        default_model: modelo usado quando o usuário não especifica um.
        compat: envia apenas o subconjunto universal da API Messages.
        docs: página de referência do provedor.
    """

    base_url: str | None = None
    api_key_env: str = "ANTHROPIC_API_KEY"
    default_model: str = "claude-opus-5"
    compat: bool = False
    docs: str = "https://docs.anthropic.com"


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    "anthropic": ProviderPreset(),
    # Moonshot AI (Kimi) — endpoint compatível com a API Messages.
    # O ID do modelo muda a cada geração: confirme o disponível na sua conta
    # e passe com --model; o padrão abaixo é apenas um ponto de partida.
    "kimi": ProviderPreset(
        base_url="https://api.moonshot.ai/anthropic",
        api_key_env="MOONSHOT_API_KEY",
        default_model="kimi-k2-turbo-preview",
        compat=True,
        docs="https://platform.kimi.ai/docs/api/overview",
    ),
    "kimi-cn": ProviderPreset(
        base_url="https://api.moonshot.cn/anthropic",
        api_key_env="MOONSHOT_API_KEY",
        default_model="kimi-k2-turbo-preview",
        compat=True,
        docs="https://platform.moonshot.cn/docs",
    ),
    # Genérico: informe --base-url e --api-key-env você mesmo.
    "compat": ProviderPreset(api_key_env="LLM_API_KEY", default_model="", compat=True),
}


def price_for(model: str) -> tuple[float, float] | None:
    for key, price in MODEL_PRICES.items():
        if model.startswith(key):
            return price
    return None


@dataclass
class UsageTracker:
    """Acumula tokens, latência e custo estimado de todas as chamadas."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_total: float = 0.0
    cost_usd: float = 0.0
    priced: bool = True  # False se algum modelo não tinha preço conhecido

    def record(self, model: str, usage: dict[str, int], latency: float = 0.0) -> None:
        self.calls += 1
        i = usage.get("input_tokens", 0)
        o = usage.get("output_tokens", 0)
        cr = usage.get("cache_read_input_tokens", 0)
        cw = usage.get("cache_creation_input_tokens", 0)
        self.input_tokens += i
        self.output_tokens += o
        self.cache_read_tokens += cr
        self.cache_write_tokens += cw
        self.latency_total += latency
        price = price_for(model)
        if price is None:
            self.priced = False
            return
        pin, pout = price
        self.cost_usd += (i * pin + o * pout + cr * pin * 0.1 + cw * pin * 1.25) / 1_000_000

    def merge(self, other: "UsageTracker") -> "UsageTracker":
        out = UsageTracker(**{k: getattr(self, k) + getattr(other, k) for k in ("calls", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "latency_total", "cost_usd")})
        out.priced = self.priced and other.priced
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "latency_total": round(self.latency_total, 3),
            "cost_usd": round(self.cost_usd, 6),
            "priced": self.priced,
        }


# --------------------------------------------------------------------- erros
class ProviderError(Exception):
    """Erro normalizado de provider. ``retryable`` orienta retries/fallback."""

    retryable: bool = False
    code: str = "provider_error"

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(redact(message))
        self.cause = cause


class ProviderAuthError(ProviderError):
    code = "auth"


class ProviderRequestError(ProviderError):
    """Pedido inválido (400/404/422): modelo inexistente, parâmetro errado..."""

    code = "bad_request"


class ProviderRateLimitError(ProviderError):
    code = "rate_limit"
    retryable = True

    def __init__(self, message: str, *, retry_after: float | None = None, cause=None) -> None:
        super().__init__(message, cause=cause)
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    code = "timeout"
    retryable = True


class ProviderUnavailableError(ProviderError):
    """5xx, overload, falha de conexão."""

    code = "unavailable"
    retryable = True


class ProviderInvalidResponseError(ProviderError):
    """Resposta sem texto, recusada ou com formato inesperado."""

    code = "invalid_response"


class ProviderRefusalError(ProviderInvalidResponseError):
    code = "refusal"


class ProviderInterrupted(ProviderError):
    """A chamada foi cancelada (Ctrl+C / cancelamento da task)."""

    code = "interrupted"


# ----------------------------------------------------------------- interface
class ModelProvider(ABC):
    """Contrato mínimo que as estratégias usam."""

    name: str = "base"
    model: str = ""
    supports_images: bool = True
    supports_tools: bool = True

    def __init__(self) -> None:
        self._usage = UsageTracker()

    @property
    def usage(self) -> UsageTracker:
        return self._usage

    @abstractmethod
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Envia a conversa ao modelo e devolve a resposta normalizada."""

    async def aclose(self) -> None:  # pragma: no cover - default vazio
        """Libera recursos (conexões HTTP)."""

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "usage": self.usage.to_dict()}


# ----------------------------------------------------------------- Anthropic
class AnthropicProvider(ModelProvider):
    """Provider real sobre o SDK oficial ``anthropic``."""

    name = "anthropic"
    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(
        self,
        model: str = "claude-opus-5",
        *,
        max_tokens: int = 16000,
        effort: str = "high",
        server_fallbacks: bool = True,
        timeout: float = 600.0,
        max_retries: int = 2,
        cache_prompts: bool = True,
        base_url: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        compat: bool = False,
        client: Any | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.compat = compat
        # Em modo de compatibilidade nada disso existe do outro lado.
        self.server_fallbacks = server_fallbacks and not compat
        self.cache_prompts = cache_prompts and not compat
        self.supports_structured_output = not compat
        self._client = client
        self._sdk: Any = None

    @classmethod
    def from_preset(cls, preset: str | ProviderPreset, **kwargs: Any) -> "AnthropicProvider":
        """Cria o provider a partir de um preset (ex.: ``"kimi"``).

        ``model`` e ``base_url`` passados em ``kwargs`` têm precedência sobre
        os do preset.
        """
        cfg = PROVIDER_PRESETS[preset] if isinstance(preset, str) else preset
        kwargs.setdefault("model", cfg.default_model)
        kwargs.setdefault("base_url", cfg.base_url)
        kwargs.setdefault("api_key_env", cfg.api_key_env)
        kwargs.setdefault("compat", cfg.compat)
        if not kwargs.get("model"):
            raise ValueError("preset sem modelo padrão: informe --model")
        return cls(**kwargs)

    # -- SDK
    def _load_sdk(self) -> Any:
        if self._sdk is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ProviderUnavailableError(
                    "SDK 'anthropic' não instalado (pip install anthropic)", cause=exc
                ) from exc
            major = int(str(getattr(anthropic, "__version__", "0")).split(".")[0] or 0)
            if major < 1:
                logger.warning(
                    "anthropic %s é anterior à série 1.x testada; recursos beta podem diferir",
                    anthropic.__version__,
                )
            self._sdk = anthropic
        return self._sdk

    def _get_client(self) -> Any:
        if self._client is None:
            sdk = self._load_sdk()
            kwargs: dict[str, Any] = {"timeout": self.timeout, "max_retries": self.max_retries}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            if self.api_key_env != "ANTHROPIC_API_KEY":
                # Credencial de outro provedor: lida do ambiente, nunca do código.
                key = os.environ.get(self.api_key_env)
                if not key:
                    raise ProviderAuthError(f"variável {self.api_key_env} não definida (exporte a chave do provedor)")
                kwargs["api_key"] = key
            try:
                # Sem api_key explícita: o SDK resolve pelo ambiente/perfil.
                self._client = sdk.AsyncAnthropic(**kwargs)
            except (TypeError, ValueError, sdk.AnthropicError) as exc:
                raise ProviderAuthError(f"não foi possível criar o cliente: {exc}", cause=exc) from exc
        return self._client

    # -- tradução de tipos
    @staticmethod
    def _to_blocks(parts: Sequence[ContentPart]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for part in parts:
            if part.type == "text":
                if part.text:
                    blocks.append({"type": "text", "text": part.text})
            elif part.type == "image":
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.media_type,
                            "data": base64.b64encode(part.data).decode("ascii"),
                        },
                    }
                )
            elif part.type == "tool_use" and part.tool_call is not None:
                blocks.append({"type": "tool_use", "id": part.tool_call.id, "name": part.tool_call.name, "input": part.tool_call.input})
            elif part.type == "tool_result":
                block: dict[str, Any] = {"type": "tool_result", "tool_use_id": part.tool_use_id, "content": part.text}
                if part.is_error:
                    block["is_error"] = True
                blocks.append(block)
        return blocks or [{"type": "text", "text": "(vazio)"}]

    def _build_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        messages = []
        for m in request.messages:
            # Um turno do assistente reenviado usa o conteúdo original do SDK
            # (preserva blocos de thinking, exigidos ao continuar após tool_use).
            content = m.raw if (m.role == "assistant" and m.raw is not None) else self._to_blocks(m.parts)
            messages.append({"role": m.role, "content": content})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens or self.max_tokens,
            "messages": messages,
        }
        if not self.compat:
            # thinking adaptativo e output_config (effort / saída estruturada)
            # são específicos da API da Anthropic.
            output_config: dict[str, Any] = {"effort": request.effort or self.effort}
            if request.output_schema is not None:
                output_config["format"] = {"type": "json_schema", "schema": request.output_schema}
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = output_config
        if request.tools:
            kwargs["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in request.tools]
        if request.system:
            if self.cache_prompts:
                # Prefixo estável (tools + system) marcado para cache; o conteúdo
                # variável (traceback, fonte) fica depois do breakpoint.
                kwargs["system"] = [{"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}]
            else:
                kwargs["system"] = request.system
        return kwargs

    def _map_exception(self, exc: BaseException) -> ProviderError:
        """Converte exceções do SDK em ``ProviderError`` (mais específica primeiro)."""
        sdk = self._sdk or self._load_sdk()
        msg = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, sdk.AuthenticationError) or isinstance(exc, sdk.PermissionDeniedError):
            return ProviderAuthError(msg, cause=exc)
        if isinstance(exc, sdk.RateLimitError):
            retry_after = None
            try:
                retry_after = float(exc.response.headers.get("retry-after", ""))
            except (ValueError, AttributeError):
                pass
            return ProviderRateLimitError(msg, retry_after=retry_after, cause=exc)
        if isinstance(exc, sdk.APITimeoutError):
            return ProviderTimeoutError(msg, cause=exc)
        if isinstance(exc, (sdk.NotFoundError, sdk.BadRequestError, sdk.UnprocessableEntityError)):
            hint = ""
            if isinstance(exc, sdk.NotFoundError):
                hint = f" (o modelo '{self.model}' existe nesta conta? ajuste --model)"
            return ProviderRequestError(msg + hint, cause=exc)
        if isinstance(exc, sdk.APIStatusError):
            if exc.status_code >= 500 or exc.status_code == 529:
                return ProviderUnavailableError(msg, cause=exc)
            return ProviderRequestError(msg, cause=exc)
        if isinstance(exc, sdk.APIConnectionError):
            return ProviderUnavailableError(msg, cause=exc)
        if isinstance(exc, sdk.APIResponseValidationError):
            return ProviderInvalidResponseError(msg, cause=exc)
        return ProviderError(msg, cause=exc)

    # -- chamada
    async def complete(self, request: ModelRequest) -> ModelResponse:
        client = self._get_client()
        sdk = self._load_sdk()
        kwargs = self._build_kwargs(request)
        started = time.perf_counter()
        try:
            if self.server_fallbacks:
                # Se o modelo recusar por política, a API reexecuta o mesmo
                # pedido num modelo substituto dentro da mesma chamada.
                async with client.beta.messages.stream(
                    betas=[self.FALLBACK_BETA], fallbacks="default", **kwargs
                ) as stream:
                    message = await stream.get_final_message()
            else:
                async with client.messages.stream(**kwargs) as stream:
                    message = await stream.get_final_message()
        except asyncio.CancelledError as exc:
            raise ProviderInterrupted("chamada ao modelo cancelada", cause=exc) from exc
        except sdk.AnthropicError as exc:
            raise self._map_exception(exc) from exc
        except (OSError, ValueError, TypeError) as exc:
            # O SDK sinaliza credenciais ausentes com TypeError na hora do pedido.
            if "authentication" in str(exc).lower() or "api_key" in str(exc).lower():
                raise ProviderAuthError(f"credenciais ausentes: defina ANTHROPIC_API_KEY ({exc})", cause=exc) from exc
            # TypeError: parâmetro não suportado pela versão do SDK; ValueError:
            # guardas do SDK (ex.: max_tokens exigindo streaming).
            raise ProviderRequestError(f"{type(exc).__name__}: {exc}", cause=exc) from exc
        return self._to_response(message, time.perf_counter() - started)

    def _to_response(self, message: Any, latency: float) -> ModelResponse:
        stop = getattr(message, "stop_reason", None)
        if stop == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise ProviderRefusalError(f"o modelo recusou o pedido (categoria={category})")
        content = getattr(message, "content", None) or []
        parts: list[ContentPart] = []
        for block in content:
            btype = getattr(block, "type", "")
            if btype == "text":
                parts.append(ContentPart.from_text(getattr(block, "text", "")))
            elif btype == "tool_use":
                raw_input = getattr(block, "input", {}) or {}
                parts.append(ContentPart.from_tool_use(ToolCall(id=str(block.id), name=str(block.name), input=dict(raw_input))))
        text = "".join(p.text for p in parts if p.type == "text")
        has_tools = any(p.type == "tool_use" for p in parts)
        if not text.strip() and not has_tools:
            raise ProviderInvalidResponseError("resposta sem bloco de texto")
        usage = getattr(message, "usage", None)
        usage_dict = {
            k: int(getattr(usage, k, 0) or 0)
            for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
            if usage is not None and getattr(usage, k, None) is not None
        }
        model = str(getattr(message, "model", self.model))
        self._usage.record(model, usage_dict, latency)
        fallback_used = any(getattr(b, "type", "") == "fallback" for b in content)
        return ModelResponse(
            text=text,
            model=model,
            stop_reason=stop,
            usage=usage_dict,
            truncated=(stop == "max_tokens"),
            fallback_used=fallback_used,
            latency=latency,
            parts=parts,
            raw=message,
            raw_content=content,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "endpoint": self.base_url or "api.anthropic.com",
            "compat": self.compat,
            "usage": self.usage.to_dict(),
        }

    async def aclose(self) -> None:
        client = self._client
        if client is not None and hasattr(client, "close"):
            result = client.close()
            if asyncio.iscoroutine(result):
                await result


# ------------------------------------------------------------------- fallback
class FallbackProvider(ModelProvider):
    """Retry com backoff + cadeia de providers alternativos.

    Política:
        * erros ``retryable`` (rate limit, timeout, indisponibilidade) são
          repetidos até ``max_retries`` vezes no mesmo provider, com backoff
          exponencial (respeitando ``retry_after`` quando informado);
        * esgotados os retries, ou em recusa/resposta inválida, passa ao
          próximo provider da lista;
        * erros de autenticação ou de pedido inválido NÃO são repetidos no
          mesmo provider (não adianta), mas ainda tentam o próximo, pois o
          próximo pode usar outro modelo;
        * ``ProviderInterrupted`` propaga imediatamente.
    """

    name = "fallback"

    def __init__(
        self,
        providers: Sequence[ModelProvider],
        *,
        max_retries: int = 2,
        base_delay: float = 0.5,
        max_delay: float = 20.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not providers:
            raise ValueError("FallbackProvider precisa de ao menos um provider")
        super().__init__()
        self.providers = list(providers)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sleep = sleep
        self.model = providers[0].model
        self.supports_images = all(p.supports_images for p in providers)
        self.supports_tools = all(p.supports_tools for p in providers)
        self.supports_structured_output = all(getattr(p, "supports_structured_output", True) for p in providers)

    @property
    def usage(self) -> UsageTracker:
        total = UsageTracker()
        for p in self.providers:
            total = total.merge(p.usage)
        return total

    async def complete(self, request: ModelRequest) -> ModelResponse:
        errors: list[ProviderError] = []
        for provider in self.providers:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await provider.complete(request)
                    if provider is not self.providers[0]:
                        response.fallback_used = True
                    return response
                except ProviderInterrupted:
                    raise
                except ProviderError as exc:
                    errors.append(exc)
                    logger.warning("provider %s falhou (tentativa %d): %s", provider.name, attempt + 1, exc)
                    if not exc.retryable or attempt == self.max_retries:
                        break
                    delay = getattr(exc, "retry_after", None) or min(
                        self.max_delay, self.base_delay * (2**attempt) + random.uniform(0, 0.25)
                    )
                    await self._sleep(delay)
        last = errors[-1] if errors else ProviderError("nenhum provider disponível")
        raise ProviderUnavailableError(
            f"todos os providers falharam ({len(errors)} erros); último: {last}", cause=last
        ) if last.retryable else last

    async def aclose(self) -> None:
        for p in self.providers:
            await p.aclose()

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "chain": [p.describe() for p in self.providers], "usage": self.usage.to_dict()}


# ----------------------------------------------------------------------- fake
class FakeProvider(ModelProvider):
    """Provider determinístico para testes e exemplos offline.

    ``responses`` pode conter strings (texto devolvido em ordem), objetos
    ``ModelResponse`` (ex.: com chamadas de ferramenta, ver ``tool_response``),
    exceções (levantadas) ou callables ``(request) -> item``. Registra todos
    os pedidos em ``requests`` para asserções.
    """

    name = "fake"

    def __init__(self, responses: Sequence[Any] | None = None, model: str = "fake-model", latency: float = 0.0):
        super().__init__()
        self.model = model
        self._responses = list(responses or [])
        self.requests: list[ModelRequest] = []
        self.latency = latency

    @staticmethod
    def tool_response(name: str, tool_input: dict[str, Any], *, call_id: str | None = None, text: str = "") -> ModelResponse:
        """Resposta que pede a execução de uma ferramenta."""
        call = ToolCall(id=call_id or f"call_{random.randrange(1 << 30):x}", name=name, input=tool_input)
        parts = ([ContentPart.from_text(text)] if text else []) + [ContentPart.from_tool_use(call)]
        return ModelResponse(text=text, model="fake-model", stop_reason="tool_use", parts=parts)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        # Guarda uma cópia rasa: loops de ferramentas reutilizam a mesma lista de mensagens.
        self.requests.append(ModelRequest(messages=list(request.messages), system=request.system, max_tokens=request.max_tokens, effort=request.effort, tools=list(request.tools), output_schema=request.output_schema, metadata=dict(request.metadata)))
        if self.latency:
            await asyncio.sleep(self.latency)
        if not self._responses:
            raise ProviderInvalidResponseError("FakeProvider sem respostas programadas")
        item = self._responses.pop(0)
        if callable(item) and not isinstance(item, BaseException):
            item = item(request)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, ModelResponse):
            item.model = self.model
            self._usage.record(self.model, {"input_tokens": 100, "output_tokens": 50}, 0.0)
            return item
        text = str(item)
        self._usage.record(self.model, {"input_tokens": 100, "output_tokens": 50}, 0.0)
        return ModelResponse(text=text, model=self.model, stop_reason="end_turn", parts=[ContentPart.from_text(text)])


# ------------------------------------------------------------------- fábrica
def build_provider(config: Any) -> ModelProvider:
    """Constrói o provider a partir de ``AgentConfig``.

    * ``llm_model`` -> provider principal;
    * ``llm_fallback_models`` -> providers alternativos (mesmo SDK, outro modelo);
    * ``llm_enable_fallbacks`` liga tanto o fallback server-side por recusa
      quanto a cadeia de retries/fallback do lado do cliente.
    """
    preset = PROVIDER_PRESETS.get(config.llm_provider)
    if preset is None:
        raise ValueError(f"provider desconhecido: {config.llm_provider} (opções: {', '.join(sorted(PROVIDER_PRESETS))})")
    common: dict[str, Any] = dict(
        max_tokens=config.llm_max_tokens,
        effort=config.llm_effort,
        timeout=config.llm_timeout,
        cache_prompts=config.llm_cache_prompts,
        base_url=config.llm_base_url or preset.base_url,
        api_key_env=config.llm_api_key_env or preset.api_key_env,
        compat=preset.compat,
    )
    model = config.llm_model or preset.default_model
    if not model:
        raise ValueError(f"o provider '{config.llm_provider}' não tem modelo padrão: informe --model")
    primary = AnthropicProvider(model=model, server_fallbacks=config.llm_enable_fallbacks, **common)
    if not config.llm_enable_fallbacks:
        return primary
    chain: list[ModelProvider] = [primary]
    for fallback_model in config.llm_fallback_models:
        chain.append(AnthropicProvider(model=fallback_model, server_fallbacks=not preset.compat, **common))
    return FallbackProvider(chain, max_retries=config.llm_max_retries)
