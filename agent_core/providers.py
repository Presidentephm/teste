"""
Camada de provider de modelo (desacoplada do SDK).

    AgentLoop -> Strategy -> ModelProvider -> Anthropic SDK -> modelo

Somente este módulo importa ``anthropic``. O restante do sistema fala com
``ModelProvider`` usando tipos próprios (``ModelRequest``/``ModelResponse``,
``ContentPart``) e recebe erros normalizados (``ProviderError`` e subclasses),
o que permite trocar o SDK, injetar um ``FakeProvider`` em testes ou encadear
providers com ``FallbackProvider`` sem tocar nas estratégias.

Compatibilidade verificada com ``anthropic`` 1.3.0:
    * ``AsyncAnthropic().messages.stream(...)`` + ``get_final_message()``;
    * ``thinking={"type": "adaptive"}`` e ``output_config={"effort": ...}``;
    * fallback server-side por recusa via ``client.beta.messages.stream(
      betas=["server-side-fallback-2026-07-01"], fallbacks="default")``;
    * blocos de imagem ``{"type": "image", "source": {"type": "base64", ...}}``.

O identificador de modelo NÃO é validado localmente: a API responde com
``NotFoundError`` (-> ``ProviderRequestError``) se o ID não existir na conta.
``claude-opus-5`` é o padrão e consta na tipagem do SDK instalado; qualquer
outro ID pode ser configurado via ``AgentConfig.llm_model`` ou ``--model``.

Credenciais: nunca são passadas em código. O SDK lê ``ANTHROPIC_API_KEY`` /
``ANTHROPIC_AUTH_TOKEN`` ou um perfil de ``ant auth login``.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Sequence

from .safety import redact

logger = logging.getLogger("agent_core.providers")

# ------------------------------------------------------------------ mensagens
PartType = Literal["text", "image"]


@dataclass
class ContentPart:
    """Parte de uma mensagem: texto ou imagem (bytes + media type)."""

    type: PartType
    text: str = ""
    data: bytes = b""
    media_type: str = "image/jpeg"

    @classmethod
    def from_text(cls, text: str) -> "ContentPart":
        return cls(type="text", text=text)

    @classmethod
    def from_image(cls, data: bytes, media_type: str = "image/jpeg") -> "ContentPart":
        return cls(type="image", data=data, media_type=media_type)


@dataclass
class ModelMessage:
    role: Literal["user", "assistant"]
    parts: list[ContentPart]

    @classmethod
    def user(cls, *parts: ContentPart | str) -> "ModelMessage":
        return cls("user", [ContentPart.from_text(p) if isinstance(p, str) else p for p in parts])


@dataclass
class ModelRequest:
    messages: list[ModelMessage]
    system: str = ""
    max_tokens: int | None = None
    effort: str | None = None
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
    raw: Any = None


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

    @abstractmethod
    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Envia a conversa ao modelo e devolve a resposta normalizada."""

    async def aclose(self) -> None:  # pragma: no cover - default vazio
        """Libera recursos (conexões HTTP)."""

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model}


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
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.server_fallbacks = server_fallbacks
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = client
        self._sdk: Any = None

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
            try:
                # Sem api_key explícita: o SDK resolve pelo ambiente/perfil.
                self._client = sdk.AsyncAnthropic(timeout=self.timeout, max_retries=self.max_retries)
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
        return blocks or [{"type": "text", "text": "(vazio)"}]

    def _build_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens or self.max_tokens,
            "messages": [{"role": m.role, "content": self._to_blocks(m.parts)} for m in request.messages],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": request.effort or self.effort},
        }
        if request.system:
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
        text = "".join(getattr(b, "text", "") for b in content if getattr(b, "type", "") == "text")
        if not text.strip():
            raise ProviderInvalidResponseError("resposta sem bloco de texto")
        usage = getattr(message, "usage", None)
        usage_dict = {
            k: int(getattr(usage, k, 0) or 0)
            for k in ("input_tokens", "output_tokens", "cache_read_input_tokens")
            if usage is not None and getattr(usage, k, None) is not None
        }
        fallback_used = any(getattr(b, "type", "") == "fallback" for b in content)
        return ModelResponse(
            text=text,
            model=str(getattr(message, "model", self.model)),
            stop_reason=stop,
            usage=usage_dict,
            truncated=(stop == "max_tokens"),
            fallback_used=fallback_used,
            latency=latency,
            raw=message,
        )

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
        self.providers = list(providers)
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._sleep = sleep
        self.model = providers[0].model
        self.supports_images = all(p.supports_images for p in providers)

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
        return {"provider": self.name, "chain": [p.describe() for p in self.providers]}


# ----------------------------------------------------------------------- fake
class FakeProvider(ModelProvider):
    """Provider determinístico para testes e exemplos offline.

    ``responses`` pode conter strings (devolvidas em ordem), exceções (levantadas)
    ou callables ``(request) -> str | Exception``. Registra todos os pedidos em
    ``requests`` para asserções.
    """

    name = "fake"

    def __init__(self, responses: Sequence[Any] | None = None, model: str = "fake-model", latency: float = 0.0):
        self.model = model
        self._responses = list(responses or [])
        self.requests: list[ModelRequest] = []
        self.latency = latency

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self.latency:
            await asyncio.sleep(self.latency)
        if not self._responses:
            raise ProviderInvalidResponseError("FakeProvider sem respostas programadas")
        item = self._responses.pop(0)
        if callable(item):
            item = item(request)
        if isinstance(item, BaseException):
            raise item
        return ModelResponse(text=str(item), model=self.model, stop_reason="end_turn")


# ------------------------------------------------------------------- fábrica
def build_provider(config: Any) -> ModelProvider:
    """Constrói o provider a partir de ``AgentConfig``.

    * ``llm_model`` -> provider principal;
    * ``llm_fallback_models`` -> providers alternativos (mesmo SDK, outro modelo);
    * ``llm_enable_fallbacks`` liga tanto o fallback server-side por recusa
      quanto a cadeia de retries/fallback do lado do cliente.
    """
    primary = AnthropicProvider(
        model=config.llm_model,
        max_tokens=config.llm_max_tokens,
        effort=config.llm_effort,
        server_fallbacks=config.llm_enable_fallbacks,
        timeout=config.llm_timeout,
    )
    if not config.llm_enable_fallbacks:
        return primary
    chain: list[ModelProvider] = [primary]
    for model in config.llm_fallback_models:
        chain.append(
            AnthropicProvider(
                model=model,
                max_tokens=config.llm_max_tokens,
                effort=config.llm_effort,
                server_fallbacks=True,
                timeout=config.llm_timeout,
            )
        )
    return FallbackProvider(chain, max_retries=config.llm_max_retries)
