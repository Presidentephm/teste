"""
Provider para endpoints compatíveis com a API OpenAI (chat completions).

    AgentLoop -> Strategy -> OpenAICompatProvider -> SDK openai -> modelo

Serve provedores que **não** falam o formato Messages da Anthropic — entre
eles a NVIDIA (``https://integrate.api.nvidia.com/v1``, catálogo do
build.nvidia.com), que hospeda modelos de terceiros como ``moonshotai/kimi-k3``.

Somente este módulo importa ``openai``; o restante do sistema continua usando
``ModelRequest``/``ModelResponse``/``ProviderError``, então estratégias, loop
de ferramentas, memória e contabilidade de uso funcionam sem alteração.

Traduções feitas aqui:
    * texto e imagens -> ``content`` em partes (``image_url`` com data URI);
    * ferramentas -> ``tools=[{type: "function", function: {...}}]`` e as
      chamadas voltam em ``message.tool_calls`` com argumentos em JSON;
    * resultados -> mensagens ``{"role": "tool", "tool_call_id": ...}``;
    * saída estruturada -> ``response_format`` com ``json_schema``;
    * **histórico de raciocínio**: o turno do assistente é reenviado com o
      dicionário original devolvido pela API, incluindo ``reasoning_content``
      quando existir. Modelos como o Kimi K3 degradam se esse histórico é
      descartado entre turnos.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from typing import Any, Sequence

from .providers import (
    ContentPart,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderAuthError,
    ProviderError,
    ProviderInterrupted,
    ProviderInvalidResponseError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ToolCall,
)


class OpenAICompatProvider(ModelProvider):
    """Provider sobre o SDK ``openai``, apontado para qualquer endpoint compatível."""

    name = "openai-compat"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        key_prefix: str = "",
        max_tokens: int = 16000,
        temperature: float | None = None,
        timeout: float = 600.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.key_prefix = key_prefix  # ex.: "nvapi-" quando a chave é guardada sem ele
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.supports_structured_output = True
        self._client = client
        self._sdk: Any = None

    # -- SDK
    def _load_sdk(self) -> Any:
        if self._sdk is None:
            try:
                import openai
            except ImportError as exc:
                raise ProviderUnavailableError("SDK 'openai' não instalado (pip install openai)", cause=exc) from exc
            self._sdk = openai
        return self._sdk

    def _credential(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise ProviderAuthError(f"variável {self.api_key_env} não definida (exporte a chave do provedor)")
        if self.key_prefix and not key.startswith(self.key_prefix):
            key = self.key_prefix + key
        return key

    def _get_client(self) -> Any:
        if self._client is None:
            sdk = self._load_sdk()
            kwargs: dict[str, Any] = {"api_key": self._credential(), "timeout": self.timeout, "max_retries": self.max_retries}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            try:
                self._client = sdk.AsyncOpenAI(**kwargs)
            except (TypeError, ValueError, sdk.OpenAIError) as exc:
                raise ProviderAuthError(f"não foi possível criar o cliente: {exc}", cause=exc) from exc
        return self._client

    # -- tradução de saída (nosso formato -> OpenAI)
    @staticmethod
    def _content(parts: Sequence[ContentPart]) -> Any:
        """Conteúdo de uma mensagem do usuário: string simples ou lista de partes."""
        blocks: list[dict[str, Any]] = []
        for part in parts:
            if part.type == "text" and part.text:
                blocks.append({"type": "text", "text": part.text})
            elif part.type == "image":
                data = base64.b64encode(part.data).decode("ascii")
                blocks.append({"type": "image_url", "image_url": {"url": f"data:{part.media_type};base64,{data}"}})
        if not blocks:
            return "(vazio)"
        if len(blocks) == 1 and blocks[0]["type"] == "text":
            return blocks[0]["text"]
        return blocks

    def _messages(self, request: ModelRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        for message in request.messages:
            if message.role == "assistant":
                # Reenvia o dicionário original (preserva reasoning_content).
                messages.append(message.raw if message.raw is not None else {"role": "assistant", "content": self._content(message.parts)})
                continue
            results = [p for p in message.parts if p.type == "tool_result"]
            others = [p for p in message.parts if p.type != "tool_result"]
            for part in results:
                messages.append({"role": "tool", "tool_call_id": part.tool_use_id, "content": part.text})
            if others or not results:
                messages.append({"role": "user", "content": self._content(others)})
        return messages

    def _build_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(request),
            "max_tokens": request.max_tokens or self.max_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if request.tools:
            kwargs["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
                for t in request.tools
            ]
        elif request.output_schema is not None:
            # response_format e tools juntos confundem vários provedores.
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "patch", "schema": request.output_schema, "strict": False},
            }
        return kwargs

    # -- erros
    def _map_exception(self, exc: BaseException) -> ProviderError:
        sdk = self._sdk or self._load_sdk()
        msg = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, (sdk.AuthenticationError, sdk.PermissionDeniedError)):
            hint = f" (confira a chave em {self.api_key_env}"
            hint += f"; este endpoint espera o prefixo '{self.key_prefix}')" if self.key_prefix else ")"
            return ProviderAuthError(msg + hint, cause=exc)
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
            hint = f" (o modelo '{self.model}' existe nesta conta? ajuste --model)" if isinstance(exc, sdk.NotFoundError) else ""
            return ProviderRequestError(msg + hint, cause=exc)
        if isinstance(exc, sdk.APIStatusError):
            status = getattr(exc, "status_code", 0)
            if status in (401, 403):
                return ProviderAuthError(msg, cause=exc)
            if status == 410:
                return ProviderRequestError(msg + " (modelo aposentado pelo provedor)", cause=exc)
            if status >= 500 or status == 529:
                return ProviderUnavailableError(msg, cause=exc)
            return ProviderRequestError(msg, cause=exc)
        if isinstance(exc, sdk.APIConnectionError):
            return ProviderUnavailableError(msg, cause=exc)
        return ProviderError(msg, cause=exc)

    # -- chamada
    async def complete(self, request: ModelRequest) -> ModelResponse:
        client = self._get_client()
        sdk = self._load_sdk()
        kwargs = self._build_kwargs(request)
        started = time.perf_counter()
        try:
            completion = await client.chat.completions.create(**kwargs)
        except asyncio.CancelledError as exc:
            raise ProviderInterrupted("chamada ao modelo cancelada", cause=exc) from exc
        except sdk.OpenAIError as exc:
            raise self._map_exception(exc) from exc
        except (OSError, ValueError, TypeError) as exc:
            raise ProviderRequestError(f"{type(exc).__name__}: {exc}", cause=exc) from exc
        return self._to_response(completion, time.perf_counter() - started)

    def _to_response(self, completion: Any, latency: float) -> ModelResponse:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise ProviderInvalidResponseError("resposta sem choices")
        choice = choices[0]
        message = choice.message
        finish = getattr(choice, "finish_reason", None)
        if finish == "content_filter":
            raise ProviderInvalidResponseError("conteúdo bloqueado pelo filtro do provedor")

        text = getattr(message, "content", None) or ""
        parts: list[ContentPart] = [ContentPart.from_text(text)] if text else []
        for call in getattr(message, "tool_calls", None) or []:
            fn = getattr(call, "function", None)
            raw_args = getattr(fn, "arguments", "") or "{}"
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError:
                arguments = {"_raw": raw_args}  # o modelo devolveu JSON inválido
            if not isinstance(arguments, dict):
                arguments = {"_value": arguments}
            parts.append(ContentPart.from_tool_use(ToolCall(id=str(call.id), name=str(getattr(fn, "name", "")), input=arguments)))
        if not parts:
            raise ProviderInvalidResponseError("resposta sem texto nem chamadas de ferramenta")

        usage_obj = getattr(completion, "usage", None)
        usage: dict[str, int] = {}
        if usage_obj is not None:
            usage["input_tokens"] = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
            usage["output_tokens"] = int(getattr(usage_obj, "completion_tokens", 0) or 0)
            details = getattr(usage_obj, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details is not None else 0
            if cached:
                usage["cache_read_input_tokens"] = int(cached)
                usage["input_tokens"] = max(0, usage["input_tokens"] - int(cached))
        model = str(getattr(completion, "model", self.model))
        self._usage.record(model, usage, latency)
        return ModelResponse(
            text=text,
            model=model,
            stop_reason=finish,
            usage=usage,
            truncated=(finish == "length"),
            latency=latency,
            parts=parts,
            raw=completion,
            raw_content=self._assistant_dict(message),
        )

    @staticmethod
    def _assistant_dict(message: Any) -> dict[str, Any]:
        """Turno do assistente para reenvio, preservando o raciocínio."""
        if hasattr(message, "model_dump"):
            data = message.model_dump(exclude_none=True)
        else:  # objetos simples em teste
            data = {k: v for k, v in vars(message).items() if v is not None}
        data["role"] = "assistant"
        data.pop("function_call", None)  # campo legado, rejeitado por alguns endpoints
        if not data.get("content") and not data.get("tool_calls"):
            data["content"] = ""
        return data

    async def aclose(self) -> None:
        client = self._client
        if client is not None and hasattr(client, "close"):
            result = client.close()
            if asyncio.iscoroutine(result):
                await result

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "endpoint": self.base_url or "api.openai.com",
            "compat": True,
            "usage": self.usage.to_dict(),
        }
