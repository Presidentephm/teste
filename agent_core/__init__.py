"""
agent_core
==========

Núcleo do agente autônomo multimodal auto-modificável.

Camadas (de baixo para cima):

    config        -> AgentConfig: parâmetros globais.
    safety        -> redação de credenciais e guarda de patches.
    backup        -> BackupManager: backups com timestamp, checkpoints e rollback.
    code_manager  -> CodeManager: leitura, análise (AST) e reescrita segura de .py.
    sandbox       -> Sandbox: execução isolada via subprocess com captura de logs.
    providers     -> ModelProvider: camada desacoplada sobre o SDK Anthropic.
    observations  -> Observation / MultimodalContext / Observer.
    vision        -> fontes visuais (câmera, tela, imagem), pipeline OpenCV, captura.
    memory        -> AgentMemory: o que já foi tentado e o resultado.
    strategies    -> FixStrategy, HeuristicFixStrategy, ModelFixStrategy, AutoStrategy.
    agent_loop    -> SelfImprovementAgent: observar -> decidir -> agir -> validar.
    cli           -> python -m agent_core ...

Uso mínimo::

    import asyncio
    from agent_core import AgentConfig, SelfImprovementAgent

    config = AgentConfig(project_root=".")
    report = asyncio.run(SelfImprovementAgent(config).run("examples/broken_script.py"))
    print(report.summary())
"""

from .config import AgentConfig, load_env_file, setup_logging
from .safety import PatchGuard, UnsafePatchError, redact
from .backup import BackupManager, BackupRecord, Checkpoint
from .code_manager import CodeManager, ModuleAnalysis, FilePatch, Replacement, InvalidSourceError, PathOutsideProjectError
from .sandbox import Sandbox, ExecutionResult, TracebackInfo
from .providers import (
    ModelProvider,
    AnthropicProvider,
    FallbackProvider,
    FakeProvider,
    ModelRequest,
    ModelResponse,
    ModelMessage,
    ContentPart,
    ProviderError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    ProviderInvalidResponseError,
    ProviderRefusalError,
    ProviderRequestError,
    ProviderInterrupted,
    ToolSpec,
    ToolCall,
    UsageTracker,
    build_provider,
)
from .openai_provider import OpenAICompatProvider
from .tools import ProjectToolbox, PATCH_SCHEMA
from .observations import (
    Observation,
    ObservationKind,
    ImageData,
    MultimodalContext,
    ContextLimits,
    Observer,
    RuntimeObserver,
    TestObserver,
    LogObserver,
    CodeObserver,
)
from .memory import AgentMemory, MemoryEntry, patch_signature
from .strategies import (
    FixStrategy,
    FixProposal,
    FailureContext,
    Decision,
    ActionKind,
    HeuristicFixStrategy,
    ModelFixStrategy,
    ToolFixStrategy,
    ClaudeFixStrategy,
    CompositeFixStrategy,
    AutoStrategy,
    Diagnosis,
    Finding,
)
from .agent_loop import SelfImprovementAgent, AgentRunReport, IterationRecord

__all__ = [
    "AgentConfig", "setup_logging", "load_env_file",
    "PatchGuard", "UnsafePatchError", "redact",
    "BackupManager", "BackupRecord", "Checkpoint",
    "CodeManager", "ModuleAnalysis", "FilePatch", "Replacement", "InvalidSourceError", "PathOutsideProjectError",
    "Sandbox", "ExecutionResult", "TracebackInfo",
    "ModelProvider", "AnthropicProvider", "FallbackProvider", "FakeProvider", "ModelRequest", "ModelResponse",
    "ModelMessage", "ContentPart", "ProviderError", "ProviderAuthError", "ProviderRateLimitError",
    "ProviderTimeoutError", "ProviderUnavailableError", "ProviderInvalidResponseError", "ProviderRefusalError",
    "ProviderRequestError", "ProviderInterrupted", "ToolSpec", "ToolCall", "UsageTracker", "build_provider",
    "ProjectToolbox", "PATCH_SCHEMA", "ToolFixStrategy", "OpenAICompatProvider",
    "Observation", "ObservationKind", "ImageData", "MultimodalContext", "ContextLimits", "Observer",
    "RuntimeObserver", "TestObserver", "LogObserver", "CodeObserver",
    "AgentMemory", "MemoryEntry", "patch_signature",
    "FixStrategy", "FixProposal", "FailureContext", "Decision", "ActionKind", "HeuristicFixStrategy",
    "ModelFixStrategy", "ClaudeFixStrategy", "CompositeFixStrategy", "AutoStrategy", "Diagnosis", "Finding",
    "SelfImprovementAgent", "AgentRunReport", "IterationRecord",
]

__version__ = "0.3.0"
