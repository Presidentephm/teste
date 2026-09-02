"""
agent_core
==========

Núcleo do agente autônomo auto-modificável.

Camadas (de baixo para cima):

    config        -> AgentConfig: parâmetros globais (raiz do projeto, limites, LLM).
    backup        -> BackupManager: cópias de segurança com timestamp + rollback.
    code_manager  -> CodeManager: leitura, análise (AST) e reescrita segura de .py.
    sandbox       -> Sandbox: execução isolada via subprocess com captura de logs.
    strategies    -> FixStrategy: quem "pensa" a correção (heurística e/ou Claude).
    agent_loop    -> SelfImprovementAgent: o loop executar -> falhar -> corrigir.
    cli           -> interface de linha de comando (python -m agent_core ...).

Uso mínimo::

    import asyncio
    from agent_core import AgentConfig, SelfImprovementAgent

    config = AgentConfig(project_root=".")
    agent = SelfImprovementAgent(config)
    report = asyncio.run(agent.run("examples/broken_script.py"))
    print(report.summary())
"""

from .config import AgentConfig
from .backup import BackupManager, BackupRecord
from .code_manager import CodeManager, ModuleAnalysis, FilePatch, Replacement
from .sandbox import Sandbox, ExecutionResult, TracebackInfo
from .strategies import (
    FixStrategy,
    FixProposal,
    FailureContext,
    HeuristicFixStrategy,
    ClaudeFixStrategy,
    CompositeFixStrategy,
)
from .agent_loop import SelfImprovementAgent, AgentRunReport, IterationRecord

__all__ = [
    "AgentConfig",
    "BackupManager",
    "BackupRecord",
    "CodeManager",
    "ModuleAnalysis",
    "FilePatch",
    "Replacement",
    "Sandbox",
    "ExecutionResult",
    "TracebackInfo",
    "FixStrategy",
    "FixProposal",
    "FailureContext",
    "HeuristicFixStrategy",
    "ClaudeFixStrategy",
    "CompositeFixStrategy",
    "SelfImprovementAgent",
    "AgentRunReport",
    "IterationRecord",
]

__version__ = "0.1.0"
