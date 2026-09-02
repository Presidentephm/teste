"""
Configuração central do agente.

Tudo que é "ajustável" vive aqui, num único dataclass imutável por convenção.
Os demais módulos recebem uma instância de ``AgentConfig`` e nunca leem
variáveis globais, o que facilita testes e execução de vários agentes em
paralelo sobre projetos diferentes.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentConfig:
    """Parâmetros globais do núcleo.

    Attributes:
        project_root: Diretório raiz do projeto que o agente pode ler/modificar.
            Todo caminho manipulado pelo agente é confinado a esta pasta.
        backup_dir_name: Nome da pasta (dentro de ``project_root``) onde os
            backups com timestamp são armazenados.
        max_backups_per_file: Quantas versões antigas manter por arquivo antes
            de podar as mais velhas (0 = ilimitado).
        sandbox_timeout: Tempo máximo (segundos) de execução de um script no
            sandbox antes de ser morto.
        sandbox_memory_limit_mb: Limite de memória virtual do processo filho
            (apenas POSIX; ``None`` desativa).
        sandbox_isolated_copy: Se ``True``, o sandbox executa o script numa
            cópia temporária do projeto, de modo que efeitos colaterais
            (arquivos criados, etc.) nunca atingem o projeto real.
        max_iterations: Número máximo de ciclos executar->corrigir do loop.
        stagnation_limit: Quantas vezes o mesmo erro pode se repetir antes de
            o loop desistir (evita ficar preso em ciclos inúteis).
        python_executable: Interpretador usado no sandbox.
        protected_paths: Caminhos relativos que o agente NUNCA modifica.
        allow_self_modification: Permite que o agente reescreva o próprio
            pacote ``agent_core``. Mantido ligado por padrão (o backup e a
            validação sintática são a rede de segurança), mas pode ser
            desligado para operar sobre projetos de terceiros.
        log_level: Nível de log padrão.
        llm_model: Modelo Claude usado pela ``ClaudeFixStrategy``.
        llm_effort: Nível de esforço de raciocínio (low/medium/high/xhigh/max).
        llm_max_tokens: Limite de tokens de saída por chamada ao modelo.
        llm_enable_fallbacks: Ativa o fallback server-side em caso de recusa
            do modelo (ver ``strategies.ClaudeFixStrategy``).
    """

    project_root: Path
    backup_dir_name: str = ".agent_backups"
    max_backups_per_file: int = 20
    sandbox_timeout: float = 30.0
    sandbox_memory_limit_mb: int | None = 512
    sandbox_isolated_copy: bool = True
    max_iterations: int = 5
    stagnation_limit: int = 2
    python_executable: str = sys.executable
    protected_paths: tuple[str, ...] = (".git",)
    allow_self_modification: bool = True
    log_level: str = "INFO"
    llm_model: str = "claude-opus-5"
    llm_effort: str = "high"
    llm_max_tokens: int = 16000
    llm_enable_fallbacks: bool = True
    llm_fallback_models: tuple[str, ...] = ()
    llm_timeout: float = 600.0
    llm_max_retries: int = 2
    extra_sandbox_env: dict[str, str] = field(default_factory=dict)

    # -- ciclo autônomo
    max_retries: int = 3               # tentativas de patch fracassadas (rollback/patch inválido) antes de desistir
    total_timeout: float | None = None # tempo máximo (s) de um ``run`` inteiro; None = sem limite
    test_command: tuple[str, ...] | None = None  # ex.: ("-m", "unittest", "discover", "-s", "tests")
    observation_interval: float = 5.0  # segundos entre observações visuais "por tempo"
    log_patterns: tuple[str, ...] = ("*.log", "logs/*.log")

    # -- visão
    vision_enabled: bool = False
    vision_source: str = "camera"      # camera | screen | image
    vision_camera_index: int = 0
    vision_monitor: int = 1
    vision_images: tuple[str, ...] = ()
    vision_fps: float = 2.0
    vision_store_frames: bool = False

    # -- memória e segurança
    memory_limit: int = 100
    memory_persist: bool = True        # grava memória em <backup_dir>/memory.json
    max_patch_files: int = 8
    max_removed_ratio: float = 0.6

    def __post_init__(self) -> None:
        # Normaliza a raiz para um Path absoluto e resolvido (sem symlinks),
        # pois toda a checagem de confinamento depende de comparações exatas.
        self.project_root = Path(self.project_root).expanduser().resolve()
        if not self.project_root.is_dir():
            raise NotADirectoryError(f"project_root não existe: {self.project_root}")
        if self.max_iterations < 1:
            raise ValueError("max_iterations deve ser >= 1")
        if self.sandbox_timeout <= 0:
            raise ValueError("sandbox_timeout deve ser > 0")
        if self.max_retries < 0:
            raise ValueError("max_retries deve ser >= 0")
        if self.observation_interval < 0:
            raise ValueError("observation_interval deve ser >= 0")
        if self.vision_source not in ("camera", "screen", "image"):
            raise ValueError("vision_source deve ser camera, screen ou image")
        if self.llm_effort not in ("low", "medium", "high", "xhigh", "max"):
            raise ValueError("llm_effort inválido")
        self.test_command = tuple(self.test_command) if self.test_command else None
        self.llm_fallback_models = tuple(self.llm_fallback_models)
        self.vision_images = tuple(str(p) for p in self.vision_images)

    @property
    def backup_dir(self) -> Path:
        """Caminho absoluto da pasta de backups."""
        return self.project_root / self.backup_dir_name

    @property
    def memory_path(self) -> Path | None:
        """Arquivo de memória persistente (ou ``None`` se desligada)."""
        return self.backup_dir / "memory.json" if self.memory_persist else None

    @property
    def all_protected_paths(self) -> tuple[str, ...]:
        """Caminhos protegidos, incluindo sempre a pasta de backups."""
        return tuple(dict.fromkeys((*self.protected_paths, self.backup_dir_name)))


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configura (uma única vez) o logger raiz do pacote e o devolve.

    O formatter redige credenciais (ver ``safety.redact``) em toda linha.
    """
    from .safety import RedactingFormatter

    logger = logging.getLogger("agent_core")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            RedactingFormatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level.upper())
    return logger
