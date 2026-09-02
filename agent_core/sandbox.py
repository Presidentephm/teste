"""
Ambiente de testes automatizados (sandbox local).

Executa scripts Python num **subprocesso isolado** e devolve um
``ExecutionResult`` estruturado com stdout, stderr, código de saída, duração
e o traceback já *parseado* (arquivo, linha, tipo e mensagem da exceção).

Camadas de isolamento aplicadas:

1. **Cópia temporária do projeto** (opcional, ligada por padrão): o script roda
   numa cópia em ``tempfile``; arquivos criados/apagados pelo script não
   afetam o projeto real. Os caminhos do traceback são mapeados de volta para
   os caminhos reais.
2. **Ambiente de variáveis mínimo** (``PATH``, ``HOME`` temporário, flags do
   Python) com ``-E -s`` para ignorar ``PYTHON*`` do usuário e o site-packages
   de usuário.
3. **Limites de recursos** (POSIX): memória virtual e CPU via ``resource``.
4. **Timeout** com kill do processo inteiro.

Nada disso é um sandbox de segurança contra código hostil (para isso use
contêineres); é um sandbox de *robustez*, que garante que erros, loops
infinitos e efeitos colaterais do código em teste não derrubem o agente nem
corrompam o projeto.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

from .config import AgentConfig

try:  # ``resource`` só existe em POSIX
    import resource  # type: ignore
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore


# --------------------------------------------------------------------- modelos
@dataclass
class TracebackFrame:
    file: str
    line: int
    function: str
    code: str | None = None


@dataclass
class TracebackInfo:
    """Traceback já interpretado."""

    exc_type: str
    message: str
    frames: list[TracebackFrame] = field(default_factory=list)

    @property
    def location(self) -> TracebackFrame | None:
        """Frame mais interno (onde a exceção estourou)."""
        return self.frames[-1] if self.frames else None

    @property
    def signature(self) -> str:
        """Assinatura estável para detectar "o mesmo erro" entre iterações."""
        loc = self.location
        where = f"{loc.file}:{loc.line}" if loc else "?"
        return f"{self.exc_type}@{where}:{self.message.strip()[:120]}"


@dataclass
class ExecutionResult:
    command: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False
    traceback: TracebackInfo | None = None

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    @property
    def signature(self) -> str:
        if self.timed_out:
            return "TIMEOUT"
        if self.traceback:
            return self.traceback.signature
        return f"EXIT:{self.returncode}"

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self, max_chars: int = 600) -> str:
        """Descrição curta para logs e prompts."""
        if self.success:
            return f"OK em {self.duration:.2f}s"
        if self.timed_out:
            return f"TIMEOUT após {self.duration:.1f}s"
        if self.traceback:
            tb = self.traceback
            loc = tb.location
            where = f" ({loc.file}:{loc.line} em {loc.function})" if loc else ""
            return f"{tb.exc_type}: {tb.message}{where}"
        return f"exit={self.returncode}; stderr={self.stderr[-max_chars:]!r}"


# ------------------------------------------------------------------- parser
_FRAME_RE = re.compile(r'^\s*File "(?P<file>[^"]+)", line (?P<line>\d+)(?:, in (?P<func>.+))?$')
_EXC_RE = re.compile(r"^(?P<type>[A-Za-z_][\w.]*)(?::\s?(?P<msg>.*))?$")


def parse_traceback(stderr: str, path_map: dict[str, str] | None = None) -> TracebackInfo | None:
    """Extrai a última exceção de um ``stderr`` de Python.

    Cobre tracebacks normais e ``SyntaxError`` (que não têm ``in <func>``).
    ``path_map`` traduz prefixos de caminho (ex.: cópia temporária -> projeto).
    """
    lines = stderr.splitlines()
    if not lines:
        return None

    # Considera apenas o último bloco "Traceback (most recent call last)" ou,
    # no caso de SyntaxError, o bloco a partir do último 'File "'.
    starts = [i for i, l in enumerate(lines) if l.startswith("Traceback (most recent call last)")]
    if starts:
        block = lines[starts[-1] + 1 :]
    else:
        file_idx = [i for i, l in enumerate(lines) if _FRAME_RE.match(l)]
        if not file_idx:
            return None
        block = lines[file_idx[-1] :]

    frames: list[TracebackFrame] = []
    i = 0
    while i < len(block):
        m = _FRAME_RE.match(block[i])
        if m:
            code = None
            # Linha de código costuma vir na próxima linha, indentada.
            if i + 1 < len(block) and block[i + 1].startswith((" ", "\t")):
                code = block[i + 1].strip()
            frames.append(
                TracebackFrame(
                    file=_map_path(m.group("file"), path_map),
                    line=int(m.group("line")),
                    function=m.group("func") or "<module>",
                    code=code,
                )
            )
        i += 1

    # A linha da exceção é a última linha não vazia que casa "Tipo: mensagem".
    exc_type, message = "UnknownError", ""
    for line in reversed(block):
        if not line.strip() or line.startswith((" ", "\t", "^", "~")):
            continue
        m = _EXC_RE.match(line.strip())
        if m and not _FRAME_RE.match(line):
            exc_type = m.group("type")
            message = m.group("msg") or ""
            break
    return TracebackInfo(exc_type=exc_type, message=message, frames=frames)


def _map_path(path: str, path_map: dict[str, str] | None) -> str:
    if not path_map:
        return path
    for src, dst in path_map.items():
        if path.startswith(src):
            return dst + path[len(src) :]
    return path


# ------------------------------------------------------------------ sandbox
class Sandbox:
    """Executor controlado de scripts/comandos Python."""

    IGNORE_ON_COPY = ("__pycache__", ".git", ".venv", "venv", "node_modules", "*.pyc")

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.root = config.project_root

    # ------------------------------------------------------------ helpers
    def _build_env(self) -> dict[str, str]:
        """Ambiente mínimo e determinístico para o processo filho."""
        env = {
            "PATH": os.environ.get("PATH", ""),
            "LANG": "C.UTF-8",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "AGENT_SANDBOX": "1",  # o código em teste pode detectar que está no sandbox
        }
        if os.name == "nt":  # pragma: no cover
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
        env.update(self.config.extra_sandbox_env)
        return env

    def _preexec(self):
        """Devolve uma função ``preexec_fn`` que aplica limites de recursos (POSIX)."""
        if resource is None or os.name == "nt":
            return None
        mem_mb = self.config.sandbox_memory_limit_mb
        cpu_s = int(self.config.sandbox_timeout) + 5

        def _limits() -> None:
            try:
                if mem_mb:
                    limit = mem_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            except (ValueError, OSError):
                pass  # limites não suportados nesta plataforma: segue sem eles

        return _limits

    def _copy_project(self, dest: Path) -> None:
        ignore = shutil.ignore_patterns(*self.IGNORE_ON_COPY, self.config.backup_dir_name)
        shutil.copytree(self.root, dest, ignore=ignore, dirs_exist_ok=True)

    # ---------------------------------------------------------------- run
    async def run_command(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        timeout: float | None = None,
        stdin: str | None = None,
        path_map: dict[str, str] | None = None,
        isolated: bool = False,
    ) -> ExecutionResult:
        """Executa um comando arbitrário com captura completa e timeout.

        Com ``isolated=True`` o projeto é copiado para um diretório temporário
        e o comando roda lá (``cwd`` relativo é reinterpretado na cópia); os
        caminhos do traceback voltam mapeados para o projeto real.

        Nunca levanta exceção por falha do comando: tudo vai para o
        ``ExecutionResult``. Só levanta se o próprio interpretador/executável
        não puder ser iniciado (``FileNotFoundError``), pois isso é erro de
        configuração do agente, não do código em teste.
        """
        if isolated:
            tmp_dir = Path(tempfile.mkdtemp(prefix="agent_sandbox_"))
            try:
                await asyncio.to_thread(self._copy_project, tmp_dir)
                rel_cwd = Path(cwd).resolve().relative_to(self.root) if cwd else Path(".")
                return await self.run_command(
                    command, cwd=tmp_dir / rel_cwd, timeout=timeout, stdin=stdin,
                    path_map={str(tmp_dir): str(self.root), **(path_map or {})},
                )
            finally:
                await asyncio.to_thread(shutil.rmtree, tmp_dir, True)
        timeout = timeout or self.config.sandbox_timeout
        cmd = [str(c) for c in command]
        started = time.perf_counter()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd or self.root),
            env=self._build_env(),
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=self._preexec(),
            start_new_session=True,  # permite matar o grupo inteiro no timeout
        )
        timed_out = False
        try:
            out_b, err_b = await asyncio.wait_for(
                proc.communicate(stdin.encode() if stdin is not None else None),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            timed_out = True
            await self._kill(proc)
            out_b, err_b = b"", b""
            try:
                out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=2)
            except Exception:
                pass
        duration = time.perf_counter() - started
        stdout = out_b.decode("utf-8", errors="replace")
        stderr = err_b.decode("utf-8", errors="replace")
        return ExecutionResult(
            command=cmd,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            duration=duration,
            timed_out=timed_out,
            traceback=parse_traceback(stderr, path_map) if not timed_out else None,
        )

    @staticmethod
    async def _kill(proc: asyncio.subprocess.Process) -> None:
        """Mata o processo e, em POSIX, todo o seu grupo de sessão."""
        try:
            if os.name != "nt":
                import signal

                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:  # pragma: no cover
                proc.kill()
        except ProcessLookupError:
            pass
        except Exception:
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    async def run_script(
        self,
        script: str | Path,
        args: Sequence[str] = (),
        *,
        timeout: float | None = None,
        isolated_copy: bool | None = None,
    ) -> ExecutionResult:
        """Executa ``python <script> <args>`` no sandbox.

        Com ``isolated_copy`` (padrão: ``config.sandbox_isolated_copy``) o
        projeto inteiro é copiado para um diretório temporário e o script roda
        lá; o traceback devolvido já aponta para os caminhos do projeto real.
        """
        isolated = self.config.sandbox_isolated_copy if isolated_copy is None else isolated_copy
        script_path = Path(script)
        if not script_path.is_absolute():
            script_path = self.root / script_path
        script_path = script_path.resolve()
        rel = script_path.relative_to(self.root)

        if not isolated:
            cmd = [self.config.python_executable, "-E", "-s", "-X", "faulthandler", str(script_path), *args]
            return await self.run_command(cmd, cwd=self.root, timeout=timeout)

        # Caminho relativo: dentro da cópia o script fica no mesmo lugar relativo.
        cmd = [self.config.python_executable, "-E", "-s", "-X", "faulthandler", str(rel), *args]
        return await self.run_command(cmd, cwd=self.root, timeout=timeout, isolated=True)

    async def run_tests(self, pattern: str = "test*.py", timeout: float | None = None, *, isolated: bool = True) -> ExecutionResult:
        """Roda a suíte ``unittest`` do projeto no sandbox (útil como critério de sucesso)."""
        cmd = [self.config.python_executable, "-E", "-s", "-m", "unittest", "discover", "-p", pattern]
        return await self.run_command(cmd, cwd=self.root, timeout=timeout or self.config.sandbox_timeout * 4, isolated=isolated)

    async def check_syntax(self, path: str | Path) -> ExecutionResult:
        """Compila um arquivo com ``py_compile`` sem executá-lo."""
        cmd = [self.config.python_executable, "-E", "-s", "-m", "py_compile", str(path)]
        return await self.run_command(cmd, timeout=10)
