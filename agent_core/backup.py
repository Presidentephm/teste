"""
Sistema de backup automático e rollback.

Regra de ouro do núcleo: **nenhum arquivo é modificado sem antes ter uma cópia
de segurança**. O ``CodeManager`` chama ``BackupManager.backup()`` antes de
qualquer escrita; o ``SelfImprovementAgent`` chama ``rollback()`` quando uma
tentativa de correção piora a situação.

Layout em disco::

    <project_root>/.agent_backups/
        manifest.json                      # índice de todos os backups
        examples/broken_script.py/         # espelha o caminho relativo
            20260902T164501_123456.bak
            20260902T164530_987654.bak

Cada backup é identificado por um timestamp com microssegundos, o que também
serve como ordenação natural (o mais recente é o último em ordem lexicográfica).
"""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

from .config import AgentConfig


@dataclass(frozen=True)
class BackupRecord:
    """Metadados de uma cópia de segurança.

    Attributes:
        original: Caminho relativo (à raiz do projeto) do arquivo original.
        backup_path: Caminho relativo (à pasta de backups) do arquivo .bak.
        timestamp: Instante ISO-8601 (UTC) em que o backup foi criado.
        existed: ``False`` se o arquivo ainda não existia no momento do backup
            (o rollback, nesse caso, remove o arquivo em vez de restaurá-lo).
        reason: Texto livre explicando o motivo da alteração (útil para auditoria).
        restored_at: Instante em que este backup foi usado num rollback, ou
            ``None``. Backups já restaurados são ignorados por ``latest()``,
            de modo que rollbacks sucessivos "caminham" para trás no histórico
            em vez de restaurar sempre a mesma versão.
    """

    original: str
    backup_path: str
    timestamp: str
    existed: bool
    reason: str = ""
    restored_at: str | None = None


@dataclass
class Checkpoint:
    """Conjunto de backups tirados juntos, antes de uma alteração importante."""

    id: str
    label: str
    timestamp: str
    records: list[BackupRecord] = field(default_factory=list)

    @property
    def files(self) -> list[str]:
        return [r.original for r in self.records]


class BackupManager:
    """Cria, lista, restaura e poda backups de arquivos do projeto."""

    MANIFEST_NAME = "manifest.json"

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.root = config.project_root
        self.backup_dir = config.backup_dir
        # Lock para serializar escritas no manifest entre corrotinas concorrentes.
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ utils
    def _relative(self, path: Path) -> str:
        """Devolve o caminho relativo à raiz do projeto em formato POSIX."""
        return path.resolve().relative_to(self.root).as_posix()

    @staticmethod
    def _now_stamp() -> tuple[str, str]:
        """Devolve (timestamp ISO, sufixo seguro para nome de arquivo)."""
        now = datetime.now(timezone.utc)
        return now.isoformat(), now.strftime("%Y%m%dT%H%M%S_%f")

    def _load_manifest(self) -> list[dict]:
        manifest = self.backup_dir / self.MANIFEST_NAME
        if not manifest.exists():
            return []
        try:
            return json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Manifest corrompido não pode travar o agente: recomeça o índice,
            # os arquivos .bak continuam no disco e podem ser recuperados à mão.
            return []

    def _save_manifest(self, entries: list[dict]) -> None:
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.backup_dir / (self.MANIFEST_NAME + ".tmp")
        tmp.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.backup_dir / self.MANIFEST_NAME)  # troca atômica

    # ---------------------------------------------------------------- backup
    async def backup(self, path: Path, reason: str = "") -> BackupRecord:
        """Cria uma cópia de segurança de ``path`` e registra no manifest.

        É seguro chamar para arquivos que ainda não existem: o registro marca
        ``existed=False`` e o rollback futuro simplesmente apaga o arquivo.
        """
        path = Path(path).resolve()
        rel = self._relative(path)
        iso, stamp = self._now_stamp()
        target_dir = self.backup_dir / rel
        target = target_dir / f"{stamp}.bak"
        existed = path.is_file()

        def _do_copy() -> None:
            target_dir.mkdir(parents=True, exist_ok=True)
            if existed:
                shutil.copy2(path, target)  # copy2 preserva mtime/permissões
            else:
                target.write_bytes(b"")  # marcador vazio: "não existia"

        async with self._lock:
            await asyncio.to_thread(_do_copy)
            record = BackupRecord(
                original=rel,
                backup_path=target.relative_to(self.backup_dir).as_posix(),
                timestamp=iso,
                existed=existed,
                reason=reason,
            )
            entries = self._load_manifest()
            entries.append(asdict(record))
            self._save_manifest(entries)
            await asyncio.to_thread(self._prune_locked, rel, entries)
        return record

    def _prune_locked(self, rel: str, entries: list[dict]) -> None:
        """Remove backups excedentes do arquivo ``rel`` (deve rodar sob o lock)."""
        limit = self.config.max_backups_per_file
        if limit <= 0:
            return
        mine = [e for e in entries if e["original"] == rel]
        excess = len(mine) - limit
        if excess <= 0:
            return
        # Os mais antigos vêm primeiro (timestamps ISO ordenam lexicograficamente).
        mine.sort(key=lambda e: e["timestamp"])
        for old in mine[:excess]:
            (self.backup_dir / old["backup_path"]).unlink(missing_ok=True)
            entries.remove(old)
        self._save_manifest(entries)

    # ------------------------------------------------------------------ list
    def list_backups(self, path: Path | None = None) -> list[BackupRecord]:
        """Lista backups (do arquivo dado, ou de todos), do mais antigo ao mais novo."""
        entries = self._load_manifest()
        if path is not None:
            rel = self._relative(Path(path))
            entries = [e for e in entries if e["original"] == rel]
        entries.sort(key=lambda e: e["timestamp"])
        return [BackupRecord(**e) for e in entries]

    def latest(self, path: Path, *, include_restored: bool = False) -> BackupRecord | None:
        """Backup mais recente (ainda não restaurado) de ``path`` ou ``None``."""
        records = self.list_backups(path)
        if not include_restored:
            records = [r for r in records if r.restored_at is None]
        return records[-1] if records else None

    # -------------------------------------------------------------- rollback
    async def rollback(self, path: Path, record: BackupRecord | None = None) -> BackupRecord:
        """Restaura ``path`` para o estado do ``record`` (ou do backup mais recente).

        A restauração é "instantânea" no sentido de que é um único ``copy``
        atômico via arquivo temporário + ``os.replace``; nunca deixa o arquivo
        alvo em estado parcial.

        Raises:
            FileNotFoundError: se não houver backup para o arquivo.
        """
        path = Path(path).resolve()
        record = record or self.latest(path)
        if record is None:
            raise FileNotFoundError(f"Nenhum backup encontrado para {path}")
        source = self.backup_dir / record.backup_path

        def _do_restore() -> None:
            if not record.existed:
                # O arquivo não existia antes da alteração: rollback = remover.
                path.unlink(missing_ok=True)
                return
            if not source.is_file():
                raise FileNotFoundError(f"Arquivo de backup ausente: {source}")
            tmp = path.with_suffix(path.suffix + ".restoring")
            shutil.copy2(source, tmp)
            tmp.replace(path)

        async with self._lock:
            await asyncio.to_thread(_do_restore)
            # Marca o backup como consumido para que o próximo rollback recue
            # mais um passo no histórico.
            stamp, _ = self._now_stamp()
            entries = self._load_manifest()
            for entry in entries:
                if entry["backup_path"] == record.backup_path:
                    entry["restored_at"] = stamp
            self._save_manifest(entries)
        return BackupRecord(**{**asdict(record), "restored_at": stamp})

    # ------------------------------------------------------------ checkpoint
    async def checkpoint(self, paths: list[Path], label: str = "") -> Checkpoint:
        """Faz backup de vários arquivos de uma vez e agrupa num ``Checkpoint``."""
        iso, stamp = self._now_stamp()
        records = [await self.backup(p, reason=f"checkpoint:{label or stamp}") for p in paths]
        return Checkpoint(id=stamp, label=label, timestamp=iso, records=records)

    async def restore(self, checkpoint: Checkpoint) -> list[BackupRecord]:
        """Restaura todos os arquivos de um checkpoint (LIFO)."""
        return await self.rollback_many(checkpoint.records)

    async def rollback_many(self, records: list[BackupRecord]) -> list[BackupRecord]:
        """Restaura vários arquivos em ordem inversa de criação (LIFO).

        Usado pelo loop do agente para desfazer um ciclo inteiro de patches.
        """
        restored: list[BackupRecord] = []
        for rec in sorted(records, key=lambda r: r.timestamp, reverse=True):
            await self.rollback(self.root / rec.original, rec)
            restored.append(rec)
        return restored
