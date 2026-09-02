"""
Gerenciador de arquivos e código (Self-Modifying Core).

Responsabilidades:

* **Confinamento**: só toca em arquivos dentro de ``project_root`` e fora dos
  caminhos protegidos. Qualquer tentativa de escapar (``../``, symlink, caminho
  absoluto externo) levanta ``PathOutsideProjectError``.
* **Leitura assíncrona** de arquivos ``.py``.
* **Análise estrutural** via ``ast``: imports, funções, classes, nomes
  definidos e erros de sintaxe, tudo num ``ModuleAnalysis`` serializável que
  as estratégias de correção usam como contexto.
* **Escrita segura**: backup automático -> validação sintática -> escrita
  atômica (arquivo temporário + ``os.replace``). Se a validação falhar nada é
  escrito e o backup registra a intenção.
* **Aplicação de patches** (``FilePatch``): substituição completa do arquivo
  ou pares busca/substituição, sempre passando pelo caminho seguro acima.
"""

from __future__ import annotations

import ast
import asyncio
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from .backup import BackupManager, BackupRecord
from .config import AgentConfig


class PathOutsideProjectError(PermissionError):
    """Tentativa de acessar um caminho fora da raiz ou protegido."""


class InvalidSourceError(ValueError):
    """O código proposto não compila; a escrita foi abortada."""


# --------------------------------------------------------------------- modelos
@dataclass
class FunctionInfo:
    name: str
    lineno: int
    end_lineno: int
    args: list[str]
    is_async: bool
    docstring: str | None
    decorators: list[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    name: str
    lineno: int
    end_lineno: int
    bases: list[str]
    methods: list[FunctionInfo]
    docstring: str | None


@dataclass
class ImportInfo:
    module: str          # "os.path" ou "json"
    names: list[str]     # [] para "import x"; ["a","b"] para "from x import a, b"
    alias: str | None    # "np" em "import numpy as np"
    lineno: int


@dataclass
class SyntaxIssue:
    message: str
    lineno: int | None
    offset: int | None
    text: str | None


@dataclass
class ModuleAnalysis:
    """Retrato estrutural de um módulo Python."""

    path: str
    line_count: int
    docstring: str | None
    imports: list[ImportInfo]
    functions: list[FunctionInfo]
    classes: list[ClassInfo]
    defined_names: set[str]
    syntax_issue: SyntaxIssue | None = None

    @property
    def is_valid(self) -> bool:
        return self.syntax_issue is None

    @property
    def imported_names(self) -> set[str]:
        """Nomes que ficam disponíveis no namespace do módulo via import."""
        names: set[str] = set()
        for imp in self.imports:
            if imp.names:
                names.update(imp.names)
            else:
                names.add(imp.alias or imp.module.split(".")[0])
        return names

    def to_dict(self) -> dict:
        data = asdict(self)
        data["defined_names"] = sorted(self.defined_names)
        return data

    def outline(self) -> str:
        """Resumo textual compacto, ideal para injetar em prompts de LLM."""
        lines = [f"# {self.path} ({self.line_count} linhas)"]
        if self.syntax_issue:
            si = self.syntax_issue
            lines.append(f"!! SyntaxError linha {si.lineno}: {si.message}")
        for imp in self.imports:
            if imp.names:
                lines.append(f"from {imp.module} import {', '.join(imp.names)}")
            else:
                suffix = f" as {imp.alias}" if imp.alias else ""
                lines.append(f"import {imp.module}{suffix}")
        for fn in self.functions:
            kind = "async def" if fn.is_async else "def"
            lines.append(f"{kind} {fn.name}({', '.join(fn.args)})  # L{fn.lineno}-{fn.end_lineno}")
        for cls in self.classes:
            bases = f"({', '.join(cls.bases)})" if cls.bases else ""
            lines.append(f"class {cls.name}{bases}:  # L{cls.lineno}-{cls.end_lineno}")
            for m in cls.methods:
                lines.append(f"    def {m.name}({', '.join(m.args)})  # L{m.lineno}")
        return "\n".join(lines)


@dataclass
class Replacement:
    """Um par busca/substituição literal (não regex)."""

    search: str
    replace: str
    count: int = 1  # quantas ocorrências substituir (0 = todas)


@dataclass
class FilePatch:
    """Alteração proposta para um arquivo.

    Exatamente um dos modos deve ser usado:

    * ``content`` preenchido -> substitui o arquivo inteiro.
    * ``replacements`` preenchido -> aplica pares busca/substituição.
    """

    path: str
    content: str | None = None
    replacements: list[Replacement] = field(default_factory=list)
    reason: str = ""

    @property
    def mode(self) -> str:
        return "replace_full" if self.content is not None else "search_replace"


# ----------------------------------------------------------------- visitante
class _Analyzer(ast.NodeVisitor):
    """Percorre a AST coletando estrutura de alto nível (sem entrar em corpos)."""

    def __init__(self) -> None:
        self.imports: list[ImportInfo] = []
        self.functions: list[FunctionInfo] = []
        self.classes: list[ClassInfo] = []
        self.defined: set[str] = set()

    # -- helpers
    @staticmethod
    def _args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        a = node.args
        names = [x.arg for x in a.posonlyargs] + [x.arg for x in a.args]
        if a.vararg:
            names.append("*" + a.vararg.arg)
        names += [x.arg for x in a.kwonlyargs]
        if a.kwarg:
            names.append("**" + a.kwarg.arg)
        return names

    @staticmethod
    def _decorators(node: ast.AST) -> list[str]:
        return [ast.unparse(d) for d in getattr(node, "decorator_list", [])]

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionInfo:
        return FunctionInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            args=self._args(node),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            docstring=ast.get_docstring(node),
            decorators=self._decorators(node),
        )

    # -- visitors (apenas nível de módulo; não recursa em corpos de função)
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(ImportInfo(alias.name, [], alias.asname, node.lineno))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = ("." * node.level) + (node.module or "")
        self.imports.append(
            ImportInfo(module, [a.asname or a.name for a in node.names], None, node.lineno)
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(self._function(node))
        self.defined.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        methods = [
            self._function(n)
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        self.classes.append(
            ClassInfo(
                name=node.name,
                lineno=node.lineno,
                end_lineno=node.end_lineno or node.lineno,
                bases=[ast.unparse(b) for b in node.bases],
                methods=methods,
                docstring=ast.get_docstring(node),
            )
        )
        self.defined.add(node.name)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            for name_node in ast.walk(target):
                if isinstance(name_node, ast.Name):
                    self.defined.add(name_node.id)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            self.defined.add(node.target.id)


# --------------------------------------------------------------- gerenciador
class CodeManager:
    """Ponto único de acesso ao sistema de arquivos do projeto."""

    def __init__(self, config: AgentConfig, backups: BackupManager | None = None) -> None:
        self.config = config
        self.root = config.project_root
        self.backups = backups or BackupManager(config)
        self._core_dir = Path(__file__).resolve().parent

    # ------------------------------------------------------------ confinamento
    def resolve(self, path: str | Path, *, for_write: bool = False) -> Path:
        """Converte ``path`` (relativo ou absoluto) num Path absoluto confinado.

        Raises:
            PathOutsideProjectError: fora da raiz, em pasta protegida, ou
                dentro do próprio núcleo com ``allow_self_modification=False``.
        """
        p = Path(path)
        candidate = (p if p.is_absolute() else self.root / p).resolve()
        try:
            rel = candidate.relative_to(self.root)
        except ValueError as exc:
            raise PathOutsideProjectError(f"{candidate} está fora de {self.root}") from exc

        rel_posix = rel.as_posix()
        for protected in self.config.all_protected_paths:
            if rel_posix == protected or rel_posix.startswith(protected.rstrip("/") + "/"):
                raise PathOutsideProjectError(f"{rel_posix} está em caminho protegido ({protected})")

        if for_write and not self.config.allow_self_modification:
            try:
                candidate.relative_to(self._core_dir)
            except ValueError:
                pass
            else:
                raise PathOutsideProjectError(
                    f"Auto-modificação desativada: não é permitido escrever em {rel_posix}"
                )
        return candidate

    def relative(self, path: str | Path) -> str:
        return self.resolve(path).relative_to(self.root).as_posix()

    # -------------------------------------------------------------- leitura
    async def read(self, path: str | Path) -> str:
        """Lê o conteúdo de um arquivo de texto do projeto."""
        target = self.resolve(path)
        return await asyncio.to_thread(target.read_text, "utf-8")

    async def list_python_files(self) -> list[Path]:
        """Todos os ``.py`` do projeto, ignorando caminhos protegidos e caches."""

        def _walk() -> list[Path]:
            found: list[Path] = []
            for p in self.root.rglob("*.py"):
                rel = p.relative_to(self.root).as_posix()
                if "__pycache__" in rel:
                    continue
                if any(
                    rel == prot or rel.startswith(prot.rstrip("/") + "/")
                    for prot in self.config.all_protected_paths
                ):
                    continue
                found.append(p)
            return sorted(found)

        return await asyncio.to_thread(_walk)

    # -------------------------------------------------------------- análise
    @staticmethod
    def analyze_source(source: str, path: str = "<memory>") -> ModuleAnalysis:
        """Analisa código-fonte em memória (síncrono, puro)."""
        line_count = source.count("\n") + (0 if source.endswith("\n") or not source else 1)
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError as exc:
            return ModuleAnalysis(
                path=path,
                line_count=line_count,
                docstring=None,
                imports=[],
                functions=[],
                classes=[],
                defined_names=set(),
                syntax_issue=SyntaxIssue(exc.msg, exc.lineno, exc.offset, exc.text),
            )
        visitor = _Analyzer()
        # Visita só os nós de nível de módulo: queremos o "esqueleto", não cada
        # atribuição dentro de funções.
        for node in tree.body:
            visitor.visit(node)
        return ModuleAnalysis(
            path=path,
            line_count=line_count,
            docstring=ast.get_docstring(tree),
            imports=visitor.imports,
            functions=visitor.functions,
            classes=visitor.classes,
            defined_names=visitor.defined,
        )

    async def analyze(self, path: str | Path) -> ModuleAnalysis:
        """Lê e analisa um arquivo do projeto."""
        target = self.resolve(path)
        source = await self.read(target)
        return self.analyze_source(source, self.relative(target))

    async def analyze_project(self) -> dict[str, ModuleAnalysis]:
        """Analisa todos os ``.py`` do projeto em paralelo."""
        files = await self.list_python_files()
        results = await asyncio.gather(*(self.analyze(f) for f in files))
        return {a.path: a for a in results}

    # -------------------------------------------------------------- escrita
    @staticmethod
    def validate_syntax(source: str, path: str = "<proposed>") -> None:
        """Levanta ``InvalidSourceError`` se o código não compilar."""
        try:
            compile(source, path, "exec")
        except SyntaxError as exc:
            raise InvalidSourceError(
                f"{path}:{exc.lineno}:{exc.offset}: {exc.msg}"
            ) from exc

    async def write(
        self,
        path: str | Path,
        content: str,
        *,
        reason: str = "",
        validate: bool = True,
        backup: bool = True,
    ) -> BackupRecord | None:
        """Escreve ``content`` em ``path`` com backup prévio e escrita atômica.

        Com ``backup=False`` (usado quando um checkpoint já cobriu o arquivo)
        devolve ``None``.

        Ordem das operações (deliberada):
            1. confinamento;
            2. validação sintática (para .py) -> nada é tocado se falhar;
            3. backup do estado atual;
            4. escrita em arquivo temporário + ``os.replace``.

        Returns:
            O ``BackupRecord`` criado, para permitir rollback pontual.
        """
        target = self.resolve(path, for_write=True)
        if validate and target.suffix == ".py":
            self.validate_syntax(content, self.relative(target))

        record = await self.backups.backup(target, reason=reason) if backup else None

        def _atomic_write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            tmp.write_text(content, encoding="utf-8")
            os.replace(tmp, target)

        await asyncio.to_thread(_atomic_write)
        return record

    async def apply_patch(self, patch: FilePatch, *, backup: bool = True) -> BackupRecord | None:
        """Aplica um ``FilePatch`` (substituição total ou busca/substituição).

        Raises:
            ValueError: se um trecho de busca não for encontrado.
            InvalidSourceError: se o resultado não compilar.
        """
        target = self.resolve(patch.path, for_write=True)
        if patch.content is not None:
            new_source = patch.content
        else:
            new_source = await self.read(target) if target.exists() else ""
            for rep in patch.replacements:
                if rep.search not in new_source:
                    raise ValueError(
                        f"Trecho de busca não encontrado em {patch.path}: {rep.search[:80]!r}"
                    )
                count = rep.count if rep.count > 0 else -1
                new_source = new_source.replace(rep.search, rep.replace, count)
        return await self.write(target, new_source, reason=patch.reason or f"patch:{patch.mode}", backup=backup)

    async def apply_patches(self, patches: Iterable[FilePatch], *, backup: bool = True) -> list[BackupRecord]:
        """Aplica vários patches em sequência. Se um falhar, reverte os anteriores.

        Com ``backup=False`` o chamador é responsável por ter feito um
        checkpoint antes (e por restaurá-lo se algo falhar).
        """
        applied: list[BackupRecord] = []
        for patch in patches:
            try:
                record = await self.apply_patch(patch, backup=backup)
            except Exception:
                await self.backups.rollback_many(applied)
                raise
            if record is not None:
                applied.append(record)
        return applied

    async def current_sources(self, paths: Iterable[str]) -> dict[str, str]:
        """Conteúdo atual dos arquivos existentes (para validação de patches)."""
        sources: dict[str, str] = {}
        for path in paths:
            try:
                target = self.resolve(path)
            except PathOutsideProjectError:
                continue
            if target.is_file():
                sources[path] = await self.read(target)
        return sources

    async def rollback(self, path: str | Path, record: BackupRecord | None = None) -> BackupRecord:
        """Atalho para ``BackupManager.rollback`` com confinamento."""
        return await self.backups.rollback(self.resolve(path, for_write=True), record)
