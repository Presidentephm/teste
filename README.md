# agent_core — núcleo do agente autônomo auto-modificável

Fundação em Python (3.10+) de um agente capaz de **ler, analisar, reescrever e
testar o próprio código**, com backup automático e rollback a cada alteração.
O núcleo usa apenas a biblioteca padrão; o SDK `anthropic` é opcional e só é
carregado pela estratégia de correção via Claude.

## Arquitetura

| Módulo | Classe principal | Papel |
| --- | --- | --- |
| `agent_core/config.py` | `AgentConfig` | Parâmetros globais (raiz do projeto, limites do sandbox, modelo). |
| `agent_core/backup.py` | `BackupManager` | Cópias `.bak` com timestamp + manifest JSON; `rollback()` instantâneo e encadeável. |
| `agent_core/code_manager.py` | `CodeManager` | Leitura/análise (AST) e escrita atômica de `.py`, confinada à raiz do projeto, sempre com backup e validação sintática. |
| `agent_core/sandbox.py` | `Sandbox` | Executa scripts em subprocesso isolado (cópia temporária do projeto, env mínimo, limites de memória/CPU, timeout) e devolve stdout, stderr e traceback parseado. |
| `agent_core/strategies.py` | `FixStrategy` e implementações | Quem formula a correção: `HeuristicFixStrategy` (offline), `ClaudeFixStrategy` (SDK oficial), `CompositeFixStrategy` (encadeia). |
| `agent_core/agent_loop.py` | `SelfImprovementAgent` | O loop executar → falhar → propor → aplicar → reexecutar, com rollback automático em regressões. |
| `agent_core/cli.py` | — | `python -m agent_core run|analyze|backups|rollback`. |

Fluxo de uma iteração:

```
sandbox.run_script ──falha──▶ FailureContext (traceback + fonte + AST + histórico)
        ▲                            │
        │                            ▼
        │                    strategy.propose ──None──▶ status "no_fix"
        │                            │
        │                            ▼
        │              code.apply_patches (backup de cada arquivo)
        │                            │
        └──── reexecuta ◀────────────┘
                 ├─ sucesso .............. "fixed"
                 ├─ SyntaxError/Import/timeout  rollback (regressão crítica)
                 ├─ mesmo erro ........... rollback (patch inócuo)
                 └─ erro diferente ....... mantém e continua (progresso)
```

## Uso rápido

```bash
# demonstração offline: dois NameError corrigidos pela heurística
python -m agent_core run examples/broken_script.py --strategy heuristic

# com o modelo (requer `pip install anthropic` e ANTHROPIC_API_KEY ou `ant auth login`)
python -m agent_core run examples/broken_script.py --strategy auto --model claude-opus-5

python -m agent_core analyze agent_core/agent_loop.py   # esqueleto via AST
python -m agent_core backups                             # lista os .bak
python -m agent_core rollback examples/broken_script.py  # desfaz a última alteração
```

Uso programático:

```python
import asyncio
from agent_core import AgentConfig, SelfImprovementAgent, CompositeFixStrategy
from agent_core import HeuristicFixStrategy, ClaudeFixStrategy

config = AgentConfig(project_root=".", max_iterations=5)
strategy = CompositeFixStrategy([HeuristicFixStrategy(), ClaudeFixStrategy(config)])
agent = SelfImprovementAgent(config, strategy)

report = asyncio.run(agent.run("examples/broken_script.py"))
print(report.summary())
# asyncio.run(agent.rollback_run(report))  # desfaz tudo o que o run alterou
```

## Redes de segurança

- Nenhum arquivo é escrito sem backup prévio; a escrita é atômica (`tmp` + `os.replace`).
- Código proposto que não compila é rejeitado antes de tocar o disco.
- Caminhos fora de `project_root`, `.git/` e a pasta de backups são inacessíveis.
- `allow_self_modification=False` proíbe alterar o próprio pacote `agent_core`.
- O sandbox roda numa cópia temporária: efeitos colaterais do código em teste
  nunca atingem o projeto. Não é uma barreira contra código hostil; para isso,
  execute o núcleo dentro de um contêiner.
- O loop detecta estagnação (mesmo erro repetido) e desiste em vez de girar em vazio.

## Testes

```bash
python -m unittest discover -s tests -v
```

## Extensão

- Nova estratégia: subclasse de `FixStrategy` com `async propose(ctx) -> FixProposal | None`.
- Entradas multimodais (screenshots, logs externos): anexe em `FailureContext.attachments`
  e consuma-as na sua estratégia.
- Outro critério de sucesso (ex.: suíte de testes em vez do script): use
  `Sandbox.run_tests()` no lugar de `run_script()` numa subclasse do agente.
