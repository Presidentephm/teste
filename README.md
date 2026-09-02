# agent_core — agente autônomo multimodal auto-modificável

Núcleo em Python (3.10+) de um agente que **observa** (código, execução, testes,
logs, visão), **entende e diagnostica**, **planeja**, **modifica o próprio
código**, **executa e testa**, **valida com nova observação** e **mantém ou
reverte** a alteração, guardando memória do que já tentou.

```
OBSERVE → ENTENDA → RACIOCINE → PLANEJE → MODIFIQUE → EXECUTE → TESTE → OBSERVE DE NOVO → VALIDE → MELHORA ou ROLLBACK → CONTINUA
```

## Arquitetura

```
                          ┌──────────────────────────────────────────────┐
                          │              SelfImprovementAgent            │
                          │              (agent_core/agent_loop.py)      │
                          │                                              │
   ┌───────────────┐      │  OBSERVE ─▶ CONTEXTO ─▶ DECIDE ─▶ AGE ─▶ VALIDA │
   │  Observers    │─────▶│     ▲                                  │      │
   │  runtime      │      │     └────────── memória / checkpoint ◀─┘      │
   │  tests        │      └───────┬───────────────┬──────────────┬────────┘
   │  logs         │              │               │              │
   │  code (AST)   │              ▼               ▼              ▼
   │  vision ──────┼──▶ MultimodalContext   FixStrategy      CodeManager + Sandbox
   └───────┬───────┘   (observations.py)   (strategies.py)   BackupManager (checkpoints)
           │                                    │                PatchGuard (safety.py)
   ┌───────┴────────────────┐                   │
   │ vision/                │             AutoStrategy
   │  VisualSource          │      analyzers: traceback, tests, logs,
   │   ├ CameraSource (cv2) │                 vision, memory  → Diagnosis
   │   ├ ScreenSource (mss) │      planners : observe, rollback,
   │   └ ImageSource        │                 heuristic, model → Decision
   │  Frame → Preprocess →  │                       │
   │  ChangeDetector →      │                       ▼
   │  VisualAnalyzer (OCR   │               ModelProvider (providers.py)
   │  opcional) → Observation│               ├ AnthropicProvider ─▶ SDK anthropic ─▶ modelo
   │  VisionCapture (async) │               ├ FallbackProvider (retry + cadeia)
   └────────────────────────┘               └ FakeProvider (testes/offline)
```

| Módulo | Classe principal | Papel |
| --- | --- | --- |
| `agent_core/config.py` | `AgentConfig` | Todos os parâmetros: limites do ciclo, sandbox, modelo, visão, memória, segurança. |
| `agent_core/safety.py` | `redact`, `PatchGuard` | Redação de credenciais em logs/memória/prompts; rejeição de patches destrutivos. |
| `agent_core/backup.py` | `BackupManager`, `Checkpoint` | Backups `.bak` com timestamp e manifest; checkpoints de vários arquivos; rollback encadeável. |
| `agent_core/code_manager.py` | `CodeManager` | Leitura, análise AST e escrita atômica de `.py`, confinada à raiz, com validação sintática. |
| `agent_core/sandbox.py` | `Sandbox` | Execução isolada (cópia temporária, env mínimo, limites, timeout) com traceback parseado. |
| `agent_core/providers.py` | `ModelProvider` | Camada desacoplada do SDK: tipos próprios, erros normalizados, retry/fallback, fake. |
| `agent_core/observations.py` | `Observation`, `MultimodalContext`, `Observer` | Contrato CAPTURE → PROCESS → CONTEXT → AGENT; observers de runtime, testes, logs e código. |
| `agent_core/vision/` | `VisualSource`, `VisualPipeline`, `VisionCapture` | Os "olhos": fontes, pipeline OpenCV, captura contínua desacoplada. |
| `agent_core/memory.py` | `AgentMemory` | Registro por ciclo (observação, diagnóstico, estratégia, ação, patch, testes, resultado, rollback, erros). |
| `agent_core/strategies.py` | `AutoStrategy`, `ModelFixStrategy`, `HeuristicFixStrategy` | Diagnóstico por evidências e decisão dinâmica (`patch`, `observe_again`, `run_tests`, `rollback`, `finish`). |
| `agent_core/agent_loop.py` | `SelfImprovementAgent` | O ciclo autônomo com checkpoints, validação, memória e limites. |
| `agent_core/cli.py` | — | `python -m agent_core run|analyze|backups|rollback|observe|ask|memory`. |

### O ciclo em detalhe

1. **Observar**: roda o script no sandbox, executa os testes (se `--tests`), lê logs, captura uma observação visual (se `--vision`). Tudo vira `Observation` num `MultimodalContext` com limites de quantidade, imagens e texto.
2. **Contexto**: localiza o arquivo do projeto onde a exceção estourou, lê fonte e AST, anexa memória e contexto ao `FailureContext`.
3. **Decidir**: `AutoStrategy` roda os analisadores (traceback, testes, logs, visão, memória) → `Diagnosis` (causa provável + necessidades) → planejadores em ordem: observar de novo (evidência insuficiente), rollback (falhas acumuladas), heurística (imports/indentação), modelo (com texto **e imagens**). Patches já fracassados na memória são recusados.
4. **Agir**: `PatchGuard` valida, `Checkpoint` salva os arquivos, `CodeManager` aplica (validação sintática, escrita atômica).
5. **Validar**: reexecuta script + testes + nova observação. `fixed` encerra; `new_error` mantém (progresso); erro igual, regressão crítica (SyntaxError/ImportError/timeout) ou testes que passavam e quebraram → **rollback automático** do checkpoint.
6. **Memória**: cada ciclo grava um `MemoryEntry`; limite configurável e persistência em `.agent_backups/memory.json`.
7. **Limites**: iterações, retries (patches fracassados), timeout total, estagnação (mesmo erro repetido), interrupção (Ctrl+C) com relatório.

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

| Dependência | Uso | Obrigatória? |
| --- | --- | --- |
| `anthropic>=1.3` | `AnthropicProvider` (SDK oficial). Testado com 1.3.0. | Só para usar o modelo |
| `opencv-python-headless`, `numpy` | Subsistema de visão | Só com `--vision` |
| `mss` | `ScreenSource` (captura de tela) | Opcional |
| `pytesseract` + binário `tesseract` (`apt install tesseract-ocr`) | OCR real no pipeline visual; a estratégia lê mensagens de erro na tela | Opcional; sem eles `text=None`, `ocr="unavailable"`; `vision_ocr=False` desliga |

O núcleo (backup, AST, sandbox, heurística, memória, contexto) usa só a biblioteca padrão.

## SDK, credenciais e modelos

- Somente `agent_core/providers.py` importa `anthropic`. Estratégias e loop usam `ModelProvider`, `ModelRequest`, `ModelResponse` e `ProviderError`.
- **Credenciais nunca vão no código.** O SDK lê `ANTHROPIC_API_KEY` (ou `ANTHROPIC_AUTH_TOKEN`, ou um perfil de `ant auth login`). Sem credenciais o provider levanta `ProviderAuthError` e a estratégia segue com heurística.
- Modelo padrão: `claude-opus-5` (presente na tipagem do SDK 1.3.0). Qualquer ID é aceito via `--model`/`AgentConfig.llm_model`; IDs inexistentes na conta retornam `ProviderRequestError` com a dica de ajustar o modelo. O ID não é validado localmente.
- Chamada: `messages.stream` + `get_final_message()`, thinking adaptativo, `output_config.effort` (`--effort low|medium|high|xhigh|max`).
- **Fallback configurável** (`--fallback`, padrão ligado; `--no-fallback` desliga):
  - server-side: `betas=["server-side-fallback-2026-07-01"], fallbacks="default"` reexecuta em modelo substituto se o pedido for recusado por política;
  - client-side: `FallbackProvider` faz retry com backoff em rate limit/timeout/5xx e passa a `--fallback-model` alternativos.
- Erros tratados e normalizados: autenticação, pedido inválido/modelo inexistente, rate limit (`retry-after`), timeout, indisponibilidade/conexão, recusa, resposta vazia/inválida, interrupção (cancelamento).
- Logs, memória e prompts passam por `redact()` (chaves `sk-ant-…`, tokens Bearer, `api_key=`…).
- **Ferramentas** (`--no-tools` desliga): por padrão o modelo recebe um prompt compacto e ferramentas de leitura (`read_file`, `list_files`, `search`, `outline`) e entrega a correção via `propose_patch`, num loop manual de `tool_use` limitado por `--max-tool-rounds`. Sem ferramentas, o esqueleto do projeto vai no prompt e a resposta usa saída estruturada (`output_config.format` com o JSON Schema do patch).
- **Esforço por tipo de erro** (`AgentConfig.effort_by_error`): `NameError`/`ImportError`/indentação → `low`; `SyntaxError`/`TypeError`/`AttributeError`/`KeyError` → `medium`; timeout, testes e o restante → `high`. Cada pedido ao modelo leva o esforço da falha atual.
- **Cache de prompt** (`--no-cache` desliga): o prefixo estável (ferramentas + sistema) recebe `cache_control`; o conteúdo variável fica depois do breakpoint.
- **Uso e custo**: cada provider acumula chamadas, tokens (entrada, saída, cache) e custo estimado pela tabela de preços por modelo; o relatório do `run` e o `bench` mostram o delta.

## Outros provedores (endpoint compatível com a API Messages)

O `ModelProvider` existe justamente para trocar o backend sem tocar nas estratégias nem no loop. Provedores que expõem um endpoint no formato Messages funcionam reaproveitando todo o código — inclusive loop de ferramentas, imagens, normalização de erros e contabilidade de uso.

```bash
cp .env.example .env      # preencha a chave; o .env é ignorado pelo Git
python -m agent_core ask "diga olá" --provider kimi
python -m agent_core run app.py --strategy auto --provider kimi --model kimi-k2.5
python -m agent_core bench --provider kimi --model kimi-k2.5
```

| Preset | Endpoint | Credencial |
| --- | --- | --- |
| `anthropic` (padrão) | `api.anthropic.com` | `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` / perfil `ant auth login` |
| `kimi` | `https://api.moonshot.ai/anthropic` | `MOONSHOT_API_KEY` |
| `kimi-cn` | `https://api.moonshot.cn/anthropic` | `MOONSHOT_API_KEY` |
| `compat` | o que você passar em `--base-url` | `--api-key-env` (padrão `LLM_API_KEY`) |

As credenciais vêm só do ambiente. A CLI carrega automaticamente o `.env` da raiz do projeto (ou o caminho em `--env-file`), sem sobrescrever variáveis já exportadas e registrando apenas os **nomes** das variáveis, nunca os valores. Nunca commite o `.env`; se uma chave for exposta, revogue-a e gere outra.

Confirme o ID do modelo no console do provedor e passe com `--model`; o padrão de cada preset é só um ponto de partida. Presets marcados como `compat` enviam apenas o subconjunto universal da API Messages, então **não** valem nesses endpoints: thinking adaptativo, `--effort` e o mapa `effort_by_error`, saída estruturada por JSON Schema (a resposta volta como texto e é lida pelo parser JSON), `cache_control` e o fallback server-side por recusa. O que continua valendo: ferramentas, imagens, retries com backoff, cadeia de `--fallback-model`, memória, checkpoints e rollback.

## Visão

```
VisualSource ──▶ Frame ──▶ FramePreprocessor (resize/gray) ──▶ ChangeDetector (diff, regiões)
             ──▶ VisualAnalyzer (resolução, brilho, contraste, bordas, elementos, cores, OCR opcional)
             ──▶ Observation(kind=vision, image=JPEG, extracted={...}, confidence)
```

- Fontes: `CameraSource(index)` (`cv2.VideoCapture`), `ScreenSource(monitor)` (`mss`), `ImageSource(paths|arrays, loop)`.
- `VisionCapture` roda em background com `fps` e `observation_interval`, emite observação quando há mudança ou o intervalo expira, grava frames relevantes (`--store-frames`) e encerra limpo.
- Dispositivo ausente **não derruba o loop**: `start()` devolve `False`, o erro fica em `status()` e o agente prossegue sem visão (`vision.unavailable` no log e no relatório).

```bash
python -m agent_core observe --vision-source image --image tela.png --json
python -m agent_core observe --vision-source camera --camera-index 0 --save frame.jpg
```

## CLI

```bash
# ciclo autônomo completo
python -m agent_core run examples/broken_script.py \
    --strategy auto --model claude-opus-5 --vision --max-iterations 10

# offline (heurística), com testes como critério de sucesso
python -m agent_core run app.py --strategy heuristic --tests "-m unittest discover -s tests"

# visão a partir de imagens (útil sem câmera), sem fallback, effort alto
python -m agent_core run app.py --strategy auto --vision --vision-source image \
    --image tela1.png --image tela2.png --observation-interval 2 --no-fallback --effort xhigh
```

| Opção (`run`) | Padrão | Efeito |
| --- | --- | --- |
| `--strategy auto\|heuristic\|claude` | `auto` | Evidências + heurística + modelo / só heurística / só modelo. |
| `--model`, `--effort`, `--max-tokens` | `claude-opus-5`, `high`, 16000 | Modelo e esforço. |
| `--fallback` / `--no-fallback`, `--fallback-model M` | ligado | Retry + fallback server/client-side. |
| `--vision` / `--no-vision`, `--vision-source`, `--camera-index`, `--monitor`, `--image`, `--fps`, `--observation-interval`, `--store-frames` | desligado | Observação visual. |
| `--max-iterations`, `--max-retries`, `--timeout`, `--total-timeout` | 5, 3, 30 s, ∞ | Limites do ciclo. |
| `--tests "…"`, `--tests-in-place` | — | Comando de testes (após `python`) que também precisa passar; por padrão roda na cópia isolada do projeto. |
| `--no-tools`, `--max-tool-rounds`, `--no-cache` | ferramentas ligadas, 8, cache ligado | Modo de consulta ao modelo. |
| `--provider`, `--base-url`, `--api-key-env` | `anthropic` | Endpoint compatível com a API Messages (ver acima). |
| `--memory-limit`, `--reset-memory` | 100 | Memória do agente. |
| `--no-self-modify` | — | Proíbe alterar `agent_core/`. |
| `--json` | — | Relatório em JSON. |

Outros comandos: `analyze <arquivo>`, `backups [arquivo]`, `rollback <arquivo>`, `observe`, `ask "<prompt>" [--image f] [--fake]`, `memory [--clear]`, `bench`.

## Benchmark

```bash
python -m agent_core bench --offline                 # FakeProvider com respostas canônicas (sem custo)
python -m agent_core bench --strategy heuristic      # só os casos que a heurística resolve
ANTHROPIC_API_KEY=... python -m agent_core bench --model claude-opus-5 --effort medium --json
```

Sete casos (`agent_core/bench.py`): import da stdlib, símbolo de módulo irmão, typo, tabs, divisão por zero, `TypeError` e teste falhando. O relatório traz status, iterações, tempo, tokens, custo total e custo por correção. Para medir o modelo real, rode com credenciais e compare `--effort` e `--no-tools`; o `bench` compartilha um único provider entre os casos, então o uso acumula.

## Uso programático

```python
import asyncio
from agent_core import AgentConfig, AutoStrategy, SelfImprovementAgent, build_provider

config = AgentConfig(project_root=".", max_iterations=8, vision_enabled=True, vision_source="screen",
                     test_command=("-m", "unittest", "discover", "-s", "tests"))
agent = SelfImprovementAgent(config, AutoStrategy(build_provider(config)))
report = asyncio.run(agent.run("app.py"))
print(report.summary())            # iterações, decisões, resultado, visão, contexto
print(agent.memory.to_prompt_text())
# asyncio.run(agent.rollback_run(report))  # desfaz o que sobreviveu
```

Extensão: `AutoStrategy(provider, analyzers=[...], planners=[...])` aceita qualquer objeto com `analyze(ctx) -> list[Finding]` ou `async plan(ctx, diagnosis) -> Decision | None`; observers novos implementam `Observer.observe()`; providers novos implementam `ModelProvider.complete()`.

## Segurança e rollback

- Nenhuma escrita sem backup/checkpoint; escrita atômica; código que não compila nunca chega ao disco.
- Confinamento à raiz do projeto; `.git/` e `.agent_backups/` são intocáveis; auto-modificação pode ser desligada.
- `PatchGuard`: no máximo N arquivos por decisão, sem esvaziar arquivos, sem remover mais de 60 % das linhas.
- Ferramentas do modelo são somente leitura e confinadas à raiz; a escrita continua passando por checkpoint, guarda e validação.
- Rollback automático em regressão crítica, patch inócuo ou testes que regrediram; ação `rollback` explícita; `rollback_run(report)`; CLI `rollback` caminha no histórico.
- Loop infinito: iterações, retries, timeout total, estagnação, recusa de patch repetido.
- O agente atua apenas dentro do workspace; não contorna permissões do sistema, autenticação ou controles externos.

## Testes

```bash
python -m unittest discover -s tests -v
```

Suítes: `test_agent_core.py` (23 originais), `test_providers.py`, `test_vision.py`, `test_multimodal.py`, `test_auto_strategy.py`, `test_loop_v2.py`, `test_memory_safety.py`, `test_tools_and_bench.py`. Todas determinísticas (fakes/mocks); chamadas reais ao modelo ficam fora da suíte (`examples/provider_call.py`, `ask`, `bench` sem `--offline`). O workflow `.github/workflows/tests.yml` roda a suíte e o benchmark offline em Python 3.10–3.12.

## Exemplos (`examples/`)

| Arquivo | Mostra |
| --- | --- |
| `provider_call.py` | Chamada ao provider (SDK real com credenciais, fake sem). |
| `vision_capture.py` | Captura contínua: câmera → tela → imagens sintéticas. |
| `frame_processing.py` | Pipeline de frame e observação estruturada. |
| `multimodal_context.py` | Contexto com código + log + runtime + visão, serialização, partes para o modelo. |
| `agent_loop_demo.py` | AgentLoop programático offline. |
| `auto_strategy_demo.py` | Strategy Auto corrigindo um `ZeroDivisionError` (fake ou `--real`). |
| `run_with_vision.py` | Ciclo completo com observações visuais. |
| `tool_strategy_demo.py` | Modelo lendo o projeto por ferramentas (`search` → `read_file` → `propose_patch`). |

## Limitações

- O sandbox isola efeitos colaterais e travamentos; não é barreira contra código hostil (use contêiner).
- OCR só existe com Tesseract instalado; sem ele a análise visual é estatística/estrutural.
- Captura de tela exige display (`mss`); em servidores headless a fonte falha de forma limpa.
- A heurística cobre imports ausentes, typos de nome e indentação; o restante depende do modelo.
- O custo é estimado por uma tabela de preços embutida (`MODEL_PRICES`); modelos fora dela aparecem como não precificados, e os valores de terceiros precisam ser conferidos no console do provedor.
- Os presets compatíveis foram implementados a partir da documentação pública do provedor e cobertos por testes com cliente falso; a validação contra o endpoint real depende de uma credencial.
- A validação com o modelo real depende de credenciais no ambiente; o harness (`bench`) está pronto, mas os números reais precisam ser medidos pelo usuário.
