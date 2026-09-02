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
| `pytesseract` + binário `tesseract` | OCR real no pipeline visual | Opcional; sem eles `text=None`, `ocr="unavailable"` |

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
| `--tests "…"` | — | Comando de testes (após `python`) que também precisa passar. |
| `--memory-limit`, `--reset-memory` | 100 | Memória do agente. |
| `--no-self-modify` | — | Proíbe alterar `agent_core/`. |
| `--json` | — | Relatório em JSON. |

Outros comandos: `analyze <arquivo>`, `backups [arquivo]`, `rollback <arquivo>`, `observe`, `ask "<prompt>" [--image f] [--fake]`, `memory [--clear]`.

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
- Rollback automático em regressão crítica, patch inócuo ou testes que regrediram; ação `rollback` explícita; `rollback_run(report)`; CLI `rollback` caminha no histórico.
- Loop infinito: iterações, retries, timeout total, estagnação, recusa de patch repetido.
- O agente atua apenas dentro do workspace; não contorna permissões do sistema, autenticação ou controles externos.

## Testes

```bash
python -m unittest discover -s tests -v
```

Suítes: `test_agent_core.py` (23 originais), `test_providers.py`, `test_vision.py`, `test_multimodal.py`, `test_auto_strategy.py`, `test_loop_v2.py`, `test_memory_safety.py`. Todas determinísticas (fakes/mocks); chamadas reais ao modelo ficam fora da suíte (`examples/provider_call.py`, `ask`).

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

## Limitações

- O sandbox isola efeitos colaterais e travamentos; não é barreira contra código hostil (use contêiner).
- OCR só existe com Tesseract instalado; sem ele a análise visual é estatística/estrutural.
- Captura de tela exige display (`mss`); em servidores headless a fonte falha de forma limpa.
- A heurística cobre imports e indentação; o restante depende do modelo.
- `--tests` roda na raiz real do projeto (não na cópia isolada).
