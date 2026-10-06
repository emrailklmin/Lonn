# Model metadata and benchmark mapping

Research snapshot: 2026-09-29. `config/models.yaml` is the only runtime
configuration. This note records the evidence and the conversion policy.

## Context windows

| NVIDIA model | Configured total tokens | Evidence |
|---|---:|---|
| Nemotron 3 Ultra 550B A55B | 1,000,000 | NVIDIA's [hosted integration cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Ultra/OpenScaffoldingResources/README.md) explicitly configures 1,000,000 for the NVIDIA endpoint. |
| Muse Glimmer 30B | 131,072 | The [hosted catalog page](https://build.nvidia.com/meta/muse-glimmer-30b) advertises 131K; its [model card](https://build.nvidia.com/meta/muse-glimmer-30b/modelcard) specifies 131,072+ combined input/output tokens. |
| GPT-OSS 20B | 131,072 | The [NVIDIA catalog page](https://build.nvidia.com/openai/gpt-oss-20b) advertises 131K; the [original model card](https://deploymentsafety.openai.com/gpt-oss/a2) gives the exact 131,072-token model context. |

These are documented configuration values, not authenticated boundary tests of
our account. The Muse/GPT exact counts resolve NVIDIA's rounded catalog labels
using model specifications. Account or deployment limits can be lower. Ultra's
1,000,000 is the hosted cookbook setting, not an assumption that 1M means
1,048,576. Context includes prompt, chat formatting, reasoning, and output;
`max_tokens` is an output budget, not a context-window setting. The executor
currently checks only the output budget against context, not the full prompt.

## Published benchmark inputs

| Model / evaluation variant | SWE-bench Verified (%) | GPQA (%) | HLE (%) | Creative Writing v3 Elo |
|---|---:|---:|---:|---:|
| Ultra NVFP4 | 69.7 | 87.9 | 26.1 | 1692.4 |
| Muse Glimmer, high reasoning | 76.0 | 83.5 | 22.0 | 1798.3 |
| GPT-OSS 20B, high reasoning | 60.7 | 71.5 | 10.9 | 665.6 |

Sources for coding/reasoning/knowledge:

- [NVIDIA Ultra model card](https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b/modelcard): NVFP4 column; GPQA and HLE without tools.
- [NVIDIA Muse model card](https://build.nvidia.com/meta/muse-glimmer-30b/modelcard): high-reasoning column; GPQA Diamond (AA) and HLE Text (AA).
- [OpenAI GPT-OSS model card, Table 3](https://deploymentsafety.openai.com/gpt-oss/a2): 20B/high column; GPQA Diamond and HLE without tools.

Writing scores come from the benchmark author's [EQ-Bench Creative Writing v3
leaderboard](https://eqbench.com/creative_writing.html), specifically the
`elo_score` rows in its [published leaderboard data](https://eqbench.com/creative_writing.js?v=1.0.91):
`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`,
`meta-models/Muse-Glimmer-30B`, and `openai/gpt-oss-20b`.
The writing evaluations are separate runs, not necessarily the reasoning
settings in the vendor tables. [EQ-Bench methodology](https://eqbench.com/about.html)
uses LLM judging and anchors the writing Elo scale to DeepSeek-R1 at 1500 and
Ministral-3B at 200; it is not a human accuracy percentage.

## Conversion to routing scores

These are explicitly **prototype proxies**, not vendor-supplied aptitude fields:

- `coding = SWE-bench Verified / 100`.
- `math_reasoning = GPQA / 100`: a science-reasoning proxy, not a math score.
- `general_knowledge_extraction = HLE / 100`: a difficult knowledge proxy, not
  an extraction benchmark. Muse's text-only HLE and the other reported HLE runs
  are not guaranteed to use identical subsets or evaluation harnesses.
- `creative_writing = 1 / (1 + 10 ** ((1500 - Elo) / 400))`, rounded to four
  decimals. This fixed logistic mapping is our routing policy; it is not a
  measured success probability. In particular, GPT-OSS's 0.0081 does not mean
  it succeeds on only 0.81% of writing requests.

Raw percentage fractions and transformed Elo values are not calibrated to a
common difficulty scale. Consequently, topic weights and alpha remain heuristic.
SWE-bench uses agent tooling that the current text executor does not provide;
reasoning settings, token budgets, quantization, and harnesses differ. Published
high-reasoning results may require much larger budgets than the executor's
4,096-token default. Do not advertise these scores as measured NVIDIA-endpoint
performance. Replace them with results on shared application-specific prompts,
including extraction tasks, when evaluating routing quality.

Prices remain the existing third-party estimates and are not NVIDIA charges.

## Providers

The pool supports two providers. The ID prefix must match `provider`:

| `provider` | ID form | API key | Typical `price_basis` |
|---|---|---|---|
| `nvidia_nim` | `nvidia_nim/<publisher>/<model>` | `NVIDIA_NIM_API_KEY` | `estimate` (NIM is free for prototyping; prices come from other providers, see `matched_provider_key`) |
| `openrouter` | `openrouter/<publisher>/<model>` | `OPENROUTER_API_KEY` | `provider` (OpenRouter's listed price for the model) |

For an OpenRouter entry, record the price from the model's OpenRouter page and
the date it was checked. Only OpenRouter's model access is used, not its routing.

## OpenRouter candidates (inactive)

Researched 2026-10-06. Eleven OpenRouter entries are in `config/models.yaml`
with `active: false`. The three NIM entries above stay active. Each candidate was
kept because it adds something per dollar; models that were worse than another
candidate in every category and also more expensive were left out (see below).

| OpenRouter ID | Role | In / out $/M | Context | Arena coding | Arena math | Arena creative | Arena expert | AA Intelligence Index | AA $ per task | AA HLE % |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `anthropic/claude-sonnet-5.5` | Frontier ceiling; #1 on Arena expert prompts | 2 / 10 | 1,000,000 | 1536 | – | 1447 | 1556 | 47 | 1.12 | – |
| `meta/muse-spark-1.3` | Frontier-level coding and math at under half of Sonnet's output price | 1.25 / 4.25 | 1,048,576 | 1539 | 1509 | 1459 | 1529 | 48 | 1.6 | 48.7 |
| `google/gemini-3.8-flash` | Best creative writing in the set; strong all-rounder at mid price | 0.75 / 3.75 | 1,048,576 | 1530 | 1517 | 1485 | 1526 | 41 | 1.24 | 47.8 |
| `moonshotai/kimi-k3` | Not for routing yet: top coding Elo, but highest output price and matched by Muse Spark 1.3 | 0.95 / 14 | 1,048,576 | 1541 | 1500 | 1460 | 1533 | 44 | 2 | 46.9 |
| `z-ai/glm-5.3-flash` | Best value: near-frontier coding and math at $0.15 / $0.50 | 0.15 / 0.5 | 1,048,576 | 1523 | 1503 | 1433 | 1511 | 42 | 0.25 | 39.9 |
| `deepseek/deepseek-v4.1-flash` | Cheapest strong reasoning; very low input price, verbose output | 0.0214 / 1.32 | 1,048,576 | 1528 | 1503 | 1440 | 1517 | 39 | 0.27 | 39.2 |
| `deepseek/deepseek-v4-pro` | Cheap balanced model with the lowest output price above 1440 Elo in creative writing | 0.2088 / 0.4176 | 1,048,576 | 1500 | 1445 | 1446 | 1480 | – | – | – |
| `nvidia/nemotron-3-ultra-550b-a55b` | OpenRouter twin of the active NIM entry, for comparing estimated and real prices | 0.5 / 2.2 | 262,144 | 1474 | 1442 | 1382 | 1466 | 23 | 0.6 | 28.4 |
| `meta/muse-glimmer-30b` | OpenRouter twin of the active NIM entry; very concise (AA $0.06 per task, second lowest in the set after gpt-oss-20b) | 0.35 / 1.5 | 131,072 | 1479 | 1446 | 1366 | 1442 | 17 | 0.06 | 22 |
| `openai/gpt-oss-120b` | Very cheap and fast for simple prompts; weak at creative writing | 0.037 / 0.17 | 131,072 | 1390 | 1381 | 1278 | 1357 | 12 | 0.11 | 20 |
| `openai/gpt-oss-20b` | Price floor; OpenRouter twin of the active NIM entry | 0.018 / 0.09 | 131,072 | 1369 | 1335 | 1240 | 1314 | 9 | 0.01 | – |

Aptitudes written to `config/models.yaml`:

| OpenRouter ID | Arena name used | coding | math_reasoning | creative_writing | general_knowledge_extraction |
|---|---|---:|---:|---:|---:|
| `anthropic/claude-sonnet-5.5` | `claude-sonnet-5.5-xhigh` | 0.6863 | – | 0.5672 | 0.7105 |
| `meta/muse-spark-1.3` | `muse-spark-1.3-max` | 0.69 | 0.6519 | 0.5841 | 0.6776 |
| `google/gemini-3.8-flash` | `gemini-3.8-flash-high` | 0.6788 | 0.6623 | 0.6199 | 0.6738 |
| `moonshotai/kimi-k3` | `kimi-k3-max` | 0.6925 | 0.6401 | 0.5855 | 0.6826 |
| `z-ai/glm-5.3-flash` | `glm-5.3-flash` | 0.67 | 0.644 | 0.5473 | 0.6545 |
| `deepseek/deepseek-v4.1-flash` | `deepseek-v4.1-flash-max` | 0.6763 | 0.644 | 0.5573 | 0.6623 |
| `deepseek/deepseek-v4-pro` | `deepseek-v4-pro` | 0.6401 | 0.5644 | 0.5658 | 0.6131 |
| `nvidia/nemotron-3-ultra-550b-a55b` | `nvidia-nemotron-3-ultra-550b-a55b-nvfp4` | 0.6049 | 0.5602 | 0.4741 | 0.5939 |
| `meta/muse-glimmer-30b` | `muse-glimmer` | 0.6118 | 0.5658 | 0.4512 | 0.5602 |
| `openai/gpt-oss-120b` | `gpt-oss-120b` | 0.4856 | 0.4727 | 0.3313 | 0.4384 |
| `openai/gpt-oss-20b` | `gpt-oss-20b` | 0.4555 | 0.4075 | 0.2847 | 0.3787 |

### Sources

- Prices and context windows: the [OpenRouter models API](https://openrouter.ai/api/v1/models), 2026-10-06.
- Arena category Elo: [coding](https://arena.ai/leaderboard/text/coding) (page cached 2026-10-05),
  [math](https://arena.ai/leaderboard/text/math) (2026-10-06),
  [creative writing](https://arena.ai/leaderboard/text/creative-writing) (2026-10-05) and
  [expert](https://arena.ai/leaderboard/text/expert) (2026-10-06). Elo comes from blind
  pairwise human votes; the ± interval on the page is about 5–30 points.
- Artificial Analysis: Intelligence Index v4.3.2 and cost per Intelligence Index task from the
  [model leaderboard](https://artificialanalysis.ai/leaderboards/models); HLE from the
  [HLE leaderboard](https://artificialanalysis.ai/evaluations/humanitys-last-exam) (top 30 only) and the
  gpt-oss-120b model page. AA measures every model with the same harness. Values are for the
  reasoning variant named on AA (Sonnet 5.5: high; Muse Spark 1.3, Kimi K3, DeepSeek V4.1 Flash: max;
  Gemini 3.8 Flash, Muse Glimmer, gpt-oss: high); HLE for Sonnet 5.5 is only published for max (55.0%)
  and is left out. No AA row was matched to OpenRouter's `deepseek-v4-pro`.

### Conversion to routing scores

OpenRouter entries use one source for all four topics, so the scores are comparable
across models:

- `coding` = Arena coding, `math_reasoning` = Arena math, `creative_writing` = Arena
  creative writing, `general_knowledge_extraction` = Arena expert (a proxy: expert-level
  prompts, not an extraction benchmark).
- `aptitude = 1 / (1 + 10 ** ((1400 - Elo) / 400))`: the Elo-implied chance of being
  preferred over a fixed 1400-Elo reference, rounded to four decimals. The reference is a
  routing policy choice, not a measured success rate.

### Considered and left out

Worse than GLM-5.3-Flash in all four Arena categories and more expensive per token:
Qwen3.7 Plus, MiniMax M3, Grok 4.7, Mistral Medium 3.5, Kimi K2.5 and K2.6. Also left out:
GLM-5.3 (max) (about equal to GLM-5.3-Flash at higher cost), Qwen3.8 Max (AA cost $5.41 per
task), GPT-5.6 Terra (below Sonnet 5.5 at a higher output price), and GPT-5.6 Sol (close to
Sonnet 5.5 at the same price; a possible second frontier model).

### Known gaps before activating any of these

- **Mixed methods.** The active NIM entries use vendor benchmarks (SWE-bench, GPQA, HLE,
  EQ-Bench); the OpenRouter entries use Arena Elo. Do not activate a mix without re-scoring
  all active entries with one method.
- **Sonnet 5.5 has no Arena math score yet** (released 2026-09-28). Its `math_reasoning` is
  `null`, and the loader will reject it if it is activated before a value is sourced.
- **Reasoning effort.** Benchmarks were run at high, xhigh or max effort. The executor sends
  no effort setting, so real quality can be lower than the aptitudes suggest. See #1.
- **Price per token is not cost per task.** Verbose models cost more than their list price
  suggests (AA: DeepSeek V4.1 Flash produced 250M output tokens on the index, Gemini 3.8
  Flash 170M, Sonnet 5.5 52M). The router only sees list prices.
- **Output budget.** Verbose reasoning models are likely to hit the executor's 4,096-token
  default and return `finish_reason == "length"`.
- **Speed is not routed on.** GLM-5.3-Flash outputs about 53 tokens/s; Gemini 3.8 Flash has
  about 17 s to first token.
- **Preliminary Elo.** Arena marks all of Gemini 3.8 Flash's scores as preliminary.
- **OpenRouter provider choice.** OpenRouter can serve a model from several providers at
  different prices; the listed price is OpenRouter's. See #6.
