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
