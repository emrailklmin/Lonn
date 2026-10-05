<p align="center">
  <img src="assets/Lonnlogo.png" width="300" alt="Lönn AI">
</p>


AI infrastructure for intelligent LLM routing, cost optimisation and observability.

## Repository layout

```text
config/models.yaml          # Single model configuration
src/lonn/                   # Runtime code
    model_pool.py           # Load and validate configuration into Model objects
    complexity.py           # Analyse prompts
    router.py               # Select a model
    executor.py             # Call the selected model
    schemas.py              # Shared data contracts
scripts/discover_models.py  # Optional catalog discovery
scripts/check_models.py     # Optional live text-availability checks
examples/                   # App and usage examples
tests/                      # Automated tests
docs/                       # Architecture
reports/                    # Generated reports (created on demand, gitignored)
```

## Setup

Requires Python 3.10 or newer.

1. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` if `.env` does not already exist.
3. Fill in your NVIDIA NIM API key (get one at https://build.nvidia.com) as
   `NVIDIA_NIM_API_KEY=nvapi-...` — that exact variable name is required, it's
   what LiteLLM's `nvidia_nim/...` provider reads. To call `openrouter/...`
   models, also set `OPENROUTER_API_KEY=sk-or-...` (https://openrouter.ai/keys).
   Only the key for the provider actually called is required.
4. `.env` is gitignored — never commit it.

## Run the prototype web app

From the repository root, after installing `requirements.txt` and setting
`NVIDIA_NIM_API_KEY` in `.env`:

```bash
source .venv/bin/activate
python3 -m streamlit run examples/app.py --server.address localhost
```

Open http://localhost:8501. Choose **Manual topic** for a quick test without
Laya, or **Automatic (Laya)** for the full analysis flow. First automatic use
may download classifier weights. **Analyze & route** displays routing without
calling NVIDIA; **Route & call NVIDIA** sends the prompt to the chosen model.
Adjust alpha, output token budget, and timeout in the sidebar. The app displays
complexity, topic probabilities, selected model, response, token usage, latency,
and estimated cost. Settings changes do not repeat a submitted API request.

Reasoning models may exhaust the output budget before producing final text.
The app reports truncation; increase the budget and retry if needed. The default
is 4,096 tokens and a 120-second timeout. API availability and account access
still require a live test. The UI keeps the API key on the Python server.

## Complexity scoring

The complexity scorer uses [Laya](https://pypi.org/project/laya/) to produce a
`ComplexityResult`: a topic probability distribution over `coding`,
`math_reasoning`, `creative_writing`, and `general_knowledge_extraction`, plus a
complexity score in [0, 1]. Laya is loaded on first use.

```python
from lonn.complexity import ComplexityScorer

result = ComplexityScorer().score_prompt(prompt)
print(result.complexity, result.topic_scores)
```

To try it by hand (add `--debug` to see Laya's raw output):

```bash
python3 examples/classify_prompt.py "Write a Python function that sorts a list"
```

## Execution

With `src` on Python's import path (`PYTHONPATH=src`), pass the router's
selected `Model` to the executor:

```python
from lonn.executor import ModelExecutor
from lonn.schemas import ModelResponse

response: ModelResponse = ModelExecutor().execute(
    prompt, selected_model, timeout=120, max_tokens=4096
)
print(response.content)
print(response.usage.total_cost, response.latency_ms)
```

The executor loads the repository `.env` without overriding existing environment
variables. The selected model ID must be the full `nvidia_nim/...` or `openrouter/...` ID
from `config/models.yaml`, and the matching key (`NVIDIA_NIM_API_KEY` or
`OPENROUTER_API_KEY`) must be set. Costs are computed from configured
per-million prices, not from provider billing. `finish_reason` exposes truncation.
Provider errors propagate to the caller without rerouting or automatic retries.

The pool loader and local Streamlit app are implemented. Model context values
and provisional benchmark-derived aptitudes are populated. See
[metadata sources and limitations](docs/model_metadata.md). Unknown values are
rejected rather than silently treated as zero.

## NVIDIA model pool

Discover the catalog without excluding models based on third-party pricing:

```bash
python3 scripts/discover_models.py
```

This writes `reports/model_catalog.json`, including unpriced models.
It is a candidate inventory, not a guarantee of text-generation availability.
Test a specific candidate with a small live request:

```bash
python3 scripts/check_models.py --model nvidia_nim/openai/gpt-oss-20b
```

With no arguments the checker probes active entries in `config/models.yaml`.
`--catalog reports/model_catalog.json` probes the entire discovered
inventory, including non-chat models; it can make many API requests. The checker
uses a 120-second timeout and at most two attempts per model, retrying transient
failures. Results go to `reports/model_availability.json`; the pool is never
rewritten. It distinguishes timeouts, rate limits, access failures, and 404s.
A successful text response does not imply tool-call support or high aptitude.

Maintain `config/models.yaml` with active models, documented context limits, and all
four topic aptitude scores. Record the source of these values alongside them.
`price_basis` must identify `provider`, `free_tier`, or `estimate`; current prices
are estimates from the providers named in `matched_provider_key`. Do not report
them as NVIDIA charges. Then load the routing contract:

```python
from lonn.model_pool import load_model_pool
from lonn.router import select_model

models = load_model_pool()
selected_model = select_model(result, models, alpha=0.7)
```

Discovery and probes are offline preparation steps, not part of each routed
request. Generated reports are ignored by Git; `config/models.yaml` is the
only curated model configuration.
See `docs/prototype_architecture.md` for the responsibility boundaries.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Executor and app tests mock provider calls and require no credentials or network access.
Complexity tests use a fake Laya; the one live Laya test is skipped when `laya`
is not installed.
