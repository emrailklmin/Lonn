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
   what LiteLLM's `nvidia_nim/...` provider reads.
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

## Python SDK for apps and benchmarks

Install the lightweight client from this checkout on the app's machine:

```bash
python -m pip install ./sdk/python
```

```python
from lonnrouter import LonnRouter

with LonnRouter(base_url="http://192.168.1.42:8000") as router:
    result = router.generate("Explain binary search", alpha=0.7)
    print(result.content)
    print(result.model.id, result.usage.total_cost)
```

Replace the example address with your running Lonn server. You can instead set
`LONN_BASE_URL` and use `LonnRouter()`. The SDK calls the HTTP API; it does not
start the server. Only the server needs Laya, NVIDIA credentials, and the model
configuration. The client package depends only on HTTPX and Pydantic.

Use `router.route(...)` for selection without generation, or `AsyncLonnRouter`
with `await router.generate(...)` inside an async app. Results expose typed
attributes and `model_dump()` for saving benchmark records. The client reuses
connections and performs no automatic retries.

See the [SDK guide](sdk/python/README.md) for async examples, installation,
timeouts, supplied signals, and error handling. This package is not yet published
to PyPI; install from the checkout or a built wheel.

## Benchmark HTTP API

Install `requirements.txt`, then start FastAPI from the repository root:

```bash
source .venv/bin/activate
python3 -m uvicorn lonn.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Interactive request/response documentation: http://127.0.0.1:8000/docs.
This API has no authentication; loopback is the default for local use. For a
benchmark runner on another device, see
[network access and SSH tunnel instructions](docs/Prototype_architecture.md#9-calling-from-another-device--new).

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Service health; does not check classifier or provider availability |
| `GET /models` | Active model pool and configured metadata |
| `POST /route` | Analyze a prompt and return the selected model without a provider call |
| `POST /generate` | Analyze, route, and call the selected NVIDIA model once |

Selection-only example, supplying a signal to avoid loading Laya:

```bash
curl http://127.0.0.1:8000/route \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Write a sorting function","alpha":0.7,"signal":{"complexity":0.5,"topic_scores":{"coding":1.0}}}'
```

Omit `signal` to analyze the prompt with Laya automatically. The signal requires
complexity and topic probabilities in [0, 1], with probabilities summing to one;
omitted topics count as zero. `alpha` defaults to 0.5, matching the core router.
Responses include `model`, `signal`, `alpha`, `analysis_source` (`provided` or
`laya`), and `routing_latency_ms` (analysis plus selection, including any wait
for the classifier).

End-to-end benchmark call from Python:

```python
import httpx

result = httpx.post(
    "http://127.0.0.1:8000/generate",
    json={"prompt": "What is 17 * 23?", "alpha": 0.7,
          "max_tokens": 4096, "timeout": 120},
    timeout=300,  # Also allow time for local prompt analysis.
)
result.raise_for_status()
data = result.json()
print(data["model"]["id"], data["response"]["content"])
print(data["response"]["usage"], data["total_latency_ms"])
```

`/generate` accepts the same optional `signal`. It additionally returns `response`
(the existing `ModelResponse`: content, model, usage, provider latency, and
finish reason) and `total_latency_ms` for server-side analysis, routing, and
execution. Neither timing includes HTTP transport; measure client elapsed time
separately. Costs are configured estimates. Preserve `finish_reason` in benchmark
results: `length` indicates truncation, potentially with empty final text.

The output budget defaults to 4096 tokens and must be below the selected model's
context window. Prompt token counting is not yet implemented. Provider timeout
defaults to 120 seconds (maximum 600); it does not limit classifier time.
Execution requires `NVIDIA_NIM_API_KEY` in the environment or repository `.env`.
There are no automatic retries or fallbacks. Invalid requests return 422;
analysis/routing failures 503; provider rate limits 429; provider timeouts 504;
other execution failures 502. Error bodies omit raw provider details.

The pool is loaded and validated at startup; restart after changing configuration.
Laya loads on the first automatic request and is reused. Warm it up with an
automatic `/route` call before measuring steady-state latency. Classifier calls
are serialized within each server process; provider calls can overlap. Each
additional Uvicorn worker loads its own classifier. This is a simple JSON API,
not an OpenAI chat-completions endpoint.

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
variables. The selected model ID must be the full `nvidia_nim/...` ID from
`config/models.yaml`. Costs are estimates based on configured per-million
prices; they are not NVIDIA billing figures. `finish_reason` exposes truncation.
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
See [the architecture and branch changes](docs/Prototype_architecture.md) for
the responsibility boundaries, HTTP request flow, and highlighted FastAPI additions.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Executor and app tests mock provider calls and require no credentials or network access.
Complexity tests use a fake Laya; the one live Laya test is skipped when `laya`
is not installed.
