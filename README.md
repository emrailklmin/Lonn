<p align="center">
  <img src="assets/Lonnlogo.png" width="300" alt="Lönn AI">
</p>


AI infrastructure for intelligent LLM routing, cost optimisation and observability.

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

Two scripts in `examples/` let you try the scorer with the real Laya model:

- `scorer_demo.py` loads Laya once, then scores each prompt you type and shows
  how long it took (add `--debug` to see Laya's raw output):

  ```bash
  python3 examples/scorer_demo.py
  ```

- `scorer_benchmark.py` measures load time and per-prompt speed on your machine:

  ```bash
  python3 examples/scorer_benchmark.py
  ```

Prompts longer than Laya's window (about 377 tokens for English) are shortened
to their beginning and end, where the actual request usually is.

## Execution

With `src` on Python's import path (`PYTHONPATH=src`), pass the router's
selected `Model` to the executor:

```python
from lonn.executor import ModelExecutor
from lonn.schemas import ModelResponse

response: ModelResponse = ModelExecutor().execute(
    prompt, selected_model, timeout=60, max_tokens=1024
)
print(response.content)
print(response.usage.total_cost, response.latency_ms)
```

The executor loads the repository `.env` without overriding existing environment
variables. The selected model ID must be the full `nvidia_nim/...` ID from
`model_pool/pool.yaml`. Costs are estimates based on configured per-million
prices; they are not NVIDIA billing figures. `finish_reason` exposes truncation.
Provider errors propagate to the caller without rerouting or automatic retries.

The pool loader and app are still unimplemented. The NVIDIA
pool needs context limits and topic aptitudes before it can populate the shared
`Model` contract for routing. See `docs/prototype_architecture.md` for details.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Executor tests mock provider calls and require no credentials or network access.
Complexity tests use a fake Laya; the one live Laya test is skipped when `laya`
is not installed.
