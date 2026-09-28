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

The pool loader, complexity scorer, and app are still unimplemented. The NVIDIA
pool needs context limits and topic aptitudes before it can populate the shared
`Model` contract for routing. See `docs/prototype_architecture.md` for details.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Executor tests mock provider calls and require no credentials or network access.
