# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Lonn AI is a prototype LLM router: it scores a prompt (topic distribution + complexity), deterministically picks a model from a curated pool (NVIDIA NIM and OpenRouter) by trading off quality against price, and executes the request through LiteLLM. `docs/Prototype_architecture.md` is the authoritative design doc (contracts, routing formula, module boundaries); `docs/model_metadata.md` documents where pool metadata comes from.

## Commands

Python 3.10+. Use the repo virtualenv (`.venv`) and install `requirements.txt`. It covers the runtime (litellm, laya), the app (streamlit), the API (fastapi, uvicorn), and an editable install of the SDK from `./sdk/python`.

```bash
source .venv/bin/activate
python3 -m pip install -r requirements.txt

# All tests (run from repo root)
python3 -m unittest discover -s tests -v
# One module / one test
python3 -m unittest discover -s tests -p test_router.py
python3 -m unittest tests.test_router.RouterTests.test_alpha_changes_selection
# Test files also run directly
python3 tests/test_router.py

# Streamlit prototype app (live calls need OPENROUTER_API_KEY in .env for the active pool)
python3 -m streamlit run examples/app.py --server.address localhost

# HTTP API for benchmarks (docs at http://127.0.0.1:8000/docs); then try the SDK
python3 -m uvicorn lonn.api:app --app-dir src --host 127.0.0.1 --port 8000
python3 tests/try_sdk.py   # manual smoke script against the running server, not a unit test

# Try the complexity scorer by hand (--debug prints raw Laya output); time it
python3 examples/scorer_demo.py
python3 examples/scorer_benchmark.py

# Fine-tune Laya on labelled prompts and publish it (see docs/fine_tuned_laya.md)
python3 scripts/finetune_laya.py
python3 scripts/push_laya.py --repo LonnAI/laya-lonn

# Benchmark prompt sets: download sources, write seeded samples to gitignored
# data/benchmark/{raw,prompts}, record data/benchmark/manifest.json (docs/benchmark_sources.md)
python3 scripts/benchmark/build_data.py            # or name datasets; --check verifies a rebuild
# Answer bank (docs/benchmark_answer_bank.md): every model on every question via OpenRouter,
# graded, appended to committed results/benchmark/. Paid calls: --dry-run first, ask before running
python3 scripts/benchmark/run.py --dry-run         # --limit N for a pilot, --budget USD cap, --regrade
python3 scripts/benchmark/check_graders.py         # graders accept reference answers (no calls)
# Task-type router (#19, docs/task_router.md): rebuild the routing table, then a cross-fitted replay (no calls)
python3 scripts/benchmark/build_routing_table.py   # writes config/routing_table.json (generated, don't hand-edit)
python3 scripts/benchmark/eval_task_router.py --name task-true-utility

# Offline pool maintenance (write to gitignored reports/). If urllib fails with
# CERTIFICATE_VERIFY_FAILED (python.org macOS build), first run:
#   export SSL_CERT_FILE=$(python3 -c 'import certifi; print(certifi.where())')
python3 scripts/discover_models.py
python3 scripts/check_models.py --model nvidia_nim/openai/gpt-oss-20b
```

There is no package install, build step, or linter config. Tests and examples put `src/` on `sys.path` themselves; for ad-hoc scripts use `PYTHONPATH=src`. Tests use `unittest` (not pytest), mock provider calls and inject a fake Laya, so they need no network or credentials. The one live Laya test is skipped when `laya` is not installed.

## Architecture

Pipeline in `src/lonn/`, orchestrated by `examples/app.py` (Streamlit) and `src/lonn/api.py` (FastAPI):

`complexity.py` (prompt → `ComplexityResult`) → `router.py` (`ComplexityResult` + `list[Model]` → `Model`) → `executor.py` (prompt + `Model` → `ModelResponse`), with `model_pool.py` loading `config/models.yaml` into `list[Model]`.

- **`schemas.py`** owns all shared dataclasses and the `TopicCategory` enum (`coding`, `math_reasoning`, `creative_writing`, `general_knowledge_extraction`). Changing a topic or contract field ripples through the scorer, the router, the pool loader's validation, `config/models.yaml` and the tests.
- **Keep modules independent**, as the architecture doc requires: the scorer knows nothing about routing, the router never calls APIs, the pool loader makes no routing decisions, and the executor does not choose models. Only the Streamlit app and the API orchestrate. `complexityold.py` is the earlier two-round scorer. Nothing imports it.
- **Scorer:** a single Laya `predict` call answers both the `category` (choice) and `complexity` (yes/no "is this task hard?") questions; complexity is Laya's probability of yes. Prompts longer than Laya's window keep their start and end. Laya is lazy-loaded and shared, and `USE_TF=0` is set before import to avoid a TensorFlow hang. `LONN_LAYA_MODEL` in `.env` swaps in a fine-tuned English checkpoint (the team's is `LonnAI/laya-lonn`, private); it is trained for the exact question wording in `complexity.py`, so retrain after changing it. Inject `ComplexityScorer(laya_router=fake)` in tests. `topic_scores` must be a probability distribution that sums to 1. See `docs/fine_tuned_laya.md`.
- **Router policy:** quality = Σ topic_prob × aptitude (a missing aptitude counts as 0). Cost = (input + output price per M), normalized by the pool's max price. The router picks the maximum of `alpha*complexity*quality - (1-alpha)*cost` (default alpha 0.5). At complexity 0, selection is effectively the cheapest model. Ties prefer lower price, then higher quality, then `(provider, id)`. Invalid input raises `ValueError`.
- **Task-type router (prototype3.0, #19):** being built next to the old router, not yet wired into the app or API. `task_types.py` loads `config/task_types.yaml` (the taxonomy, `docs/task_types.md`). `routing_table.py` holds accuracy and mean billed cost per (task type, difficulty, model), built from the answer bank. `task_router.py` takes a `TaskSignal` (task-type probabilities + optional complexity) and picks the maximum of `accuracy - lam * cost_usd` (default lam 20, complexity off). Router evaluations from the bank must be cross-fitted (`eval_task_router.py`), because the table is built from the same answers.
- **HTTP API (`api.py`):** `GET /health`, `GET /models`, `POST /route` (no provider call), and `POST /generate` (one execution, no retries). A request may include a `signal` (complexity + topic_scores) to skip Laya. The pool loads at startup, so restart after editing `models.yaml`. Laya calls are serialized behind a lock. Provider errors map to 429/504/502 with sanitized bodies, and analysis failures map to 503. There is no authentication; bind to loopback.
- **SDK (`sdk/python`, package `lonnrouter`):** `LonnRouter`/`AsyncLonnRouter`, a thin httpx + pydantic client of the HTTP API. It does not depend on `lonn`. `tests/test_sdk.py` runs it against the real FastAPI app with the pool, scorer, and executor mocked. When you change an API response shape, update `sdk/python/src/lonnrouter/models.py` too.
- **Pool (`config/models.yaml`):** this is the only curated model config. Every active entry must have `provider` ∈ {nvidia_nim, openrouter} (see `PROVIDER_API_KEYS` in `schemas.py`) and an ID of the form `<provider>/<publisher>/<model>` with a matching prefix, known prices, `price_basis` ∈ {provider, estimate, free_tier}, a positive `context_window`, and all four aptitudes in [0, 1]. The loader rejects unknown values rather than defaulting them to zero, and it reports every invalid entry at once. Inactive entries may stay incomplete. Aptitudes are benchmark proxies. NIM prices are third-party estimates (`matched_provider_key`) rather than NVIDIA billing; OpenRouter prices are OpenRouter's listed prices. OpenRouter is used only for model access with credits, not for its routing. Record the source of any metadata you add.
- **Executor:** makes a non-streaming LiteLLM call. It loads the repo `.env` without overriding existing environment variables and requires only the key for the model's provider: `NVIDIA_NIM_API_KEY` or `OPENROUTER_API_KEY`. Cost is tokens / 1e6 × the configured price. `finish_reason == "length"` signals truncation, which is common for reasoning models under the default 4096-token budget and 120 s timeout. It does no retries, rerouting, streaming or tool calls, and provider exceptions propagate to the caller.
- **Benchmark (`scripts/benchmark/`)** is offline tooling and is not imported by `src/lonn`. `build_data.py` writes the prompt sets. `run.py` calls OpenRouter directly (not the executor) with identical per-family system prompts (`templates.py`; bump `TEMPLATE_VERSION` after any wording change). `graders.py` grades answers and runs generated code only inside the macOS `sandbox-exec` profile. `results.py` appends to and reads the answer bank.
- **Discovery and availability scripts** are offline preparation. They write only to `reports/` and never modify `config/models.yaml`.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues on `emrailklmin/Lonn`, through the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default labels are used as-is: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
