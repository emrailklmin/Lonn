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
python3 scripts/benchmark/answer_checks.py         # check every saved answer once (sandbox), then eval_task_router.py --cascade
# Experiments that did not ship (docs/oracle_gap.md); paid calls, --dry-run first
python3 scripts/benchmark/cheap_judge.py --dry-run  # generic writing judge
python3 scripts/complexity_labels.py --dry-run      # then train_complexity.py: predicted complexity head
# Classifier training data (docs/task_data.md): streamed from HF, ~9 MB local, shared as private LonnAI/lonn-task-data
python3 scripts/build_task_data.py --include-gpqa   # --push LonnAI/lonn-task-data to upload; --clean to delete local copy
python3 scripts/classifier_bakeoff.py              # compare classifiers (docs/classifier_bakeoff.md), ~45 min
python3 scripts/train_task_classifier.py           # writes config/task_classifier.json (generated, don't hand-edit)
# Everyday prompts from WildChat (docs/task_data.md): sample is free; label makes paid calls (--dry-run, ask first)
python3 scripts/everyday_prompts.py sample          # then: label --set eval|train --model ... --dry-run; status

# Offline pool maintenance (write to gitignored reports/). If urllib fails with
# CERTIFICATE_VERIFY_FAILED (python.org macOS build), first run:
#   export SSL_CERT_FILE=$(python3 -c 'import certifi; print(certifi.where())')
python3 scripts/discover_models.py
python3 scripts/check_models.py --model nvidia_nim/openai/gpt-oss-20b
```

There is no package install, build step, or linter config. Tests and examples put `src/` on `sys.path` themselves; for ad-hoc scripts use `PYTHONPATH=src`. Tests use `unittest` (not pytest), mock provider calls and inject a fake Laya, so they need no network or credentials. The one live Laya test is skipped when `laya` is not installed.

## Architecture

Pipeline in `src/lonn/`, orchestrated by `examples/app.py` (Streamlit) and `src/lonn/api.py` (FastAPI). Prototype 4.0 uses the task-type pipeline (`task_classifier.py` → `task_router.py` → `executor.py`, see the task-type router bullet); the original pipeline below remains in the code but is no longer orchestrated:

`complexity.py` (prompt → `ComplexityResult`) → `router.py` (`ComplexityResult` + `list[Model]` → `Model`) → `executor.py` (prompt + `Model` → `ModelResponse`), with `model_pool.py` loading `config/models.yaml` into `list[Model]`.

- **`schemas.py`** owns all shared dataclasses and the `TopicCategory` enum (`coding`, `math_reasoning`, `creative_writing`, `general_knowledge_extraction`). Changing a topic or contract field ripples through the scorer, the router, the pool loader's validation, `config/models.yaml` and the tests.
- **Keep modules independent**, as the architecture doc requires: the scorer knows nothing about routing, the router never calls APIs, the pool loader makes no routing decisions, and the executor does not choose models. Only the Streamlit app and the API orchestrate. `complexityold.py` is the earlier two-round scorer. Nothing imports it.
- **Legacy scorer:** a single Laya `predict` call answers both the `category` and `complexity` questions. It remains available for experiments, but the app and API use the task classifier. See `docs/fine_tuned_laya.md`.
- **Legacy router policy:** `router.py` uses the original four topics, static aptitudes and alpha. It remains for historical comparisons and is not the active task router.
- **Task-type router (prototype4.0):** used by the app and API. `task_classifier.py` embeds the prompt and applies the linear layer in `config/task_classifier.json`. `routing_table.py` loads the eight measured task rows, overlays six estimates from `config/task_aptitudes.yaml`, and derives `other` from all 14. Supplied/derived categories use the measured global cost. Model responses expose score and cost provenance. `task_router.py` maximizes quality minus `lam * cost_usd` (default lam 20, complexity off). Evaluations must cross-fit measured rows and derive `other` within each training fold; see `docs/prototype4.md`.
- **HTTP API (`api.py`):** `GET /health`, `GET /models`, `POST /route` (no provider call), and `POST /generate` (falls back down the ranking when a provider call fails; then `answer_check.py` checks the answer, cut off, empty, or code failing the prompt's examples in the sandbox from `code_sandbox.py`, and escalates once via `task_router.escalation_target` when a second call is worth its cost; `verify: false` skips this). Requests take `lam` (default 20) and may include a `signal` (`task_scores` + optional `complexity`) to skip the classifier; responses include per-candidate `estimates`. The pool and routing table load at startup, so restart after editing `models.yaml` or rebuilding the table. Classifier calls are serialized behind a lock. Provider errors map to 429/504/502 with sanitized bodies, and analysis failures map to 503. There is no authentication; bind to loopback.
- **SDK (`sdk/python`, package `lonnrouter`):** `LonnRouter`/`AsyncLonnRouter`, a thin httpx + pydantic client of the HTTP API. It does not depend on `lonn`. `tests/test_sdk.py` runs it against the real FastAPI app with the pool, routing table, classifier, and executor mocked. When you change an API response shape, update `sdk/python/src/lonnrouter/models.py` too.
- **Pool (`config/models.yaml`):** active entries require a supported provider, matching ID, known prices, price basis, context window, and the legacy four topic scores. `config/task_aptitudes.yaml` is the single source of the six user-supplied task estimates; the eight measured scores remain in `config/routing_table.json`. NIM prices are third-party estimates; OpenRouter prices are listed prices. Record the source of any metadata you add.
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
