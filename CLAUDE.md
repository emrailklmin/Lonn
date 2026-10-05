# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Lönn AI is a prototype LLM router: it scores a prompt (topic distribution + complexity), deterministically picks a model from a curated pool (NVIDIA NIM and OpenRouter) by trading off quality against price, and executes the request through LiteLLM. `docs/Prototype_architecture.md` is the authoritative design doc (contracts, routing formula, module boundaries); `docs/model_metadata.md` documents where pool metadata comes from.

## Commands

Python 3.10+. Use the repo virtualenv (`.venv`) and install `requirements.txt` (litellm, python-dotenv, PyYAML, laya, streamlit).

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

# Streamlit prototype app (live calls need the selected model's provider key in .env)
python3 -m streamlit run examples/app.py --server.address localhost

# Try the complexity scorer by hand (--debug prints raw Laya output)
python3 examples/classify_prompt.py "Write a Python function that sorts a list"

# Offline pool maintenance (write to gitignored reports/). If urllib fails with
# CERTIFICATE_VERIFY_FAILED (python.org macOS build), first run:
#   export SSL_CERT_FILE=$(python3 -c 'import certifi; print(certifi.where())')
python3 scripts/discover_models.py
python3 scripts/check_models.py --model nvidia_nim/openai/gpt-oss-20b
```

There is no package install, build step, or linter config. Tests and examples put `src/` on `sys.path` themselves; for ad-hoc scripts use `PYTHONPATH=src`. Tests use `unittest` (not pytest), mock provider calls and inject a fake Laya, so they need no network or credentials. The one live Laya test is skipped when `laya` is not installed.

## Architecture

Pipeline in `src/lonn/`, orchestrated by `examples/app.py`:

`complexity.py` (prompt → `ComplexityResult`) → `router.py` (`ComplexityResult` + `list[Model]` → `Model`) → `executor.py` (prompt + `Model` → `ModelResponse`), with `model_pool.py` loading `config/models.yaml` into `list[Model]`.

- **`schemas.py`** owns all shared dataclasses and the `TopicCategory` enum (`coding`, `math_reasoning`, `creative_writing`, `general_knowledge_extraction`). Changing a topic or contract field ripples through the scorer, the router, the pool loader's validation, `config/models.yaml` and the tests.
- **Keep modules independent**, as the architecture doc requires: the scorer knows nothing about routing, the router never calls APIs, the pool loader makes no routing decisions, and the executor does not choose models. The app is the only orchestrator.
- **Scorer:** a single Laya `predict` call answers both the `category` (choice) and `complexity` (score over 5 levels) questions. Laya is lazy-loaded and shared, and `USE_TF=0` is set before import to avoid a TensorFlow hang. Inject `ComplexityScorer(laya_router=fake)` in tests. `topic_scores` must be a probability distribution that sums to 1.
- **Router policy:** quality = Σ topic_prob × aptitude (a missing aptitude counts as 0). Cost = (input + output price per M), normalized by the pool's max price. The router picks the maximum of `alpha*quality - (1-alpha)*cost` (default alpha 0.5). Ties prefer lower price, then higher quality, then `(provider, id)`. Complexity is validated but does not affect selection unless the caller passes `alpha=signal.complexity`. Invalid input raises `ValueError`.
- **Pool (`config/models.yaml`):** this is the only curated model config. Every active entry must have `provider` ∈ {nvidia_nim, openrouter} (see `PROVIDER_API_KEYS` in `schemas.py`) and an ID of the form `<provider>/<publisher>/<model>` with a matching prefix, known prices, `price_basis` ∈ {provider, estimate, free_tier}, a positive `context_window`, and all four aptitudes in [0, 1]. The loader rejects unknown values rather than defaulting them to zero, and it reports every invalid entry at once. Inactive entries may stay incomplete. Aptitudes are benchmark proxies. NIM prices are third-party estimates (`matched_provider_key`) rather than NVIDIA billing; OpenRouter prices are OpenRouter's listed prices. OpenRouter is used only for model access with credits, not for its routing. Record the source of any metadata you add.
- **Executor:** makes a non-streaming LiteLLM call. It loads the repo `.env` without overriding existing environment variables and requires only the key for the model's provider: `NVIDIA_NIM_API_KEY` or `OPENROUTER_API_KEY`. Cost is tokens / 1e6 × the configured price. `finish_reason == "length"` signals truncation, which is common for reasoning models under the default 4096-token budget and 120 s timeout. It does no retries, rerouting, streaming or tool calls, and provider exceptions propagate to the caller.
- **Discovery and availability scripts** are offline preparation. They write only to `reports/` and never modify `config/models.yaml`.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues on `emrailklmin/Lonn`, through the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five default labels are used as-is: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
