# Lönn customer demo

This branch extends `prototype3.0`. The original Streamlit routing lab and benchmark API remain available. The customer demo uses React + TypeScript, Vite, Tailwind, a shadcn-style Radix/CVA button component, Recharts, FastAPI, SQLite + SQLModel, and the existing HTTPX Python SDK.

## First-time setup

Requires Python 3.10+ and Node.js 20+ with npm or pnpm. Python 3.12 is tested.

```bash
git clone https://github.com/emrailklmin/Lonn.git
cd Lonn
git checkout demo/customer-portal
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/demo.py
```

The startup command installs/builds the frontend and starts one loopback-only FastAPI process serving both applications. pnpm uses the committed lockfile; npm is supported but resolves from package.json on its first install. For subsequent starts with an unchanged frontend:

```bash
python scripts/demo.py --skip-build
```

Open **http://127.0.0.1:8000** for the Lönn workspace and **http://127.0.0.1:8000/customer** for the separate Nordic Notes customer application. Keep one server process running. Stop with Ctrl+C.

Automatic analysis uses the existing local task classifier. Its pinned embedding weights must be downloaded once. Prefetch before presenting:

```bash
python scripts/demo.py prepare
```

After dependencies and weights are cached, OpenRouter is the only runtime external service. Manual task mode uses the same router with an explicit task distribution and needs no classifier download.

## Presenter walkthrough

1. Select **Enterprise administrator** and sign in with the prefilled demo credentials. Nordic Studio starts with 64 deterministic synthetic requests, with models ranked by the existing router at λ = 0.20 for a varied synthetic task mix and token counts.
2. Inspect the overview, model usage and request history. Use the source selector to distinguish seeded and live requests. Click a request ID for routing details.
3. In **Settings → Presenter configuration**, enter your OpenRouter key. The field is masked; the API returns only whether a key is configured. This credential is not the customer API key.
4. Reveal/copy the generated **Lönn API key**. The example Python snippet deliberately uses a placeholder; paste the customer key or use an environment variable.
5. Save routing preferences. The slider directly sets the router's λ between **0 and 1**, default **0.20**. The logarithmic-looking curve is visual only. Prototype 3.0 ranks models by expected accuracy minus λ times expected cost, so higher λ favors cost and lower λ favors accuracy. The prototype's original benchmark default is 20; the demo's 0.20 default therefore has a different weighting. If organization enforcement is enabled, the server uses this setting; otherwise an explicit SDK request `lam` applies, followed by the saved personal preference or default. Legacy demo settings outside 0–1 are reset to 0.20 at startup; in-range values are preserved.
6. Review **Model pool**. All configured models are shown, including inactive ones. Account exclusions and unknown privacy metadata are explained. Individual radar charts show the existing normalized aptitude scores. Select up to four models in the comparison arena; the dashed pool outline is the maximum eligible catalog score per category, not a predicted ensemble quality. Search/filter the catalog and enable/disable models directly.
7. Open **Customer application** for the Nordic Notes chat workspace and Python integration pane. A signed-in portal user connects automatically to an assigned application; the connection popover also accepts a personal key with an application, or an application key. Use the composer’s **+** menu for automatic routing (`router1.0`), direct selection from eligible models, and λ when unlocked. The snippet uses the existing SDK’s automatic `generate()` call or its `model` argument. Messages are independent requests.
8. Return to the dashboard. It refreshes every five seconds and records successful and failed requests.
9. Restart the server and sign in again. Account settings, keys and usage persist. Browser login sessions expire on restart.

No verified endpoint-level ZDR, training, region or retention metadata exists in the prototype catalog. Strict privacy requirements therefore exclude **all** models. This is intentional fail-closed behavior, not a provider compliance claim. Clear those requirements to demonstrate inference. Disabled models remain excluded from ranking, fallback and escalation.

## Python customer integration

The existing SDK is installed by requirements.txt; client-only environments can use `python -m pip install ./sdk/python`.

```bash
export LONN_API_KEY='paste-your-customer-key-here'
export LONN_BASE_URL='http://127.0.0.1:8000'
export LONN_APPLICATION='engineering-app'  # required for personal keys
python examples/customer_requests.py
# Same router, explicit task signals, no classifier download:
python examples/customer_requests.py --manual-tasks
```

The script sends mathematical reasoning, code generation, creative writing and general-knowledge prompts. It makes real provider requests when configured; no mock inference mode is enabled in the running product.

```python
import os
from lonnrouter import LonnRouter

with LonnRouter(api_key=os.environ['LONN_API_KEY'], application='engineering-app') as client:
    result = client.generate('Explain how neural networks learn.')
    print(result.content)
    print(result.model.id, result.total_cost_usd)
```

`AsyncLonnRouter` accepts the same `api_key` parameter. Both clients also read `LONN_API_KEY`. HTTP requests use `Authorization: Bearer <customer key>` on `/generate`, `/route` and `/models`. `/generate` returns the original typed response and an `X-Lonn-Request-ID` header. Failed authentication is rejected before routing. The miniature React app calls that same authenticated `/generate` endpoint, never OpenRouter.

## Persistence and reset

Runtime files are gitignored under `.local/demo/`:

- `demo.sqlite3`: accounts, users with hashed passwords, teams, generated personal/application keys, saved settings, catalog snapshot and usage records.
- `provider.key`: presenter credential; mode 0600, atomically replaced and never returned by the API. It is local plaintext, not an encrypted credential vault.

`LONN_DEMO_DIR` can select another local data directory. Keep it outside version control. No actual key is in seed data. `data/demo/seed.json` plus the existing `config/models.yaml` deterministically create the historical dataset; the customer key is freshly generated at initialization/reset. Catalog changes require a reset to refresh the database snapshot.

Stop the server before resetting:

```bash
python scripts/demo.py reset
python scripts/demo.py --skip-build
```

Reset restores seeded accounts, users, teams, settings/history and generates fresh personal/application keys. It retains the presenter credential; remove it through Settings when finished. Previously copied customer keys stop working.

## Architecture and limitations

- `src/lonn/demo/store.py`: SQLModel persistence and reproducible seed generation, with customer IDs on usage records.
- `src/lonn/demo/policy.py`: validated account policy and model eligibility reasons; no fabricated compliance data.
- `src/lonn/demo/api.py`: demo login session, separate customer API-key authentication, account management and usage recording. The original unauthenticated generation routes are removed from this application.
- `src/lonn/api.py`: small reusable hooks pass a request-local model pool and executor into the existing classifier/ranking/fallback/escalation flow. No global model-pool mutation; concurrent requests use a settings snapshot.
- `src/lonn/executor.py`: accepts an explicit provider key, keeping account requests independent of environment mutations. Original callers retain environment/.env behavior. The demo deliberately uses only the presenter-configured credential.
- `frontend/`: portal and customer application, served from the same local origin after a Vite build.
- `sdk/python/`: existing sync/async SDK with optional customer-key authentication.

Model cost uses the configured input/output token rates. Estimated top-model cost is `model cost × (top model benchmark cost / selected model benchmark cost)`, using the supplied relative costs per 690 prompts. For multiple completed calls, the demo estimates each call against the same highest-quality eligible model, then sums model costs and baseline costs. The Lönn fee is `0.20 × max(0, estimated top-model cost − model cost)`, clamped once per request. Total customer cost is model cost plus that fee. Missing benchmark data leaves the estimate unavailable and charges no savings fee.

Both synthetic fixtures and new live requests use the same Decimal-based calculation. Request details and the SDK's `result.billing` distinguish model cost, estimated baseline/savings, Lönn fee, rate, fee basis, and customer total. Historical model/pool choices, benchmark values, and per-attempt token/rate snapshots are retained. Existing real recorded invoices remain unchanged; only outdated synthetic fixtures are refreshed at startup. OpenRouter's reported `usage.cost`, when available, is separate upstream telemetry and does not replace configured-rate model cost. Token counts are provider-reported for completed responses and summed across successful escalation attempts. Failed provider calls may have unreported usage or charges; request details expose `usage_complete=false` in that case. Prompt text, response text, raw errors and upstream keys are omitted from stored telemetry. A successful HTTP response can still be truncated; inspect `finish_reason` and answer-check details.

Model identifiers, prices and benchmark statistics are reused from Prototype 3.0 and have not been refreshed or live-verified. Historical benchmark estimates are not quality measurements or savings claims for customer requests.

The login uses database users, PBKDF2-SHA256 password hashes and expiring server sessions, with loopback binding, trusted-host checks and browser-origin checks. Seeded credentials are public demo credentials. This is not a production identity service: no SSO, recovery, invitation email or brute-force protection. Management sessions live in one process. Do not expose the demo on a public interface. The customer app holds keys only in page memory; a signed-in session can reconnect after refresh, while manually entered keys must be entered again. Backend role checks separate enterprise administration, team-member views and private accounts. The presenter configures the shared upstream credential through the seeded enterprise administrator. Private accounts cannot read or change that credential.

The existing generated-code verifier uses macOS sandbox-exec. A small compatibility fix permits the Python standard-library directory for uv-managed environments without opening the rest of the home folder. Other platforms retain Prototype 3.0's verification limitations.

## Development and verification

For frontend hot reload, run the backend as above, then `cd frontend && pnpm dev` (or `npm run dev`). Vite proxies API calls to port 8000. Production-like startup serves the built frontend directly; no second server is needed.

```bash
python -m unittest discover -s tests -v
cd frontend
pnpm run build
```

Customer-demo tests mock provider calls, test sync/async SDK integration, seeded persistence/reset, key rotation, missing credentials, credential isolation, failures, account policy enforcement and restricted escalation with aggregate token/cost reporting. They use disposable temporary databases. Real inference verification was intentionally deferred at the user's request; configure OpenRouter and follow the presenter walkthrough to perform it.


## Enterprise teams and user logins

The sign-in screen provides three seeded personas. All use the public **demo-only** password `LonnDemo2026!`:

| Login | Role | Scope |
| --- | --- | --- |
| `admin@nordic.demo` | Enterprise administrator | Nordic Studio settings, all organization usage, teams, API keys and users |
| `robin@nordic.demo` | Enterprise team member | assigned application usage, personal key, effective model pool; delegated admins may enforce their scope |
| `sam@personal.demo` | Private user | Separate personal account, key, routing settings and usage; no enterprise visibility |

Three seeded teams each have a default project and application. Under **Teams & access**, organization administrators manage teams and model restrictions. Under **Projects & apps**, they create projects and applications, view or rotate application keys, assign users to multiple teams, projects, and applications, and delegate team/application administrator roles. Project and application λ settings can inherit their parent, set a local default, or lock a value for lower levels. Users can inspect the effective λ and its policy source.

Each user has one personal key. Supply `application="engineering-app"` (or another authorized application name or ID) to `LonnRouter` or `AsyncLonnRouter`; the server validates team, project, and application membership. An application key is bound to one application and cannot select another. Legacy team keys are migrated to the corresponding seeded default application's key. Existing administrator and private account keys become personal keys and now require an application selector. Newly created teams receive a default project and application.

λ priority is organization enforcement → team enforcement → project lock → application lock → explicit request value → personal preference → application default → project default → default 0.20. Higher-level locks prevent editing lower-level settings. The existing router receives that literal value. An application key has no individual user, so user preference does not apply. Model pool restrictions from the organization and team are combined before selection, fallback and escalation.

The dashboard filters the existing usage records by organization, team, project, application, and user. Requests made with personal keys record the user; requests made with application keys do not invent one. Historical live records created before this feature may have unassigned project/application/user fields and remain unchanged. Seeded records have deterministic application assignments. Database changes are additive and preserve existing state; reset creates fresh demo keys.

## Cumulative spending comparison

For each completed request, the demo identifies the eligible, context-compatible model with the highest overall quality in the routing table's global row (an equal-weighted mean across measured task types). Equal quality is broken by benchmark cost, then provider and model ID. This pool-wide choice is independent of the request's task classification. The chosen model and eligible model IDs are saved with the request. No extra inference call is made.

The saved benchmark baseline uses **model cost before the Lönn fee**, avoiding a circular estimate. The relative costs come from the supplied benchmark of 690 prompts, and the calculation does not assume identical token counts across models. These benchmark-wide ratios may not match a customer's workload and do not establish measured savings.

Daily sums are accumulated chronologically. Estimated gross savings are baseline minus model cost, while estimated customer savings subtract total customer spending (including the Lönn fee); percentage customer savings are `100 × (1 − customer total/baseline)` when baseline is positive. Negative savings remain negative. Requests missing a safe historical pool-wide baseline or benchmark data are excluded from the comparable series and counted in the coverage note; ordinary usage totals still include them. Seeded requests are synthetic. Real historical recorded charges retain their original tariff.

The seeded traffic uses the existing router at λ = 0.20 for a varied task mix. Prototype 3.0 does not use its optional complexity signal by default, so the demo does not claim to select different models based on difficulty within the same task type.
