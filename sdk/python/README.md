# Lonn Router Python SDK

Call a running Lonn API from a Python app or benchmark runner. Requires Python
3.10+, HTTPX, and Pydantic. This client does not install the inference server,
load models, or require a NVIDIA key on the client device.

## Install

From a checkout of the Lonn repository on the client machine:

```bash
python -m pip install ./sdk/python
```

For SDK development, use `python -m pip install -e ./sdk/python`. The package
is not published to PyPI; `pip install lonnrouter` is not the installation path
for this implementation. You can also distribute a wheel built with
`python -m pip wheel --no-deps ./sdk/python --wheel-dir dist`.

## Generate a response

```python
from lonnrouter import LonnRouter

with LonnRouter(base_url="http://192.168.1.42:8000") as router:
    result = router.generate("Explain binary search", lam=0.50)
    print(result.content)
    print(result.model.id)
    print(result.usage.total_tokens, result.usage.total_cost)
    print(result.finish_reason, result.total_latency_ms)
```

Replace the example IP with the running server's address. Alternatively set
`LONN_BASE_URL` and construct `LonnRouter()` with no arguments. The default is
`http://127.0.0.1:8000`. All model routing and execution happen on the server.
The server needs the key for the selected model's provider:
`OPENROUTER_API_KEY` for `openrouter/...` models or `NVIDIA_NIM_API_KEY` for
`nvidia_nim/...` models. The SDK does not start or deploy the server.

`generate` checks the answer on the server before returning it (cut off, empty, or
code that fails the prompt's worked examples). If it fails and a second model is
worth its expected cost, that model answers instead: `result.escalated`,
`result.attempts` and `result.total_cost_usd` show what happened and what it cost.
Pass `verify=False` to skip the check.

`lam` (lambda) defaults to 0.50 and ranges from 0 (cost only) to 1
(quality only). It weights expected aptitude against normalized benchmark cost.
`result.estimates` lists each ranked candidate's expected aptitude and cost. The returned
`model` contains the selected model's metadata. `usage` reports tokens and
estimated costs. `finish_reason == "length"` signals a truncated response and
may accompany empty content. The complete wire response is available through
`result.model_dump()`; execution metadata is also under `result.response`.

## Route without generating

```python
from lonnrouter import LonnRouter, Signal

with LonnRouter() as router:
    selection = router.route("Write a sorting function", lam=0.65)
    print(selection.model.id, selection.routing_latency_ms)

    # Isolate the router from classifier inference for repeatable benchmarks.
    selection = router.route(
        "Write a sorting function",
        signal=Signal(task_scores={"code_function": 1.0}),
    )
    print(router.health().status)
    print([model.id for model in router.models()])
```

Both `route` and `generate` accept `signal` as a `Signal` or matching dictionary.
A signal holds task-type probabilities (`task_scores`, summing to 1, with IDs
from the server's `config/task_types.yaml`) and an optional `complexity`. Omit it
for automatic classification. The server validates request parameters;
invalid requests raise `LonnAPIError` with `status_code == 422`.

## Async apps

```python
import asyncio
from lonnrouter import AsyncLonnRouter

async def main():
    async with AsyncLonnRouter(base_url="http://192.168.1.42:8000") as router:
        result = await router.generate("Explain binary search", lam=0.50)
        print(result.content)

asyncio.run(main())
```

The async client has the same `route`, `generate`, `models`, and `health`
methods. Reuse one client across requests in an app and call `close()` (sync)
or `await aclose()` (async) during shutdown, or use the context managers shown
above. An injected `http_client` is owned and closed by the caller. Connection
reuse and context management follow the
[HTTPX client model](https://www.python-httpx.org/advanced/clients/).

## Timeouts and failures

```python
from lonnrouter import LonnAPIError, LonnConnectionError, LonnRouter

with LonnRouter(request_timeout=300) as router:
    try:
        result = router.generate("Explain binary search",
                                 max_tokens=4096, provider_timeout=120)
    except LonnAPIError as error:
        print("Server returned", error.status_code)
    except LonnConnectionError:
        print("Check the Lonn server address and connectivity")
```

`request_timeout` is the HTTPX network-operation timeout (default 300 seconds),
not a total wall-clock deadline. `provider_timeout` is sent to the server's
model executor (default 120 seconds, maximum 600). Allow extra client time for
classification, particularly the classifier's first-use loading. The first automatic
`route` call can be used to warm the classifier before a benchmark.

- `LonnAPIError`: non-success HTTP response; inspect `status_code` (422 invalid
  input, 429 rate limit, 502 execution failure, 503 routing failure, 504 provider
  timeout).
- `LonnConnectionError`: network or transport failure.
- `LonnTimeoutError`: subclass of `LonnConnectionError` for network timeouts.
- `LonnResponseError`: a successful HTTP response had an unexpected format.
- `LonnError`: base class for all these request/response errors.

Exceptions omit response bodies that could contain prompt data. There are no
automatic retries. A timed-out or disconnected request may still execute on
the server, so retrying generation can duplicate provider usage.

For another device on a trusted LAN, start the server with
`python3 -m uvicorn lonn.api:app --app-dir src --host 0.0.0.0 --port 8000` from
the repository root in its server environment. The API currently has no
authentication: use a trusted private network or an SSH tunnel. Through a
tunnel such as `ssh -N -L 8001:127.0.0.1:8000 USER@HOST`, use
`base_url="http://127.0.0.1:8001"` and keep the server bound to loopback.

## Local customer demo authentication and pricing

Use `LonnRouter(api_key="lonn_demo_…", base_url="http://127.0.0.1:8000")`
(or `AsyncLonnRouter`) with the customer key from the portal. Both clients also
read `LONN_API_KEY`. Never pass an OpenRouter credential to the customer SDK.
The demo server uses `lam` as a request preference unless an organization, team,
project, or application administrator enforces a value; enforced values take precedence.

Customer-demo `total_cost_usd` is model token cost plus 20% of positive estimated
savings against the highest-quality eligible model. Benchmark ratios estimate
that baseline from each completed model call's cost. `result.billing` separates
`model_cost_usd`, `service_fee_usd`, `total_cost_usd`, and the estimated baseline
and savings. Missing benchmark data produces no savings fee. `cost_basis="calculated"`
identifies this tariff; upstream reported `provider_cost` remains separate telemetry. See the repository's
[customer demo guide](../../docs/customer_demo.md).

## Organization applications

A personal key must select an authorized application by name or ID:

```python
with LonnRouter(api_key="lonn_personal_...", application="engineering-app") as client:
    print(client.generate("Explain binary search.").content)
```

An application key is bound to its application and needs no `application`
argument. Application keys cannot select another application. The async client
supports the same `application` argument. Set `LONN_APPLICATION` as an
alternative. The local server resolves enforced λ in organization → team → project → application order;
otherwise an explicit request `lam` overrides the saved user preference, followed by
unlocked application and project defaults and finally 0.50.

## Model discovery and direct generation (customer demo API)

`client.model_pool()` lists active, measured models allowed by organization and
team model settings before privacy restrictions. `client.models()` lists only
models currently eligible after privacy filtering. Both return an empty list
when there are no matches. Pass `model=model_id` to `client.generate()` to call
one eligible model without falling back to another. The server rejects models
outside the eligible pool. These additions are also available on the async client.
