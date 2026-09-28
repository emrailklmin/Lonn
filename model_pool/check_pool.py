"""Verify every model in pool.yaml (active or not) can do schema-valid tool calls.

10 trials/model through LiteLLM, backoff on 429, abort-and-mark-unavailable on 404.
Writes gate_pass_rate and verified_at back into pool.yaml. Does not pick, reorder,
or otherwise edit any other field.
"""
import datetime
import json
import time

import litellm
import yaml

litellm.suppress_debug_info = True

POOL_PATH = "pool.yaml"
N_TRIALS = 10
MAX_ATTEMPTS_PER_TRIAL = 6
BASE_BACKOFF = 1.0
INTER_TRIAL_SLEEP = 0.3

TOOL = {
    "type": "function",
    "function": {
        "name": "get_order_details",
        "description": "Get the status and details of an order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order id, e.g. '#W0000000'."},
            },
            "required": ["order_id"],
        },
    },
}
USER_MSG = "Can you check the status of my order? The order id is #W1234567."
EXPECTED_PARAMS = set(TOOL["function"]["parameters"]["properties"].keys())


def run_trial(model: str):
    """Returns (verdict, n_429, latency_ms, hard_unavailable)."""
    attempt, backoff, n_429 = 0, BASE_BACKOFF, 0
    while True:
        attempt += 1
        t0 = time.time()
        try:
            resp = litellm.completion(
                model=model, messages=[{"role": "user", "content": USER_MSG}],
                tools=[TOOL], tool_choice="auto", timeout=30,
            )
            latency_ms = (time.time() - t0) * 1000
            tool_calls = getattr(resp.choices[0].message, "tool_calls", None) or []
            if len(tool_calls) != 1:
                return "FAIL", n_429, latency_ms, False
            tc = tool_calls[0]
            if tc.function.name != TOOL["function"]["name"]:
                return "FAIL", n_429, latency_ms, False
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                return "FAIL", n_429, latency_ms, False
            if set(args.keys()) != EXPECTED_PARAMS:
                return "FAIL", n_429, latency_ms, False
            return "PASS", n_429, latency_ms, False
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            status = getattr(e, "status_code", None)
            if status == 404 or "not found" in str(e).lower():
                return "FAIL", n_429, latency_ms, True
            if status == 429 or (status and 500 <= status < 600) or "timeout" in str(e).lower():
                n_429 += 1 if status == 429 else 0
                if attempt >= MAX_ATTEMPTS_PER_TRIAL:
                    return "FAIL", n_429, latency_ms, False
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
            return "FAIL", n_429, latency_ms, False


def check_model(model_id: str) -> dict:
    n_pass = n_429_total = 0
    latencies = []
    for _ in range(N_TRIALS):
        verdict, n_429, latency_ms, unavailable = run_trial(model_id)
        n_429_total += n_429
        if unavailable:
            return {"gate_pass_rate": 0.0, "mean_latency_ms": None, "n_429": n_429_total, "unavailable": True}
        if verdict == "PASS":
            n_pass += 1
            latencies.append(latency_ms)
        time.sleep(INTER_TRIAL_SLEEP)
    mean_latency = sum(latencies) / len(latencies) if latencies else None
    return {
        "gate_pass_rate": round(n_pass / N_TRIALS, 3),
        "mean_latency_ms": round(mean_latency, 1) if mean_latency else None,
        "n_429": n_429_total,
        "unavailable": False,
    }


def main():
    with open(POOL_PATH) as f:
        pool = yaml.safe_load(f)

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rows = []
    for entry in pool:
        result = check_model(entry["id"])
        entry["gate_pass_rate"] = result["gate_pass_rate"]
        entry["verified_at"] = now
        rows.append({**entry, **result})

    with open(POOL_PATH, "w") as f:
        yaml.dump(pool, f, sort_keys=False, default_flow_style=False)

    header = f"{'id':50s} {'tier':10s} {'active':7s} {'pass_rate':10s} {'mean_lat_ms':12s} {'n_429':6s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        status = "UNAVAILABLE" if r["unavailable"] else f"{r['gate_pass_rate']:.2f}"
        lat = f"{r['mean_latency_ms']:.0f}" if r["mean_latency_ms"] else "-"
        print(f"{r['id']:50s} {r['tier']:10s} {str(r['active']):7s} {status:10s} {lat:12s} {r['n_429']:<6d}")


if __name__ == "__main__":
    main()
