"""List NVIDIA NIM chat/instruct models that have a known list price.

Fetches the live NIM catalog, drops anything that isn't a general chat/instruct
model by its id alone, then tries to price each survivor by matching it against
litellm.model_cost under together_ai/, groq/, deepinfra/, fireworks_ai/.

Writes models.json (priced) and models_unpriced.json (no match found).
"""
import json
import os
import re

import requests
import litellm

API_KEY = os.environ["NVIDIA_NIM_API_KEY"]
CATALOG_URL = "https://integrate.api.nvidia.com/v1/models"
PRICE_PREFIXES = ["together_ai/", "groq/", "deepinfra/", "fireworks_ai/"]
FIREWORKS_ACCOUNT_PATHS = ["accounts/fireworks/models/", "accounts/fireworks/routers/"]

# Substrings in a NIM model id that mark it as NOT a general chat/instruct model.
NOT_CHAT_SUBSTRINGS = [
    "embed", "rerank", "guard", "safety", "vision", "-vlm", "vlm-",
    "kosmos", "neva", "vila", "fuyu", "cosmos", "clip", "deplot",
    "diffusion", "detector", "calibration", "parse", "omni", "ocr",
    "speech", "asr", "tts", "translate", "nemoretriever", "reward",
]


def fetch_catalog() -> list[str]:
    resp = requests.get(CATALOG_URL, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=30)
    resp.raise_for_status()
    return [m["id"] for m in resp.json()["data"]]


def is_chat_model(model_id: str) -> bool:
    low = model_id.lower()
    return not any(s in low for s in NOT_CHAT_SUBSTRINGS)


def normalize(name: str) -> str:
    name = name.lower()
    name = re.sub(r"(\d)p(\d)", r"\1\2", name)  # fireworks spells "2.5" as "2p5"
    return re.sub(r"[^a-z0-9]", "", name)


def build_price_lookup() -> dict[str, dict]:
    """normalized leaf model name -> cheapest matching litellm.model_cost entry"""
    lookup: dict[str, list[dict]] = {}
    for key, info in litellm.model_cost.items():
        prefix = next((p for p in PRICE_PREFIXES if key.startswith(p)), None)
        if prefix is None:
            continue
        leaf = key[len(prefix):]
        for account_path in FIREWORKS_ACCOUNT_PATHS:
            if leaf.startswith(account_path):
                leaf = leaf[len(account_path):]
        if info.get("input_cost_per_token") is None:
            continue
        lookup.setdefault(normalize(leaf), []).append({"key": key, **info})
    return {n: min(hits, key=lambda h: h["input_cost_per_token"]) for n, hits in lookup.items()}


def main():
    catalog = fetch_catalog()
    chat_models = [m for m in catalog if is_chat_model(m)]
    price_lookup = build_price_lookup()

    priced, unpriced = [], []
    for model_id in chat_models:
        _, _, leaf = model_id.partition("/")
        hit = price_lookup.get(normalize(model_id)) or price_lookup.get(normalize(leaf))
        if hit is None:
            unpriced.append(model_id)
            continue
        priced.append({
            "nim_model_id": f"nvidia_nim/{model_id}",
            "matched_provider_key": hit["key"],
            "usd_per_m_in": round(hit["input_cost_per_token"] * 1_000_000, 4),
            "usd_per_m_out": round(hit["output_cost_per_token"] * 1_000_000, 4),
        })
    priced.sort(key=lambda m: m["usd_per_m_in"])

    with open("models.json", "w") as f:
        json.dump(priced, f, indent=2)
    with open("models_unpriced.json", "w") as f:
        json.dump(sorted(unpriced), f, indent=2)

    print(f"catalog:       {len(catalog)} models")
    print(f"chat/instruct: {len(chat_models)}")
    print(f"priced:        {len(priced)}")
    print(f"unpriced:      {len(unpriced)}")
    print()
    print(f"{'nim_model_id':50s} {'$/M in':>10s} {'$/M out':>10s}")
    for m in priced:
        print(f"{m['nim_model_id']:50s} {m['usd_per_m_in']:>10.4f} {m['usd_per_m_out']:>10.4f}")


if __name__ == "__main__":
    main()
