"""Discover all NVIDIA catalog IDs, without price or model-name filtering.

Discovery is not proof of chat availability. Output is a candidate inventory,
not routing configuration. No inference calls or pool changes are performed.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.request import Request, urlopen

CATALOG_URL = 'https://integrate.api.nvidia.com/v1/models'
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'reports' / 'model_catalog.json'


def build_inventory(catalog: dict) -> dict:
    models = []
    seen = set()
    for item in catalog['data']:
        model_id = item['id']
        if not isinstance(model_id, str) or not model_id or model_id in seen:
            raise ValueError('Catalog contains an invalid or duplicate model ID')
        seen.add(model_id)
        models.append({'id': 'nvidia_nim/' + model_id,
                       'publisher': item.get('owned_by'), 'status': 'untested'})
    return {'discovered_at': datetime.now(timezone.utc).isoformat(),
            'source': CATALOG_URL, 'models': sorted(models, key=lambda m: m['id'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    request = Request(CATALOG_URL, headers={'Accept': 'application/json'})
    with urlopen(request, timeout=30) as response:
        inventory = build_inventory(json.load(response))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2) + '\n')
    print(f"Saved {len(inventory['models'])} candidates to {args.output}")


if __name__ == '__main__':
    main()
