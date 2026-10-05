"""Upload a fine-tuned Laya checkpoint to a private Hugging Face model repo.

Usage:
    hf auth login                       # once, with a write token
    python3 scripts/push_laya.py --repo <hf-user>/laya-lonn

Uploads the directory written by scripts/finetune_laya.py (weights, config, tokenizer,
encoder config, metrics.json and the model card). The repo is created private if it does
not exist; an existing repo keeps its visibility. Colleagues need read access and
`hf auth login`, then set LONN_LAYA_MODEL=<repo> in .env.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("rl_agent_config.json", "model.safetensors", "tokenizer", "encoder")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="Hugging Face repo id, e.g. <user>/laya-lonn")
    parser.add_argument("--dir", type=Path, default=ROOT / "models/laya-lonn")
    parser.add_argument("--message", default="Upload fine-tuned Laya checkpoint")
    args = parser.parse_args()

    missing = [name for name in REQUIRED if not (args.dir / name).exists()]
    if missing:
        parser.error(f"{args.dir} is not a complete checkpoint (missing {', '.join(missing)}); "
                     "run scripts/finetune_laya.py first")

    from huggingface_hub import HfApi

    api = HfApi()
    url = api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    commit = api.upload_folder(folder_path=str(args.dir), repo_id=args.repo, repo_type="model",
                               commit_message=args.message)
    print(f"Uploaded {args.dir} to {url}")
    print(f"Commit: {commit.oid}")


if __name__ == "__main__":
    main()
