#!/usr/bin/env python3
"""Download published base checkpoints, or only a tokenizer for preprocessing."""
import argparse
import json
from pathlib import Path

MODELS = {
    "math_1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "math_7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "code_1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "tool_1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "tool_3b": "Qwen/Qwen2.5-3B-Instruct",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profiles", nargs="+", choices=list(MODELS) + ["all"])
    p.add_argument("--output", "--model-dir", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    p.add_argument("--tokenizer-only", action="store_true")
    a = p.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    names = MODELS if "all" in a.profiles else a.profiles
    for model in sorted({MODELS[name] for name in names}):
        revision = HfApi().model_info(model).sha
        destination = a.output / model.split("/")[-1]
        patterns = ["*.json", "*.model", "*.tiktoken", "*.txt"] if a.tokenizer_only else None
        snapshot_download(model, revision=revision, local_dir=destination, allow_patterns=patterns)
        (destination / "download_revision.json").write_text(json.dumps({"repository": model, "revision": revision}, indent=2))
        print(destination)


if __name__ == "__main__":
    main()
