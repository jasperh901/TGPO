#!/usr/bin/env python3
"""Download and prepare the three paper datasets from their public sources."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PRIME_REVISION = "18ad596f08d487bb546d80d738d99ec697bd2e75"
TOOLRL_REVISION = "8cee13ec0ca72f0461da372a93a6fd8140dbb840"
MATH_FILES = {
    "AI-MO/aimo-validation-aime/aimo-validation-aime.jsonl": "7db6f782e60f62dafcc492d5c12f59e2a4e7ae74544d98224c5e25afdff32d5a",
    "AI-MO/aimo-validation-amc/aimo-validation-amc.jsonl": "f4d4c28866b1b7dee913538f5bf9760b69532fe0c25eb1b765905546bd61846e",
    "math500/math_test_cleaned.json": "8b7943482dadca3c0c9819db270aee203ab6d51a461e6ff59b99d5edc7f529ec",
    "minerva_math/test.jsonl": "0e656b430d3a0a1dab6aa853f453ba3805e28e2cabe4fc72486a897cc09d8ba2",
    "olympiadbench/test.jsonl": "bb8c1e38a16ea9eacb77530315a674d883f267fbff79183effc53681b4fcc416",
}


def download(url, output, expected=None):
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        temporary = output.with_suffix(output.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as target:
            shutil.copyfileobj(response, target)
        temporary.replace(output)
    actual = hashlib.sha256(output.read_bytes()).hexdigest()
    if expected and actual != expected:
        raise ValueError(f"Dataset checksum mismatch: {output}")
    return actual


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=("math", "code", "tool", "all"))
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--model-dir", type=Path, default=ROOT / "models")
    a = p.parse_args()
    tasks = ("math", "code", "tool") if a.task == "all" else (a.task,)
    from huggingface_hub import hf_hub_download
    raw = a.data_dir.resolve() / "raw"
    def run(script, *args):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)], check=True)
    if "math" in tasks:
        source = Path(hf_hub_download("agentica-org/DeepScaleR-Preview-Dataset", "deepscaler.json",
                                    repo_type="dataset", revision="b6ae8c6", local_dir=raw / "deepscaler"))
        if hashlib.sha256(source.read_bytes()).hexdigest() != "4e9e7b18248d982b4498a54a6bdb37b1bb653cf5f77bc309fa67a6c2533c8ba4":
            raise ValueError("DeepScaleR source differs from the paper dataset")
        run("prepare_math_data.py", "--source", source, "--model", a.model_dir / "DeepSeek-R1-Distill-Qwen-1.5B",
            "--output-dir", a.data_dir / "math")
        for name, digest in MATH_FILES.items():
            download(f"https://raw.githubusercontent.com/PRIME-RL/PRIME/{PRIME_REVISION}/eval/data/{name}",
                     raw / "math-eval" / name, digest)
        run("prepare_math_eval_data.py", "--source-root", raw / "math-eval", "--output-dir", a.data_dir / "math/eval")
    if "code" in tasks:
        for split in ("train", "validation"):
            hf_hub_download("PRIME-RL/Eurus-2-RL-Data", f"{split}.parquet", repo_type="dataset",
                            revision="9776b13", local_dir=raw / "eurus")
        # The historical filter used the 7B tokenizer for both model scales.
        tokenizer = a.model_dir / "DeepSeek-R1-Distill-Qwen-7B"
        if not (tokenizer / "tokenizer.json").exists():
            raise ValueError("Run scripts/download_models.py math_7b --tokenizer-only before preparing code data")
        run("prepare_code_data.py", "--source", raw / "eurus", "--tokenizer", tokenizer,
            "--output", a.data_dir / "code")
    if "tool" in tasks:
        download(f"https://raw.githubusercontent.com/qiancheng0/ToolRL/{TOOLRL_REVISION}/dataset/rlla_4k/train.parquet",
                 a.data_dir / "tool/train.parquet", "f2e728ad4a379550887711870db5f97154a0ecc45efbe5d43db8a99dc3b9e1c8")
        print(a.data_dir / "tool/train.parquet")


if __name__ == "__main__":
    main()
