#!/usr/bin/env python3
"""Fetch the BFCL-v3 scorer and install the paper's local ToolRL adapter."""
import argparse
from pathlib import Path
import shutil
import subprocess

REVISION = "ea13468e4423454d0c213704fb87cf7cb3990433"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "third_party/BFCL-v3")
    args = parser.parse_args()
    repo = args.destination.resolve()
    if not (repo / ".git").is_dir():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                        "https://github.com/ShishirPatil/gorilla.git", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", REVISION], check=True)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != REVISION:
        raise ValueError(f"BFCL must be checked out at {REVISION}; found {head}")
    patch = ROOT / "eval/bfcl/adapter.patch"
    already = subprocess.run(["git", "-C", str(repo), "apply", "--reverse", "--check", str(patch)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not already:
        subprocess.run(["git", "-C", str(repo), "apply", "--check", str(patch)], check=True)
        subprocess.run(["git", "-C", str(repo), "apply", str(patch)], check=True)
    shutil.copy2(ROOT / "eval/bfcl/toolrl_gdpo.py",
                 repo / "berkeley-function-call-leaderboard/bfcl_eval/model_handler/local_inference/toolrl_gdpo.py")
    print(f"BFCL-v3 and TGPO adapter ready: {repo}")


if __name__ == "__main__":
    main()
