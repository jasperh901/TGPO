#!/usr/bin/env python3
"""Generate four responses per math question and run the paper's graders."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("math", "aime24", "amc22_23", "minerva", "olympiad")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/math/eval")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / "runtime/math"), CUDA_VISIBLE_DEVICES=a.gpus,
               RAY_USAGE_STATS_ENABLED="0", TOKENIZERS_PARALLELISM="false")
    env.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    output = a.output.resolve()
    for suite in SUITES:
        command = [sys.executable, "-u", "-m", "verl.trainer.main_generation",
                   "trainer.nnodes=1", f"trainer.n_gpus_per_node={len(a.gpus.split(','))}",
                   f"data.path={a.data_dir.resolve() / (suite + '.parquet')}",
                   f"data.output_path={output / (suite + '.parquet')}",
                   "data.n_samples=4", "data.sample_seeds=[0,1,2,3]", "data.batch_size=2048",
                   f"model.path={a.checkpoint.resolve()}", "rollout.temperature=0.6",
                   "rollout.seed=0", "rollout.response_length=2048", "rollout.prompt_length=1536",
                   "rollout.top_k=-1", "rollout.top_p=0.95", "rollout.gpu_memory_utilization=0.85",
                   "rollout.tensor_model_parallel_size=1", "rollout.max_num_batched_tokens=4096",
                   "rollout.max_num_seqs=128", "rollout.enforce_eager=true", "rollout.free_cache_engine=true"]
        if a.dry_run:
            print(command); continue
        if (output / f"{suite}.parquet").exists():
            p.error(f"Generation already exists: {output / (suite + '.parquet')}")
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="tgpo-ray-") as ray_dir:
            env["RAY_TEMP_DIR"] = ray_dir
            subprocess.run(command, env=env, cwd=ROOT / "runtime/math", check=True)
    command = [sys.executable, str(ROOT / "scripts/score_math_eval_robust.py"),
               "--generation-dir", str(output), "--output-dir", str(output / "scored"),
               "--expected-samples", "4", "--length-limit", "1024", "--symbolic-timeout", "10",
               "--retry-timeout", "60", "--controlled", "--require-unique-seeds"]
    if a.dry_run:
        print(command)
    else:
        subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
