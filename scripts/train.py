#!/usr/bin/env python3
"""Launch one published TGPO configuration with portable paths."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("math_1.5b", "math_7b", "code_1.5b", "tool_1.5b", "tool_3b")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("profile", choices=PROFILES)
    p.add_argument("--algorithm", choices=("tgpo", "gdpo", "dvao"), default="tgpo")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--model-dir", type=Path, default=ROOT / "models")
    p.add_argument("--output", type=Path)
    p.add_argument("--gpus", help="Comma-separated physical GPU IDs")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--override", action="append", default=[], help="Hydra key=value; repeatable")
    a = p.parse_args()
    from omegaconf import OmegaConf
    task = a.profile.split("_")[0]
    runtime = ROOT / "runtime" / ("tool_3b" if a.profile == "tool_3b" else task)
    output = (a.output or ROOT / "outputs" / f"{a.profile}_{a.algorithm}").resolve()
    env = os.environ.copy()
    env.update(TGPO_DATA_DIR=str(a.data_dir.resolve()), TGPO_MODEL_DIR=str(a.model_dir.resolve()),
               TGPO_OUTPUT_DIR=str(output), PYTHONPATH=os.pathsep.join([str(runtime), str(ROOT / "src")]),
               RAY_USAGE_STATS_ENABLED="0", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1")
    env.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    for key in ("WITHLENGTH", "REFINEDREWARD", "COARSEREWARD", "STRICTMATCH", "CORRECTMAX1",
                "MAX1STEP30MAX3", "SCHEDULEREWARD", "SCHEDULELENGTH"):
        env[key] = "0"
    overrides = list(a.override)
    if a.gpus:
        devices = a.gpus.split(",")
        if len(set(devices)) != len(devices) or any(not x.isdigit() for x in devices):
            p.error("--gpus must contain unique numeric GPU IDs")
        env["CUDA_VISIBLE_DEVICES"] = a.gpus
        overrides.append(f"trainer.n_gpus_per_node={len(devices)}")
    if a.algorithm != "tgpo":
        overrides += [f"algorithm.adv_estimator={a.algorithm}",
                      "actor_rollout_ref.actor.moco_cagrad_shadow.enabled=false",
                      "actor_rollout_ref.actor.moco_cagrad_shadow.apply_update=false"]
        if task == "tool":
            overrides.append("++actor_rollout_ref.actor.historical_tool_objective=true")
    os.environ.update({k: env[k] for k in ("TGPO_DATA_DIR", "TGPO_MODEL_DIR", "TGPO_OUTPUT_DIR")})
    cfg = OmegaConf.load(ROOT / "configs" / f"{a.profile}.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([s.lstrip("+") for s in overrides]))
    OmegaConf.resolve(cfg)
    # Freeze interpolations before passing the config to Ray workers.
    command = [sys.executable, "-u", "-m", f"verl.trainer.main_{task}_ppo",
               "--config-path", str(output / "config"), "--config-name", "train"]
    if a.dry_run:
        print(OmegaConf.to_yaml(cfg)); print("Command:", json.dumps(command)); return
    if not Path(cfg.data.train_files).is_file():
        p.error(f"Missing dataset: {cfg.data.train_files}; run scripts/download_data.py")
    if not (Path(cfg.actor_rollout_ref.model.path) / "config.json").is_file():
        p.error(f"Missing model: {cfg.actor_rollout_ref.model.path}; run scripts/download_models.py")
    if output.exists() and any(output.iterdir()):
        p.error(f"Output directory already contains files: {output}; choose a new --output")
    (output / "config").mkdir(parents=True)
    OmegaConf.save(cfg, output / "config/train.yaml")
    with (output / "train.log").open("w") as log:
        process = subprocess.Popen(command, cwd=runtime, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True); log.write(line); log.flush()
        raise SystemExit(process.wait())


if __name__ == "__main__":
    main()
