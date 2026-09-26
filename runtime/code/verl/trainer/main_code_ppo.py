"""Matched multi-reward PPO entry point for the GDPO code task."""

from __future__ import annotations

import json
import base64
import os
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
import numpy as np
import ray
import torch
from omegaconf import open_dict

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.reward_score.gdpo_code import CodeGraderError, score_code_response


def score_code_inputs(scoring_inputs, target_length: int, num_workers: int,
                      score_fn=score_code_response, sandbox=False,
                      sandbox_timeout_per_response: float = 75.0,
                      sandbox_responses_per_process: int = 16):
    if target_length < 1 or num_workers < 1:
        raise ValueError("target_length and num_workers must be positive")
    if sandbox:
        return score_code_inputs_sandboxed(
            scoring_inputs,
            target_length,
            num_workers,
            timeout_per_response=sandbox_timeout_per_response,
            responses_per_process=sandbox_responses_per_process,
        )

    def score_one(values):
        response, ground_truth, response_tokens = values
        return score_fn(
            response=response,
            ground_truth=ground_truth,
            response_tokens=response_tokens,
            target_length=target_length,
        )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(score_one, scoring_inputs))


def _sandbox_command():
    from verl.utils.code_sandbox import sandbox_command
    return sandbox_command()


def score_code_inputs_sandboxed(scoring_inputs, target_length: int, num_workers: int,
                                timeout_per_response: float = 75.0,
                                responses_per_process: int = 16):
    """Score in read-only, networkless worker namespaces; fail closed on any loss."""
    if not scoring_inputs:
        return []
    if not np.isfinite(timeout_per_response) or timeout_per_response <= 0:
        raise ValueError("timeout_per_response must be finite and positive")
    if int(responses_per_process) < 1:
        raise ValueError("responses_per_process must be positive")
    shards = []
    for index, (response, ground_truth, response_tokens) in enumerate(scoring_inputs):
        if index % int(responses_per_process) == 0:
            shards.append([])
        shards[-1].append({
            "id": index,
            "response": response,
            "ground_truth": ground_truth,
            "response_tokens": int(response_tokens),
            "target_length": int(target_length),
        })

    command = _sandbox_command()

    def run_shard(tasks):
        # PRIME already bounds each generated program. This outer deadline is
        # deliberately wider and catches namespace/manager failures that would
        # otherwise suspend a complete PPO step forever.
        shard_timeout = 30.0 + timeout_per_response * len(tasks)
        def execute(batch):
            batch_payload = "".join(
                base64.b64encode(
                    json.dumps(task, allow_nan=False, ensure_ascii=False).encode("utf-8")
                ).decode("ascii") + "\n"
                for task in batch
            )
            try:
                completed = subprocess.run(
                    command,
                    input=batch_payload,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=30.0 + timeout_per_response * len(batch),
                )
            except subprocess.TimeoutExpired as error:
                raise CodeGraderError(
                    f"sandbox shard exceeded {30.0 + timeout_per_response * len(batch):.1f}s "
                    f"for {len(batch)} responses"
                ) from error
            if completed.returncode != 0:
                raise CodeGraderError(
                    f"sandbox worker exited {completed.returncode}: {completed.stderr[-1000:]}"
                )
            rows = []
            for line in completed.stdout.splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise CodeGraderError("sandbox worker emitted non-JSON output") from error
                if "error" in row:
                    raise CodeGraderError(
                        f"sandbox task {row.get('id')} failed with {row.get('error_type')}: "
                        f"{row['error']}"
                    )
                rows.append(row)
            if len(rows) != len(batch):
                raise CodeGraderError(
                    f"sandbox worker returned {len(rows)}/{len(batch)} results"
                )
            return rows

        try:
            return execute(tasks)
        except CodeGraderError:
            # Long generated programs can expose a transient line-protocol
            # truncation in the namespace boundary. Retry each record in its
            # own short-lived worker; scoring semantics remain unchanged.
            if len(tasks) == 1:
                raise
            rows = []
            for task in tasks:
                rows.extend(execute([task]))
            return rows

    with ThreadPoolExecutor(max_workers=min(int(num_workers), len(shards))) as executor:
        shard_results = list(executor.map(run_shard, shards))
    ordered = [None] * len(scoring_inputs)
    for rows in shard_results:
        for row in rows:
            index = int(row["id"])
            if not 0 <= index < len(ordered) or ordered[index] is not None:
                raise CodeGraderError(f"sandbox returned invalid or duplicate task id {index}")
            ordered[index] = row["score"]
    if any(score is None for score in ordered):
        raise CodeGraderError("sandbox result set is incomplete")
    return ordered


class CodeRewardManager:
    def __init__(self, tokenizer, target_length=4000, num_examine=0, num_workers=32,
                 sandbox=True, sandbox_timeout_per_response=75.0,
                 sandbox_responses_per_process=16):
        self.tokenizer = tokenizer
        self.target_length = int(target_length)
        self.num_examine = int(num_examine)
        self.num_workers = int(num_workers)
        self.sandbox = bool(sandbox)
        self.sandbox_timeout_per_response = float(sandbox_timeout_per_response)
        self.sandbox_responses_per_process = int(sandbox_responses_per_process)

    def __call__(self, data: DataProto, step: int):
        shape = data.batch["responses"].shape
        total_tensor = torch.zeros(shape, dtype=torch.float32)
        component_tensors = {
            "pass": torch.zeros(shape, dtype=torch.float32),
            "conditioned_length": torch.zeros(shape, dtype=torch.float32),
            "bug_free": torch.zeros(shape, dtype=torch.float32),
        }
        scoring_inputs = []
        for index in range(len(data)):
            item = data[index]
            prompt_length = item.batch["prompts"].shape[-1]
            valid_response_length = int(item.batch["attention_mask"][prompt_length:].sum().item())
            response_ids = item.batch["responses"][:valid_response_length]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
            scoring_inputs.append((response, ground_truth, valid_response_length))

        scores = score_code_inputs(
            scoring_inputs,
            target_length=self.target_length,
            num_workers=self.num_workers,
            sandbox=self.sandbox,
            sandbox_timeout_per_response=self.sandbox_timeout_per_response,
            sandbox_responses_per_process=self.sandbox_responses_per_process,
        )
        for index, (score, scoring_input) in enumerate(zip(scores, scoring_inputs, strict=True)):
            response, _, response_tokens = scoring_input
            target_position = max(response_tokens - 1, 0)
            total_tensor[index, target_position] = score["reward"]
            for name, tensor in component_tensors.items():
                tensor[index, target_position] = score[name]
            if index < self.num_examine:
                print(
                    f"step={step} sample={index} tokens={response_tokens} "
                    f"pass={score['pass']:.3f} conditioned={score['conditioned_length']:.0f} "
                    f"bug_free={score['bug_free']:.0f}\n{response}",
                    flush=True,
                )

        means = {
            name: float(tensor.sum(dim=-1).mean().item())
            for name, tensor in component_tensors.items()
        }
        print(
            f"code reward audit step={step} responses={len(data)} grader_failures=0 "
            f"pass={means['pass']:.6f} conditioned_length={means['conditioned_length']:.6f} "
            f"bug_free={means['bug_free']:.6f}",
            flush=True,
        )
        return total_tensor, component_tensors


@hydra.main(config_path="config", config_name="code_ppo_scppo_8gpu_400", version_base=None)
def main(config):
    if not ray.is_initialized():
        worker_env = {
            "TOKENIZERS_PARALLELISM": os.getenv("TOKENIZERS_PARALLELISM", "false"),
            "NCCL_DEBUG": os.getenv("NCCL_DEBUG", "WARN"),
            "EXPERIMENT_NAME": config.trainer.experiment_name,
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        }
        for name in (
            "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "NCCL_IB_DISABLE",
            "NCCL_P2P_DISABLE", "PYTORCH_CUDA_ALLOC_CONF", "VLLM_ATTENTION_BACKEND",
            "CUDA_DEVICE_MAX_CONNECTIONS", "OMP_NUM_THREADS",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        ):
            if name in os.environ:
                worker_env[name] = os.environ[name]
        ray_kwargs = {
            "address": "local",
            "include_dashboard": False,
            "runtime_env": {"env_vars": worker_env},
        }
        if os.getenv("RAY_TEMP_DIR"):
            ray_kwargs["_temp_dir"] = os.environ["RAY_TEMP_DIR"]
        ray.init(**ray_kwargs)
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    seed = int(config.trainer.get("seed", 1))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    from omegaconf import OmegaConf
    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

    print(OmegaConf.to_yaml(config, resolve=True), flush=True)
    OmegaConf.resolve(config)
    resume_path = config.trainer.get('resume_from_checkpoint', None)
    if resume_path:
        with open_dict(config.actor_rollout_ref.actor):
            config.actor_rollout_ref.actor.resume_from_checkpoint = str(resume_path)
    model_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(model_path)
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }
    pool_id = "code_pool"
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec={pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
        mapping={Role.ActorRollout: pool_id, Role.Critic: pool_id, Role.RefPolicy: pool_id},
    )
    reward_fn = CodeRewardManager(
        tokenizer=tokenizer,
        target_length=config.algorithm.get("length_limit", 4000),
        num_examine=config.trainer.get("reward_samples_to_print", 0),
        num_workers=config.trainer.get("reward_workers", 32),
        sandbox=config.trainer.get("reward_sandbox", True),
        sandbox_timeout_per_response=config.trainer.get(
            "reward_sandbox_timeout_per_response", 75.0
        ),
        sandbox_responses_per_process=config.trainer.get(
            "reward_sandbox_responses_per_process", 16
        ),
    )
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
        reward_fn=reward_fn,
        val_reward_fn=None,
    )
    trainer.init_workers()
    trainer.fit()


if __name__ == "__main__":
    main()
