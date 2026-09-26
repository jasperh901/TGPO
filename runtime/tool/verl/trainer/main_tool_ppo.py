import os
import random

import hydra
import numpy as np
import ray
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.reward_score import rlla


class ToolRewardManager:
    """Exact ToolRL format/correctness reward adapter with auditable failures."""

    def __init__(self, tokenizer, num_examine=0):
        self.tokenizer = tokenizer
        self.num_examine = int(num_examine)

    def __call__(self, data: DataProto, step: int):
        shape = data.batch['responses'].shape
        reward_tensor = torch.zeros(shape, dtype=torch.float32)
        format_tensor = torch.zeros(shape, dtype=torch.float32)
        correctness_tensor = torch.zeros(shape, dtype=torch.float32)
        length_tensor = torch.zeros(shape, dtype=torch.float32)
        failures = []

        for index in range(len(data)):
            item = data[index]
            prompt_ids = item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(item.batch['attention_mask'][:prompt_length].sum().item())
            valid_response_length = int(item.batch['attention_mask'][prompt_length:].sum().item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            valid_response_ids = item.batch['responses'][:valid_response_length]
            sequence = torch.cat((valid_prompt_ids, valid_response_ids))
            sequence_text = self.tokenizer.decode(sequence)
            ground_truth = item.non_tensor_batch['reward_model']['ground_truth']
            try:
                score, format_score, correctness_score, length_score = rlla.compute_score(
                    solution_str=sequence_text,
                    ground_truth=ground_truth,
                    step=step,
                )
            except Exception as error:
                failures.append((index, error))
                continue

            target_position = max(valid_response_length - 1, 0)
            reward_tensor[index, target_position] = score
            format_tensor[index, target_position] = format_score
            correctness_tensor[index, target_position] = correctness_score
            length_tensor[index, target_position] = length_score
            if index < self.num_examine:
                print(sequence_text, flush=True)

        print(
            f'tool reward audit step={step} responses={len(data)} '
            f'grader_failures={len(failures)}',
            flush=True,
        )
        if failures:
            preview = ', '.join(
                f'index={index}:{type(error).__name__}' for index, error in failures[:8]
            )
            raise RuntimeError(
                f'Tool reward grader produced {len(failures)} unresolved errors ({preview})'
            )
        return reward_tensor, format_tensor, correctness_tensor, length_tensor


@hydra.main(config_path='config', config_name='tool_ppo_scppo_8gpu_100', version_base=None)
def main(config):
    if not ray.is_initialized():
        worker_env = {
            'TOKENIZERS_PARALLELISM': os.getenv('TOKENIZERS_PARALLELISM', 'false'),
            'NCCL_DEBUG': os.getenv('NCCL_DEBUG', 'WARN'),
            'EXPERIMENT_NAME': config.trainer.experiment_name,
            'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
        }
        for name in (
            'HF_HOME', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'NCCL_IB_DISABLE',
            'NCCL_P2P_DISABLE', 'PYTORCH_CUDA_ALLOC_CONF', 'RLLA_MEMORY_DEBUG',
            'VLLM_ATTENTION_BACKEND', 'CUDA_DEVICE_MAX_CONNECTIONS',
            'OMP_NUM_THREADS', 'TORCH_NCCL_ASYNC_ERROR_HANDLING', 'RAY_DEDUP_LOGS',
            'WITHLENGTH',
            'REFINEDREWARD', 'COARSEREWARD', 'STRICTMATCH', 'CORRECTMAX1',
            'MAX1STEP30MAX3', 'SCHEDULEREWARD', 'SCHEDULELENGTH',
        ):
            if name in os.environ:
                worker_env[name] = os.environ[name]
        ray_kwargs = {
            'address': 'local',
            'include_dashboard': False,
            'runtime_env': {'env_vars': worker_env},
        }
        if os.getenv('RAY_TEMP_DIR'):
            ray_kwargs['_temp_dir'] = os.environ['RAY_TEMP_DIR']
        ray.init(**ray_kwargs)
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    seed = int(config.trainer.get('seed', 1))
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
    model_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(model_path)
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }
    pool_id = 'tool_pool'
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec={pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
        mapping={
            Role.ActorRollout: pool_id,
            Role.Critic: pool_id,
            Role.RefPolicy: pool_id,
        },
    )
    reward_fn = ToolRewardManager(
        tokenizer=tokenizer,
        num_examine=config.trainer.get('reward_samples_to_print', 0),
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


if __name__ == '__main__':
    main()
