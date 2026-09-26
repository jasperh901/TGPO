import os
import random
from concurrent.futures import ThreadPoolExecutor

import hydra
import numpy as np
import ray
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.reward_score.deepscaler_math import compute_score as deepscaler_math_score


def score_math_response(response: str, ground_truth: str, response_tokens: int,
                        length_limit: int = 4000, symbolic_timeout: float = 10,
                        propagate_grader_errors: bool = False) -> dict:
    try:
        is_correct, extracted_answer = deepscaler_math_score(
            response,
            ground_truth,
            symbolic_timeout=symbolic_timeout,
            propagate_errors=propagate_grader_errors,
        )
        format_correct = extracted_answer is not None
    except Exception as error:
        if propagate_grader_errors:
            raise
        print(f'math grader failure: {type(error).__name__}: {error}', flush=True)
        is_correct, format_correct, extracted_answer = False, False, None
    correctness = float(bool(is_correct))
    length = float(response_tokens <= length_limit)
    return {
        'reward': correctness + length,
        'correctness': correctness,
        'length': length,
        'format': float(bool(format_correct)),
        'extracted_answer': extracted_answer,
    }


def score_math_inputs(scoring_inputs, length_limit: int, num_workers: int,
                      symbolic_timeout: float, retry_timeout: float,
                      score_fn=score_math_response):
    """Score in parallel, then deterministically retry timeout cases in input order."""
    if symbolic_timeout <= 0 or retry_timeout <= symbolic_timeout:
        raise ValueError('reward retry timeout must exceed the initial symbolic timeout')

    def score_one(values):
        response, ground_truth, valid_response_length = values
        try:
            scores = score_fn(
                response=response,
                ground_truth=ground_truth,
                response_tokens=valid_response_length,
                length_limit=length_limit,
                symbolic_timeout=symbolic_timeout,
                propagate_grader_errors=True,
            )
            return 'ok', scores, None
        except TimeoutError as error:
            return 'timeout', None, error
        except Exception as error:
            return 'failure', None, error

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        first_pass = list(executor.map(score_one, scoring_inputs))

    all_scores = [None] * len(scoring_inputs)
    timeout_indices = []
    failures = []
    for index, (status, scores, error) in enumerate(first_pass):
        if status == 'ok':
            all_scores[index] = scores
        elif status == 'timeout':
            timeout_indices.append(index)
        else:
            failures.append((index, error))

    retry_resolved = 0
    retry_failures = []
    for index in timeout_indices:
        response, ground_truth, valid_response_length = scoring_inputs[index]
        try:
            all_scores[index] = score_fn(
                response=response,
                ground_truth=ground_truth,
                response_tokens=valid_response_length,
                length_limit=length_limit,
                symbolic_timeout=retry_timeout,
                propagate_grader_errors=True,
            )
            retry_resolved += 1
        except Exception as error:
            retry_failures.append((index, error))

    audit = {
        'responses': len(scoring_inputs),
        'fast_timeouts': len(timeout_indices),
        'retry_resolved': retry_resolved,
        'retry_failures': len(retry_failures),
        'grader_failures': len(failures),
    }
    if failures or retry_failures:
        details = failures + retry_failures
        preview = ', '.join(
            f'index={index}:{type(error).__name__}' for index, error in details[:8]
        )
        raise RuntimeError(
            f'reward grader produced {len(details)} unresolved errors ({preview})'
        )
    if any(scores is None for scores in all_scores):
        raise RuntimeError('reward scoring left an unresolved response')
    return all_scores, audit


class MathRewardManager:
    def __init__(self, tokenizer, length_limit=4000, num_examine=0, num_workers=48,
                 symbolic_timeout=10, retry_timeout=60):
        self.tokenizer = tokenizer
        self.length_limit = int(length_limit)
        self.num_examine = int(num_examine)
        self.num_workers = int(num_workers)
        self.symbolic_timeout = float(symbolic_timeout)
        self.retry_timeout = float(retry_timeout)

    def __call__(self, data: DataProto, step: int):
        shape = data.batch['responses'].shape
        reward_tensor = torch.zeros(shape, dtype=torch.float32)
        format_tensor = torch.zeros(shape, dtype=torch.float32)
        correctness_tensor = torch.zeros(shape, dtype=torch.float32)
        length_tensor = torch.zeros(shape, dtype=torch.float32)
        correct_count = 0
        overlong_count = 0

        scoring_inputs = []
        for index in range(len(data)):
            item = data[index]
            prompt_length = item.batch['prompts'].shape[-1]
            valid_response_length = int(item.batch['attention_mask'][prompt_length:].sum().item())
            if valid_response_length > 0:
                valid_response_ids = item.batch['responses'][:valid_response_length]
                response = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            else:
                response = ''

            ground_truth = item.non_tensor_batch['reward_model']['ground_truth']
            scoring_inputs.append((response, ground_truth, valid_response_length))

        all_scores, audit = score_math_inputs(
            scoring_inputs=scoring_inputs,
            length_limit=self.length_limit,
            num_workers=self.num_workers,
            symbolic_timeout=self.symbolic_timeout,
            retry_timeout=self.retry_timeout,
        )
        print(
            f'reward grader audit step={step} responses={audit["responses"]} '
            f'fast_timeouts={audit["fast_timeouts"]} '
            f'retry_resolved={audit["retry_resolved"]} '
            f'retry_failures={audit["retry_failures"]} '
            f'grader_failures={audit["grader_failures"]}',
            flush=True,
        )

        for index, (scores, scoring_input) in enumerate(zip(all_scores, scoring_inputs)):
            response, _, valid_response_length = scoring_input
            target_position = max(valid_response_length - 1, 0)
            reward_tensor[index, target_position] = scores['reward']
            format_tensor[index, target_position] = scores['format']
            correctness_tensor[index, target_position] = scores['correctness']
            length_tensor[index, target_position] = scores['length']
            correct_count += int(scores['correctness'])
            overlong_count += int(not scores['length'])

            if index < self.num_examine:
                print(
                    f'step={step} sample={index} tokens={valid_response_length} '
                    f'correct={scores["correctness"]} answer={scores["extracted_answer"]!r}\n{response}',
                    flush=True,
                )

        print(
            f'reward summary step={step} responses={len(data)} '
            f'correct={correct_count / max(len(data), 1):.6f} '
            f'exceed={overlong_count / max(len(data), 1):.6f}',
            flush=True,
        )
        return reward_tensor, format_tensor, correctness_tensor, length_tensor


@hydra.main(config_path='config', config_name='math_ppo', version_base=None)
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
            'OMP_NUM_THREADS', 'TORCH_NCCL_ASYNC_ERROR_HANDLING',
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
    pool_id = 'math_pool'
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec={pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
        mapping={
            Role.ActorRollout: pool_id,
            Role.Critic: pool_id,
            Role.RefPolicy: pool_id,
        },
    )
    reward_fn = MathRewardManager(
        tokenizer=tokenizer,
        length_limit=config.algorithm.get('length_limit', 4000),
        num_examine=config.trainer.get('reward_samples_to_print', 0),
        num_workers=config.trainer.get('reward_workers', 48),
        symbolic_timeout=config.trainer.get('reward_symbolic_timeout', 10),
        retry_timeout=config.trainer.get('reward_retry_timeout', 60),
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
