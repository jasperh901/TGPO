# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""
import ray
import numpy as np
import hydra
import os
import math

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


def resolve_sample_seeds(configured_seeds, sample_count, base_seed):
    """Return explicit, unique per-slot seeds for reproducible evaluation."""
    if configured_seeds is None:
        seeds = [int(base_seed) + sample_id for sample_id in range(sample_count)]
    else:
        seeds = [int(seed) for seed in configured_seeds]
    if len(seeds) != sample_count:
        raise ValueError(
            f'data.sample_seeds must contain exactly {sample_count} entries, got {len(seeds)}'
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError('data.sample_seeds must be unique to avoid pseudo-replication')
    return seeds


@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    if not ray.is_initialized():
        ray_kwargs = {'address': 'local', 'include_dashboard': False}
        if os.getenv('RAY_TEMP_DIR'):
            ray_kwargs['_temp_dir'] = os.environ['RAY_TEMP_DIR']
        ray.init(**ray_kwargs)
    local_path = copy_local_path_from_hdfs(config.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.rollout.temperature == 0.:
        assert config.data.n_samples == 1, 'When temperature=0, n_samples must be 1.'

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    chat_lst = dataset[config.data.prompt_key].tolist()

    chat_lst = [chat.tolist() for chat in chat_lst]

    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    # FSDP/NCCL requires one rank per physical GPU.  RayResourcePool defaults
    # to five colocated workers, which can assign multiple ranks to one GPU.
    resource_pool = RayResourcePool(
        process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        max_colocate_count=1,
    )
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()

    total_samples = len(dataset)
    # real_batch_size = data.batch['input_ids'].shape[0]
    config_batch_size = config.data.batch_size
    dp_size = wg.world_size // config.rollout.tensor_model_parallel_size
    num_batch = math.ceil(total_samples / config_batch_size)
    output_lst = [[] for _ in range(config.data.n_samples)]
    output_length_lst = [[] for _ in range(config.data.n_samples)]
    sample_seeds = resolve_sample_seeds(
        configured_seeds=config.data.get('sample_seeds'),
        sample_count=int(config.data.n_samples),
        base_seed=int(config.rollout.get('seed', 0)),
    )
    print(f'evaluation sample seeds: {sample_seeds}', flush=True)

    for batch_idx in range(num_batch):
        print(f'[{batch_idx+1}/{num_batch}] Start to process.')
        batch_chat_lst = chat_lst[batch_idx * config_batch_size:(batch_idx + 1) * config_batch_size]
        inputs = tokenizer.apply_chat_template(batch_chat_lst,
                                               add_generation_prompt=True,
                                               padding=True,
                                               truncation=True,
                                               max_length=config.rollout.prompt_length,
                                               return_tensors='pt',
                                               return_dict=True,
                                               tokenize=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = compute_position_id_with_mask(attention_mask)

        batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

        data = DataProto.from_dict(batch_dict)
        real_batch_size = data.batch['input_ids'].shape[0]
        if real_batch_size % dp_size != 0:
            dummy_data_size = dp_size - real_batch_size % dp_size
            dummy_data = data[:dummy_data_size]
            data = DataProto.concat([data, dummy_data])
            print(
                f'dp_size {dp_size} is not divisible by real_batch_size {real_batch_size}, add {dummy_data_size} dummy data'
            )

        batch_size = data.batch['input_ids'].shape[0]
        assert batch_size % dp_size == 0, f'batch_size {batch_size} is not divisible by dp_size {dp_size}'

        print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
        # START TO GENERATE FOR n_samples TIMES
        for i in range(config.data.n_samples):
            data.meta_info['sampling_seed'] = sample_seeds[i]
            output = wg.generate_sequences(data)
            # remove dummy data
            output = output[:real_batch_size]
            response_mask = output.batch['attention_mask'][:, -config.rollout.response_length:]
            output_length_lst[i].extend(response_mask.sum(-1).cpu().tolist())
            output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                                 skip_special_tokens=False)

            # remove the padding
            pad_token = tokenizer.pad_token
            output_text_unpad = []
            for text in output_text:
                output_text_unpad.append(text.replace(pad_token, ''))

            output_lst[i].extend(output_text_unpad)

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()
    output_length_lst = np.array(output_length_lst, dtype=object)
    output_length_lst = np.transpose(output_length_lst, axes=(1, 0)).tolist()

    # add to the data frame
    dataset[f'responses'] = output_lst
    dataset[f'response_token_lengths'] = output_length_lst
    dataset[f'response_sampling_seeds'] = [sample_seeds for _ in range(len(dataset))]
    generation_metadata = {
        'generation_model_path': str(config.model.path),
        'generation_temperature': float(config.rollout.temperature),
        'generation_top_p': float(config.rollout.top_p),
        'generation_top_k': int(config.rollout.top_k),
        'generation_prompt_length': int(config.rollout.prompt_length),
        'generation_response_length': int(config.rollout.response_length),
        'generation_tensor_parallel_size': int(config.rollout.tensor_model_parallel_size),
        'generation_sequence_parallel_size': int(config.actor.ulysses_sequence_parallel_size),
        'generation_n_samples': int(config.data.n_samples),
    }
    for key, value in generation_metadata.items():
        dataset[key] = value

    # write to a new parquet
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(config.data.output_path)

    ray.shutdown()
    return output_text


if __name__ == '__main__':
    main()
