import os
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

import verl.utils.torch_functional as verl_F
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask


def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}
    for data in data_list:
        for key, value in data.items():
            target = tensors if isinstance(value, torch.Tensor) else non_tensors
            target.setdefault(key, []).append(value)
    tensors = {key: torch.stack(value, dim=0) for key, value in tensors.items()}
    non_tensors = {key: np.asarray(value, dtype=object) for key, value in non_tensors.items()}
    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """Minimal verl-compatible parquet dataset for text-only RL rollouts."""

    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        prompt_key='prompt',
        max_prompt_length=1024,
        use_chat_template=True,
        filter_prompts=True,
        cache_dir='~/.cache/verl/rlhf',
        chat_template_func=None,
        return_raw_chat=False,
        truncation='error',
    ):
        if not isinstance(parquet_files, (list, ListConfig)):
            parquet_files = [parquet_files]
        self.parquet_files = list(parquet_files)
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.prompt_key = prompt_key
        self.max_prompt_length = max_prompt_length
        self.use_chat_template = use_chat_template
        self.filter_prompts = filter_prompts
        self.chat_template_func = chat_template_func
        self.return_raw_chat = return_raw_chat
        self.truncation = truncation
        self._download()
        self._read_files()

    def _download(self):
        for index, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[index] = copy_local_path_from_hdfs(
                src=parquet_file, cache_dir=self.cache_dir
            )

    def _read_files(self):
        frames = [pd.read_parquet(path) for path in self.parquet_files]
        self.dataframe = pd.concat(frames, ignore_index=True)
        print(f'dataset len: {len(self.dataframe)}')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row = self.dataframe.iloc[item].to_dict()
        chat = row.pop(self.prompt_key)
        if isinstance(chat, np.ndarray):
            chat = chat.tolist()
        if self.use_chat_template:
            prompt = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = '\n\n'.join(message['content'].strip() for message in chat).strip()

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row['input_ids'] = input_ids[0]
        row['attention_mask'] = attention_mask[0]
        row['position_ids'] = compute_position_id_with_mask(attention_mask)[0]
        if self.return_raw_chat:
            row['raw_prompt'] = chat
        row['index'] = row.get('extra_info', {}).get('index', item)
        return row
