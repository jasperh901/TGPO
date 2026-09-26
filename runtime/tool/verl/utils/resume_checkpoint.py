"""Read an actor snapshot without reducing its stored tensor precision."""

import json
from pathlib import Path

import torch


def load_actor_state_dict(directory):
    directory = Path(directory)
    index_path = directory / 'model.safetensors.index.json'
    if index_path.is_file():
        from safetensors.torch import load_file

        weight_map = json.loads(index_path.read_text())['weight_map']
        if not weight_map:
            raise ValueError('Empty model shard index')
        state = {}
        for filename in sorted(set(weight_map.values())):
            if Path(filename).name != filename:
                raise ValueError(f'Invalid shard filename: {filename}')
            shard = load_file(str(directory / filename), device='cpu')
            expected = {key for key, source in weight_map.items() if source == filename}
            if set(shard) != expected:
                raise ValueError(f'Tensor keys disagree with index: {filename}')
            state.update(shard)
        return state
    if (directory / 'model.safetensors').is_file():
        from safetensors.torch import load_file

        return load_file(str(directory / 'model.safetensors'), device='cpu')
    return torch.load(directory / 'pytorch_model.bin', map_location='cpu', weights_only=True)
