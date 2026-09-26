#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


MATH_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(sorted_values, fraction):
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def main():
    parser = argparse.ArgumentParser(description='Prepare the local DeepScaleR data for verl.')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-prompt-length', type=int, default=1024)
    args = parser.parse_args()

    source = args.source.resolve()
    model = args.model.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(source.read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)

    rows = []
    prompt_lengths = []
    for index, record in enumerate(records):
        question = f"{record['problem']} {MATH_INSTRUCTION}"
        prompt = [{'role': 'user', 'content': question}]
        prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        prompt_lengths.append(len(prompt_ids))
        rows.append({
            'data_source': 'DeepScaleR-Preview',
            'prompt': prompt,
            'ability': 'math',
            'reward_model': {'style': 'rule', 'ground_truth': str(record['answer'])},
            'extra_info': {'index': index},
        })

    parquet_path = output_dir / 'train.parquet'
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    sorted_lengths = sorted(prompt_lengths)
    manifest = {
        'source': str(source),
        'source_sha256': sha256(source),
        'model': str(model),
        'output': str(parquet_path),
        'output_sha256': sha256(parquet_path),
        'rows': len(rows),
        'prompt_instruction': MATH_INSTRUCTION,
        'prompt_template_example': tokenizer.apply_chat_template(
            rows[0]['prompt'], tokenize=False, add_generation_prompt=True
        ),
        'max_prompt_length': args.max_prompt_length,
        'prompts_over_limit': sum(length > args.max_prompt_length for length in prompt_lengths),
        'prompt_token_length': {
            'min': min(sorted_lengths),
            'p50': percentile(sorted_lengths, 0.50),
            'p90': percentile(sorted_lengths, 0.90),
            'p95': percentile(sorted_lengths, 0.95),
            'p99': percentile(sorted_lengths, 0.99),
            'max': max(sorted_lengths),
        },
    }
    manifest_path = output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
