#!/usr/bin/env python3
"""Build the five paper evaluation sets from the locally mirrored PRIME files."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from prepare_math_data import MATH_INSTRUCTION


EXPECTED_ROWS = {
    'math': 500,
    'aime24': 30,
    'amc22_23': 83,
    'minerva': 272,
    'olympiad': 675,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def last_boxed(solution: str) -> str:
    for marker in ('\\boxed', '\\fbox'):
        start = solution.rfind(marker)
        if start < 0:
            continue
        left = solution.find('{', start)
        if left < 0:
            continue
        depth = 0
        for index in range(left, len(solution)):
            if solution[index] == '{':
                depth += 1
            elif solution[index] == '}':
                depth -= 1
                if depth == 0:
                    return solution[left + 1:index].strip()
    raise ValueError('solution has no complete boxed answer')


def normalize_number(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def make_row(source: str, index: int, problem: str, answer, source_id) -> dict:
    question = f'{problem} {MATH_INSTRUCTION}'
    return {
        'data_source': source,
        'prompt': [{'role': 'user', 'content': question}],
        'ability': 'math',
        'reward_model': {'style': 'rule', 'ground_truth': answer},
        'extra_info': {'index': index, 'source_id': str(source_id)},
    }


def load_suites(source_root: Path) -> tuple[dict[str, list[dict]], dict[str, Path]]:
    paths = {
        'aime': source_root / 'AI-MO/aimo-validation-aime/aimo-validation-aime.jsonl',
        'amc': source_root / 'AI-MO/aimo-validation-amc/aimo-validation-amc.jsonl',
        'math': source_root / 'math500/math_test_cleaned.json',
        'minerva': source_root / 'minerva_math/test.jsonl',
        'olympiad': source_root / 'olympiadbench/test.jsonl',
    }

    math_columns = json.loads(paths['math'].read_text(encoding='utf-8'))
    math_keys = sorted(math_columns['problem'], key=lambda value: int(value))
    math_rows = [
        make_row(
            'MATH-500', index, math_columns['problem'][key],
            str(math_columns['expected_answer'][key]), math_columns['id'][key],
        )
        for index, key in enumerate(math_keys)
    ]

    aime_source = [row for row in read_jsonl(paths['aime']) if '2024_AIME' in row['url']]
    aime_rows = [
        make_row('AIME-2024', index, row['question'], str(row['answer']), row['url'])
        for index, row in enumerate(aime_source)
    ]

    amc_source = read_jsonl(paths['amc'])
    amc_rows = [
        make_row('AMC-2022-2023', index, row['question'], normalize_number(row['answer']), row['url'])
        for index, row in enumerate(amc_source)
    ]

    minerva_source = read_jsonl(paths['minerva'])
    minerva_rows = [
        make_row('MINERVA', index, row['problem'], last_boxed(row['solution']), row['idx'])
        for index, row in enumerate(minerva_source)
    ]

    olympiad_source = read_jsonl(paths['olympiad'])
    olympiad_rows = []
    for index, row in enumerate(olympiad_source):
        problem = row['question']
        if row.get('context'):
            problem = f"{row['context']}\n\n{problem}"
        answers = [str(answer).strip('$') for answer in row['final_answer']]
        olympiad_rows.append(make_row('OLYMPIAD_BENCH', index, problem, answers, row['id']))

    suites = {
        'math': math_rows,
        'aime24': aime_rows,
        'amc22_23': amc_rows,
        'minerva': minerva_rows,
        'olympiad': olympiad_rows,
    }
    return suites, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suites, source_paths = load_suites(source_root)

    manifest = {
        'prompt_instruction': MATH_INSTRUCTION,
        'source_files': {name: {'path': str(path), 'sha256': sha256(path)} for name, path in source_paths.items()},
        'suites': {},
    }
    for name, rows in suites.items():
        if len(rows) != EXPECTED_ROWS[name]:
            raise ValueError(f'{name}: expected {EXPECTED_ROWS[name]} rows, found {len(rows)}')
        output_path = output_dir / f'{name}.parquet'
        pd.DataFrame(rows).to_parquet(output_path, index=False)
        manifest['suites'][name] = {
            'rows': len(rows),
            'path': str(output_path),
            'sha256': sha256(output_path),
        }

    manifest_path = output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
