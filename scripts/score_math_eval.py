#!/usr/bin/env python3
"""Score mathematical responses with the DeepScaleR and PRIME graders."""

import argparse
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from verl.utils.reward_score.deepscaler_math import compute_score as deepscaler_score
from verl.utils.reward_score.prime_math import compute_score as prime_score


SUITES = ('math', 'aime24', 'amc22_23', 'minerva', 'olympiad')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def as_list(value) -> list:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def ground_truths(value) -> list[str]:
    if isinstance(value, dict):
        value = value['ground_truth']
    return [str(item) for item in as_list(value)]


def score_prime_one(values):
    response, answers = values
    prime_correct = False
    prime_extracted = None
    for answer in answers:
        try:
            correct, _, extracted = prime_score(response, answer)
        except Exception:
            correct, extracted = False, None
        if prime_extracted is None:
            prime_extracted = extracted
        if correct:
            prime_correct = True
            prime_extracted = extracted
            break
    return prime_correct, prime_extracted


def score_deepscaler_one(values, symbolic_timeout=10, propagate_errors=False):
    response, answers = values
    deepscaler_correct = False
    deepscaler_extracted = None
    for answer in answers:
        try:
            correct, extracted = deepscaler_score(
                response,
                answer,
                symbolic_timeout=symbolic_timeout,
                propagate_errors=propagate_errors,
            )
        except Exception:
            if propagate_errors:
                raise
            correct, extracted = False, None
        if deepscaler_extracted is None:
            deepscaler_extracted = extracted
        if correct:
            deepscaler_correct = True
            deepscaler_extracted = extracted
            break
    return deepscaler_correct, deepscaler_extracted


def score_one(values):
    prime_correct, prime_extracted = score_prime_one(values)
    deepscaler_correct, deepscaler_extracted = score_deepscaler_one(values)
    return prime_correct, deepscaler_correct, prime_extracted, deepscaler_extracted


def score_deepscaler_tasks(tasks, workers, symbolic_timeout=10, retry_timeout=60,
                           score_fn=score_deepscaler_one):
    if symbolic_timeout <= 0 or retry_timeout <= symbolic_timeout:
        raise ValueError('evaluation retry timeout must exceed initial timeout')

    def first_pass(task):
        try:
            return 'ok', score_fn(
                task, symbolic_timeout=symbolic_timeout, propagate_errors=True
            ), None
        except TimeoutError as error:
            return 'timeout', None, error
        except Exception as error:
            return 'failure', None, error

    with ThreadPoolExecutor(max_workers=workers) as executor:
        attempted = list(executor.map(first_pass, tasks))
    scores = [None] * len(tasks)
    timeout_indices = []
    failures = []
    for index, (status, value, error) in enumerate(attempted):
        if status == 'ok':
            scores[index] = value
        elif status == 'timeout':
            timeout_indices.append(index)
        else:
            failures.append((index, error))
    retry_resolved = 0
    retry_failures = []
    for index in timeout_indices:
        try:
            scores[index] = score_fn(
                tasks[index], symbolic_timeout=retry_timeout, propagate_errors=True
            )
            retry_resolved += 1
        except Exception as error:
            retry_failures.append((index, error))
    audit = {
        'responses': len(tasks),
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
            f'evaluation grader produced {len(details)} unresolved errors ({preview})'
        )
    if any(score is None for score in scores):
        raise RuntimeError('evaluation grading left an unresolved response')
    return scores, audit


def mean_se(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, 0.0
    return mean, float(values.std(ddof=1) / math.sqrt(len(values)))


def summarize(scored: pd.DataFrame, target: dict | None = None, length_limit: int = 4000) -> dict:
    prime_question = np.array([np.mean(as_list(value)) for value in scored['prime_correct']], dtype=float)
    deepscaler_question = np.array(
        [np.mean(as_list(value)) for value in scored['deepscaler_correct']], dtype=float
    )
    exceed_question = np.array(
        [np.mean(np.asarray(as_list(value), dtype=int) > length_limit)
         for value in scored['response_token_lengths']],
        dtype=float,
    )
    prime_mean, prime_se = mean_se(prime_question)
    deepscaler_mean, deepscaler_se = mean_se(deepscaler_question)
    exceed_mean, exceed_se = mean_se(exceed_question)
    prime_ci95 = [prime_mean - 1.96 * prime_se, prime_mean + 1.96 * prime_se]
    deepscaler_ci95 = [
        deepscaler_mean - 1.96 * deepscaler_se,
        deepscaler_mean + 1.96 * deepscaler_se,
    ]
    exceed_ci95 = [exceed_mean - 1.96 * exceed_se, exceed_mean + 1.96 * exceed_se]
    result = {
        'questions': len(scored),
        'samples_per_question': int(len(as_list(scored.iloc[0]['responses']))),
        'primary_accuracy': deepscaler_mean,
        'primary_accuracy_se': deepscaler_se,
        'primary_accuracy_ci95': deepscaler_ci95,
        'prime_accuracy': prime_mean,
        'prime_accuracy_se': prime_se,
        'prime_accuracy_ci95': prime_ci95,
        'deepscaler_accuracy': deepscaler_mean,
        'deepscaler_accuracy_se': deepscaler_se,
        'deepscaler_accuracy_ci95': deepscaler_ci95,
        'exceed': exceed_mean,
        'exceed_se': exceed_se,
        'exceed_ci95': exceed_ci95,
        'grader_accuracy_delta': prime_mean - deepscaler_mean,
    }
    if target is not None:
        target_accuracy = target['accuracy'] / 100.0
        target_exceed = target['exceed'] / 100.0
        accuracy_consistent = deepscaler_ci95[0] <= target_accuracy <= deepscaler_ci95[1]
        exceed_consistent = exceed_ci95[0] <= target_exceed <= exceed_ci95[1]
        result.update({
            'paper_target_accuracy': target_accuracy,
            'paper_target_exceed': target_exceed,
            'accuracy_delta_vs_paper': deepscaler_mean - target_accuracy,
            'exceed_delta_vs_paper': exceed_mean - target_exceed,
            'paper_accuracy_statistically_consistent': bool(accuracy_consistent),
            'paper_exceed_statistically_consistent': bool(exceed_consistent),
            'paper_pair_statistically_consistent': bool(accuracy_consistent and exceed_consistent),
        })
    return result


def score_suite(input_path: Path, output_path: Path, workers: int,
                expected_samples_count: int = 16,
                require_unique_seeds: bool = False,
                symbolic_timeout: float = 10,
                retry_timeout: float = 60) -> tuple[pd.DataFrame, dict]:
    data = pd.read_parquet(input_path)
    required = {'responses', 'response_token_lengths', 'reward_model'}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f'{input_path}: missing columns {sorted(missing)}')

    sample_counts = {len(as_list(value)) for value in data['responses']}
    if sample_counts != {expected_samples_count}:
        raise ValueError(
            f'{input_path}: expected exactly {expected_samples_count} responses per question, '
            f'found {sample_counts}'
        )
    if 'response_sampling_seeds' in data.columns:
        seed_rows = [tuple(int(seed) for seed in as_list(value))
                     for value in data['response_sampling_seeds']]
        if any(len(seeds) != expected_samples_count for seeds in seed_rows):
            raise ValueError(f'{input_path}: sampling seed count does not match response count')
        if any(len(set(seeds)) != len(seeds) for seeds in seed_rows):
            raise ValueError(f'{input_path}: repeated sampling seeds create pseudo-replication')
        if len(set(seed_rows)) != 1:
            raise ValueError(f'{input_path}: questions do not share the registered common seed table')
    elif require_unique_seeds:
        raise ValueError(f'{input_path}: missing response_sampling_seeds audit column')

    tasks = []
    row_ranges = []
    offset = 0
    for row in data.itertuples(index=False):
        responses = [str(value) for value in as_list(row.responses)]
        answers = ground_truths(row.reward_model)
        tasks.extend((response, answers) for response in responses)
        row_ranges.append((offset, offset + len(responses)))
        offset += len(responses)

    # DeepScaleR is the paper-comparison grader and is safe to run in threads.
    # PRIME uses multiprocessing timeouts internally, so keep that audit on the
    # main thread rather than forking concurrently from worker threads.
    deepscaler_scores, grader_audit = score_deepscaler_tasks(
        tasks,
        workers=workers,
        symbolic_timeout=symbolic_timeout,
        retry_timeout=retry_timeout,
    )
    prime_scores = [score_prime_one(task) for task in tasks]
    scores = [
        (prime[0], deepscaler[0], prime[1], deepscaler[1])
        for prime, deepscaler in zip(prime_scores, deepscaler_scores)
    ]

    data = data.copy()
    data['prime_correct'] = [[bool(scores[i][0]) for i in range(start, end)] for start, end in row_ranges]
    data['deepscaler_correct'] = [[bool(scores[i][1]) for i in range(start, end)] for start, end in row_ranges]
    data['prime_extracted'] = [[scores[i][2] for i in range(start, end)] for start, end in row_ranges]
    data['deepscaler_extracted'] = [[scores[i][3] for i in range(start, end)] for start, end in row_ranges]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output_path, index=False)
    return data, grader_audit


def print_report(report: dict):
    if report['mode'] == 'controlled':
        print('suite         DS primary  PRIME audit  exceed')
        for suite in SUITES:
            item = report['suites'][suite]
            print(
                f'{suite:13s} {item["primary_accuracy"] * 100:10.2f}  '
                f'{item["prime_accuracy"] * 100:11.2f}  {item["exceed"] * 100:7.2f}'
            )
        macro = report['macro_average']
        print(
            f'{"macro_average":13s} {macro["primary_accuracy"] * 100:10.2f}  '
            f'{macro["prime_accuracy"] * 100:11.2f}  {macro["exceed"] * 100:7.2f}'
        )
        return

    print('suite         DS primary  PRIME audit  exceed   paper acc  paper exceed  acc delta  exceed delta')
    for suite in SUITES:
        item = report['suites'][suite]
        print(
            f'{suite:13s} {item["primary_accuracy"] * 100:10.2f}  '
            f'{item["prime_accuracy"] * 100:11.2f}  {item["exceed"] * 100:7.2f}  '
            f'{item["paper_target_accuracy"] * 100:9.2f}  {item["paper_target_exceed"] * 100:12.2f}  '
            f'{item["accuracy_delta_vs_paper"] * 100:+9.2f}  '
            f'{item["exceed_delta_vs_paper"] * 100:+12.2f}'
        )
    macro = report['macro_average']
    print(
        f'{"macro_average":13s} {macro["primary_accuracy"] * 100:10.2f}  '
        f'{macro["prime_accuracy"] * 100:11.2f}  {macro["exceed"] * 100:7.2f}  '
        f'{macro["paper_target_accuracy"] * 100:9.2f}  {macro["paper_target_exceed"] * 100:12.2f}  '
        f'{macro["accuracy_delta_vs_paper"] * 100:+9.2f}  '
        f'{macro["exceed_delta_vs_paper"] * 100:+12.2f}'
    )
    assessment = report['reproduction_assessment']
    print(
        'Paper consistency (question-clustered 95% CI): '
        f'macro pair={assessment["macro_pair_statistically_consistent"]}, '
        f'all five suite pairs={assessment["all_suite_pairs_statistically_consistent"]}'
    )


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--generation-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--targets', type=Path, help='Optional reference metrics for a separate comparison.'
    )
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--expected-samples', type=int, default=4)
    parser.add_argument('--length-limit', type=int, default=1024)
    parser.add_argument('--symbolic-timeout', type=float, default=10)
    parser.add_argument('--retry-timeout', type=float, default=60)
    parser.add_argument(
        '--require-unique-seeds', action='store_true',
        help='Reject generations without a unique, common per-slot sampling seed table.',
    )
    parser.add_argument(
        '--controlled', action='store_true',
        help='Report raw controlled metrics without comparing against the paper targets.',
    )
    args = parser.parse_args()
    args.controlled = args.controlled or args.targets is None

    target_metrics = None
    if not args.controlled:
        targets = json.loads(args.targets.read_text(encoding='utf-8'))
        target_metrics = targets['metrics_percent']
    report = {
        'mode': 'controlled' if args.controlled else 'paper_reproduction',
        'grader': 'DeepScaleR boxed-answer primary; PRIME robustness audit',
        'length_limit': args.length_limit,
        'expected_samples_per_question': args.expected_samples,
        'sampling_seed_policy': (
            'unique slots required; common random-number table across methods'
            if args.require_unique_seeds else 'audited when present; legacy files allowed'
        ),
        'suites': {},
        'artifacts': {},
    }
    if target_metrics is not None:
        report['targets'] = str(args.targets.resolve())
    for suite in SUITES:
        input_path = args.generation_dir / f'{suite}.parquet'
        output_path = args.output_dir / f'{suite}.scored.parquet'
        scored, grader_audit = score_suite(
            input_path, output_path, args.workers,
            expected_samples_count=args.expected_samples,
            require_unique_seeds=args.require_unique_seeds,
            symbolic_timeout=args.symbolic_timeout,
            retry_timeout=args.retry_timeout,
        )
        target = target_metrics[suite] if target_metrics is not None else None
        report['suites'][suite] = summarize(scored, target, args.length_limit)
        report['artifacts'][suite] = {
            'generation_path': str(input_path.resolve()),
            'generation_sha256': sha256(input_path),
            'scored_path': str(output_path.resolve()),
            'scored_sha256': sha256(output_path),
            'grader_audit': grader_audit,
        }
        if 'response_sampling_seeds' in scored.columns:
            report['artifacts'][suite]['sampling_seeds'] = [
                int(seed) for seed in as_list(scored.iloc[0]['response_sampling_seeds'])
            ]
        print(f'scored {suite}: {len(scored)} questions', flush=True)

    macro = {}
    for key in ('primary_accuracy', 'prime_accuracy', 'deepscaler_accuracy', 'exceed'):
        macro[key] = float(np.mean([report['suites'][suite][key] for suite in SUITES]))
        se_key = f'{key}_se'
        macro[se_key] = math.sqrt(sum(
            report['suites'][suite][se_key] ** 2 for suite in SUITES
        )) / len(SUITES)
        macro[f'{key}_ci95'] = [
            macro[key] - 1.96 * macro[se_key],
            macro[key] + 1.96 * macro[se_key],
        ]
    macro['grader_accuracy_delta'] = macro['prime_accuracy'] - macro['deepscaler_accuracy']
    report['macro_average'] = macro
    if target_metrics is not None:
        macro['paper_target_accuracy'] = target_metrics['macro_average']['accuracy'] / 100.0
        macro['paper_target_exceed'] = target_metrics['macro_average']['exceed'] / 100.0
        macro['accuracy_delta_vs_paper'] = macro['primary_accuracy'] - macro['paper_target_accuracy']
        macro['exceed_delta_vs_paper'] = macro['exceed'] - macro['paper_target_exceed']
        macro_accuracy_consistent = (
            macro['primary_accuracy_ci95'][0]
            <= macro['paper_target_accuracy']
            <= macro['primary_accuracy_ci95'][1]
        )
        macro_exceed_consistent = (
            macro['exceed_ci95'][0]
            <= macro['paper_target_exceed']
            <= macro['exceed_ci95'][1]
        )
        macro['paper_accuracy_statistically_consistent'] = bool(macro_accuracy_consistent)
        macro['paper_exceed_statistically_consistent'] = bool(macro_exceed_consistent)
        macro['paper_pair_statistically_consistent'] = bool(
            macro_accuracy_consistent and macro_exceed_consistent
        )
        report['reproduction_assessment'] = {
            'criterion': (
                'A paper point estimate is statistically consistent when it lies inside this run\'s '
                'question-clustered 95% confidence interval. Paper-run uncertainty is unavailable.'
            ),
            'macro_pair_statistically_consistent': macro['paper_pair_statistically_consistent'],
            'all_suite_pairs_statistically_consistent': bool(all(
                report['suites'][suite]['paper_pair_statistically_consistent'] for suite in SUITES
            )),
            'all_suite_accuracy_statistically_consistent': bool(all(
                report['suites'][suite]['paper_accuracy_statistically_consistent'] for suite in SUITES
            )),
            'all_suite_exceed_statistically_consistent': bool(all(
                report['suites'][suite]['paper_exceed_statistically_consistent'] for suite in SUITES
            )),
        }

    report_path = args.output_dir / 'report.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    print_report(report)
    print(f'report: {report_path.resolve()}')


if __name__ == '__main__':
    main()
