#!/usr/bin/env python3
"""Export auditable question-level endpoints from one completed BFCL-v3 run."""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path


from bfcl_eval.model_handler.local_inference.toolrl_gdpo import (  # noqa: E402
    is_valid_toolrl_response,
    paired_question_seed,
)


MODEL_DIR = "TGPO_ToolRL"
SIMPLE_CATEGORIES = ("simple", "java", "javascript")
NONLIVE_DIRECT_CATEGORIES = (
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
)
LIVE_CATEGORIES = (
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
)
MULTI_CATEGORIES = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
)
ALL_CATEGORIES = (
    *SIMPLE_CATEGORIES,
    *NONLIVE_DIRECT_CATEGORIES,
    *LIVE_CATEGORIES,
    *MULTI_CATEGORIES,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstruct BFCL correctness and strict-format endpoints per question."
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        help="Override only for legacy runs whose manifest predates paired seeds.",
    )
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path}:{line_number}") from exc
    return rows


def iter_response_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from iter_response_strings(item)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_bfcl_components(category_accuracy, category_count):
    missing = sorted(set(ALL_CATEGORIES) - set(category_accuracy))
    if missing:
        raise ValueError(f"Missing BFCL categories: {missing}")
    simple = sum(category_accuracy[name] for name in SIMPLE_CATEGORIES) / 3
    non_live = (
        simple
        + sum(category_accuracy[name] for name in NONLIVE_DIRECT_CATEGORIES)
    ) / 5
    live_total = sum(category_count[name] for name in LIVE_CATEGORIES)
    live = sum(
        category_accuracy[name] * category_count[name] for name in LIVE_CATEGORIES
    ) / live_total
    multi_turn = sum(category_accuracy[name] for name in MULTI_CATEGORIES) / 4
    return {
        "non_live": non_live,
        "live": live,
        "multi_turn": multi_turn,
        "overall": (non_live + live + multi_turn) / 3,
    }


def bfcl_question_weight(category, category_count):
    if category in SIMPLE_CATEGORIES:
        return 1 / (3 * 5 * 3 * category_count[category])
    if category in NONLIVE_DIRECT_CATEGORIES:
        return 1 / (3 * 5 * category_count[category])
    if category in LIVE_CATEGORIES:
        live_total = sum(category_count[name] for name in LIVE_CATEGORIES)
        return 1 / (3 * live_total)
    if category in MULTI_CATEGORIES:
        return 1 / (3 * 4 * category_count[category])
    raise ValueError(f"Unexpected BFCL category: {category}")


def parse_percentage(value):
    if not isinstance(value, str) or not value.endswith("%"):
        raise ValueError(f"Expected BFCL percentage, got {value!r}")
    return float(value[:-1]) / 100


def read_official_overall(score_root):
    path = score_root / "data_overall.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one model row in {path}, got {len(rows)}")
    return parse_percentage(rows[0]["Overall Acc"])


def resolve_evaluation_seed(manifest, override):
    recorded = manifest.get("evaluation_seed")
    if override is not None and recorded is not None and int(override) != int(recorded):
        raise ValueError(
            f"Evaluation seed override {override} disagrees with manifest {recorded}"
        )
    seed = override if override is not None else recorded
    if seed is None:
        return None
    seed = int(seed)
    if not 0 <= seed < 2**31 - 1:
        raise ValueError(f"Invalid evaluation seed: {seed}")
    return seed


def extract_run(run_root, output_dir=None, evaluation_seed=None):
    run_root = run_root.resolve()
    manifest_path = run_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"BFCL run is not complete: {manifest_path}")
    evaluation_seed = resolve_evaluation_seed(manifest, evaluation_seed)

    result_root = run_root / "result" / MODEL_DIR
    score_root = run_root / "score" / MODEL_DIR
    if not result_root.is_dir() or not score_root.is_dir():
        raise ValueError(f"Missing BFCL result/score directory below {run_root}")

    category_rows = {}
    category_accuracy = {}
    category_count = {}
    source_hashes = {}
    for category in ALL_CATEGORIES:
        result_path = result_root / f"BFCL_v3_{category}_result.json"
        score_path = score_root / f"BFCL_v3_{category}_score.json"
        if not result_path.is_file() or not score_path.is_file():
            raise ValueError(f"Missing result or score file for BFCL category {category}")
        source_hashes[str(result_path.relative_to(run_root))] = sha256(result_path)
        source_hashes[str(score_path.relative_to(run_root))] = sha256(score_path)

        results = read_jsonl(result_path)
        scores = read_jsonl(score_path)
        if not scores or not {"accuracy", "correct_count", "total_count"} <= scores[0].keys():
            raise ValueError(f"Missing BFCL score summary: {score_path}")
        summary = scores[0]
        ids = [row.get("id") for row in results]
        if any(not isinstance(question_id, str) for question_id in ids):
            raise ValueError(f"Missing question id in {result_path}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate result id in {result_path}")
        failed_ids = {row.get("id") for row in scores[1:]}
        if None in failed_ids:
            raise ValueError(f"Missing failed question id in {score_path}")
        if not failed_ids <= set(ids):
            raise ValueError(f"Score file contains ids absent from result file: {score_path}")
        expected_failures = int(summary["total_count"]) - int(summary["correct_count"])
        if len(failed_ids) != expected_failures:
            raise ValueError(
                f"Cannot reconstruct {category}: {len(failed_ids)} unique failures, "
                f"expected {expected_failures}"
            )
        if len(results) != int(summary["total_count"]):
            raise ValueError(f"Result/score count mismatch for {category}")

        rows = []
        for result in results:
            responses = list(iter_response_strings(result.get("result")))
            valid = [bool(is_valid_toolrl_response(response)) for response in responses]
            rows.append(
                {
                    "question_id": result["id"],
                    "category": category,
                    "correct": int(result["id"] not in failed_ids),
                    "strict_valid_responses": int(sum(valid)),
                    "response_count": len(valid),
                    "full_trajectory_format": int(bool(valid) and all(valid)),
                    "is_multi_turn": int(category in MULTI_CATEGORIES),
                    "sampling_seed": (
                        paired_question_seed(evaluation_seed, result["id"])
                        if evaluation_seed is not None
                        else None
                    ),
                }
            )
        reconstructed = sum(row["correct"] for row in rows) / len(rows)
        if not math.isclose(reconstructed, float(summary["accuracy"]), abs_tol=1e-12):
            raise ValueError(f"Reconstructed category accuracy mismatch for {category}")
        category_rows[category] = rows
        category_accuracy[category] = reconstructed
        category_count[category] = len(rows)

    rows = []
    for category in ALL_CATEGORIES:
        weight = bfcl_question_weight(category, category_count)
        for row in category_rows[category]:
            row["bfcl_overall_weight"] = weight
            rows.append(row)

    components = official_bfcl_components(category_accuracy, category_count)
    weighted_overall = sum(row["bfcl_overall_weight"] * row["correct"] for row in rows)
    if not math.isclose(weighted_overall, components["overall"], abs_tol=1e-12):
        raise ValueError("Question weights do not reconstruct BFCL overall")
    if not math.isclose(
        sum(row["bfcl_overall_weight"] for row in rows), 1.0, abs_tol=1e-12
    ):
        raise ValueError("BFCL question weights do not sum to one")

    official_displayed = read_official_overall(run_root / "score")
    if abs(official_displayed - components["overall"]) > 0.000051:
        raise ValueError(
            "Reconstructed BFCL overall disagrees with official CSV beyond display rounding: "
            f"{components['overall']:.8f} vs {official_displayed:.8f}"
        )

    response_total = sum(row["response_count"] for row in rows)
    response_valid = sum(row["strict_valid_responses"] for row in rows)
    multi_rows = [row for row in rows if row["is_multi_turn"]]
    protected = {
        "overall_accuracy": components["overall"],
        "strict_format_rate": response_valid / response_total,
        "multi_turn_full_trajectory_format_rate": sum(
            row["full_trajectory_format"] for row in multi_rows
        )
        / len(multi_rows),
    }

    format_path = run_root / "format_compliance.json"
    if format_path.is_file():
        recorded_format = json.loads(format_path.read_text(encoding="utf-8"))
        recorded_ratio = recorded_format["response_level"]["ratio"]
        if not math.isclose(protected["strict_format_rate"], recorded_ratio, abs_tol=1e-12):
            raise ValueError("Question export disagrees with recorded strict-format ratio")

    summary = {
        "status": "PASS",
        "run_root": str(run_root),
        "bfcl_commit": manifest.get("bfcl_commit"),
        "evaluation_code": manifest.get("evaluation_code"),
        "checkpoint": manifest.get("checkpoint"),
        "evaluation_seed": evaluation_seed,
        "temperature": manifest.get("temperature"),
        "top_p": manifest.get("top_p"),
        "question_count": len(rows),
        "response_count": response_total,
        "protected_endpoints": protected,
        "official_components": components,
        "official_displayed_overall": official_displayed,
        "category_accuracy": category_accuracy,
        "category_count": category_count,
        "source_sha256": source_hashes,
        "definitions": {
            "overall_accuracy": "Exact BFCL-v3 official category aggregation.",
            "strict_format_rate": (
                "Valid ToolRL responses divided by all emitted response strings."
            ),
            "multi_turn_full_trajectory_format_rate": (
                "Fraction of multi-turn questions for which every emitted response "
                "string is strict ToolRL format and at least one response was emitted."
            ),
            "resampling_unit": "Whole BFCL question; response counts stay clustered.",
        },
    }

    output_dir = (output_dir or (run_root / "question_metrics")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "question_metrics.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary, rows_path, summary_path


def main():
    args = parse_args()
    summary, rows_path, summary_path = extract_run(
        args.run_root, args.output_dir, args.evaluation_seed
    )
    print(
        f"BFCL question export PASS: n={summary['question_count']} "
        f"overall={summary['protected_endpoints']['overall_accuracy']:.6f} "
        f"strict={summary['protected_endpoints']['strict_format_rate']:.6f} "
        "multi_full="
        f"{summary['protected_endpoints']['multi_turn_full_trajectory_format_rate']:.6f}"
    )
    print(f"Rows: {rows_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
