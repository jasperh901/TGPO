#!/usr/bin/env python3
"""Score every code test in an OS sandbox and aggregate the paper endpoints."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from code_eval_common import DATA_ERRORS, MAX_TOKENS, load_rows, question_identity, read_jsonl, request_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime/code"))


def judge(tasks, workers):
    from verl.utils.code_sandbox import sandbox_command
    command = sandbox_command("verl.utils.reward_score.native_code_eval_worker")
    if not tasks:
        return []
    shards = [tasks[i::workers] for i in range(min(workers, len(tasks)))]
    def run(shard):
        payload = "".join(base64.b64encode(json.dumps(t).encode()).decode() + "\n" for t in shard)
        timeout = 60 + sum(5 * (len(json.loads(t["ground_truth"])["inputs"]) + 1) + 15 for t in shard)
        result = subprocess.run(command, input=payload, text=True, capture_output=True, timeout=timeout, check=True)
        scores = [json.loads(line) for line in result.stdout.splitlines()]
        if [s.get("id") for s in scores] != [t["id"] for t in shard] or any("error" in s for s in scores):
            raise RuntimeError(f"Incomplete judge output: {result.stderr[-1000:]}")
        return scores
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sorted([r for shard in pool.map(run, shards) for r in shard], key=lambda r: r["id"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generation-dir", type=Path)
    p.add_argument("--data", type=Path, default=ROOT / "data/code/validation.parquet")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--smoke-test", action="store_true")
    a = p.parse_args()
    if a.smoke_test:
        tasks = [dict(id=0, response="```python\nprint(input())\n```", ground_truth=json.dumps({"inputs": ["ok"], "outputs": ["ok"]})),
                 dict(id=1, response="```python\nprint(input())\n```", ground_truth=json.dumps({"inputs": [str(i) for i in range(11)], "outputs": [str(i) for i in range(10)] + ["wrong"]}))]
        assert [r["score"]["strict_pass"] for r in judge(tasks, 2)] == [True, False]
        print("Sandbox and all-test coverage checks passed"); return
    if a.generation_dir is None:
        p.error("--generation-dir is required")
    rows = load_rows(a.data)
    generated = sorted([r for path in a.generation_dir.glob("rank*.jsonl") for r in read_jsonl(path)],
                       key=lambda r: (r["position"], r["sample"]))
    if [(r["position"], r["sample"]) for r in generated] != [(i, s) for i in range(1016) for s in range(4)]:
        raise ValueError("Expected exactly four distinct responses for each of 1,016 questions")
    tasks = []
    for i, record in enumerate(generated):
        pos, sample = record["position"], record["sample"]
        if record["question_id"] != question_identity(rows[pos], pos) or record["seed"] != request_seed(pos, sample):
            raise ValueError("Question or sampling identity mismatch")
        if not 0 < record["response_tokens"] <= MAX_TOKENS:
            raise ValueError("Invalid response length")
        tasks.append(dict(id=i, response=record["response"], ground_truth=rows[pos]["reward_model"]["ground_truth"], per_test_timeout=5))
    scored = judge(tasks, a.workers)
    valid = []
    for record, result in zip(generated, scored):
        score = result["score"]
        if (score["outcome"] == "dataset_error") != (record["position"] in DATA_ERRORS):
            raise ValueError("Malformed-record identities differ from the published data protocol")
        if score["strict_pass"] is True and not score["tests_executed"] == score["prefix_passed"] == score["test_count"]:
            raise ValueError("Successful response has incomplete test coverage")
        if record["position"] not in DATA_ERRORS:
            valid.append((record, score))
    n = len(valid)
    report = {"questions": n // 4, "responses": n,
              "pass_at_1": sum(s["strict_pass"] is True for r, s in valid) / n,
              "correct_within_4000": sum(s["strict_pass"] is True and r["response_tokens"] <= 4000 for r, s in valid) / n,
              "overlength": sum(r["response_tokens"] > 4000 for r, s in valid) / n,
              "syntax_valid": sum(s["syntax_valid"] for r, s in valid) / n,
              "mean_tokens": sum(r["response_tokens"] for r, s in valid) / n}
    (a.generation_dir / "scores.json").write_text(json.dumps(scored))
    (a.generation_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
