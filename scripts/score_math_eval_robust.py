#!/usr/bin/env python3
"""Run Math-RL scoring with deterministic timeout-worker recovery."""

from __future__ import annotations

import importlib.util
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from verl.utils.py_functional import TimeoutWorkerError


MATH_SCORER = Path(os.environ.get(
    "MATH_RL_SCORER", str(Path(__file__).with_name("score_math_eval.py"))))
SPEC = importlib.util.spec_from_file_location("math_rl_score_math_eval", MATH_SCORER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load Math-RL scorer: {MATH_SCORER}")
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)


def score_deepscaler_tasks(
    tasks,
    workers,
    symbolic_timeout=10,
    retry_timeout=60,
    score_fn=None,
):
    """Score concurrently, then retry process-worker failures in input order."""
    if symbolic_timeout <= 0 or retry_timeout <= symbolic_timeout:
        raise ValueError("evaluation retry timeout must exceed initial timeout")
    final_retry_timeout = 300
    score_fn = SCORER.score_deepscaler_one if score_fn is None else score_fn

    def first_pass(task):
        try:
            return "ok", score_fn(
                task, symbolic_timeout=symbolic_timeout, propagate_errors=True
            ), None
        except TimeoutError as error:
            return "timeout", None, error
        except TimeoutWorkerError as error:
            return "worker_failure", None, error
        except Exception as error:
            return "failure", None, error

    with ThreadPoolExecutor(max_workers=workers) as executor:
        attempted = list(executor.map(first_pass, tasks))

    scores = [None] * len(tasks)
    timeout_indices = []
    worker_failure_indices = []
    failures = []
    for index, (status, value, error) in enumerate(attempted):
        if status == "ok":
            scores[index] = value
        elif status == "timeout":
            timeout_indices.append(index)
        elif status == "worker_failure":
            worker_failure_indices.append(index)
        else:
            failures.append((index, error))

    retry_resolved = 0
    worker_retry_resolved = 0
    retry_failures = []
    extended_retries = 0
    extended_retry_resolved = 0
    retry_cases = sorted(
        [(index, "timeout") for index in timeout_indices]
        + [(index, "worker_failure") for index in worker_failure_indices]
    )
    for index, origin in retry_cases:
        try:
            scores[index] = score_fn(
                tasks[index], symbolic_timeout=retry_timeout, propagate_errors=True
            )
            if origin == "timeout":
                retry_resolved += 1
            else:
                worker_retry_resolved += 1
        except (TimeoutError, TimeoutWorkerError):
            extended_retries += 1
            try:
                scores[index] = score_fn(
                    tasks[index],
                    symbolic_timeout=final_retry_timeout,
                    propagate_errors=True,
                )
                if origin == "timeout":
                    retry_resolved += 1
                else:
                    worker_retry_resolved += 1
                extended_retry_resolved += 1
            except Exception as final_error:
                retry_failures.append((index, final_error))
        except Exception as error:
            retry_failures.append((index, error))

    audit = {
        "responses": len(tasks),
        "fast_timeouts": len(timeout_indices),
        "retry_resolved": retry_resolved,
        "retry_failures": len(retry_failures),
        "grader_failures": len(failures),
        "fast_worker_failures": len(worker_failure_indices),
        "worker_retry_resolved": worker_retry_resolved,
        "extended_retries": extended_retries,
        "extended_retry_resolved": extended_retry_resolved,
    }
    if failures or retry_failures:
        details = failures + retry_failures
        preview = ", ".join(
            f"index={index}:{type(error).__name__}"
            for index, error in details[:8]
        )
        raise RuntimeError(
            f"evaluation grader produced {len(details)} unresolved errors ({preview})"
        )
    if any(score is None for score in scores):
        raise RuntimeError("evaluation grading left an unresolved response")
    return scores, audit


SCORER.score_deepscaler_tasks = score_deepscaler_tasks

if __name__ == "__main__":
    SCORER.main()
