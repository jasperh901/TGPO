"""GDPO paper rewards derived from the official PRIME code evaluator."""

from __future__ import annotations

import ast
import json

from verl.utils.reward_score.prime_code import evaluate_code


class CodeGraderError(RuntimeError):
    """The grader failed to return a complete, interpretable result."""


def _test_result(metadata: dict) -> tuple[bool, bool]:
    """Return ``(passed, execution_error)`` for one PRIME test case."""
    try:
        raw = metadata["test_case"]["res"]
        result = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (KeyError, SyntaxError, ValueError, TypeError) as error:
        raise CodeGraderError("PRIME metadata is missing a valid test_case.res") from error
    if not isinstance(result, list) or len(result) != 1:
        raise CodeGraderError(f"expected one PRIME result per case, got {result!r}")
    value = result[0]
    passed = value is True
    execution_error = isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0
    if not passed and not execution_error and value is not False:
        raise CodeGraderError(f"unknown PRIME result value: {value!r}")
    return passed, execution_error


def score_code_response(response: str, ground_truth: str | dict, response_tokens: int,
                        target_length: int = 4000) -> dict[str, float]:
    """Compute pass-rate, conditioned-length, and bug-free rewards."""
    try:
        test_cases = ground_truth if isinstance(ground_truth, dict) else json.loads(ground_truth)
    except (TypeError, json.JSONDecodeError) as error:
        raise CodeGraderError("ground truth is not valid PRIME test-case JSON") from error
    if not isinstance(test_cases, dict) or not isinstance(test_cases.get("inputs"), list):
        raise CodeGraderError("ground truth does not contain a PRIME inputs list")
    if len(test_cases["inputs"]) != len(test_cases.get("outputs", [])):
        raise CodeGraderError("PRIME input/output test counts differ")
    tested = min(len(test_cases["inputs"]), 10)
    if tested < 1:
        raise CodeGraderError("PRIME problem contains no test cases")

    success, metadata = evaluate_code(response, test_cases)
    if bool(success):
        pass_reward = 1.0
        bug_reward = 1.0
    else:
        if not isinstance(metadata, list) or len(metadata) != tested:
            raise CodeGraderError(
                f"PRIME returned {type(metadata).__name__} metadata for {tested} cases"
            )
        outcomes = [_test_result(item) for item in metadata]
        pass_reward = sum(passed for passed, _ in outcomes) / tested
        bug_reward = float(not any(execution_error for _, execution_error in outcomes))

    conditioned_length = float(pass_reward == 1.0 and response_tokens <= target_length)
    return {
        "reward": pass_reward + conditioned_length + bug_reward,
        "pass": pass_reward,
        "conditioned_length": conditioned_length,
        "bug_free": bug_reward,
        "length_exceeded": float(response_tokens > target_length),
    }
