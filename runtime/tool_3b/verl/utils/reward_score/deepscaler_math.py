"""DeepScaleR-compatible boxed-answer reward used during math RL training."""

from verl.utils.reward_score.prime_math import (
    _is_frac,
    _last_boxed_only_string,
    _normalize,
    _str_is_int,
    are_equal_under_sympy,
    math_normalize,
    split_tuple,
)
from verl.utils.py_functional import timeout_limit


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    return math_normalize.normalize_answer(given_answer) == math_normalize.normalize_answer(ground_truth)


def _sympy_equal_with_timeout(ground_truth: str, given_answer: str,
                              timeout_seconds: float) -> bool:
    if timeout_seconds <= 0:
        raise ValueError('symbolic timeout must be positive')
    if timeout_seconds == 10:
        return are_equal_under_sympy(ground_truth, given_answer)

    raw_sympy_equal = getattr(are_equal_under_sympy, '__wrapped__', None)
    if raw_sympy_equal is None:
        raise RuntimeError('PRIME symbolic equality implementation is not retryable')
    return timeout_limit(seconds=timeout_seconds)(raw_sympy_equal)(
        ground_truth, given_answer
    )


def grade_answer_sympy(given_answer: str, ground_truth: str,
                       symbolic_timeout: float = 10,
                       propagate_errors: bool = False) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)
    if ground_truth_normalized is None or given_normalized is None:
        return False
    if ground_truth_normalized == given_normalized:
        return True
    if not given_normalized:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)
    if (
        len(ground_truth_elems) > 1
        and (
            ground_truth_normalized[0] != given_normalized[0]
            or ground_truth_normalized[-1] != given_normalized[-1]
        )
    ) or len(ground_truth_elems) != len(given_elems):
        return False

    # Symbolic simplification can take unbounded time for malformed model
    # outputs. Keep PRIME's process-isolated timeout around the same equality
    # logic so one response cannot stall the complete training batch.
    for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
        if _is_frac(ground_truth_elem) and _is_frac(given_elem):
            is_correct = ground_truth_elem == given_elem
        elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
            is_correct = False
        else:
            try:
                is_correct = _sympy_equal_with_timeout(
                    ground_truth_elem, given_elem, symbolic_timeout
                )
            except Exception as error:
                if propagate_errors:
                    raise
                print(
                    f"Symbolic comparison failed: {type(error).__name__}: {error}",
                    flush=True,
                )
                is_correct = False
        if not is_correct:
            return False
    return True


def compute_score(model_output: str, ground_truth: str,
                  symbolic_timeout: float = 10,
                  propagate_errors: bool = False):
    extracted_answer = _last_boxed_only_string(str(model_output))
    if extracted_answer is None:
        return False, None
    ground_truth = str(ground_truth)
    is_correct = grade_answer_mathd(extracted_answer, ground_truth) or grade_answer_sympy(
        extracted_answer,
        ground_truth,
        symbolic_timeout=symbolic_timeout,
        propagate_errors=propagate_errors,
    )
    return bool(is_correct), extracted_answer
