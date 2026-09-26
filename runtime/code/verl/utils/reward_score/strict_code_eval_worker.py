"""Evaluation-only, fail-fast full-suite judge; run in the existing OS sandbox.

This does not change the historical training reward. A wrong answer can stop
testing early without changing the binary all-tests-pass result. No partial
test-pass reward is inferred from that prefix.
"""
from __future__ import annotations

import ast
import base64
import contextlib
import copy
import json
import multiprocessing as mp
import os
import re
import resource
import signal
import sys
import time

from .prime_code import testing_util


def extract_code(response):
    final = response.rsplit("</think>", 1)[-1]
    if "<think>" in final:
        return ""
    final = final.replace("<｜end▁of▁sentence｜>", "").strip()
    blocks = re.findall(r"```([^\n`]*)\n(.*?)(?:```|\Z)", final, re.S)
    python_blocks = [body for language, body in blocks
                     if language.strip().lower() in ("python", "python3", "py", "")]
    return (python_blocks[-1] if python_blocks else final).strip()


def raise_timeout(signum, frame):
    raise testing_util.TimeoutException("per-test wall-time limit exceeded")


def normalize_tests(tests):
    """Adapt structured arguments/line arrays without changing their values."""
    tests = copy.deepcopy(tests)
    if tests.get("fn_name") is not None:
        inputs, outputs = [], []
        for arguments, expected in zip(tests["inputs"], tests["outputs"], strict=True):
            if isinstance(arguments, list):
                inputs.append("\n".join(json.dumps(value) for value in arguments))
                outputs.append(json.dumps(expected))
            elif isinstance(arguments, str):
                # Old APPS representation: one JSON argument per line.
                for line in arguments.split("\n"):
                    json.loads(line)
                json.loads(expected)
                inputs.append(arguments)
                outputs.append(expected)
            else:
                raise ValueError("unsupported call-based argument representation")
        tests["inputs"], tests["outputs"] = inputs, outputs
    else:
        for key in ("inputs", "outputs"):
            for i, value in enumerate(tests[key]):
                if isinstance(value, list) and all(isinstance(line, str) for line in value):
                    tests[key][i] = "\n".join(value)
                elif not isinstance(value, str):
                    raise ValueError("unsupported standard-input test representation")
    return tests


def _execute(connection, tests, code, timeout):
    # Child-only changes: run_test disables OS operations and mutates inputs.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    signal.signal(signal.SIGALRM, raise_timeout)
    try:
        with open(os.devnull, "w") as null, contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
            results, metadata = testing_util.run_test(copy.deepcopy(tests), code, timeout=timeout)
        connection.send({"results": [int(value) for value in results],
                         "metadata": {key: str(value)[:1000] for key, value in metadata.items()}})
    except (SystemExit, KeyboardInterrupt) as error:
        connection.send({"results": [-1], "metadata": {"program_termination": repr(error)[:1000]}})
    except BaseException as error:
        connection.send({"results": [], "metadata": {"exception": repr(error)[:1000]}})
    finally:
        signal.alarm(0)
        connection.close()


def score(task, *, program_timeout_signal=None):
    started = time.monotonic()
    tests = task["ground_truth"]
    if isinstance(tests, str):
        tests = json.loads(tests)
    count = len(tests["inputs"])
    if not count or len(tests["outputs"]) != count:
        raise ValueError("empty or misaligned test suite")
    timeout = int(task.get("per_test_timeout", 5))
    if timeout < 1:
        raise ValueError("invalid per-test timeout")
    code = extract_code(task["response"])
    result = {"strict_pass": False, "test_count": count, "tests_executed": 0,
              "prefix_passed": 0, "syntax_valid": False,
              "call_based": tests.get("fn_name") is not None}
    try:
        tests = normalize_tests(tests)
    except (ValueError, TypeError) as error:
        # Apply independently of the answer; never select exclusions by model.
        return {**result, "strict_pass": None, "outcome": "dataset_error",
                "detail": str(error), "seconds": time.monotonic() - started}
    try:
        if not code:
            raise SyntaxError("no final code")
        ast.parse(code)
    except (SyntaxError, ValueError, RecursionError) as error:
        return {**result, "outcome": "invalid_code", "detail": str(error)[:200],
                "seconds": time.monotonic() - started}
    except MemoryError:
        # CPython 3.10's parser raises MemoryError on excessive parenthesis
        # nesting even with ample free RAM. This answer did not compile; it
        # must not abort the worker or be treated as executable Python.
        return {**result, "outcome": "invalid_code",
                "detail": "MemoryError while parsing generated source (compiler resource/nesting limit)",
                "seconds": time.monotonic() - started}
    result["syntax_valid"] = True
    context = mp.get_context("fork")
    reader, writer = context.Pipe(duplex=False)
    child = context.Process(target=_execute, args=(writer, tests, code, timeout))
    child.start()
    writer.close()
    payload = None
    try:
        # Compilation plus EVERY testcase gets its own time allowance. The
        # old wrapper incorrectly allowed only six seconds for the whole suite.
        if not reader.poll(timeout * (count + 1) + 10):
            # Infrastructure/global-watchdog failures must not become model errors.
            raise RuntimeError("full-suite watchdog expired; evaluation incomplete")
        try:
            payload = reader.recv()
        except EOFError:
            pass
    finally:
        reader.close()
        child.join(timeout=1)
        if child.is_alive():
            child.kill()
            child.join()
    if payload is None:
        exit_code = child.exitcode
        if program_timeout_signal is not None and exit_code == -program_timeout_signal:
            return {**result, "outcome": "timeout", "process_exit_code": exit_code,
                    "detail": "per-test OS alarm terminated generated program; partial coverage unknown",
                    "seconds": time.monotonic() - started}
        program_faults = {-signal.SIGSEGV, -signal.SIGABRT, -signal.SIGFPE, -signal.SIGILL, -signal.SIGBUS}
        if exit_code in program_faults or (exit_code is not None and exit_code >= 0):
            return {**result, "outcome": "execution_error", "process_exit_code": exit_code,
                    "detail": "generated program exited before returning test results",
                    "seconds": time.monotonic() - started}
        # SIGKILL could be host OOM or operator intervention: do not silently
        # count it as a model mistake. Preserve a failing infrastructure gate.
        raise RuntimeError(f"judge child vanished with unclassified exit code {exit_code}")
    values, metadata = payload["results"], payload["metadata"]
    if "exception" in metadata:
        raise RuntimeError(f"judge exception requires audit: {metadata['exception']}")
    passed = len(values) == count and all(value == 1 for value in values)
    detail = json.dumps(metadata, ensure_ascii=False)[:1200]
    if passed:
        outcome = "passed"
    elif "timeoutexception" in detail.lower():
        outcome = "timeout"
    elif values and values[-1] == 0:
        outcome = "wrong_answer"
    else:
        outcome = "execution_error"
    return {**result, "strict_pass": passed, "tests_executed": len(values),
            "prefix_passed": sum(value == 1 for value in values),
            "outcome": outcome, "detail": detail,
            "seconds": time.monotonic() - started}


def main():
    for line in sys.stdin:
        task = json.loads(base64.b64decode(line.strip(), validate=True))
        try:
            result = {"id": task["id"], "score": score(task)}
        except BaseException as error:
            result = {"id": task["id"], "error": repr(error)}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
