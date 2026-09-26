"""Full-suite evaluation using script globals and real text/binary stdin.

Evaluation only. Keep the frozen PRIME output comparator and function adapter;
avoid rewriting generated scripts as nested functions. Run only in OS sandbox.
"""
import builtins
import io
import resource
import signal
import sys
from types import ModuleType
from unittest.mock import patch, mock_open

from . import strict_code_eval_worker as legacy

_legacy_execute = legacy._execute
_legacy_score = legacy.score
_legacy_run_test = legacy.testing_util.run_test
_legacy_guard = legacy.testing_util.reliability_guard
ADDRESS_SPACE_LIMIT = 8 * 1024**3


class BinaryOutput:
    def __init__(self, stream):
        self.stream = stream

    def write(self, value):
        self.stream.write(value.decode("utf-8"))
        return len(value)

    def flush(self):
        self.stream.flush()


class TextOutput:
    def __init__(self, stream):
        self.stream = stream
        self.buffer = BinaryOutput(stream)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def script_call(method, inputs):
    stream = io.TextIOWrapper(io.BytesIO(inputs.encode("utf-8")), encoding="utf-8")
    with patch("sys.stdin", stream), patch("sys.stdout", TextOutput(sys.stdout)), \
            patch("builtins.open", mock_open(read_data=inputs)):
        try:
            method()
        except SystemExit as error:
            if error.code not in (None, 0):
                raise RuntimeError(f"program exit status {error.code}") from error


def native_run_test(in_outs, test=None, debug=False, timeout=5):
    if in_outs.get("fn_name") is not None:
        return _legacy_run_test(in_outs, test=test, debug=debug, timeout=timeout)
    try:
        compiled = compile(test, "<generated_script>", "exec")
    except (SyntaxError, ValueError) as error:
        return [-2], {"error": repr(error)}

    def script_module(name, ignored_transformed_source):
        module = ModuleType(name)

        def code():
            namespace = {"__name__": "__main__", "__builtins__": builtins.__dict__}
            exec(compiled, namespace, namespace)

        module.code = code
        return module

    with patch.object(legacy.testing_util, "module_from_string", script_module), \
            patch.object(legacy.testing_util, "call_method", script_call):
        return _legacy_run_test(in_outs, test=test, debug=debug, timeout=timeout)


def native_guard():
    _legacy_guard()
    # A Python exception handler cannot interrupt some C-level allocations or
    # computations. The kernel must enforce the SAME per-test alarm deadline.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    # Normal Python exits are caught by the test runner. Disabling these made
    # valid early-return scripts fail with NoneType-is-not-callable errors.
    builtins.exit = sys.exit
    builtins.quit = sys.exit


def native_execute(connection, tests, code, timeout):
    # Includes imported runtime mappings (~4.5 GiB virtual, much less resident).
    # Identical cap for every method; prevent one answer exhausting host RAM.
    resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE_LIMIT, ADDRESS_SPACE_LIMIT))
    legacy.testing_util.run_test = native_run_test
    legacy.testing_util.reliability_guard = native_guard
    _legacy_execute(connection, tests, code, timeout)


def score(task):
    with patch.object(legacy, "_execute", native_execute):
        result = _legacy_score(task, program_timeout_signal=signal.SIGALRM)
    if result["outcome"] == "execution_error" and "memoryerror" in result.get("detail", "").lower():
        result["outcome"] = "memory_limit"
    return result


if __name__ == "__main__":
    legacy.score = score
    legacy.main()
