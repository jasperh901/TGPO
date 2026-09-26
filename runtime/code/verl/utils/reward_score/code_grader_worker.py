"""JSON-lines PRIME scorer intended to run inside the code sandbox."""

from __future__ import annotations

import contextlib
import base64
import json
import os
import sys

from verl.utils.reward_score.gdpo_code import score_code_response


def main():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        for line in sys.stdin:
            # The parent sends one URL-safe base64 record per line.  Keeping
            # the transport alphabet separate from generated code and PRIME
            # JSON prevents embedded control characters from corrupting the
            # line-delimited protocol.
            payload = base64.b64decode(line.strip(), validate=True)
            task = json.loads(payload.decode("utf-8"))
            try:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    score = score_code_response(
                        response=task["response"],
                        ground_truth=task["ground_truth"],
                        response_tokens=int(task["response_tokens"]),
                        target_length=int(task["target_length"]),
                    )
                result = {"id": int(task["id"]), "score": score}
            except BaseException as error:
                result = {
                    "id": int(task.get("id", -1)),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
