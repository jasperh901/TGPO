#!/usr/bin/env bash
set -euo pipefail
TGPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TGPO_ENV="${TGPO_EVAL_ENV:-$TGPO_ROOT/.venv-eval}"
python3 "$TGPO_ROOT/scripts/setup_bfcl.py"
uv venv --python 3.10 "$TGPO_ENV"
uv pip sync --python "$TGPO_ENV/bin/python" --extra-index-url https://download.pytorch.org/whl/cu124 \
  --index-strategy unsafe-best-match "$TGPO_ROOT/env/requirements-eval.lock"
uv pip install --python "$TGPO_ENV/bin/python" --no-deps \
  -e "$TGPO_ROOT/third_party/BFCL-v3/berkeley-function-call-leaderboard"
uv pip check --python "$TGPO_ENV/bin/python"
echo "Activate with: source $TGPO_ENV/bin/activate"
