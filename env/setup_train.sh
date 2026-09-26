#!/usr/bin/env bash
set -euo pipefail
TGPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TGPO_ENV="${TGPO_TRAIN_ENV:-$TGPO_ROOT/.venv-train}"
uv venv --python 3.10 "$TGPO_ENV"
uv pip sync --python "$TGPO_ENV/bin/python" --extra-index-url https://download.pytorch.org/whl/cu121 \
  --index-strategy unsafe-best-match "$TGPO_ROOT/env/requirements-train.lock"
if [[ -n "${TGPO_FLASH_WHEEL:-}" ]]; then
  uv pip install --python "$TGPO_ENV/bin/python" --no-deps "$TGPO_FLASH_WHEEL"
else
  uv pip install --python "$TGPO_ENV/bin/python" --no-build-isolation --no-deps flash-attn==2.6.3
fi
uv pip install --python "$TGPO_ENV/bin/python" --no-deps -e "$TGPO_ROOT/runtime/math"
uv pip check --python "$TGPO_ENV/bin/python"
echo "Activate with: source $TGPO_ENV/bin/activate"
