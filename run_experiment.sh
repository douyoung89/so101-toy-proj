#!/usr/bin/env bash
# One command: import existing data if needed -> prepare -> train -> evaluate -> GIF.
set -euo pipefail
cd "$(dirname "$0")"

# Activate conda so101 first, or override PYTHON explicitly.
PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
printf 'Python: %s\n' "$PYTHON"
if [[ ! -f data/source.json ]]; then
  "$PYTHON" scripts/import_data.py
fi
"$PYTHON" scripts/setup_assets.py
"$PYTHON" scripts/prepare_data.py
"$PYTHON" train.py
"$PYTHON" evaluate.py "$@"
"$PYTHON" demo.py
