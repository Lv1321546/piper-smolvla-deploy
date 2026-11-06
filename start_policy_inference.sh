#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
PY=${PIPER_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}
if [[ ! -x "$PY" ]]; then
  echo "Set PIPER_PYTHON to the E14 LeRobot Python executable." >&2
  exit 1
fi
if [[ ! -f config.local.inference.json ]]; then
  cp config.inference.example.json config.local.inference.json
  echo "Created config.local.inference.json using the two training camera identities."
fi
exec "$PY" -u -m e14.deploy_policy --config config.local.inference.json "$@"
