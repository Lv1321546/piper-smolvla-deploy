#!/usr/bin/env bash
# Default: validate latest episode without hardware. --execute: original-speed replay.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PY=${JOINT_DEMO_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}
exec "$PY" -m e14.replay_joint --root "${JOINT_DEMO_OUTPUT:-data/piper_raw_joint}" \
  --config "${JOINT_DEMO_CONFIG:-config.local.json}" "$@"
