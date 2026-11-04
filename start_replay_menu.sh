#!/usr/bin/env bash
# Input a recording number to replay; --check-only never opens hardware.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PY=${JOINT_DEMO_PYTHON:-$HOME/miniconda3/envs/lerobot/bin/python}
exec "$PY" -m e14.replay_menu "$@"
