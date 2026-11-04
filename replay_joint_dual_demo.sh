#!/usr/bin/env bash
# Same smooth joint replay; reads the separate dual-camera recording root.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export JOINT_DEMO_CONFIG=${JOINT_DEMO_CONFIG:-config.joint.dual.local.json}
export JOINT_DEMO_OUTPUT=${JOINT_DEMO_OUTPUT:-data/piper_raw_joint_dual}
exec ./replay_joint_demo.sh "$@"
