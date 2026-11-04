#!/usr/bin/env bash
# Dabai + Astra Pro RGB, same joint mapping and startup pose as the single-camera demo.
# --check: passive preflight; --mock: simulated devices. Default starts real joint control.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -f config.joint.dual.local.json && -z "${JOINT_DEMO_CONFIG:-}" ]]; then
  if [[ " $* " == *" --mock "* ]]; then
    export JOINT_DEMO_CONFIG=config.joint.dual.example.json
  else
  echo "请先复制 config.joint.dual.example.json 为 config.joint.dual.local.json 并填写两路相机设备。" >&2
  exit 1
  fi
fi
export JOINT_DEMO_CONFIG=${JOINT_DEMO_CONFIG:-config.joint.dual.local.json}
export JOINT_DEMO_OUTPUT=${JOINT_DEMO_OUTPUT:-data/piper_raw_joint_dual}
export PREVIEW_PORT=${PREVIEW_PORT:-8767}
echo "双 RGB 视角：front=Dabai DC1，astra=Astra Pro；分别保存，不拼接训练图片。"
echo "双画面控制页面：http://127.0.0.1:$PREVIEW_PORT"
exec ./start_joint_demo.sh "$@"
