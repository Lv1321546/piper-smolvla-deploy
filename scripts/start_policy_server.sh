#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PY=${PIPER_SERVER_PYTHON:-$HOME/.conda/envs/piper_smolvla_20261105/bin/python}
if [[ ! -x "$PY" ]]; then
  echo "Set PIPER_SERVER_PYTHON to a server Python with the training-compatible LeRobot environment." >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES=${PIPER_GPU:-${CUDA_VISIBLE_DEVICES:-1}}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
# Check physical GPU ownership before importing torch. Never evict another job.
"$PY" - "$CUDA_VISIBLE_DEVICES" <<'PYGPU'
import subprocess,sys
selected=sys.argv[1]
if ',' in selected or not selected:raise SystemExit('Select exactly one physical GPU index or UUID')
lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
cards=[line.split(',') for line in lines.splitlines()]
matches=[c for c in cards if selected in (c[0].strip(),c[1].strip())]
if len(matches)!=1:raise SystemExit('Selected GPU not found: '+selected)
card=matches[0];uuid=card[1].strip()
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
if any(line.split(',')[0].strip()==uuid for line in apps.splitlines()) or float(card[2])>=256:
    raise SystemExit('Selected GPU is occupied; inference server not started: '+uuid)
print('Using idle physical GPU:',card[0].strip(),uuid,flush=True)
PYGPU
exec "$PY" -u -m server.inference_server --config config.inference.example.json --warmup 2 "$@"
