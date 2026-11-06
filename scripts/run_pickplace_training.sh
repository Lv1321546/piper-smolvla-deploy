#!/usr/bin/env bash
set -euo pipefail
GPU_UUID=$1
RUN_TAG=$2
WORK=$3
PY=$4
CODE=$5
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export HF_HOME="$WORK/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$WORK/cache/torch"
export TMPDIR="$WORK/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$CODE"
export OMP_NUM_THREADS=4
mkdir -p "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TMPDIR" "$WORK/reports/$RUN_TAG"
cd "$CODE"
exec > >(tee -a "$WORK/reports/$RUN_TAG/workflow.log") 2>&1
STAGE=starting
finish() { result=$?; printf '%s stage=%s exit=%s\n' "$(date -Is)" "$STAGE" "$result" > "$WORK/reports/$RUN_TAG/workflow_status.txt"; }
trap finish EXIT
SMOKE="$WORK/runs/smoke_$RUN_TAG"
BASELINE="$WORK/runs/baseline_seed42_$RUN_TAG"
BUNDLE=/mnt/nas_ailab_434/Personal_File/eureka/models/smolvla_bundle
common=(--dataset-root "$WORK/datasets/pickplace_dual_v1" --base-model "$BUNDLE/smolvla_base" --vlm-path "$BUNDLE/smolvlm" --steps 20000 --batch-size 16 --seed 42 --num-workers 4 --skip-final-evaluation --execute)
wait_for_gpu() {
  "$PY" - "$GPU_UUID" <<'PYGPU'
import sys,time
from server.training_queue import scan
uuid=sys.argv[1]
while True:
    try:
        if uuid in {g['uuid'] for g in scan(None)}:
            time.sleep(3)
            if uuid in {g['uuid'] for g in scan(None)}:break
    except Exception as exc:print('GPU check failed:',exc,flush=True)
    print('Waiting for selected GPU to be idle:',uuid,flush=True)
    time.sleep(60)
PYGPU
}
printf '%s GPU=%s\n' "$(date -Is)" "$GPU_UUID"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
STAGE=smoke_200
wait_for_gpu
"$PY" -m server.train_policy "${common[@]}" --output-dir "$SMOKE" --stop-after 200 --save-freq 200
STAGE=resume_to_300
wait_for_gpu
"$PY" -m server.train_policy "${common[@]}" --output-dir "$SMOKE" --stop-after 300 --save-freq 200 --resume "$SMOKE/checkpoints/last"
STAGE=smoke_acceptance
"$PY" -m server.training_acceptance --run-dir "$SMOKE"
STAGE=smoke_validation
wait_for_gpu
"$PY" -m server.evaluate_training --run-dir "$SMOKE" --checkpoint "$SMOKE/checkpoints/last/pretrained_model" --anchors-per-episode 2 --output "$WORK/reports/$RUN_TAG/smoke_validation"
STAGE=baseline_20000
wait_for_gpu
"$PY" -m server.train_policy "${common[@]}" --output-dir "$BASELINE" --stop-after 20000 --save-freq 2000
STAGE=checkpoint_selection
wait_for_gpu
"$PY" -m server.evaluate_training --run-dir "$BASELINE" --anchors-per-episode 50 --full-finalists --export-selected --output "$WORK/reports/$RUN_TAG/baseline_validation"
STAGE=archive
ARCHIVE=/mnt/nas_ailab_434/Personal_File/eureka/piper_smolvla/archive/$RUN_TAG
mkdir -p "$ARCHIVE/reports" "$ARCHIVE/code"
rsync -r --inplace --no-perms --no-owner --no-group "$WORK/reports/$RUN_TAG/" "$ARCHIVE/reports/"
rsync -r --inplace --no-perms --no-owner --no-group --exclude=__pycache__ "$CODE/server" "$CODE/shared" "$ARCHIVE/code/"
"$PY" - "$WORK/reports/$RUN_TAG/baseline_validation/selection.json" "$ARCHIVE" <<'PYARCHIVE'
import json,subprocess,sys
from pathlib import Path
selection=json.loads(Path(sys.argv[1]).read_text())
for name in ('selected','runner_up'):
    if selection.get(name):
        target=Path(sys.argv[2])/name
        target.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-r','--inplace','--no-perms','--no-owner','--no-group',selection[name]+'/',str(target)+'/'],check=True)
PYARCHIVE
STAGE=complete
