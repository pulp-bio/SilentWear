#!/usr/bin/env bash
#
# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Run one experiment group for one seed, with its own output tree, log and manifest entry.
#
# Usage:
#   bash reproduce_paper_scripts/40_run_seed_job.sh <seed> <job> <data_dir> <win_and_feats> <artifacts_root>
#
# Jobs:
#   global_and_rf        global SpeechNet + global RF + inter-session RF (1400 ms)
#   inter_session_sweep  inter-session SpeechNet, windows 0.4..1.4 s
#   ft_and_tfs           incremental fine-tuning + train-from-scratch (800, 1400 ms)
#   random_labels        random-label control (inter-session, SpeechNet)
#   scaling_silent       subject-scaling analysis, silent
#   scaling_vocalized    subject-scaling analysis, vocalized
#
# Outputs: <artifacts_root>/seed_<seed>/{models,tables,figures,logs}
# Manifest: <artifacts_root>/runs_manifest.csv (start, end, seed, job, git commit, exit code)

set -o pipefail

SEED=$1
JOB=$2
DATA=$3
WAF=$4
ROOT=$5
PY=${PYTHON:-python}

if [ -z "$ROOT" ]; then
    echo "Usage: $0 <seed> <job> <data_dir> <win_and_feats> <artifacts_root>"
    exit 2
fi

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || exit 2
OUT=$ROOT/seed_$SEED
mkdir -p "$OUT/logs"

# Local copy of the paper config pointing to the windows folder of this machine
CFG=$OUT/paper_models_config.yaml
sed "s|win_and_feats: .*|win_and_feats: $WAF|" config/paper_models_config.yaml > "$CFG"

SPEECHNET=config/models_configs/speechnet_config.yaml
RF=config/models_configs/random_forest_config.yaml
RUN="$PY reproduce_paper_scripts/30_run_experiments.py --base_config $CFG --data_dir $DATA --artifacts_dir $OUT --seed $SEED"

COMMIT=$(git rev-parse --short HEAD)
DIRTY=$(git diff --quiet HEAD -- . ':!artifacts_rebuttal' 2>/dev/null || echo "-dirty")
START=$(date -Iseconds)
LOG=$OUT/logs/$JOB.log
echo "[$START] seed=$SEED job=$JOB commit=$COMMIT$DIRTY" | tee "$LOG"

case $JOB in
    global_and_rf)
        $RUN --model_config $SPEECHNET --experiment global &&
        $RUN --model_config $RF --experiment global &&
        $RUN --model_config $RF --experiment inter_session --inter_session_windows_s 1.4
        ;;
    inter_session_sweep)
        $RUN --model_config $SPEECHNET --experiment inter_session
        ;;
    ft_and_tfs)
        $RUN --model_config $SPEECHNET --experiment inter_session_ft train_from_scratch \
            --ft_config config/paper_ft_config.yaml --tfs_config config/paper_train_from_scratch_config.yaml
        ;;
    random_labels)
        $PY offline_experiments/V_random_label_control.py --data_dir "$DATA" --win_and_feats "$WAF" \
            --artifacts_dir "$OUT" --seed "$SEED"
        ;;
    scaling_silent | scaling_vocalized)
        $PY offline_experiments/VI_subject_scaling_experiment.py --data_dir "$DATA" --win_and_feats "$WAF" \
            --artifacts_dir "$OUT" --conditions "${JOB#scaling_}" --seed "$SEED"
        ;;
    *)
        echo "Unknown job: $JOB"
        exit 2
        ;;
esac 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}

END=$(date -Iseconds)
MANIFEST=$ROOT/runs_manifest.csv
[ -f "$MANIFEST" ] || echo "start,end,seed,job,commit,exit_code,log" > "$MANIFEST"
echo "$START,$END,$SEED,$JOB,$COMMIT$DIRTY,$STATUS,$LOG" >> "$MANIFEST"
echo "[$END] seed=$SEED job=$JOB exit=$STATUS" | tee -a "$LOG"
exit $STATUS
