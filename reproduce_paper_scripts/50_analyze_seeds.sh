#!/usr/bin/env bash
#
# Copyright ETH Zurich 2026
# Licensed under Apache v2.0 see LICENSE for details.
#
# SPDX-License-Identifier: Apache-2.0
#
# Pool a multi-seed run and produce all paper tables/figures, per seed and pooled.
#
# Usage:
#   bash reproduce_paper_scripts/50_analyze_seeds.sh <artifacts_root> <seed> [<seed> ...]
#   e.g. bash reproduce_paper_scripts/50_analyze_seeds.sh artifacts_rebuttal 42 52 62
#
# Steps:
#   1. pool:    <root>/seeds_pooled/models   (per fold/session: mean over seeds)
#   2. analyze: tables/ and figures/ inside every <root>/seed_<s>/ and <root>/seeds_pooled/
#               with the unchanged analysis scripts of utils/III_results_analysis
#   3. spread:  <root>/seeds_pooled/seed_spread/  (mean and std across seeds of every table)

set -o pipefail

ROOT=$1
shift
SEEDS=("$@")
PY=${PYTHON:-python}
if [ -z "$ROOT" ] || [ ${#SEEDS[@]} -lt 2 ]; then
    echo "Usage: $0 <artifacts_root> <seed> <seed> [<seed> ...]"
    exit 2
fi

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO" || exit 2
A=utils/III_results_analysis
SEED_DIRS=()
for s in "${SEEDS[@]}"; do SEED_DIRS+=("$ROOT/seed_$s"); done
POOLED=$ROOT/seeds_pooled
export MPLBACKEND=Agg

run() {
    echo "+ $*"
    "$@" || { echo "[FAILED] $*"; FAILED=1; }
}

analyze() {
    local D=$1
    echo -e "\n========== analysis: $D =========="
    for m in speechnet random_forest; do
        run $PY $A/I_global_intersession_analysis.py --artifacts_dir "$D" --experiment global \
            --model_name $m --model_name_id w1400ms --plot_confusion_matrix
        run $PY $A/I_global_intersession_analysis.py --artifacts_dir "$D" --experiment inter_session \
            --model_name $m --model_name_id w1400ms --plot_confusion_matrix
    done
    run $PY $A/I_global_intersession_analysis.py --artifacts_dir "$D" --experiment inter_session \
        --model_name speechnet --windows_s 0.4 1.4
    run $PY $A/I_global_intersession_analysis.py --artifacts_dir "$D" --experiment inter_session_random_labels \
        --model_name speechnet --model_name_id w1400ms
    run $PY $A/II_infotransrate.py --artifacts_dir "$D" --model_name speechnet
    for w in w1400ms w800ms; do
        run $PY $A/III_ft_results.py --artifacts_dir "$D" --model_base_id $w
    done
    run $PY $A/IV_subject_scaling_analysis.py --artifacts_dir "$D"
    run $PY $A/V_confusion_matrix_figure.py --artifacts_dir "$D"
}

FAILED=0
echo "========== 1. pool seeds ${SEEDS[*]} =========="
$PY $A/00_pool_seeds.py --seed_dirs "${SEED_DIRS[@]}" --out_dir "$POOLED" || exit 1

echo "========== 2. analyze =========="
for D in "${SEED_DIRS[@]}" "$POOLED"; do analyze "$D"; done

echo "========== 3. seed spread =========="
run $PY $A/00_pool_seeds.py --mode spread --seed_dirs "${SEED_DIRS[@]}" --out_dir "$POOLED"

if [ $FAILED = 1 ]; then echo "[DONE WITH FAILURES] see [FAILED] lines above"; exit 1; fi
echo "[DONE] pooled results: $POOLED/{tables,figures}; seed spread: $POOLED/seed_spread"
