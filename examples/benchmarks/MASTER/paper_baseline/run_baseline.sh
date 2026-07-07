#!/bin/bash
# MASTER baseline — run CSI300 (CN) and SP500 (US).
#
# Usage (from anywhere):
#   bash run_baseline.sh smoke     # 1 epoch, seed 0, both markets  (pipeline check, ~minutes)
#   bash run_baseline.sh full      # 5 seeds x 40 epochs, both markets  (LONG, GPU-intensive)
#
# Results are written to ./model/ (checkpoints), ./mlruns/ (qlib/mlflow records), and
# handler_*.pkl (cached preprocessed features). Mean ± std across seeds is printed at the end.
set -e
cd "$(dirname "$0")"

# activate the qlib-master env (self-contained — works without manual activation)
eval "$(conda shell.bash hook)" 2>/dev/null
conda activate qlib-master

MODE="${1:-smoke}"

run_market () {
    local market=$1
    if [ "$MODE" = "smoke" ]; then
        echo "=== [SMOKE] $market (1 epoch, seed 0) ==="
        python -u run_baseline.py --market "$market" --smoke
    elif [ "$MODE" = "full" ]; then
        # Run each seed in a FRESH python process. Running multiple seeds in one process
        # accumulates RAM (dataset/model/recorder not fully released between seeds) and can
        # OOM on large markets — e.g. SP500 peaks ~11GB/seed and exhausts a 15GB machine by
        # seed 2. One process per seed fully releases memory between seeds.
        echo "=== [FULL] $market (5 seeds x 40 epochs, one fresh process per seed) ==="
        for s in 0 1 2 3 4; do
            echo "--- $market seed $s ---"
            python -u run_baseline.py --market "$market" --seed-start "$s" --seeds 1
        done
    else
        echo "Unknown mode: '$MODE' (use 'smoke' or 'full')"; exit 1
    fi
}

run_market csi300
run_market sp500
echo "=== done ($MODE) ==="
