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
        echo "=== [FULL] $market (5 seeds x 40 epochs) ==="
        python -u run_baseline.py --market "$market" --seeds 5
    else
        echo "Unknown mode: '$MODE' (use 'smoke' or 'full')"; exit 1
    fi
}

run_market csi300
run_market sp500
echo "=== done ($MODE) ==="
