#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -n "${MASTER_PROTOCOL_CACHE:-}" ]]; then
    cache_dir=$MASTER_PROTOCOL_CACHE
    mkdir -p "$cache_dir"
else
    cache_dir=$(mktemp -d /tmp/master_protocol.XXXXXX)
    trap 'rm -rf "$cache_dir"' EXIT
fi

cd "$script_dir"
python export_protocol_results.py --prepare-cache "$cache_dir" "$@"
while IFS=$'\t' read -r market tag score_path; do
    output_path="$cache_dir/${market}_${tag}.pkl"
    if [[ ! -s "$output_path" ]]; then
        python export_protocol_results.py \
            --worker-market "$market" \
            --worker-score "$score_path" \
            --worker-output "$output_path"
        sleep 10
    fi
done < "$cache_dir/jobs.tsv"
python export_protocol_results.py --backtest-cache "$cache_dir" "$@"
