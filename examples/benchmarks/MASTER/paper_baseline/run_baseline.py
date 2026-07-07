"""MASTER baseline runner — CSI300 (CN) & SP500 (US), 5 seeds, Qlib backtest.

Adapted from ``examples/benchmarks/MASTER/main.py`` with these changes:
  - ``--market {csi300,sp500}`` selects the config; ``--seeds N`` (default 5);
    ``--smoke`` runs 1 epoch / seed 0 only (fast pipeline check).
  - ``qlib.init`` reads provider/region from the yaml's ``qlib_init`` (no hardcoded CN).
  - No ``sed`` mutation of the yaml (market/universe are fixed per-config-file).
  - Aggregates IC / Rank IC / annualized_return / information_ratio (with & without cost)
    across seeds and prints mean ± std.

Usage (run from this directory, inside the qlib-master conda env):
    python run_baseline.py --market csi300            # full: 5 seeds x 40 epochs
    python run_baseline.py --market sp500 --seeds 5
    python run_baseline.py --market csi300 --smoke    # 1 epoch, seed 0
    python run_baseline.py --market sp500 --only_backtest   # reuse trained checkpoints
"""
import os
import sys
import argparse
from pathlib import Path

import yaml
import numpy as np
import pprint as pp

# Make this directory importable so the yaml `module_path: us_handlers` resolves
# (the SP500 config uses Alpha158US defined in us_handlers.py).
DIRNAME = Path(__file__).absolute().resolve().parent
sys.path.insert(0, str(DIRNAME))

import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord

_MARKET_CONFIG = {
    "csi300": "workflow_config_master_csi300.yaml",
    "sp500": "workflow_config_master_sp500.yaml",
}

METRIC_KEYS = [
    "IC",
    "ICIR",
    "Rank IC",
    "Rank ICIR",
    "1day.excess_return_without_cost.annualized_return",
    "1day.excess_return_without_cost.information_ratio",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.information_ratio",
]


def parse_args():
    p = argparse.ArgumentParser(description="MASTER baseline runner (CSI300 / SP500)")
    p.add_argument("--market", choices=list(_MARKET_CONFIG), default="csi300")
    p.add_argument("--config", default=None, help="override config filename (in this dir)")
    p.add_argument("--seeds", type=int, default=5, help="number of seeds (0 .. N-1)")
    p.add_argument("--only_backtest", action="store_true", help="load saved model, skip training")
    p.add_argument("--smoke", action="store_true", help="1 epoch, seed 0 only (pipeline check)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = DIRNAME / (args.config or _MARKET_CONFIG[args.market])
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    qlib_init = config["qlib_init"]
    qlib.init(**qlib_init)
    print(
        f"[{args.market}] config={cfg_path.name}  "
        f"provider={qlib_init['provider_uri']}  region={qlib_init['region']}"
    )

    if args.smoke:
        config["task"]["model"]["kwargs"]["n_epochs"] = 1
        seeds = [0]
        print("[SMOKE] n_epochs=1, seeds=[0]  (pipeline check only)")
    else:
        seeds = list(range(args.seeds))

    # Cache the preprocessed handler so all seeds share it. The filename is derived from the
    # segment dates via .strftime(), so the yaml dates MUST stay unquoted (datetime.date).
    seg_kwargs = config["task"]["dataset"]["kwargs"]
    h_conf = seg_kwargs["handler"]
    h_path = (
        DIRNAME
        / f'handler_{args.market}_{seg_kwargs["segments"]["train"][0].strftime("%Y%m%d")}'
        f'_{seg_kwargs["segments"]["test"][1].strftime("%Y%m%d")}.pkl'
    )
    if not h_path.exists():
        h = init_instance_by_config(h_conf)
        h.to_pickle(h_path, dump_all=True)
        print("Save preprocessed data to", h_path)
    seg_kwargs["handler"] = f"file://{h_path}"
    dataset = init_instance_by_config(config["task"]["dataset"])

    os.makedirs("./model", exist_ok=True)

    all_metrics = {k: [] for k in METRIC_KEYS}
    for seed in seeds:
        print("------------------------")
        print(f"[{args.market}] seed: {seed}")

        config["task"]["model"]["kwargs"]["seed"] = seed
        model = init_instance_by_config(config["task"]["model"])

        if not args.only_backtest:
            model.fit(dataset=dataset)
        else:
            model.load_model(f"./model/{config['market']}master_{seed}.pkl")

        with R.start(experiment_name=f"{config['market']}_MASTER_seed{seed}"):
            recorder = R.get_recorder()
            SignalRecord(model, dataset, recorder).generate()
            SigAnaRecord(recorder).generate()
            PortAnaRecord(recorder, config["port_analysis_config"], "day").generate()

            metrics = recorder.list_metrics()
            print(metrics)
            for k in all_metrics:
                if k in metrics:
                    all_metrics[k].append(metrics[k])
            pp.pprint(all_metrics)

    print(f"\n================ Summary [{args.market}] ================")
    for k in all_metrics:
        if all_metrics[k]:
            vals = np.array(all_metrics[k], dtype=float)
            print(f"{k}: {vals.mean():.6f} ± {vals.std():.6f}  (n={len(vals)})")
        else:
            print(f"{k}: (not recorded)")


if __name__ == "__main__":
    main()
