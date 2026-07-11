"""Export existing MASTER artifacts to Baseline Results Protocol v1.0."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluation_metrics import portfolio_metrics, prediction_metrics


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
MARKET_CONFIGS = {
    "csi300": HERE / "workflow_config_master_csi300.yaml",
    "sp500": HERE / "workflow_config_master_sp500.yaml",
}
MODEL = "master"
SEED_COLUMNS = [
    "market", "model", "seed", "IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "MDD",
    "Sharpe", "Sortino", "Calmar", "num_test_days", "pred_path_or_ckpt_path",
]
METRICS = ["IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"]
STANDARD_STRATEGY = {
    "class": "TopkDropoutStrategy",
    "module_path": "qlib.contrib.strategy.signal_strategy",
    "kwargs": {
        "topk": 30,
        "n_drop": 5,
        "method_sell": "bottom",
        "method_buy": "top",
        "hold_thresh": 1,
        "only_tradable": False,
        "forbid_all_trade_at_limit": True,
        "risk_degree": 0.95,
    },
}
STANDARD_EXECUTOR = {
    "class": "SimulatorExecutor",
    "module_path": "qlib.backtest.executor",
    "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--ensemble-method", choices=["avg_none", "avg_zscore", "avg_rank"], default="avg_none")
    parser.add_argument("--skip-ensemble", action="store_true", help="export seed results without running ensemble backtests")
    parser.add_argument("--prepare-cache", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--backtest-cache", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-market", choices=list(MARKET_CONFIGS), help=argparse.SUPPRESS)
    parser.add_argument("--worker-score", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def relpath(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def discover_runs() -> dict[str, dict[int, Path]]:
    runs: dict[str, dict[int, Path]] = {}
    for experiment_meta in (HERE / "mlruns").glob("*/meta.yaml"):
        experiment = yaml.safe_load(experiment_meta.read_text())
        name = experiment.get("name", "")
        if "_MASTER_seed" not in name:
            continue
        market, seed_text = name.split("_MASTER_seed", 1)
        candidates = []
        for run_meta in experiment_meta.parent.glob("*/meta.yaml"):
            metadata = yaml.safe_load(run_meta.read_text())
            run_dir = run_meta.parent
            required = [
                run_dir / "artifacts/pred.pkl",
                run_dir / "artifacts/label.pkl",
                run_dir / "artifacts/portfolio_analysis/report_normal_1day.pkl",
            ]
            if metadata.get("status") == 3 and all(path.is_file() for path in required):
                candidates.append((int(metadata.get("end_time") or 0), run_dir))
        if candidates:
            runs.setdefault(market, {})[int(seed_text)] = max(candidates)[1]
    return runs


def load_series(path: Path) -> pd.Series:
    value = pd.read_pickle(path)
    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise ValueError(f"expected one column in {path}")
        return value.iloc[:, 0]
    return value


def normalized_ensemble(predictions: list[pd.Series], method: str) -> pd.Series:
    frame = pd.concat(predictions, axis=1, join="inner")
    if method == "avg_zscore":
        frame = frame.groupby(level="datetime").transform(
            lambda values: (values - values.mean()) / (values.std(ddof=0) or np.nan)
        )
    elif method == "avg_rank":
        frame = frame.groupby(level="datetime").rank(pct=True)
    result = frame.mean(axis=1)
    result.name = "score"
    return result.dropna()


def standard_backtest_config(score: pd.Series, config: dict) -> dict:
    test_start, test_end = config["task"]["dataset"]["kwargs"]["segments"]["test"]
    native_exchange = config["port_analysis_config"]["backtest"]["exchange_kwargs"]
    strategy = json.loads(json.dumps(STANDARD_STRATEGY))
    strategy["kwargs"]["signal"] = score.to_frame()
    return {
        "strategy": strategy,
        "executor": STANDARD_EXECUTOR,
        "start_time": str(test_start),
        "end_time": str(test_end),
        "account": 100000000,
        "benchmark": config["benchmark"],
        "exchange_kwargs": {
            "freq": "day",
            "codes": config["market"],
            "limit_threshold": native_exchange.get("limit_threshold"),
            "deal_price": native_exchange.get("deal_price", "close"),
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 0,
        },
    }


def run_standard_backtest(score: pd.Series, config: dict) -> pd.DataFrame:
    from qlib.backtest import backtest

    portfolio, _ = backtest(**standard_backtest_config(score, config))
    report, _ = portfolio["1day"]
    curve = pd.DataFrame(index=report.index)
    curve.index.name = "datetime"
    curve["daily_ret_gross"] = report["return"]
    curve["cost"] = report["cost"]
    curve["daily_ret_net"] = curve["daily_ret_gross"] - curve["cost"]
    curve["bench_ret"] = report["bench"]
    curve["nav"] = (1 + curve["daily_ret_net"]).cumprod()
    curve["bench_nav"] = (1 + curve["bench_ret"]).cumprod()
    return curve.reset_index()


def run_standard_backtest_isolated(score: pd.Series, market: str, tag: str, cache: Path | None) -> pd.DataFrame:
    if cache is not None:
        path = cache / f"{market}_{tag}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"missing standard backtest cache: {path}")
        return pd.read_pickle(path)
    with tempfile.TemporaryDirectory(prefix="master_protocol_") as temp_dir:
        score_path = Path(temp_dir) / "score.pkl"
        output_path = Path(temp_dir) / "curve.pkl"
        score.to_pickle(score_path)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-market", market,
                "--worker-score", str(score_path),
                "--worker-output", str(output_path),
            ],
            cwd=HERE,
            check=True,
        )
        return pd.read_pickle(output_path)


def worker_main(args: argparse.Namespace) -> None:
    import qlib

    config = yaml.safe_load(MARKET_CONFIGS[args.worker_market].read_text())
    qlib.init(**config["qlib_init"])
    score = pd.read_pickle(args.worker_score)
    run_standard_backtest(score, config).to_pickle(args.worker_output)


def prepare_cache(cache: Path, runs: dict[str, dict[int, Path]], ensemble_method: str, skip_ensemble: bool) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    jobs = []
    for market, seed_runs in sorted(runs.items()):
        predictions = []
        for seed, run_dir in sorted(seed_runs.items()):
            score = load_series(run_dir / "artifacts/pred.pkl")
            score_path = cache / f"{market}_seed{seed}_score.pkl"
            score.to_pickle(score_path)
            jobs.append((market, f"seed{seed}", score_path))
            predictions.append(score)
        if not skip_ensemble and len(predictions) >= 2:
            score_path = cache / f"{market}_ensemble_score.pkl"
            normalized_ensemble(predictions, ensemble_method).to_pickle(score_path)
            jobs.append((market, f"ensemble_{ensemble_method}", score_path))
    with (cache / "jobs.tsv").open("w") as stream:
        for market, tag, score_path in jobs:
            stream.write(f"{market}\t{tag}\t{score_path}\n")


def aggregate(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (market, model), group in seed_metrics.groupby(["market", "model"], sort=True):
        row = {"market": market, "model": model}
        for metric in METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)
    columns = ["market", "model"] + [item for metric in METRICS for item in (f"{metric}_mean", f"{metric}_std")]
    return pd.DataFrame(rows, columns=columns)


def display_table(aggregates: pd.DataFrame) -> pd.DataFrame:
    table = aggregates[["market", "model"]].copy()
    for metric in METRICS:
        table[metric] = aggregates.apply(lambda row: f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_std']:.4f}", axis=1)
    return table


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.worker_market:
        if args.worker_score is None or args.worker_output is None:
            raise ValueError("worker mode requires score and output paths")
        worker_main(args)
        return
    out = args.out.resolve()
    for subdir in ("metrics", "tables", "metadata", "diagnostics"):
        (out / subdir).mkdir(parents=True, exist_ok=True)
    runs = discover_runs()
    configs = {market: yaml.safe_load(path.read_text()) for market, path in MARKET_CONFIGS.items()}
    expected = {market: sorted(seed_runs) for market, seed_runs in runs.items()}
    if set(expected) != set(MARKET_CONFIGS) or any(len(seeds) < 1 for seeds in expected.values()):
        raise RuntimeError(f"incomplete market discovery: {expected}")
    if args.prepare_cache is not None:
        prepare_cache(args.prepare_cache.resolve(), runs, args.ensemble_method, args.skip_ensemble)
        return

    seed_rows = []
    selected_artifacts = {}
    prediction_coverage = {}
    for market in sorted(runs):
        config = configs[market]
        selected_artifacts[market] = {}
        prediction_coverage[market] = {}
        for seed, run_dir in sorted(runs[market].items()):
            pred_path = run_dir / "artifacts/pred.pkl"
            label_path = run_dir / "artifacts/label.pkl"
            report_path = run_dir / "artifacts/portfolio_analysis/report_normal_1day.pkl"
            pred = load_series(pred_path)
            label = load_series(label_path)
            native_report = pd.read_pickle(report_path)
            calendar = pd.DatetimeIndex(native_report.index).sort_values()
            pred_dates = pd.DatetimeIndex(pred.index.get_level_values("datetime").unique()).sort_values()
            coverage_complete = pred_dates.equals(calendar)
            if not coverage_complete:
                raise RuntimeError(
                    f"{market} seed {seed} prediction dates do not cover the test calendar: "
                    f"prediction={pred_dates.min()}..{pred_dates.max()} ({len(pred_dates)}), "
                    f"calendar={calendar.min()}..{calendar.max()} ({len(calendar)})"
                )
            curve = run_standard_backtest_isolated(pred, market, f"seed{seed}", args.backtest_cache)
            row = {"market": market, "model": MODEL, "seed": seed}
            row.update(prediction_metrics(pred, label))
            row.update(portfolio_metrics(curve["daily_ret_net"]))
            row["pred_path_or_ckpt_path"] = relpath(pred_path)
            seed_rows.append(row)
            selected_artifacts[market][str(seed)] = {
                "run_id": run_dir.name,
                "prediction": relpath(pred_path),
                "label": relpath(label_path),
                "daily_backtest": relpath(report_path),
            }
            prediction_coverage[market][str(seed)] = {
                "prediction_start": str(pred_dates.min().date()),
                "prediction_end": str(pred_dates.max().date()),
                "test_calendar_start": str(calendar.min().date()),
                "test_calendar_end": str(calendar.max().date()),
                "prediction_days": len(pred_dates),
                "test_calendar_days": len(calendar),
                "calendar_source": "native Qlib report index from the same declared test split",
                "complete": coverage_complete,
            }
    seed_frame = pd.DataFrame(seed_rows, columns=SEED_COLUMNS)
    seed_frame.to_csv(out / "metrics/seed_metrics.csv", index=False)
    aggregate_frame = aggregate(seed_frame)
    aggregate_frame.to_csv(out / "metrics/aggregate_metrics.csv", index=False)
    display_table(aggregate_frame).to_csv(out / "tables/seed_mean_std.csv", index=False)

    ensemble_enabled = not args.skip_ensemble and all(len(seeds) >= 2 for seeds in expected.values())
    ensemble_rows = []
    if ensemble_enabled:
        curve_dir = out / "curves/ensemble"
        curve_dir.mkdir(parents=True, exist_ok=True)
        for market in sorted(runs):
            seed_runs = runs[market]
            predictions = [load_series(seed_runs[seed] / "artifacts/pred.pkl") for seed in sorted(seed_runs)]
            score = normalized_ensemble(predictions, args.ensemble_method)
            label = load_series(seed_runs[sorted(seed_runs)[0]] / "artifacts/label.pkl").reindex(score.index)
            curve = run_standard_backtest_isolated(score, market, f"ensemble_{args.ensemble_method}", args.backtest_cache)
            curve.to_csv(curve_dir / f"{market}_{MODEL}.csv", index=False)
            row = {"market": market, "model": MODEL, "ensemble_method": args.ensemble_method}
            row.update(prediction_metrics(score, label))
            row.update(portfolio_metrics(curve["daily_ret_net"]))
            row["seeds"] = ",".join(map(str, sorted(seed_runs)))
            row["pred_paths"] = ",".join(relpath(seed_runs[seed] / "artifacts/pred.pkl") for seed in sorted(seed_runs))
            ensemble_rows.append(row)
        ensemble_columns = ["market", "model", "ensemble_method"] + METRICS + ["num_test_days", "seeds", "pred_paths"]
        ensemble_frame = pd.DataFrame(ensemble_rows, columns=ensemble_columns)
        ensemble_frame.to_csv(out / "metrics/ensemble_metrics.csv", index=False)
        ensemble_table = ensemble_frame.copy()
        ensemble_table[METRICS] = ensemble_table[METRICS].map(lambda value: f"{value:.4f}")
        ensemble_table.to_csv(out / "tables/ensemble.csv", index=False)

    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=True).stdout.strip()
    try:
        qlib_version = importlib.metadata.version("pyqlib")
    except importlib.metadata.PackageNotFoundError:
        qlib_version = "unknown"
    market_details = {}
    for market, config in configs.items():
        dataset = config["task"]["dataset"]["kwargs"]
        backtest = config["port_analysis_config"]["backtest"]
        market_details[market] = {
            "provider_uri": config["qlib_init"]["provider_uri"],
            "region": config["qlib_init"]["region"],
            "benchmark": backtest["benchmark"],
            "stock_pool": config["market"],
            "instruments": "Qlib dynamic historical constituents for the configured stock pool",
            "deal_price": backtest["exchange_kwargs"].get("deal_price", "close"),
            "limit_threshold": backtest["exchange_kwargs"].get("limit_threshold"),
            "untradable_handling": "Qlib Exchange order checks; only_tradable=false for ranking candidates",
            "data_end_date": str(config["data_handler_config"]["end_time"]),
            "market_gate_indices": config["market_data_handler_config"]["market_indices"],
        }
    eval_config = {
        "schema_version": "1.0",
        "baseline": "MASTER",
        "models": [MODEL],
        "markets": sorted(configs),
        "seeds": expected,
        "periods": {"train": ["2009-01-01", "2020-12-31"], "valid": ["2021-01-01", "2022-12-31"], "test": ["2023-01-01", "2025-12-31"]},
        "label": "Ref($close, -5) / Ref($close, -1) - 1",
        "standard_backtest": {
            "strategy": STANDARD_STRATEGY,
            "executor": STANDARD_EXECUTOR,
            "frequency": "day",
            "account": 100000000,
            "costs": {"open_cost": 0.0005, "close_cost": 0.0015, "min_cost": 0},
            "deal_price": "close",
            "trade_unit": "Qlib region market configuration; not overridden",
            "positioning": "long-only, no leverage, risk_degree=0.95",
            "qlib_version": qlib_version,
            "applies_to": ["seed", "ensemble"],
        },
        "market_details": market_details,
        "signal_alignment": {"signal_date": "t-1", "trade_date": "t", "qlib_internal_shift": 1, "manual_shift_applied": False, "label_horizon": "5 trading days forward: Ref($close,-5)/Ref($close,-1)-1"},
        "return_semantics": {"return_field": "report.return", "return_is_gross": True, "cost_field": "report.cost", "net_formula": "report.return - report.cost", "evidence": "qlib/workflow/record_temp.py computes with-cost excess return as report.return - report.bench - report.cost"},
        "metric_convention": {"annualization": 252, "return_type": "daily simple return converted with log1p", "std_ddof": 1, "risk_free_rate": 0, "MAR_daily": 0, "AR": "exp(mean(log1p(r_net))*252)-1", "STD": "std(log1p(r_net),ddof=1)*sqrt(252)", "MDD": "min([1,exp(cumsum(g))]/cummax-1)", "Sharpe": "sqrt(252)*mean(g)/std(g,ddof=1)", "Sortino": "sqrt(252)*mean(g)/sqrt(mean(min(g,0)^2))", "Calmar": "AR/abs(MDD)", "ICIR_annualized": False},
        "ensemble": {"enabled": ensemble_enabled, "methods": [args.ensemble_method] if ensemble_enabled else [], "join": "inner", "normalize": args.ensemble_method.removeprefix("avg_"), "score_formula": "row-wise mean across aligned seed scores", "ranking_metrics_source": "recomputed from ensemble score and aligned test label", "backtest": "same Qlib strategy, costs, stock pool, and test period as seeds"},
        "selected_artifacts": selected_artifacts,
        "prediction_coverage": prediction_coverage,
        "run_selection": "latest FINISHED MLflow run with prediction, label, and daily backtest artifacts for each market/seed",
        "portfolio_metric_source": "fresh standard Qlib backtest from each selected prediction; native MLflow backtest retained for traceability only",
        "git_commit": git_commit,
    }
    write_json(out / "metadata/eval_config.json", eval_config)
    files = {
        "seed_metrics": "metrics/seed_metrics.csv", "aggregate_metrics": "metrics/aggregate_metrics.csv",
        "seed_table": "tables/seed_mean_std.csv", "eval_config": "metadata/eval_config.json",
        "validation": "diagnostics/validation.json",
    }
    primary_keys = {"seed_metrics": ["market", "model", "seed"], "aggregate_metrics": ["market", "model"]}
    if ensemble_enabled:
        primary_keys["ensemble_metrics"] = ["market", "model", "ensemble_method"]
        files.update({"ensemble_metrics": "metrics/ensemble_metrics.csv", "ensemble_table": "tables/ensemble.csv", "ensemble_curves": "curves/ensemble/*.csv"})
    write_json(out / "metadata/manifest.json", {"schema_version": "1.0", "baseline": "MASTER", "description": "MASTER on CSI300 and SP500 using Qlib TopkDropoutStrategy", "primary_keys": primary_keys, "files": files})

    validator = HERE / "inspect_eval_results.py"
    result = subprocess.run([sys.executable, str(validator), "--results", str(out)], cwd=REPO_ROOT)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
