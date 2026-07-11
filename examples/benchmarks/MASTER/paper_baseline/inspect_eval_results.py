"""Validate Baseline Results Protocol v1.0 output."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation_metrics import portfolio_metrics, prediction_metrics
from export_protocol_results import METRICS, aggregate, display_table, load_series, normalized_ensemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[4] / "results")
    return parser.parse_args()


class Validator:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def check(self, name: str, condition: bool, detail: str) -> None:
        self.checks.append({"name": name, "passed": bool(condition), "detail": detail})

    def close(self, output: Path) -> bool:
        failures = sum(not item["passed"] for item in self.checks)
        report = {"passed": failures == 0, "passes": len(self.checks) - failures, "failures": failures, "checks": self.checks}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        return failures == 0


def frames_close(left: pd.DataFrame, right: pd.DataFrame, atol: float = 1e-12) -> bool:
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        return False
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
            if not np.allclose(left[column], right[column], atol=atol, rtol=1e-10, equal_nan=True):
                return False
        elif not left[column].astype(str).equals(right[column].astype(str)):
            return False
    return True


def main() -> None:
    args = parse_args()
    root = args.results.resolve()
    validator = Validator()
    manifest_path = root / "metadata/manifest.json"
    config_path = root / "metadata/eval_config.json"
    validator.check("core metadata exists", manifest_path.is_file() and config_path.is_file(), str(root))
    if not manifest_path.is_file() or not config_path.is_file():
        raise SystemExit(1)
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(config_path.read_text())
    validation_path = root / "diagnostics/validation.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.touch()
    for name, relative in manifest["files"].items():
        if "*" not in relative:
            validator.check(f"manifest file: {name}", (root / relative).is_file(), relative)

    seed = pd.read_csv(root / manifest["files"]["seed_metrics"])
    expected_keys = {(market, "master", int(value)) for market, values in config["seeds"].items() for value in values}
    actual_keys = set(seed[["market", "model", "seed"]].itertuples(index=False, name=None))
    validator.check("seed key coverage", actual_keys == expected_keys, f"expected={sorted(expected_keys)}, actual={sorted(actual_keys)}")
    validator.check("seed primary key unique", not seed.duplicated(["market", "model", "seed"]).any(), f"rows={len(seed)}")
    numeric = seed[METRICS + ["num_test_days"]]
    validator.check("seed metrics finite", np.isfinite(numeric.to_numpy()).all(), "no allowed undefined exceptions in this export")
    bounds = (seed["IC"].abs() <= 1).all() and (seed["RankIC"].abs() <= 1).all() and (seed["STD"] >= 0).all() and (seed["MDD"] <= 0).all()
    validator.check("seed metric bounds", bounds, "|IC|, |RankIC| <= 1; STD >= 0; MDD <= 0")

    actual_aggregate = pd.read_csv(root / manifest["files"]["aggregate_metrics"])
    expected_aggregate = aggregate(seed)
    validator.check("aggregate recomputation", frames_close(actual_aggregate, expected_aggregate), "mean/std recomputed with ddof=1")
    actual_table = pd.read_csv(root / manifest["files"]["seed_table"])
    validator.check("seed table formatting", actual_table.equals(display_table(expected_aggregate)), "four-decimal mean ± std")

    if config["ensemble"]["enabled"]:
        ensemble = pd.read_csv(root / manifest["files"]["ensemble_metrics"])
        expected_ensemble_keys = {(market, "master", method) for market in config["markets"] for method in config["ensemble"]["methods"]}
        actual_ensemble_keys = set(ensemble[["market", "model", "ensemble_method"]].itertuples(index=False, name=None))
        validator.check("ensemble key coverage", actual_ensemble_keys == expected_ensemble_keys and not ensemble.duplicated(["market", "model", "ensemble_method"]).any(), str(actual_ensemble_keys))
        table = pd.read_csv(root / manifest["files"]["ensemble_table"], dtype=str)
        table_ok = len(table) == len(ensemble)
        for metric in METRICS:
            table_ok &= np.allclose(table[metric].astype(float), ensemble[metric], atol=5.1e-5, rtol=0)
        validator.check("ensemble table formatting", table_ok, "four-decimal numeric display")
        repo_root = Path(__file__).resolve().parents[4]
        for row in ensemble.itertuples(index=False):
            curve_path = root / f"curves/ensemble/{row.market}_{row.model}.csv"
            curve = pd.read_csv(curve_path, parse_dates=["datetime"])
            dates_ok = curve["datetime"].is_monotonic_increasing and curve["datetime"].is_unique
            finite = np.isfinite(curve.drop(columns="datetime").to_numpy()).all()
            validator.check(f"{row.market} curve dates and values", dates_ok and finite, f"rows={len(curve)}")
            net_ok = np.allclose(curve["daily_ret_net"], curve["daily_ret_gross"] - curve["cost"], atol=1e-12)
            nav_ok = np.allclose(curve["nav"], (1 + curve["daily_ret_net"]).cumprod(), atol=1e-10) and np.allclose(curve["bench_nav"], (1 + curve["bench_ret"]).cumprod(), atol=1e-10)
            validator.check(f"{row.market} curve accounting", net_ok and nav_ok, "net=gross-cost; NAV includes first return")
            recomputed = portfolio_metrics(curve["daily_ret_net"])
            portfolio_ok = all(np.isclose(recomputed[metric], getattr(row, metric), atol=1e-10, rtol=1e-9) for metric in ["AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar"])
            validator.check(f"{row.market} portfolio metrics", portfolio_ok and recomputed["num_test_days"] == row.num_test_days, "recomputed from curve")
            artifacts = config["selected_artifacts"][row.market]
            seeds = [int(value) for value in str(row.seeds).split(",")]
            predictions = [load_series(repo_root / artifacts[str(seed_value)]["prediction"]) for seed_value in seeds]
            score = normalized_ensemble(predictions, row.ensemble_method)
            label = load_series(repo_root / artifacts[str(seeds[0])]["label"]).reindex(score.index)
            ranking = prediction_metrics(score, label)
            ranking_ok = all(np.isclose(ranking[metric], getattr(row, metric), atol=1e-12, rtol=1e-10) for metric in ["IC", "ICIR", "RankIC", "RankICIR"])
            validator.check(f"{row.market} ensemble ranking source", ranking_ok, "recomputed from aligned ensemble score and label")

    passed = validator.close(validation_path)
    print(f"validation: {sum(item['passed'] for item in validator.checks)} passed, {sum(not item['passed'] for item in validator.checks)} failed")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
