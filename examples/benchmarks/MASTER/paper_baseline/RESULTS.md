# MASTER Baseline Results (CSI300 & SP500)

Paper baseline for the **MASTER** model — *Market-Guided Stock Transformer for Stock Price Forecasting* ([arXiv 2312.15235](https://arxiv.org/abs/2312.15235)) — on two markets. Code/configs live in `examples/benchmarks/MASTER/paper_baseline/`.

## 1. Experimental Setup

| Item | Value |
|---|---|
| Model | MASTER (default architecture: `d_feat=158, d_model=256, t_nhead=4, s_nhead=2, dropout=0.5, n_epochs=40, lr=8e-6, train_stop_loss_thred=0.95`) |
| Markets | A-share **CSI300** (region `cn`, `~/.qlib/qlib_data/cn_data`) · US **SP500** (region `us`, `~/.qlib/qlib_data/us_data`) |
| Stock features | Alpha158 (158 factors). CSI300 → stock `Alpha158`; SP500 → `Alpha158US` (`$vwap`→typical-price proxy, since us_data has no `$vwap`) |
| Market-guided gate | 63 market features from 3 indices: CSI300 → `sh000300/sh000903/sh000905`; SP500 → `^gspc/^dji/^ndx` (architecture unchanged) |
| Label | `Ref($close,-5)/Ref($close,-1)-1` (5-day forward return) — both markets |
| Split | train `2009-01-01 ~ 2020-12-31` · valid `2021-01-01 ~ 2022-12-31` · test `2023-01-01 ~ 2025-12-31` |
| Strategy | Qlib `TopkDropoutStrategy` (`topk=30, n_drop=5`); benchmark `SH000300` / `^gspc` |
| Transaction costs | buy `open_cost=0.0005` (万分之五), sell `close_cost=0.0015` (万分之十五), `min_cost=5` |
| Seeds | 5 (`0–4`); results are mean ± std |
| Hardware | NVIDIA RTX 5060 Ti (CUDA) |

## 2. Results — 5 seeds, test 2023-01-01 ~ 2025-12-31 (mean ± std)

### CSI300 🇨🇳
| Metric | mean ± std | per-seed (0..4) |
|---|---|---|
| IC | **0.0402 ± 0.0068** | 0.0498, 0.0465, 0.0377, 0.0338, 0.0330 |
| Rank IC | **0.0618 ± 0.0069** | 0.0715, 0.0676, 0.0606, 0.0557, 0.0535 |
| ICIR | 0.2186 ± 0.0383 | |
| Rank ICIR | 0.3393 ± 0.0351 | |
| Annualized excess return (with cost) | **+12.27% ± 2.21%** | 15.38, 9.97, 13.48, 13.01, 9.53 |
| Annualized excess return (without cost) | +13.94% ± 2.22% | 17.05, 11.62, 15.17, 14.68, 11.19 |
| Information ratio (with cost) | 0.657 ± 0.118 | |
| Information ratio (without cost) | 0.745 ± 0.118 | |

### SP500 🇺🇸
| Metric | mean ± std | per-seed (0..4) |
|---|---|---|
| IC | **0.0019 ± 0.0042** (≈0) | 0.0099, −0.0003, −0.0002, −0.0020, 0.0021 |
| Rank IC | **0.0036 ± 0.0045** | 0.0122, 0.0016, 0.0033, −0.0005, 0.0013 |
| ICIR | 0.0158 ± 0.0358 | |
| Rank ICIR | 0.0290 ± 0.0372 | |
| Annualized excess return (with cost) | **−6.48% ± 2.16%** | −3.74, −7.49, −10.04, −6.03, −5.10 |
| Annualized excess return (without cost) | −4.71% ± 2.17% | −1.96, −5.72, −8.28, −4.25, −3.31 |
| Information ratio (with cost) | −0.345 ± 0.130 | |
| Information ratio (without cost) | −0.252 ± 0.127 | |

### Interpretation
- **CSI300**: MASTER is a strong baseline on its home market (CN); IC ≈ 0.06 and +12% annualized excess return are consistent with the paper's scale.
- **SP500**: IC ≈ 0 and negative excess return. MASTER is designed for CN — the US adaptation (VWAP proxy + US market indices) yields no predictive edge, and the 2023–2025 US bull market made the `^gspc` benchmark hard to beat. This is expected behavior, not a bug; it indicates a stronger US-specific method is needed under this setup.

## 3. Reproduction

### Environment
```bash
conda activate qlib-master   # pyqlib installed editable from this repo; torch+CUDA; numpy 2.0
```

### Train + backtest (each seed trains, then predicts + backtests)
```bash
cd examples/benchmarks/MASTER/paper_baseline

# Single seed, e.g. seed 0:
python run_baseline.py --market csi300 --seed-start 0 --seeds 1
python run_baseline.py --market sp500  --seed-start 0 --seeds 1

# Full 5 seeds — one fresh process per seed (avoids cross-seed RAM accumulation / OOM):
bash run_baseline.sh smoke   # quick check: 1 epoch, seed 0, both markets (~minutes)
bash run_baseline.sh full    # 5 seeds × 40 epochs × both markets (LONG, hours)
```

Per seed the runner does: train 40 epochs → predict on test → `SignalRecord` + `SigAnaRecord` (IC / Rank IC) + `PortAnaRecord` (backtest with the costs above). Checkpoints → `model/{market}master_{seed}.pkl`; experiment records → `mlruns/`.

### Backtest only (reuse a trained checkpoint, skip training)
```bash
python run_baseline.py --market csi300 --seed-start 0 --seeds 1 --only_backtest
```

### Resume — skip already-finished seeds
```bash
# e.g. seed 0 already done, run seeds 1–4 only:
python run_baseline.py --market sp500 --seed-start 1 --seeds 4
```

### View results
```bash
mlflow ui     # open the printed URL; experiments are named {market}_MASTER_seed{0..4}
```

## 4. Configs & files

- `workflow_config_master_csi300.yaml` / `workflow_config_master_sp500.yaml` — all hyperparameters, segments, costs, strategy.
- `us_handlers.py` — `Alpha158US` (VWAP → typical-price proxy for US).
- `run_baseline.py` — multi-seed runner (`--market`, `--seeds`, `--seed-start`, `--smoke`, `--only_backtest`); prints mean ± std across seeds.
- `run_baseline.sh` — `smoke` | `full` wrapper.

## 5. Caveats

- **US adaptation**: MASTER is natively CN-only. SP500 required two backward-compatible changes — `Alpha158US` (no `$vwap` in us_data) and `marketDataHandler.market_indices = [^gspc, ^dji, ^ndx]` (the gate's 3 market indices). Model architecture unchanged (158 stock + 63 market features).
- **Memory (≈15 GB machines)**: running >1 seed in a single python process can OOM on SP500 (~11 GB/seed, RAM accumulates across seeds). `run_baseline.sh full` runs each seed in a fresh process. If you abort a run, clean orphan workers with `pkill -9 -f run_baseline.py`.
- **Costs**: `open_cost=0.0005`, `close_cost=0.0015` are set in each yaml's `port_analysis_config.backtest.exchange_kwargs`; both `with_cost` and `without_cost` metrics are reported.
- **Dates must stay unquoted** in the yaml files (`run_baseline.py` builds the handler-cache filename via `.strftime()` on the parsed date).
