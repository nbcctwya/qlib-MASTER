## overview
This is an alternative version of the MASTER benchmark. 

paper: [MASTER: Market-Guided Stock Transformer for Stock Price Forecasting](https://arxiv.org/abs/2312.15235) 

codes: [https://github.com/SJTU-Quant/MASTER](https://github.com/SJTU-Quant/MASTER)

## config
We recommend you to use conda to config the environment and run the codes:
> Note that you should install `torch` and by your self.
```
bash config.sh
```

## run
You can directly use the bash script to run the codes (you can set the `universe` and `only_backtest` flag in `run.sh`), this `main.py` will test the model with 3 random seeds:
```
conda activate MASTER
bash run.sh
```

## Baseline Results Protocol v1.0

The paper baseline under `paper_baseline/` exports its existing MLflow prediction, label, and daily Qlib backtest artifacts into the repository-level `results/` directory. It does not retrain MASTER or alter the native `TopkDropoutStrategy` (`topk=30`, `n_drop=5`).

```bash
conda activate qlib-master
cd examples/benchmarks/MASTER/paper_baseline
python export_protocol_results.py
python inspect_eval_results.py --results ../../../../results
```

`results/metrics/` contains numeric seed, aggregate, and ensemble metrics; `results/tables/` contains four-decimal display tables; `results/curves/` contains ensemble daily returns and NAV; `results/metadata/` records artifact selection and evaluation conventions; and `results/diagnostics/validation.json` contains reproducible checks.

Prediction IC and RankIC are daily cross-sectional correlations. Portfolio metrics use `r_net = report.return - report.cost`, daily log returns, annualization 252, sample standard deviation (`ddof=1`), zero risk-free rate, and zero daily MAR. The default ensemble inner-joins all seed scores by `(datetime, instrument)`, averages raw scores, and reruns the same market-specific Qlib backtest.
<!-- or you can just directly use `qrun` tp run the codes (note that you should modify your `qlib`, since we add or modify some files in `qlib/contrib/data/dataset.py`, `qlib/data/dataset/__init__.py`, `qlib/data/dataset/processor.py` and `qlib/contrib/model/pytorch_master.py`):
```
qrun workflow_config_master_Alpha158.yaml
``` -->
