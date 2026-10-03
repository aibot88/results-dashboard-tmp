# Monthly company characteristics → Embedding32 → NN3

This is an exploratory chronological next-month return prediction experiment, executed on `100.80.28.19:/workspace/AssertBert` on 2026-10-03. It does not use test returns to select architecture, seeds, calibration, or an early-stopping epoch.

## Models

- **RAW**: 16 observed characteristics plus 16 missing-value flags → NN3 hidden widths 64, 32, 16 → scalar return.
- **EMB32**: the same 32 inputs → encoder 64, 32 → NN3 hidden widths 64, 32, 16 → scalar return. Encoder and prediction head jointly learn from MSE.
- Fixed seeds 42, 43, 44. The ensemble averages predictions, not embeddings. Each seed's embedding coordinates have their own meaning; do not average independently trained vectors as if their axes were aligned.
- Adam learning rate 0.001, weight decay 0.0001, dropout 0.1, batch size 4096, maximum 80 epochs. Early stop after 10 non-improving epochs using validation MSE. Target mean/std come only from training rows. Final chosen epochs were RAW 1/2/1 and EMB32 1/1/1; this short optimum is reported rather than replaced by a more favorable test checkpoint.

## Time split

Periods refer to **outcome months**, not formation months. A formation-month feature predicts the following month. Training outcomes 2016-06–2021-12; validation outcomes 2022-01–2022-12; primary test 2023-01–2024-11 (23 months); extended test through 2025-04 (28 months). The test calendar has appeared in prior project experiments, so this is exploratory rather than a newly untouched holdout.

Contemporaneous cross-sectional preprocessing uses 1%/99% clipping, monthly median imputation, and standardization. Entirely missing features are set to zero with missing flags. Missing future outcomes are removed **after** feature preprocessing, so future label availability does not alter normalization. Their exclusion still limits the backtest universe.

## Data requirements

Private source files remain on the server; the public page publishes aggregate results and source code, not raw licensed data or per-stock predictions.

Under `/workspace/AssertBert/data/raw/ashare/market_financials/`:

- `月个股回报率文件/TRD_Mnth.csv`: formation-month float capitalization in thousands (converted to yuan), and monthly returns excluding reinvested cash dividends.
- `基本面数据_update/公司文件1990_2026/TRD_Co.csv`: market type, listing month.
- `基本面数据_update/盈利能力2015_2025/FI_T5.csv`: ROA, ROE, ROIC, R&D / sales.
- `基本面数据_update/偿债能力2015_2025/FI_T1.csv`: current ratio, leverage.
- `发展能力update/FI_T8.csv`: asset growth.
- `资产负债表/FS_Combas.csv`: book equity.
- The prior CH3 preparation output `sparkEval/ch3_proxy_20261003/monthly_characteristics.csv`: monthly repriced E/P proxy.
- `sparkEval/ch3_proxy_20261003/ch3_reference.csv`: CH3 data privately extracted from the supplied local Excel. Columns `month,rf_mon,mktrf,SMB,VMG`; used for lagged market beta and ex-post Alpha, not future prediction features.

The companion `prepare_ch3_inputs.py` generates the prerequisite characteristics in the existing CH3 folder, assuming its private reference CSV has already been supplied. No raw input tables are fetched by these scripts.

Financial reports are annual consolidated regular reports, assumed observable by Apr 30 of the next year, used in predicting May onward. Actual publication dates are unavailable. Annual E/P is approximated using positive PE and subsequent non-dividend returns; loss-making-company E/P is unavailable. Book equity / float capitalization is also a proxy. There is no quarterly backfill or retrospective monthly interpolation.

## Run on the server

```sh
cd /workspace/AssertBert
python3 -m venv sparkEval/firm_characteristic_nn3_20261003/.venv
sparkEval/firm_characteristic_nn3_20261003/.venv/bin/python -m pip install -r sparkEval/firm_characteristic_nn3_20261003/requirements.txt
sparkEval/firm_characteristic_nn3_20261003/.venv/bin/python -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
python3 sparkEval/firm_characteristic_nn3_20261003/prepare_ch3_inputs.py
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 sparkEval/firm_characteristic_nn3_20261003/.venv/bin/python -u sparkEval/firm_characteristic_nn3_20261003/run.py
sparkEval/firm_characteristic_nn3_20261003/.venv/bin/python sparkEval/firm_characteristic_nn3_20261003/dashboard.py
cd sparkEval/firm_characteristic_nn3_20261003
.venv/bin/python -m unittest -v test_run
```

The installed runtime is CPU PyTorch; GPUs were not needed for these small networks. A broken inherited network proxy had to be unset during installation; the scripts do not modify persistent proxy settings.

## Evaluation

- Universe: Shanghai/Shenzhen A shares including ChiNext and STAR, listing age at least six calendar months and positive formation capitalization. This return prediction experiment does **not** remove the smallest 30% stocks. Current company-file historical completeness is not fully verified.
- Sort by predicted return, with stock code breaking ties deterministically. Lowest/highest `floor(N/10)` stocks form P1/P10; remainder stays in the middle. EW uses equal weights, VW uses formation-month float capitalization. Missing outcome returns cannot be evaluated; suspend/delist completeness is not established.
- R² = `1 − sum((actual − predicted)^2) / sum(actual^2)`; monthly Rank IC is average Spearman correlation.
- Spread mean and CH3 intercept t-statistics use Newey–West/HAC with 3 lags. The local reference CH3 factors, not the earlier reconstructed proxies, are used for Alpha. A zero-investment long-short return is not reduced by the risk-free rate a second time.
- Cost scenario: subtract 10bp multiplied by the sum of absolute signed target-weight changes, including the initial portfolio trades. This is a target-weight turnover approximation; it omits drift adjustment, borrowing costs, slippage, and execution feasibility. The cumulative chart compounds monthly spreads as an illustrative series, not a margin-account simulation.

## Primary ensemble results

| Model | Weight | Monthly gross P10−P1 | Monthly net, 10bp | Rank IC | OOS R² | Monthly CH3 Alpha | Alpha HAC t |
|---|---|---:|---:|---:|---:|---:|---:|
| RAW | EW | 1.861% | 1.670% | 0.0785 | 0.691% | 1.147% | 2.978 |
| EMB32 | EW | 1.854% | 1.636% | 0.0854 | 0.580% | 0.867% | 2.917 |
| RAW | VW | 1.624% | 1.394% | 0.0785 | 0.691% | 0.839% | 1.256 |
| EMB32 | VW | 1.672% | 1.406% | 0.0854 | 0.580% | 0.532% | 0.809 |

Embedding improves Rank IC here, but does not materially improve EW spread and has a lower EW Sharpe than the raw-feature baseline. The VW Alpha t-statistic is weak. These estimates do not establish causation or future performance, and W2V uses a different stock universe/input pipeline despite sharing the test calendar.

## Outputs

Server folder: `/workspace/AssertBert/sparkEval/firm_characteristic_nn3_20261003/outputs/`.

- `embedding32_seed42.npz`, `embedding32_seed43.npz`, `embedding32_seed44.npz`: 128,160 sample-aligned 32-dimensional vectors for the extended test set, with `asset_id` and `formation_month` keys. Each vector is generated by a checkpoint selected using 2022 validation only.
- Six `.pt` checkpoints, per-stock predictions, and `monthly_panel.csv` stay on the server.
- `metrics.csv`, `portfolio_monthly.csv`, `results.json`, and `index.html`: public aggregate outputs.

Three tests verify prediction gradients reach the encoder, monthly preprocessing ignores outcome labels and future-month values, and decile/cost calculations match a known example. Additional post-run checks validate temporal splits, 32-dimensional vector alignment, ensemble averaging, factor alignment, and independently recomputed metrics.
