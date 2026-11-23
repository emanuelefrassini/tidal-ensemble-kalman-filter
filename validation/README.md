# Rolling-origin forecast validation

The original project reports analysis error at a held-out water-level station, Bath, and two eight-hour free forecasts. The analysis RMSE measures state estimation **while other stations continue to be assimilated**. This additional experiment tests how those state estimates perform when observations stop arriving.

Run from the repository root after `python -m pip install -e '.[test]'`:

```bash
python -m pytest tests/test_rolling_forecasts.py -v
python -m experiments.rolling_forecasts --repeats 3
```

The default experiment runs a 100-member EnKF, with observations from Vlissingen, Terneuzen and Hansweert, and **always withholds Bath**. At cutoffs +8, +12, ..., +40 hours, the filter's augmented analysis mean is frozen. The linear shallow-water model is propagated without new assimilated measurements for eight hours. The conditional mean of the AR(1) boundary error decays by `alpha` at each step, with zero mean for future innovations. The future *astronomical tide* input is treated as predictable/known forcing. This is not a real-time forecast using unknown future storm water levels.

The no-assimilation comparator propagates the same deterministic shallow-water model from the start of the record with the astronomical tide forcing. Forecast and baseline errors are evaluated on the **same future Bath observations**. The observation at the initial cutoff is excluded from all forecast scores; only leads 10 minutes through eight hours contribute. Each origin has its own RMSE, and the report also provides pointwise lead scores at 1, 2, 4 and 8 hours.

`pooled_relative_rmse_reduction_percent = 100 * (1 - RMSE_EnKF / RMSE_baseline)`. Positive values favour EnKF; **negative values mean the EnKF is worse**. A pooled RMSE aggregates all valid timestamps from all origins; partially overlapping eight-hour windows are not independent samples. The three-seed option measures exploratory random-ensemble sensitivity, **not uncertainty across independent storms**. Missing Bath observations are excluded from scoring; missing assimilation observations retain the original filter's all-or-nothing update rule.

The output file, `validation/results/rolling_forecasts.json`, is ignored by Git until reviewed. The existing model, filter, figures, `results.json` and published numbers are left unchanged.

This study does **not** establish probabilistic calibration or credible intervals: the original filter returns a diagonal state variance, which cannot be used to obtain the exact interpolated station variance without accounting for cross-covariances. A future improvement is to record full projected ensemble spread at each station, then evaluate held-out prediction intervals and innovation statistics.

The observational data originate in the original TU Delft coursework; the corresponding `data/README.md` discusses provenance and release terms. This script does not download or redistribute additional data.
