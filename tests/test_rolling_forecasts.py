"""Synthetic checks for strict rolling-origin evaluation without future updates."""
import numpy as np
import pytest

from experiments.rolling_forecasts import free_forecast, rolling_origin_validation
from tidal_enkf.filter import EnsembleKalmanFilter, FilterConfig
from tidal_enkf.model import ModelConfig, ShallowWater1D


def setup_case():
    model = ShallowWater1D(ModelConfig(cells=20, dt_s=600.0))
    t = np.arange(61) * model.config.dt_s
    base = 0.7 * np.sin(2.0 * np.pi * t / (12.0 * 3600.0))
    locations = np.array([20_000., 45_000., 70_000., 90_000.])
    rng = np.random.default_rng(53)
    truth = model.run(base + 0.15 * np.sin(2 * np.pi * t / (7.0 * 3600.0)))
    obs = model.sample_heights(truth, locations) + rng.normal(0, 0.03, (len(t), 4))
    return model, base, locations, obs


def test_zero_boundary_error_matches_no_assimilation_rollout():
    model, base, locs, _ = setup_case()
    deterministic = model.run(base)
    cutoff, steps = 10, 12
    state = np.r_[deterministic[cutoff], 0.0]
    predicted = free_forecast(model, base, state, alpha=0.95, cutoff=cutoff,
                              steps=steps, station_m=locs[3])
    expected = model.sample_heights(deterministic[cutoff:cutoff+steps+1], [locs[3]])[:, 0]
    np.testing.assert_allclose(predicted, expected, atol=1e-10)


def test_eta_decay_uses_no_future_random_innovations():
    model, base, locs, _ = setup_case()
    cutoff, steps = 11, 5
    state = np.r_[model.run(base)[cutoff], 0.4]
    predicted = free_forecast(model, base, state, alpha=0.8, cutoff=cutoff,
                              steps=steps, station_m=locs[3])
    x = state[:-1].copy()
    expected = [model.sample_heights(x, [locs[3]])[0]]
    for k in range(1, steps+1):
        x = model.step(x, base[cutoff+k] + 0.4 * 0.8**k)
        expected.append(model.sample_heights(x, [locs[3]])[0])
    np.testing.assert_allclose(predicted, expected, atol=1e-12)


def test_future_measurement_changes_cannot_affect_frozen_forecast():
    model, base, locs, obs = setup_case()
    cutoff = 24
    changed = obs.copy()
    changed[cutoff+1:, :3] += 20.0
    cfg = FilterConfig(ensemble_size=20, seed=12)
    a = EnsembleKalmanFilter(model, cfg).assimilate(base, obs, locs, (0, 1, 2))
    b = EnsembleKalmanFilter(model, cfg).assimilate(base, changed, locs, (0, 1, 2))
    np.testing.assert_array_equal(a["analysis_states"][cutoff], b["analysis_states"][cutoff])
    np.testing.assert_array_equal(
        free_forecast(model, base, a["analysis_states"][cutoff], alpha=a["alpha"], cutoff=cutoff, steps=12, station_m=locs[3]),
        free_forecast(model, base, b["analysis_states"][cutoff], alpha=b["alpha"], cutoff=cutoff, steps=12, station_m=locs[3]),
    )


def test_rolling_summary_and_relative_skill_identity():
    model, base, locs, obs = setup_case()
    filtered = EnsembleKalmanFilter(model, FilterConfig(ensemble_size=18, seed=31)).assimilate(base, obs, locs, (0, 1, 2))
    summary = rolling_origin_validation(
        model, base, obs, filtered["analysis_states"], alpha=filtered["alpha"],
        station_m=locs[3], observation_column=3,
        first_cutoff_hour=2., last_cutoff_hour=6., spacing_hour=2.,
        horizon_hour=2., lead_hours=(1., 2.),
    )
    assert summary["n_origins"] == 3
    assert summary["n_pooled_observations"] == 3 * 12
    assert len(summary["per_origin"]) == 3
    assert summary["by_lead_hour"]["1.0"]["n_origins_with_observation"] == 3
    a, b = summary["pooled_enkf_free_rmse_m"], summary["pooled_baseline_free_rmse_m"]
    np.testing.assert_allclose(summary["pooled_relative_rmse_reduction_percent"], 100 * (1 - a / b))


def test_missing_holdout_observation_excluded_and_no_cutoff_scored():
    model, base, locs, obs = setup_case()
    deterministic = model.run(base)
    states = np.column_stack((deterministic, np.zeros(len(base))))
    obs[:, 3] = model.sample_heights(deterministic, [locs[3]])[:, 0]
    obs[13, 3] = np.nan  # +1 step past +2 h cutoff
    summary = rolling_origin_validation(
        model, base, obs, states, alpha=0.8, station_m=locs[3],
        observation_column=3, first_cutoff_hour=2., last_cutoff_hour=2.,
        spacing_hour=1., horizon_hour=1., lead_hours=(1.,),
    )
    assert summary["n_pooled_observations"] == 5
    assert summary["per_origin"][0]["n_valid_future_observations"] == 5
    assert summary["pooled_baseline_free_rmse_m"] == pytest.approx(0., abs=1e-10)
    assert summary["pooled_relative_rmse_reduction_percent"] is None


def test_invalid_cutoffs_are_rejected():
    model, base, locs, obs = setup_case()
    states = np.column_stack((model.run(base), np.zeros(len(base))))
    options = dict(alpha=0.9, station_m=locs[3], observation_column=3,
                   first_cutoff_hour=2, last_cutoff_hour=4, spacing_hour=1,
                   horizon_hour=2, lead_hours=(1., 2.))
    with pytest.raises(ValueError, match="forecast horizon"):
        rolling_origin_validation(model, base, obs, states, **(options | {"last_cutoff_hour": 9}))
    with pytest.raises(ValueError, match="integer number"):
        rolling_origin_validation(model, base, obs, states, **(options | {"spacing_hour": 0.11}))
    with pytest.raises(ValueError, match="Lead"):
        rolling_origin_validation(model, base, obs, states, **(options | {"lead_hours": (1., 4.)}))
    with pytest.raises(ValueError, match="augmented"):
        free_forecast(model, base, states[2, :-1], alpha=0.9, cutoff=2, steps=3, station_m=locs[3])
