"""Rolling-origin, no-update forecasts at a withheld station.

This is an independent validation experiment; it does not modify the EnKF,
underlying model, or the original results.json. Only the analysis state
available at each origin is used. Future astronomical tide is treated as
known forcing; future measured water levels are not assimilated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tidal_enkf.data import STATIONS, STATION_X_M, load_observations, load_western_tide
from tidal_enkf.filter import EnsembleKalmanFilter, FilterConfig
from tidal_enkf.model import ModelConfig, ShallowWater1D

ROOT = Path(__file__).resolve().parents[1]


def _steps_for_hours(hours: float, dt_s: float) -> int:
    """Reject hours not exactly representable by the model timestep."""
    if not np.isfinite(hours) or hours <= 0:
        raise ValueError("Hours must be positive and finite")
    steps = hours * 3600.0 / dt_s
    rounded = round(steps)
    if rounded < 1 or not np.isclose(steps, rounded, rtol=0.0, atol=1e-9):
        raise ValueError("Hours must represent an integer number of time steps")
    return int(rounded)


def free_forecast(
    model: ShallowWater1D,
    base_boundary_m: np.ndarray,
    state_at_cutoff: np.ndarray,
    *,
    alpha: float,
    cutoff: int,
    steps: int,
    station_m: float,
) -> np.ndarray:
    """Conditional mean forecast, including the analysis estimate at cutoff.

    ``state_at_cutoff`` is [h0,u0,...,hN,uN,eta]. The future innovations
    in eta_{k+1}=alpha*eta_k+w_k have expectation zero. The underlying
    shallow-water model is linear, so propagation of this augmented mean
    gives the conditional mean future height at the requested station.
    """
    base = np.asarray(base_boundary_m, dtype=float)
    augmented = np.asarray(state_at_cutoff, dtype=float)
    if base.ndim != 1 or augmented.shape != (model.nstate + 1,):
        raise ValueError("Expected one-dimensional forcing and one augmented state")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(augmented)):
        raise ValueError("Forcing and initial state must be finite")
    if not 0 <= alpha <= 1 or not np.isfinite(alpha):
        raise ValueError("AR(1) parameter must be in [0, 1]")
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, np.integer)) or cutoff < 0:
        raise ValueError("Cutoff must be a nonnegative integer")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("Number of forecast steps must be a positive integer")
    if cutoff + steps >= len(base):
        raise ValueError("Forecast window extends past the forcing data")
    H = model.height_operator(np.asarray([station_m], dtype=float))[0]
    x = augmented[:-1].copy()
    eta = float(augmented[-1])
    out = np.empty(steps + 1, dtype=float)
    out[0] = H @ x
    for k in range(1, steps + 1):
        eta *= alpha
        x = model.step(x, base[cutoff + k] + eta)
        out[k] = H @ x
    return out


def _rmse(errors: list[float] | np.ndarray) -> float | None:
    a = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(a * a))) if a.size else None


def rolling_origin_validation(
    model: ShallowWater1D,
    base_boundary_m: np.ndarray,
    observations_m: np.ndarray,
    analysis_states: np.ndarray,
    *,
    alpha: float,
    station_m: float,
    observation_column: int,
    first_cutoff_hour: float = 8.0,
    last_cutoff_hour: float = 40.0,
    spacing_hour: float = 4.0,
    horizon_hour: float = 8.0,
    lead_hours: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
) -> dict:
    """Compare held-out EnKF free forecasts with no-assimilation forecasts.

    The observation column must have been excluded from analysis updates.
    Origin-specific trajectories may overlap, so the pooled errors are
    descriptive and are not independent samples for significance tests.
    """
    base = np.asarray(base_boundary_m, dtype=float)
    obs = np.asarray(observations_m, dtype=float)
    states = np.asarray(analysis_states, dtype=float)
    if base.ndim != 1 or obs.ndim != 2 or obs.shape[0] != len(base):
        raise ValueError("Forcing/observations have incompatible shapes")
    if states.shape != (len(base), model.nstate + 1):
        raise ValueError("Analysis states must have shape (times, nstate+1)")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(states)):
        raise ValueError("Forcing and analysis states must be finite")
    if isinstance(observation_column, bool) or not isinstance(observation_column, (int, np.integer)) or not 0 <= observation_column < obs.shape[1]:
        raise ValueError("Invalid observation column")
    if last_cutoff_hour < first_cutoff_hour:
        raise ValueError("Last cutoff must not precede first cutoff")

    dt = model.config.dt_s
    first = _steps_for_hours(first_cutoff_hour, dt)
    last = _steps_for_hours(last_cutoff_hour, dt)
    spacing = _steps_for_hours(spacing_hour, dt)
    horizon = _steps_for_hours(horizon_hour, dt)
    leads = {str(float(h)): _steps_for_hours(h, dt) for h in lead_hours}
    if not leads or len(leads) != len(lead_hours) or max(leads.values()) > horizon:
        raise ValueError("Lead hours must be distinct and within the forecast horizon")
    if last + horizon >= len(base):
        raise ValueError("No observations/forcing for the requested last forecast horizon")

    baseline = model.sample_heights(model.run(base), np.asarray([station_m]))[:, 0]
    y = obs[:, observation_column]
    origins = []
    errors_enkf, errors_base = [], []
    by_lead = {name: {"enkf": [], "baseline": []} for name in leads}

    for cutoff in range(first, last + 1, spacing):
        # Forecast only from the frozen analysis at this origin, not future analyses.
        predicted = free_forecast(
            model, base, states[cutoff], alpha=alpha,
            cutoff=cutoff, steps=horizon, station_m=station_m,
        )[1:]
        measured = y[cutoff + 1:cutoff + horizon + 1]
        ref = baseline[cutoff + 1:cutoff + horizon + 1]
        valid = np.isfinite(measured)
        forecast_error = (predicted - measured)[valid]
        reference_error = (ref - measured)[valid]
        errors_enkf.extend(forecast_error.tolist())
        errors_base.extend(reference_error.tolist())
        for name, lead in leads.items():
            if np.isfinite(measured[lead - 1]):
                by_lead[name]["enkf"].append(float(predicted[lead - 1] - measured[lead - 1]))
                by_lead[name]["baseline"].append(float(ref[lead - 1] - measured[lead - 1]))
        origins.append({
            "cutoff_hour": float(cutoff * dt / 3600),
            "n_valid_future_observations": int(valid.sum()),
            "enkf_free_rmse_m": _rmse(forecast_error),
            "baseline_free_rmse_m": _rmse(reference_error),
        })
    enkf_rmse, base_rmse = _rmse(errors_enkf), _rmse(errors_base)
    skill = (100.0 * (1.0 - enkf_rmse / base_rmse)
             if enkf_rmse is not None and base_rmse is not None and base_rmse > 0 else None)
    return {
        "study": "descriptive rolling-origin free forecasts, no observations after each cutoff",
        "future_forcing": "astronomical tide assumed known; zero-mean future AR(1) innovations",
        "warning": "Overlapping forecast windows and a single storm event do not provide independent evidence of generalisation.",
        "configuration": {
            "first_cutoff_hour": float(first_cutoff_hour),
            "last_cutoff_hour": float(last_cutoff_hour),
            "spacing_hour": float(spacing_hour),
            "horizon_hour": float(horizon_hour),
            "heldout_station_position_m": float(station_m),
            "heldout_observation_column": int(observation_column),
        },
        "n_origins": len(origins),
        "n_pooled_observations": len(errors_enkf),
        "pooled_enkf_free_rmse_m": enkf_rmse,
        "pooled_baseline_free_rmse_m": base_rmse,
        "pooled_relative_rmse_reduction_percent": skill,
        "by_lead_hour": {
            name: {
                "n_origins_with_observation": len(group["enkf"]),
                "enkf_rmse_m": _rmse(group["enkf"]),
                "baseline_rmse_m": _rmse(group["baseline"]),
            }
            for name, group in by_lead.items()
        },
        "per_origin": origins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--first-cutoff-hour", type=float, default=8.0)
    parser.add_argument("--last-cutoff-hour", type=float, default=40.0)
    parser.add_argument("--spacing-hour", type=float, default=4.0)
    parser.add_argument("--horizon-hour", type=float, default=8.0)
    parser.add_argument("--output", type=Path, default=ROOT / "validation" / "results" / "rolling_forecasts.json")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    times, observed = load_observations(ROOT / "data", "waterlevel")
    tide_times, tide = load_western_tide(ROOT / "data")
    if not np.array_equal(times, tide_times):
        raise ValueError("Station and boundary timestamps do not align")

    model = ShallowWater1D(ModelConfig())
    reports = []
    for k in range(args.repeats):
        seed = args.seed + k
        out = EnsembleKalmanFilter(
            model, FilterConfig(ensemble_size=args.ensemble_size, seed=seed)
        ).assimilate(tide, observed, STATION_X_M, assimilated_columns=(0, 1, 2))
        assert 3 not in out["assimilated_columns"], "Bath must remain withheld"
        report = rolling_origin_validation(
            model, tide, observed, out["analysis_states"],
            alpha=out["alpha"], station_m=float(STATION_X_M[3]),
            observation_column=3,
            first_cutoff_hour=args.first_cutoff_hour,
            last_cutoff_hour=args.last_cutoff_hour,
            spacing_hour=args.spacing_hour,
            horizon_hour=args.horizon_hour,
        )
        reports.append({"seed": seed, "results": report})
        fmt = lambda v: f"{v:.3f}" if v is not None else "n/a"
        skill = report['pooled_relative_rmse_reduction_percent']
        print(
            f"seed={seed} | n_origins={report['n_origins']} | "
            f"EnKF RMSE={fmt(report['pooled_enkf_free_rmse_m'])} m | "
            f"baseline RMSE={fmt(report['pooled_baseline_free_rmse_m'])} m | "
            f"relative reduction={f'{skill:.1f}%' if skill is not None else 'n/a'}"
        )
    report = {
        "validation": "exploratory single-event forecast comparison; not a statistical generalisation claim",
        "seeds": list(range(args.seed, args.seed + args.repeats)),
        "ensemble_size": args.ensemble_size,
        "model_dt_seconds": model.config.dt_s,
        "assimilation_stations": list(STATIONS[:3]),
        "heldout_station": STATIONS[3],
        "repeats": reports,
    }
    if args.repeats > 1:
        skills = [r["results"]["pooled_relative_rmse_reduction_percent"] for r in reports]
        if all(x is not None for x in skills):
            report["seed_sensitivity"] = {
                "mean_relative_rmse_reduction_percent": float(np.mean(skills)),
                "sample_std_percentage_points": float(np.std(skills, ddof=1)),
                "n_seeds": args.repeats,
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
