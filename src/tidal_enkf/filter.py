"""Stochastic EnKF with an augmented, AR(1) boundary-error state.

The AR(1) forcing is part of the state to preserve Markovianity. Observations
are water levels at selected stations. Gaussian perturbed observations yield
a stochastic EnKF analysis update; known boundary constraints are reinstated.
"""
from dataclasses import dataclass

import numpy as np
from .model import ShallowWater1D


@dataclass(frozen=True)
class FilterConfig:
    ensemble_size: int = 100
    boundary_sd_m: float = 0.2
    correlation_hours: float = 6.0
    observation_sd_m: float = 0.12
    seed: int = 2026

    def __post_init__(self):
        if self.ensemble_size < 3:
            raise ValueError("Need at least three ensemble members")
        if min(self.boundary_sd_m, self.correlation_hours, self.observation_sd_m) <= 0:
            raise ValueError("Uncertainties and correlation time must be positive")


class EnsembleKalmanFilter:
    """Physical state dimension 2*n, augmented with stochastic boundary eta."""

    def __init__(self, model: ShallowWater1D, config=FilterConfig()):
        self.model, self.config = model, config
        self.rng = np.random.default_rng(config.seed)
        self.alpha = np.exp(-model.config.dt_s / (3600 * config.correlation_hours))
        self.innovation_sd_m = config.boundary_sd_m * np.sqrt(1-self.alpha**2)

    def assimilate(self, base_boundary_m, observed_m, observation_locations_m,
                   assimilated_columns=None):
        """Run EnKF. Does not use withheld station data in any analysis update.

        Returns dict with full analysis means, forecasts before updates, and
        augmented ensemble variance. The first time point is an initial state.
        """
        base = np.asarray(base_boundary_m, float)
        obs = np.asarray(observed_m, float)
        if obs.ndim != 2 or obs.shape[0] != len(base):
            raise ValueError("observed_m must be shape (n_times, n_stations)")
        locs = np.asarray(observation_locations_m, float)
        if len(locs) != obs.shape[1]:
            raise ValueError("Each observation column needs a location")
        cols = list(range(len(locs))) if assimilated_columns is None else list(assimilated_columns)
        if not cols or min(cols)<0 or max(cols)>=len(locs) or len(set(cols))!=len(cols):
            raise ValueError("Invalid list of assimilated columns")
        H = self.model.height_operator(locs[cols])
        N = self.config.ensemble_size
        R_sd = self.config.observation_sd_m
        nstate = self.model.nstate
        # A column is an augmented state ensemble member [h,u,...,eta].
        ensemble = np.zeros((nstate+1,N))
        ensemble[-1] = self.rng.normal(0, self.config.boundary_sd_m, N)
        ensemble[0] = base[0] + ensemble[-1]
        result = np.empty((len(base), nstate+1))
        prior = np.empty((len(base), nstate+1))
        variances = np.empty((len(base), nstate+1))
        result[0] = ensemble.mean(axis=1)
        prior[0] = result[0]
        variances[0] = ensemble.var(axis=1, ddof=1)
        for time_idx in range(1,len(base)):
            # ENSEMBLE FORECAST
            ensemble[-1] = (self.alpha*ensemble[-1]
                            + self.innovation_sd_m*self.rng.standard_normal(N))
            ensemble[:-1] = self.model.step(ensemble[:-1], base[time_idx]+ensemble[-1])
            prior[time_idx] = ensemble.mean(axis=1)
            # ANALYSIS UPDATE (not using withheld columns)
            if np.all(np.isfinite(obs[time_idx,cols])):
                mean = ensemble.mean(axis=1, keepdims=True)
                anomalies = ensemble - mean
                Y_anom = H @ anomalies[:-1]
                P_xy = anomalies @ Y_anom.T / (N-1)
                P_yy = Y_anom @ Y_anom.T / (N-1) + (R_sd**2)*np.eye(len(cols))
                K = np.linalg.solve(P_yy, P_xy.T).T
                noise = R_sd*self.rng.standard_normal((len(cols),N))
                residual = obs[time_idx,cols,None] + noise - H@ensemble[:-1]
                ensemble += K@residual
                # Boundary consistency: h(0) is determined by the AR(1) forcing;
                # velocity at the eastern boundary is prescribed as 0.
                ensemble[0] = base[time_idx]+ensemble[-1]
                ensemble[nstate-1] = 0.
            result[time_idx] = ensemble.mean(axis=1)
            variances[time_idx] = ensemble.var(axis=1,ddof=1)
        return {"analysis_states":result, "forecast_states":prior,
                "analysis_variance":variances, "alpha":self.alpha,
                "innovation_sd_m":self.innovation_sd_m,
                "assimilated_columns":cols}
