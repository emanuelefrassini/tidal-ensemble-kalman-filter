"""Transparent forecast and assimilation diagnostics."""
import numpy as np


def metrics(estimate, observed, start_index=0):
    """Per-station bias and RMSE in metres; observations use rows as timestamps."""
    e = np.asarray(estimate, dtype=float)
    y = np.asarray(observed, dtype=float)
    if e.shape != y.shape or e.ndim != 2:
        raise ValueError("Estimate and observations must be (n_times, n_stations)")
    error = e[start_index:] - y[start_index:]
    return {"bias_m": np.mean(error, axis=0),
            "rmse_m": np.sqrt(np.mean(error**2, axis=0))}
