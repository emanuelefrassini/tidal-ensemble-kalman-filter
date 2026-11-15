"""Input readers for the supplied MATROOS / Rijkswaterstaat text series."""
from datetime import datetime
from pathlib import Path

import numpy as np

STATIONS = ("vlissingen", "terneuzen", "hansweert", "bath")
# IDEALIZED positions from the TU Delft exercise, not geodesic distances.
STATION_X_M = np.array([25_000., 50_000., 75_000., 99_000.])


def read_series(path):
    timestamps, heights = [], []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            date, height = line.split()[:2]
            timestamps.append(datetime.strptime(date, "%Y%m%d%H%M"))
            heights.append(float(height))
    if not timestamps:
        raise ValueError(f"No records in {path}")
    return np.array(timestamps, dtype=object), np.array(heights, dtype=float)


def load_observations(data_dir, kind="waterlevel"):
    if kind not in ("waterlevel", "tide"):
        raise ValueError("kind must be 'waterlevel' or 'tide'")
    data_dir = Path(data_dir)
    times = None
    columns = []
    for station in STATIONS:
        t, y = read_series(data_dir / f"{kind}_{station}.txt")
        if times is not None and not np.array_equal(times, t):
            raise ValueError(f"Timestamp mismatch for {station}")
        times = t
        columns.append(y)
    return times, np.column_stack(columns)


def load_western_tide(data_dir):
    return read_series(Path(data_dir) / "tide_cadzand.txt")
