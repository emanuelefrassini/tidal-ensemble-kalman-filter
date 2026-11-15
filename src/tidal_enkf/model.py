"""One-dimensional linear shallow-water model.

    h_t + D u_x = 0
    u_t + g h_x + f u = 0

Water elevations h_i and velocities u_{i+1/2} lie on a staggered grid.
Crank--Nicolson timestepping handles the linear dynamics. Boundary conditions
are a prescribed western water level and a closed eastern velocity boundary.
The computational domain and boundary conditions are idealisations of the estuary.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import eye, lil_matrix
from scipy.sparse.linalg import splu


@dataclass(frozen=True)
class ModelConfig:
    length_m: float = 100_000.0
    depth_m: float = 20.0
    gravity: float = 9.81
    friction_per_s: float = 1.0 / (0.06 * 86400.0)
    cells: int = 100
    dt_s: float = 600.0

    def __post_init__(self):
        if min(self.length_m, self.depth_m, self.gravity, self.dt_s) <= 0:
            raise ValueError("Length, depth, gravity and time step must be positive")
        if self.cells < 4 or self.friction_per_s < 0:
            raise ValueError("At least four cells and nonnegative friction are required")


class ShallowWater1D:
    """Semi-discrete staggered scheme with state [h0,u0,h1,u1,...]."""

    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config
        n = config.cells
        self.nstate = 2 * n
        # n water points and n velocity points; final velocity = 0 at x=L
        self.dx_m = config.length_m / (n - 0.5)
        self.height_positions_m = np.arange(n) * self.dx_m
        self.velocity_positions_m = (np.arange(n) + 0.5) * self.dx_m
        self.h_idx = 2 * np.arange(n)
        self.u_idx = self.h_idx + 1

        J = lil_matrix((self.nstate, self.nstate), dtype=float)
        D, g, f, dx = (config.depth_m, config.gravity,
                        config.friction_per_s, self.dx_m)
        for i in range(1, n):  # h0 is prescribed, so skip its ODE
            J[2*i, 2*i+1] = -D / dx
            J[2*i, 2*i-1] = D / dx
        for i in range(n-1):  # final velocity is identically zero
            J[2*i+1, 2*(i+1)] = -g / dx
            J[2*i+1, 2*i] = g / dx
            J[2*i+1, 2*i+1] = -f

        I = eye(self.nstate, format="csr")
        A = (I - 0.5 * config.dt_s * J.tocsr()).tolil()
        B = (I + 0.5 * config.dt_s * J.tocsr()).tolil()
        for idx in (0, self.nstate-1):
            A.rows[idx] = [idx]
            A.data[idx] = [1.0]
            B.rows[idx] = []
            B.data[idx] = []
        self._B = B.tocsr()
        self._lu = splu(A.tocsc())

    @property
    def analytical_wave_speed_m_s(self):
        return np.sqrt(self.config.gravity * self.config.depth_m)

    def initial_state(self, west_height_m=0.0):
        x = np.zeros(self.nstate)
        x[0] = west_height_m
        return x

    def step(self, state: np.ndarray, west_height_m):
        """Advance one time step; state is (2*n,) or (2*n, nensemble)."""
        x = np.asarray(state, dtype=float)
        if x.ndim not in (1,2) or x.shape[0] != self.nstate:
            raise ValueError("State must have shape (2*n,) or (2*n, ensemble_size)")
        rhs = self._B @ x
        rhs[0] = west_height_m
        rhs[-1] = 0.0
        return self._lu.solve(np.asarray(rhs))

    def run(self, west_levels_m, initial_state=None):
        """Simulation (n_times, 2*n); array includes initial state at t0."""
        forcing = np.asarray(west_levels_m, dtype=float)
        if forcing.ndim != 1 or len(forcing) < 2:
            raise ValueError("Provide at least two boundary samples")
        x = self.initial_state(forcing[0]) if initial_state is None else np.array(initial_state, dtype=float).copy()
        if x.shape != (self.nstate,):
            raise ValueError("Initial state size mismatch")
        x[0] = forcing[0]
        x[-1] = 0.0
        result = np.empty((len(forcing), self.nstate))
        result[0] = x
        for i in range(1, len(forcing)):
            x = self.step(x, forcing[i])
            result[i] = x
        return result

    def height_operator(self, x_positions_m):
        """Linear interpolation to locations within the water-level grid."""
        locs = np.asarray(x_positions_m, dtype=float)
        if locs.ndim != 1 or np.any(locs < 0) or np.any(locs > self.height_positions_m[-1]):
            raise ValueError("Observation locations outside model height grid")
        H = np.zeros((len(locs), self.nstate))
        for row, loc in enumerate(locs):
            k = min(int(loc/self.dx_m), self.config.cells-2)
            fraction = (loc - self.height_positions_m[k]) / self.dx_m
            H[row, 2*k] = 1 - fraction
            H[row, 2*(k+1)] = fraction
        return H

    def sample_heights(self, states, locations_m):
        H = self.height_operator(locations_m)
        return np.asarray(states) @ H.T
