"""
Synthetic causal system for Specialist A (world-model) and, later,
Specialist B (causal reasoning).

Design choices:
- 3 hidden latent variables (z1, z2, z3) with Lorenz-style nonlinear
  coupling. Chosen because it's a well-understood chaotic system:
  sensitive to initial conditions, nonlinear, and has a natural
  "regime shift" lever (the rho parameter) that qualitatively changes
  behavior without requiring a custom-designed nonlinearity.
- The agent never sees z directly. It only sees `observe(z)`, a noisy
  nonlinear projection down to observation space. This is deliberate:
  the world-model has to learn to predict in a space that hides the
  true causal state, which is the realistic case.
- `intervene()` exists now, unused by the world-model, but required
  later for the causal module's validation (do(z_i = v) vs just
  observing z_i = v historically).
"""

import numpy as np


class SyntheticSystem:
    def __init__(
        self,
        dt: float = 0.01,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        obs_dim: int = 6,
        obs_noise_std: float = 0.05,
        seed: int | None = None,
    ):
        self.dt = dt
        self.sigma = sigma
        self.rho = rho          # this is the regime-shift lever
        self.beta = beta
        self.obs_dim = obs_dim
        self.obs_noise_std = obs_noise_std

        self.rng = np.random.default_rng(seed)

        # Fixed random nonlinear projection from latent -> observation space.
        # Fixed (not learned) because the environment's job is just to
        # hide the latent, not to be interesting on its own.
        self.W = self.rng.normal(scale=0.6, size=(obs_dim, 3))

        self.z = self.rng.normal(scale=1.0, size=3).astype(np.float64)

    def reset(self, z0: np.ndarray | None = None) -> np.ndarray:
        if z0 is not None:
            self.z = np.array(z0, dtype=np.float64)
        else:
            self.z = self.rng.normal(scale=1.0, size=3).astype(np.float64)
        return self.observe(self.z)

    def _dynamics(self, z: np.ndarray, action: float) -> np.ndarray:
        z1, z2, z3 = z
        dz1 = self.sigma * (z2 - z1) + action
        dz2 = z1 * (self.rho - z3) - z2
        dz3 = z1 * z2 - self.beta * z3
        return np.array([dz1, dz2, dz3], dtype=np.float64)

    def observe(self, z: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(scale=self.obs_noise_std, size=self.obs_dim)
        return np.tanh(self.W @ z) + noise

    def step(self, action: float = 0.0) -> np.ndarray:
        """Advance one timestep via forward Euler integration."""
        dz = self._dynamics(self.z, action)
        self.z = self.z + self.dt * dz
        return self.observe(self.z)

    def intervene(self, index: int, value: float) -> np.ndarray:
        """
        do(z_index = value): forcibly overwrite one latent variable,
        bypassing normal dynamics for this step, then let the system
        continue evolving from the new state on subsequent step() calls.
        """
        if not 0 <= index < 3:
            raise ValueError(f"index must be in [0, 3), got {index}")
        self.z[index] = value
        return self.observe(self.z)

    def set_regime(self, rho: float) -> None:
        """Change the causal driver (regime shift lever)."""
        self.rho = rho

    def true_state(self) -> np.ndarray:
        """Exposed only for debugging/eval, never fed to the model."""
        return self.z.copy()