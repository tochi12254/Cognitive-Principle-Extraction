"""
Structural Causal Model (SCM) for Specialist B.

Uses the discovered equations from CausalDiscovery to make three kinds of
predictions:

1. OBSERVATIONAL prediction: predict next state from current state using the
   discovered structural equations (same as a good regression model).

2. INTERVENTIONAL prediction: predict what happens after do(z_i = v) —
   Pearl's Hierarchy Level 2. Forces z_i to value v, applies the structural
   equations to propagate the effect through the system. This is NOT the same
   as conditioning on z_i = v observed historically (Level 1).

3. REGIME SCORE: compute the residual of current observed dynamics under the
   fitted SCM. High residual = current dynamics don't match the trained model
   = regime has likely shifted. This is the signal the binding layer will use
   to tell the world-model predictor that a shift has occurred.

Why SCM and not just the regression model (full reasoning in research log):
    A regression model fitted by LASSO is an observational model — it predicts
    well on data similar to training but treats all correlations as equal.
    An SCM uses the STRUCTURAL equations discovered by LASSO to reason about
    interventions. The difference: if we observe z1=10 historically, z2 and z3
    take specific correlated values (because z1=10 occurs naturally in specific
    parts of the Lorenz attractor). If we DO(z1=10) — force z1 to 10 regardless
    of the normal dynamics — z2 and z3 evolve from wherever they are, driven
    by the structural equations with z1 fixed at 10. These are different
    predictions. The SCM gets the interventional one right; a pure regression
    model gets it wrong.

Key property (Pearl's Causal Hierarchy, from source thesis):
    "Facts at the interventional or counterfactual level are not recoverable
    from observational data alone because observational data has no information
    about what would happen if something had been changed regardless of dataset
    size."
    The SCM encodes the structural equations precisely to go beyond this limit.
"""

from __future__ import annotations

import numpy as np

from specialists.causal_module.discovery import CausalGraph, build_library


class StructuralCausalModel:
    """
    Structural Causal Model built from a discovered CausalGraph.

    Args:
        graph: CausalGraph from CausalDiscovery.fit()
        dt:    Integration timestep (must match SyntheticSystem.dt, default 0.01)
    """

    def __init__(self, graph: CausalGraph, dt: float = 0.01):
        self.graph = graph
        self.dt    = dt

    def _apply_equations(
        self,
        z:      np.ndarray,   # (3,) current state
        action: float = 0.0,
    ) -> np.ndarray:
        """
        Compute dz/dt from the discovered structural equations.

        Args:
            z:      (3,) current latent state
            action: scalar action applied to z1 dynamics

        Returns:
            dz_dt: (3,) estimated derivative vector
        """
        theta = build_library(z[np.newaxis], np.array([action]))[0]  # (10,)
        return self.graph.coefficients @ theta   # (3,)

    def predict_next(
        self,
        z:      np.ndarray,
        action: float = 0.0,
    ) -> np.ndarray:
        """
        Predict state at t+1 using Euler integration of structural equations.

        This is an OBSERVATIONAL prediction — no interventions applied.

        Args:
            z:      (3,) current state
            action: scalar action

        Returns:
            z_next: (3,) predicted next state
        """
        dz_dt = self._apply_equations(z, action)
        return z + self.dt * dz_dt

    def rollout(
        self,
        z0:      np.ndarray,
        n_steps: int,
        actions: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Rollout the SCM forward from initial state z0.

        Args:
            z0:      (3,) initial state
            n_steps: number of steps to simulate
            actions: (n_steps,) action sequence, or None for zero actions

        Returns:
            states: (n_steps + 1, 3) — includes z0 at index 0
        """
        if actions is None:
            actions = np.zeros(n_steps)

        states    = np.empty((n_steps + 1, 3))
        states[0] = z0.copy()

        for t in range(n_steps):
            states[t + 1] = self.predict_next(states[t], float(actions[t]))

        return states

    def predict_under_intervention(
        self,
        z0:     np.ndarray,
        do:     dict,
        n_steps: int,
        actions: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Rollout under do(z_i = v) — Pearl's Hierarchy Level 2.

        Forces specified variables to given values at EVERY step of the
        rollout (a "hard intervention" / "atomic intervention"). The
        structural equations still govern the other variables.

        This is different from conditioning: we force z_i = v and let
        the dynamics propagate — we don't select trajectories where
        z_i happened to equal v historically.

        Args:
            z0:      (3,) initial state (before intervention)
            do:      {var_index: value} — variables to force at every step.
                     Example: {0: 10.0} forces z1 = 10.0 at every step.
            n_steps: number of steps to simulate
            actions: (n_steps,) optional action sequence

        Returns:
            states: (n_steps + 1, 3) — trajectory under intervention.
                    Index 0 is z0 with do-values applied.
        """
        if actions is None:
            actions = np.zeros(n_steps)

        states    = np.empty((n_steps + 1, 3))
        states[0] = z0.copy()
        # Apply intervention to initial state
        for var_idx, value in do.items():
            states[0, var_idx] = value

        for t in range(n_steps):
            z_current = states[t].copy()
            # Apply structural equations
            z_next = self.predict_next(z_current, float(actions[t]))
            # Re-apply hard intervention: forced variable stays at value
            for var_idx, value in do.items():
                z_next[var_idx] = value
            states[t + 1] = z_next

        return states

    def residual_score(
        self,
        Z:       np.ndarray,
        actions: np.ndarray,
        dZ_dt:   np.ndarray,
    ) -> float:
        """
        Compute mean squared residual of the SCM on observed dynamics.

        Regime shift detection: fit SCM on rho=28 data, then call
        residual_score() on rho=60 data. The score will be significantly
        higher because the z1 coefficient in the z2 equation is wrong (28
        instead of 60). This is the quantitative signal for regime shift.

        Args:
            Z:       (N, 3)  — latent states
            actions: (N,)    — actions
            dZ_dt:   (N, 3)  — observed finite-difference derivatives

        Returns:
            Scalar mean squared residual across all variables and timesteps.
            Lower = dynamics match the SCM well (same regime).
            Higher = dynamics don't match (regime may have shifted).
        """
        Theta  = build_library(Z, actions)              # (N, 10)
        dZ_pred = Theta @ self.graph.coefficients.T      # (N, 3)
        residuals = dZ_dt - dZ_pred
        return float(np.mean(residuals ** 2))

    def per_variable_residual(
        self,
        Z:       np.ndarray,
        actions: np.ndarray,
        dZ_dt:   np.ndarray,
    ) -> np.ndarray:
        """
        Per-variable residual scores — (3,) array.

        Useful for diagnosing which equation is most affected by the regime
        shift. Expect z2 to show the largest residual since rho only appears
        in the dz2/dt equation.
        """
        Theta   = build_library(Z, actions)
        dZ_pred = Theta @ self.graph.coefficients.T
        residuals = dZ_dt - dZ_pred
        return np.mean(residuals ** 2, axis=0)

    def estimate_rho_from_window(
        self,
        Z:       np.ndarray,
        actions: np.ndarray,
        dZ_dt:   np.ndarray,
    ) -> float:
        """
        Estimate rho from a window of observed dynamics via OLS on the z2 equation.

        This gives a pointwise rho estimate from any trajectory window —
        the primary regime-detection signal. If the estimate differs significantly
        from the trained SCM's rho_estimate, the regime has likely shifted.

        Uses ordinary least squares (not LASSO) on the full z2 equation to get
        the best unbiased estimate of the z1 coefficient from the available window.

        Args:
            Z:       (N, 3)  — latent states (N >= 10 recommended for stability)
            actions: (N,)    — actions
            dZ_dt:   (N, 3)  — observed derivatives

        Returns:
            Estimated rho (z1 coefficient in dz2/dt equation).
        """
        from sklearn.linear_model import LinearRegression

        Theta  = build_library(Z, actions)
        target = dZ_dt[:, 1]   # dz2/dt

        lr = LinearRegression(fit_intercept=False)
        lr.fit(Theta, target)
        return float(lr.coef_[0])   # z1 coefficient = rho estimate