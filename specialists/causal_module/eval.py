"""
Evaluation of Specialist B — causal reasoning module.

Three validation tests, corresponding to Pearl's Causal Hierarchy:

LEVEL 1 — OBSERVATIONAL: Does the discovered SCM accurately predict
    dynamics on held-out observational data (trajectories not used during
    fitting)? Measured by R² and mean prediction error on true-state sequences.

LEVEL 2 — INTERVENTIONAL: Does the SCM correctly predict what happens
    when we forcibly set a variable, bypassing normal dynamics? Compared
    against actual system behavior under synthetic_system.intervene().
    This is the test that pure regression models fail.

REGIME DETECTION: Does the residual score spike when dynamics shift from
    rho=28 to rho=60? This is the primary signal the binding layer will use
    to tell the world-model predictor that a structural change has occurred.
    Per-window rho estimates are computed to quantify the detected change.

Why the interventional test matters (from the source thesis, Pearl's theorem):
    "Current LLMs and traditional machine learning train on observational data
    and can never truly understand cause and effect just by reading off the
    internet." An SCM fitted by SINDy can answer interventional queries by
    construction — the structural equations define exactly what changes when
    a variable is forced to a value, independent of its observational distribution.

Expected results (validated pre-coding):
    - Observational R² > 0.999 on held-out data
    - Interventional prediction closer to actual system than observational baseline
    - Residual score on rho=60 data >> residual score on rho=28 data
    - Per-window rho estimates tracking the true rho change
"""

from __future__ import annotations

import numpy as np
from envs.synthetic_system import SyntheticSystem
from specialists.causal_module.discovery import (
    CausalDiscovery,
    CausalGraph,
    generate_discovery_data,
)
from specialists.causal_module.scm import StructuralCausalModel


# ─────────────────────────────────────────────────────────────────────────────
# Level 1: Observational validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_observational(
    scm:           StructuralCausalModel,
    Z:             np.ndarray,
    actions:       np.ndarray,
    dZ_dt:         np.ndarray,
) -> dict:
    """
    Validate SCM observational predictions on held-out trajectories.

    Args:
        scm:     Fitted StructuralCausalModel
        Z:       (N, 3) held-out latent states
        actions: (N,)   held-out actions
        dZ_dt:   (N, 3) held-out derivative approximations

    Returns:
        {
          r2_per_var:     (3,)   R² for each variable equation
          mean_pred_error: float  mean abs prediction error for state transitions
          residual_score:  float  mean squared derivative residual
        }
    """
    from specialists.causal_module.discovery import build_library

    Theta   = build_library(Z, actions)
    dZ_pred = Theta @ scm.graph.coefficients.T

    r2 = np.zeros(3)
    for i in range(3):
        ss_res = np.sum((dZ_dt[:, i] - dZ_pred[:, i]) ** 2)
        ss_tot = np.sum((dZ_dt[:, i] - dZ_dt[:, i].mean()) ** 2)
        r2[i]  = 1.0 - ss_res / (ss_tot + 1e-12)

    # Mean absolute one-step prediction error
    dt = scm.dt
    z_pred_next = Z + dt * dZ_pred
    z_true_next = Z + dt * dZ_dt          # ground truth via finite diff
    pred_error  = float(np.mean(np.abs(z_pred_next - z_true_next)))

    return {
        "r2_per_var":      r2,
        "mean_pred_error": pred_error,
        "residual_score":  scm.residual_score(Z, actions, dZ_dt),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Level 2: Interventional validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_interventional(
    scm:         StructuralCausalModel,
    rho:         float = 28.0,
    n_trials:    int   = 100,
    seed:        int   = 77,
) -> dict:
    """
    Validate single-step interventional predictions against actual system.

    Pearl Level 2 test (do-calculus vs conditioning):
        For each trial, we have state z = (z1_curr, z2_curr, z3_curr).
        We force z1 to a new value v (do(z1=v)) and observe what z2 does
        in the NEXT step.

        Three predictions are compared:
        1. SCM: uses structural equation dz2/dt = rho*v - z2_curr - v*z3_curr
                (correctly uses z3_curr — whatever z3 actually IS, not what it
                historically tends to be when z1≈v)
        2. Naive baseline: predicts z2 stays at z2_curr (no-change prediction)
        3. Ground truth: actual system.intervene(0, v) then system.step()

        Why SCM should beat the naive baseline:
            The structural equation predicts the DIRECTION of z2's change given
            the intervention. When z1 is forced to a very different value, z2
            changes substantially and in a predictable direction — the SCM
            captures this, the naive baseline doesn't.

        Design note on multi-step interventions:
            A sustained do(z1=v for T steps) creates a fundamentally different
            dynamical system (Lorenz with z1 pinned). That trajectory diverges
            rapidly from any real system trajectory — not a useful test. The
            single-step test correctly isolates the effect of one intervention.

    Returns:
        {
          scm_error:        mean abs z2 prediction error (SCM vs true)
          naive_error:      mean abs z2 prediction error (naive vs true)
          improvement:      naive_error / scm_error  (>1 = SCM better)
          n_trials:         int
          mean_z2_change:   mean |z2_next - z2_curr| (magnitude of effect)
        }
    """
    rng = np.random.default_rng(seed)
    sys = SyntheticSystem(rho=rho, obs_noise_std=0.0, seed=seed)

    # Warm up system to get a diverse state distribution
    sys.reset()
    for _ in range(500):
        sys.step(0.0)

    scm_errors   = []
    naive_errors = []
    z2_changes   = []

    for _ in range(n_trials):
        # Advance a few steps for diversity
        for _ in range(5):
            sys.step(float(rng.normal(0, 0.1)))

        z_curr = sys.true_state().copy()   # (z1_curr, z2_curr, z3_curr)

        # Force z1 to a value far from its current position —
        # this maximises the difference between do() and observe()
        v = float(rng.uniform(20.0, 40.0))   # Lorenz z1 typically in [-20, 20]

        # Ground truth: intervene on a fresh copy, take one step
        sys_copy = SyntheticSystem(rho=rho, obs_noise_std=0.0, seed=0)
        sys_copy.z = z_curr.copy()
        sys_copy.intervene(0, v)              # do(z1 = v)
        sys_copy.step(0.0)
        z2_true_next = sys_copy.true_state()[1]

        # SCM prediction: structural equation for z2 with z1 forced to v
        z_intervened     = z_curr.copy()
        z_intervened[0]  = v
        z2_scm_next  = scm.predict_next(z_intervened, action=0.0)[1]

        # Naive baseline: z2 stays where it is (no-change prediction)
        z2_naive_next = z_curr[1]

        scm_err   = abs(z2_scm_next   - z2_true_next)
        naive_err = abs(z2_naive_next  - z2_true_next)

        scm_errors.append(scm_err)
        naive_errors.append(naive_err)
        z2_changes.append(abs(z2_true_next - z_curr[1]))

    mean_scm   = float(np.mean(scm_errors))
    mean_naive = float(np.mean(naive_errors))

    return {
        "scm_error":      mean_scm,
        "naive_error":    mean_naive,
        "improvement":    mean_naive / (mean_scm + 1e-10),
        "n_trials":       n_trials,
        "mean_z2_change": float(np.mean(z2_changes)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Regime detection
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_regime_detection(
    scm:             StructuralCausalModel,
    window_size:     int   = 50,
    n_windows_each:  int   = 30,
    seed:            int   = 55,
) -> dict:
    """
    Test whether residual scores and rho estimates correctly identify regime.

    Generates windows from two stationary regimes (rho=28 and rho=60) and
    computes per-window residual scores and rho estimates. A successful
    detector should show clearly separated distributions.

    Args:
        scm:             SCM fitted on rho=28 data
        window_size:     number of steps per test window
        n_windows_each:  number of test windows per regime
        seed:            RNG seed

    Returns:
        {
          rho28_residuals:  (n_windows,) residual scores on rho=28 windows
          rho60_residuals:  (n_windows,) residual scores on rho=60 windows
          rho28_estimates:  (n_windows,) estimated rho from rho=28 windows
          rho60_estimates:  (n_windows,) estimated rho from rho=60 windows
          separation_ratio: mean(rho60_residuals) / mean(rho28_residuals)
          rho_estimate_error_28: mean |estimated - true| for rho=28 windows
          rho_estimate_error_60: mean |estimated - true| for rho=60 windows
        }
    """
    def get_windows(rho_val, rng):
        Z_w, a_w, dZ_w = generate_discovery_data(
            rho=rho_val,
            n_traj=n_windows_each,
            n_steps=window_size,
            seed=int(rng.integers(0, 9999)),
        )
        # Split into per-window chunks
        residuals  = []
        rho_ests   = []
        for i in range(n_windows_each):
            sl = slice(i * window_size, (i + 1) * window_size)
            Zw = Z_w[sl]; aw = a_w[sl]; dZw = dZ_w[sl]
            residuals.append(scm.residual_score(Zw, aw, dZw))
            rho_ests.append(scm.estimate_rho_from_window(Zw, aw, dZw))
        return np.array(residuals), np.array(rho_ests)

    rng = np.random.default_rng(seed)
    res28, est28 = get_windows(28.0, rng)
    res60, est60 = get_windows(60.0, rng)

    return {
        "rho28_residuals":        res28,
        "rho60_residuals":        res60,
        "rho28_estimates":        est28,
        "rho60_estimates":        est60,
        "separation_ratio":       float(res60.mean() / (res28.mean() + 1e-12)),
        "rho_estimate_error_28":  float(np.mean(np.abs(est28 - 28.0))),
        "rho_estimate_error_60":  float(np.mean(np.abs(est60 - 60.0))),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    n_traj_fit:   int = 30,
    n_traj_test:  int = 10,
    seed:         int = 0,
) -> dict:
    """
    Full evaluation of the causal module.

    1. Fits SCM on rho=28 training data
    2. Validates observationally on held-out rho=28 data
    3. Validates interventionally against actual system
    4. Tests regime detection on rho=28 vs rho=60 windows

    Returns full results dict and prints a formatted report.
    """
    print("=" * 60)
    print("  CAUSAL MODULE EVALUATION")
    print("=" * 60)

    # ── Fit ────────────────────────────────────────────────────────────────
    print("\n── Fitting SCM on rho=28 training data ──────────────────────")
    disc = CausalDiscovery()
    graph = disc.fit_from_system(rho=28.0, n_traj=n_traj_fit, seed=seed)
    scm   = StructuralCausalModel(graph)
    print(graph)

    # ── Level 1: Observational ─────────────────────────────────────────────
    print("\n── Level 1: Observational validation (held-out rho=28) ──────")
    Z_test, a_test, dZ_test = generate_discovery_data(
        rho=28.0, n_traj=n_traj_test, seed=seed + 999
    )
    obs_results = validate_observational(scm, Z_test, a_test, dZ_test)
    print(f"  R² per variable:   {obs_results['r2_per_var'].round(5)}")
    print(f"  Mean pred error:   {obs_results['mean_pred_error']:.6f}")
    print(f"  Residual score:    {obs_results['residual_score']:.4f}")
    print("  → Expect R² ≈ 1.0, small prediction error")

    # ── Level 2: Interventional ────────────────────────────────────────────
    print("\n── Level 2: Interventional validation ───────────────────────")
    int_results = validate_interventional(scm, rho=28.0, n_trials=30, seed=seed)
    print(f"  Mean z2 change per intervention:      {int_results['mean_z2_change']:.4f}")
    print(f"  SCM error (do(z1=v) → z2_next):      {int_results['scm_error']:.5f}")
    print(f"  Naive baseline error (z2 unchanged):  {int_results['naive_error']:.5f}")
    print(f"  SCM improvement over naive baseline:  {int_results['improvement']:.2f}×")
    if int_results["improvement"] > 1.0:
        print("  ✓  SCM predicts post-intervention z2 better than no-change baseline.")
        print("     Structural equations correctly capture do(z1=v) → dz2/dt effect.")
    else:
        print("  ~  SCM not outperforming naive baseline — check equation recovery.")

    # ── Regime detection ────────────────────────────────────────────────────
    print("\n── Regime detection (rho=28 vs rho=60) ──────────────────────")
    regime_results = evaluate_regime_detection(scm, seed=seed)
    print(f"  Mean residual (rho=28 windows): {regime_results['rho28_residuals'].mean():.4f}")
    print(f"  Mean residual (rho=60 windows): {regime_results['rho60_residuals'].mean():.4f}")
    print(f"  Separation ratio:               {regime_results['separation_ratio']:.1f}×")
    print(f"  Rho estimate error (rho=28):    {regime_results['rho_estimate_error_28']:.3f}")
    print(f"  Rho estimate error (rho=60):    {regime_results['rho_estimate_error_60']:.3f}")
    if regime_results["separation_ratio"] > 10.0:
        print("  ✓  Regime shift clearly detectable from residual scores.")
        print("     This is the signal the binding layer will pass to the world-model.")
    else:
        print("  ~  Separation ratio < 10 — regime detection signal is weak.")

    print("\n── Specialist B status ──────────────────────────────────────")
    print("  The causal module is VALIDATED when:")
    print("  1. Observational R² > 0.99  ✓" if obs_results["r2_per_var"].min() > 0.99 else
          "  1. Observational R² > 0.99  ✗")
    print("  2. Regime separation ratio > 10×  ✓" if regime_results["separation_ratio"] > 10.0 else
          "  2. Regime separation ratio > 10×  ✗")
    print("  3. Interventional improvement > 1.0×  ✓" if int_results["improvement"] > 1.0 else
          "  3. Interventional improvement > 1.0×  ✗")
    print()
    print("  Next: bind Specialist A + B via integration/binding_v0_1_conditioning.py")
    print("  Expected POC result: bound pair crossing error < standalone predictor baseline (1.15×)")

    return {
        "graph":           graph,
        "scm":             scm,
        "obs_results":     obs_results,
        "int_results":     int_results,
        "regime_results":  regime_results,
    }


if __name__ == "__main__":
    evaluate()