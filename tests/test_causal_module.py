"""
Test suite for specialists/causal_module/

Covers discovery.py:
- build_library returns correct shape and feature names
- generate_discovery_data returns correct shapes and finite values
- CausalDiscovery.fit recovers correct causal structure
- CausalGraph r2_scores are high (> 0.99) for Lorenz system
- CausalGraph rho_estimate close to true rho (within 5%)
- Adjacency matrix correctly reflects known Lorenz structure
- rho=28 and rho=60 produce clearly different rho_estimates

Covers scm.py:
- predict_next returns correct shape
- rollout returns correct shape including z0
- predict_under_intervention holds forced variable throughout
- residual_score lower on same-regime data, higher on shifted-regime data
- per_variable_residual: z2 equation shows largest residual on rho=60
- estimate_rho_from_window recovers rho accurately
- StructuralCausalModel is deterministic (same input → same output)

Covers eval.py:
- validate_observational returns expected keys and R² > 0.99
- validate_interventional: SCM improvement > 1.0×
- evaluate_regime_detection: separation_ratio >> 1.0
"""

import numpy as np
import pytest

from specialists.causal_module.discovery import (
    CausalDiscovery,
    CausalGraph,
    build_library,
    generate_discovery_data,
    LIBRARY_NAMES,
    N_LIBRARY,
)
from specialists.causal_module.scm import StructuralCausalModel
from specialists.causal_module.eval import (
    validate_observational,
    validate_interventional,
    evaluate_regime_detection,
)
from envs.synthetic_system import SyntheticSystem


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fitted_graph():
    """Fit once, reuse across tests."""
    disc = CausalDiscovery()
    return disc.fit_from_system(rho=28.0, n_traj=20, seed=0)


@pytest.fixture(scope="module")
def scm_28(fitted_graph):
    return StructuralCausalModel(fitted_graph)


@pytest.fixture(scope="module")
def test_data_28():
    return generate_discovery_data(rho=28.0, n_traj=5, seed=99)


@pytest.fixture(scope="module")
def test_data_60():
    return generate_discovery_data(rho=60.0, n_traj=5, seed=99)


# ─────────────────────────────────────────────────────────────────────────────
# build_library tests
# ─────────────────────────────────────────────────────────────────────────────

def test_build_library_shape():
    Z = np.random.randn(50, 3)
    a = np.random.randn(50)
    Theta = build_library(Z, a)
    assert Theta.shape == (50, N_LIBRARY), f"Expected (50, {N_LIBRARY}), got {Theta.shape}"


def test_build_library_correct_terms():
    """Spot-check a known state: verify cross terms are z1*z2 etc."""
    Z = np.array([[2.0, 3.0, 5.0]])
    a = np.array([0.1])
    Theta = build_library(Z, a)[0]
    names = LIBRARY_NAMES
    assert abs(Theta[names.index("z1z2")] - 6.0)  < 1e-9   # 2*3
    assert abs(Theta[names.index("z1z3")] - 10.0) < 1e-9   # 2*5
    assert abs(Theta[names.index("z2z3")] - 15.0) < 1e-9   # 3*5
    assert abs(Theta[names.index("z1^2")] - 4.0)  < 1e-9   # 2^2
    assert abs(Theta[names.index("action")] - 0.1) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# generate_discovery_data tests
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_discovery_data_shapes():
    Z, a, dZ = generate_discovery_data(rho=28.0, n_traj=3, n_steps=50, seed=0)
    N = 3 * 50
    assert Z.shape  == (N, 3)
    assert a.shape  == (N,)
    assert dZ.shape == (N, 3)


def test_generate_discovery_data_finite():
    Z, a, dZ = generate_discovery_data(rho=28.0, n_traj=3, n_steps=50, seed=0)
    assert np.isfinite(Z).all()
    assert np.isfinite(a).all()
    assert np.isfinite(dZ).all()


def test_generate_discovery_data_reproducible():
    Z1, a1, dZ1 = generate_discovery_data(rho=28.0, n_traj=3, seed=42)
    Z2, a2, dZ2 = generate_discovery_data(rho=28.0, n_traj=3, seed=42)
    np.testing.assert_array_equal(Z1, Z2)
    np.testing.assert_array_equal(a1, a2)


# ─────────────────────────────────────────────────────────────────────────────
# CausalDiscovery tests
# ─────────────────────────────────────────────────────────────────────────────

def test_causal_discovery_r2_scores_high(fitted_graph):
    """R² > 0.99 for all three Lorenz equations."""
    assert fitted_graph.r2_scores.min() > 0.99, (
        f"R² scores too low: {fitted_graph.r2_scores}"
    )


def test_causal_discovery_rho_estimate_accurate(fitted_graph):
    """Estimated rho within 5% of true rho=28."""
    true_rho = 28.0
    assert abs(fitted_graph.rho_estimate - true_rho) < 0.05 * true_rho, (
        f"rho estimate {fitted_graph.rho_estimate:.2f} too far from {true_rho}"
    )


def test_causal_discovery_recovers_z2_parents(fitted_graph):
    """
    z2 equation must have z1 and z3 as causal parents (from z1*z3 cross term).
    True equation: dz2/dt = rho*z1 - z2 - z1*z3
    """
    # z1 coefficient in z2 equation (index 1, library index 0)
    assert abs(fitted_graph.coefficients[1, 0]) > 1.0, (
        "z1 coefficient in dz2/dt too small — rho not recovered."
    )
    # z1z3 cross term coefficient (library index 4)
    assert abs(fitted_graph.coefficients[1, 4]) > 0.5, (
        "z1z3 coefficient in dz2/dt too small — nonlinear coupling not recovered."
    )


def test_causal_discovery_recovers_z3_parents(fitted_graph):
    """
    z3 equation: dz3/dt = z1*z2 - beta*z3
    z1z2 cross term (lib idx 3) and z3 linear term (lib idx 2) must be present.
    """
    assert abs(fitted_graph.coefficients[2, 3]) > 0.5, "z1z2 coefficient in dz3 missing."
    assert abs(fitted_graph.coefficients[2, 2]) > 1.0, "z3 self-decay coefficient in dz3 missing."


def test_causal_discovery_distinguishes_rho_28_vs_60():
    """
    SCMs fit on rho=28 and rho=60 must produce clearly different rho estimates.
    Separation must be > 20 (true separation is 32).
    """
    disc = CausalDiscovery()
    g28  = disc.fit_from_system(rho=28.0, n_traj=15, seed=1)
    g60  = disc.fit_from_system(rho=60.0, n_traj=15, seed=1)
    sep  = abs(g60.rho_estimate - g28.rho_estimate)
    assert sep > 20.0, (
        f"rho estimate separation {sep:.2f} too small — "
        f"g28: {g28.rho_estimate:.2f}, g60: {g60.rho_estimate:.2f}"
    )


def test_causal_graph_n_samples(fitted_graph):
    assert fitted_graph.n_samples == 20 * 300


# ─────────────────────────────────────────────────────────────────────────────
# StructuralCausalModel tests
# ─────────────────────────────────────────────────────────────────────────────

def test_scm_predict_next_shape(scm_28):
    z = np.array([1.0, 2.0, 3.0])
    z_next = scm_28.predict_next(z, action=0.0)
    assert z_next.shape == (3,)


def test_scm_predict_next_finite(scm_28):
    z = np.array([5.0, -3.0, 10.0])
    z_next = scm_28.predict_next(z, action=0.1)
    assert np.isfinite(z_next).all()


def test_scm_rollout_shape(scm_28):
    z0 = np.array([1.0, 1.0, 1.0])
    states = scm_28.rollout(z0, n_steps=20)
    assert states.shape == (21, 3), f"Expected (21, 3), got {states.shape}"


def test_scm_rollout_starts_at_z0(scm_28):
    z0 = np.array([3.0, -2.0, 7.0])
    states = scm_28.rollout(z0, n_steps=10)
    np.testing.assert_array_equal(states[0], z0)


def test_scm_intervention_holds_forced_variable(scm_28):
    """
    do(z1=10.0) must hold z1=10.0 at EVERY step of the rollout.
    """
    z0 = np.array([1.0, 5.0, 8.0])
    traj = scm_28.predict_under_intervention(z0, do={0: 10.0}, n_steps=15)
    assert traj.shape == (16, 3)
    for t in range(16):
        assert abs(traj[t, 0] - 10.0) < 1e-9, (
            f"z1 not held at 10.0 at step {t}: {traj[t, 0]}"
        )


def test_scm_intervention_other_vars_evolve(scm_28):
    """
    Non-forced variables must still change under the intervention.
    """
    z0 = np.array([1.0, 5.0, 8.0])
    traj = scm_28.predict_under_intervention(z0, do={0: 10.0}, n_steps=10)
    # z2 should change (driven by z1=10 which is large)
    assert not np.allclose(traj[:, 1], traj[0, 1]), (
        "z2 did not change under intervention — structural equations not applied."
    )


def test_scm_residual_score_lower_for_same_regime(scm_28, test_data_28, test_data_60):
    """
    Residual score on rho=28 data must be much lower than on rho=60 data.
    This is the regime detection signal.
    """
    Z28, a28, dZ28 = test_data_28
    Z60, a60, dZ60 = test_data_60
    score_28 = scm_28.residual_score(Z28, a28, dZ28)
    score_60 = scm_28.residual_score(Z60, a60, dZ60)
    assert score_60 > score_28 * 100, (
        f"Regime detection weak: rho=28 score={score_28:.4f}, "
        f"rho=60 score={score_60:.4f}, ratio={score_60/score_28:.1f}×"
    )


def test_scm_z2_shows_largest_residual_on_rho60(scm_28, test_data_60):
    """
    rho only appears in the z2 equation — per-variable residual for z2
    must be the largest of the three on rho=60 data.
    """
    Z60, a60, dZ60 = test_data_60
    per_var = scm_28.per_variable_residual(Z60, a60, dZ60)
    assert per_var[1] == per_var.max(), (
        f"z2 residual ({per_var[1]:.2f}) not the largest: {per_var}"
    )


def test_scm_estimate_rho_from_window_accurate(scm_28, test_data_28, test_data_60):
    """
    estimate_rho_from_window must recover rho accurately for both regimes.
    Error must be < 2% of true rho.
    """
    Z28, a28, dZ28 = test_data_28
    Z60, a60, dZ60 = test_data_60
    est28 = scm_28.estimate_rho_from_window(Z28, a28, dZ28)
    est60 = scm_28.estimate_rho_from_window(Z60, a60, dZ60)
    assert abs(est28 - 28.0) < 0.02 * 28.0, f"rho=28 estimate {est28:.3f} too far off."
    assert abs(est60 - 60.0) < 0.02 * 60.0, f"rho=60 estimate {est60:.3f} too far off."


def test_scm_deterministic(scm_28):
    """Same input always gives same prediction."""
    z = np.array([3.0, -1.0, 5.0])
    out1 = scm_28.predict_next(z, action=0.0)
    out2 = scm_28.predict_next(z, action=0.0)
    np.testing.assert_array_equal(out1, out2)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation function tests
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_observational_returns_keys(scm_28, test_data_28):
    Z, a, dZ = test_data_28
    result = validate_observational(scm_28, Z, a, dZ)
    assert "r2_per_var"      in result
    assert "mean_pred_error" in result
    assert "residual_score"  in result


def test_validate_observational_r2_high(scm_28, test_data_28):
    Z, a, dZ = test_data_28
    result = validate_observational(scm_28, Z, a, dZ)
    assert result["r2_per_var"].min() > 0.99, (
        f"R² too low on held-out data: {result['r2_per_var']}"
    )


def test_validate_interventional_scm_outperforms_naive(scm_28):
    result = validate_interventional(scm_28, rho=28.0, n_trials=50, seed=1)
    assert result["improvement"] > 1.0, (
        f"SCM improvement {result['improvement']:.2f}× — should be > 1.0"
    )


def test_validate_interventional_returns_keys(scm_28):
    result = validate_interventional(scm_28, n_trials=20, seed=2)
    for key in ("scm_error", "naive_error", "improvement", "n_trials", "mean_z2_change"):
        assert key in result


def test_evaluate_regime_detection_separation(scm_28):
    result = evaluate_regime_detection(scm_28, window_size=50, n_windows_each=10, seed=7)
    assert result["separation_ratio"] > 10.0, (
        f"Regime separation too weak: {result['separation_ratio']:.1f}×"
    )


def test_evaluate_regime_detection_rho_estimate_error(scm_28):
    """Rho estimates must be accurate to within 5% of true values."""
    result = evaluate_regime_detection(scm_28, window_size=100, n_windows_each=10, seed=8)
    assert result["rho_estimate_error_28"] < 0.05 * 28.0
    assert result["rho_estimate_error_60"] < 0.05 * 60.0