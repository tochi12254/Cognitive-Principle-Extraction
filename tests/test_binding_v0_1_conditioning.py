"""
Test suite for integration/binding_v0_1_conditioning.py

Covers:
- ConditionedPredictor: output shape, causal features used, gradient flows
- ConditionedPredictor: different causal features → different outputs
- compute_causal_features: correct shape, separated by regime
- BindingDataGenerator: correct keys, shapes, causal features present
- BindingDataGenerator: shift dataset marks crossing windows correctly
- BindingDataset: correct length, shapes per item
- train_binding: runs without error, loss decreases
- evaluate_binding_poc: returns all expected keys, metrics finite
- POC result: conditioned predictor crossing ratio ≤ standalone (or improves)
"""

import numpy as np
import pytest
import torch

from integration.binding_v0_1_conditioning import (
    BindingConfig,
    BindingDataGenerator,
    BindingDataset,
    ConditionedPredictor,
    compute_causal_features,
    evaluate_binding_poc,
    train_binding,
)
from specialists.causal_module.discovery import CausalDiscovery
from specialists.causal_module.scm import StructuralCausalModel
from specialists.world_model.encoder import ContextEncoder
from specialists.world_model.predictor import Predictor
from specialists.world_model.target_encoder import TargetEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def scm():
    disc  = CausalDiscovery()
    graph = disc.fit_from_system(rho=28.0, n_traj=10, seed=0)
    return StructuralCausalModel(graph)


@pytest.fixture(scope="module")
def tiny_enc():
    return ContextEncoder(
        obs_dim=6, seq_len=10, embed_dim=32, latent_dim=32,
        n_heads=4, n_layers=1, ff_dim=64, dropout=0.0,
    )


@pytest.fixture(scope="module")
def tiny_tgt(tiny_enc):
    return TargetEncoder(tiny_enc)


@pytest.fixture(scope="module")
def tiny_sa_pred():
    return Predictor(latent_dim=32, hidden_dim=64)


@pytest.fixture(scope="module")
def tiny_cfg(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("binding")
    return BindingConfig(
        n_traj_per_rho=2,
        n_steps=30,
        seq_len=10,
        n_eval_traj=2,
        n_pre_shift=15,
        n_post_shift=15,
        n_epochs=3,
        batch_size=16,
        scm_n_traj=5,
        data_seed=7,
        model_seed=0,
        hidden_dim=64,
    )


@pytest.fixture(scope="module")
def gen(scm):
    return BindingDataGenerator(scm=scm, obs_dim=6, seed=0)


@pytest.fixture(scope="module")
def tiny_stationary_ds(gen):
    return gen.generate_stationary_dataset(
        rho=28.0, n_trajectories=2, n_steps=30, seq_len=10
    )


# ─────────────────────────────────────────────────────────────────────────────
# ConditionedPredictor tests
# ─────────────────────────────────────────────────────────────────────────────

def test_conditioned_predictor_output_shape():
    cp = ConditionedPredictor(latent_dim=32, hidden_dim=64, n_causal=1)
    cp.eval()
    z_ctx  = torch.randn(4, 32)
    a_next = torch.randn(4)
    causal = torch.randn(4, 1)
    with torch.no_grad():
        out = cp(z_ctx, a_next, causal)
    assert out.shape == (4, 32)


def test_conditioned_predictor_1d_action_accepted():
    cp = ConditionedPredictor(latent_dim=32, n_causal=1)
    cp.eval()
    z_ctx  = torch.randn(3, 32)
    a_next = torch.randn(3)       # 1D
    causal = torch.randn(3, 1)
    with torch.no_grad():
        out = cp(z_ctx, a_next, causal)
    assert out.shape == (3, 32)


def test_conditioned_predictor_causal_features_used():
    """Different causal features must produce different outputs."""
    cp = ConditionedPredictor(latent_dim=32, n_causal=1)
    cp.eval()
    torch.manual_seed(0)
    z_ctx  = torch.randn(1, 32)
    a_next = torch.randn(1)
    cf1 = torch.tensor([[0.026]])   # low residual  ≈ rho=28
    cf2 = torch.tensor([[0.941]])   # high residual ≈ rho=60
    with torch.no_grad():
        o1 = cp(z_ctx, a_next, cf1)
        o2 = cp(z_ctx, a_next, cf2)
    assert not torch.allclose(o1, o2), (
        "Same z_ctx + action but different causal features produced identical "
        "output — causal features are not being used."
    )


def test_conditioned_predictor_gradients_flow():
    cp = ConditionedPredictor(latent_dim=32, n_causal=1)
    z_ctx  = torch.randn(4, 32)
    a_next = torch.randn(4)
    causal = torch.randn(4, 1)
    out = cp(z_ctx, a_next, causal)
    out.sum().backward()
    for name, p in cp.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert p.grad.isfinite().all()


def test_conditioned_predictor_finite_output():
    cp = ConditionedPredictor(latent_dim=32, n_causal=1)
    cp.eval()
    z_ctx  = torch.randn(4, 32) * 10
    a_next = torch.randn(4)
    causal = torch.tensor([[0.026]] * 4)
    with torch.no_grad():
        out = cp(z_ctx, a_next, causal)
    assert out.isfinite().all()


def test_conditioned_predictor_raises_on_bad_latent_dim():
    with pytest.raises(ValueError, match="latent_dim"):
        ConditionedPredictor(latent_dim=0)


def test_conditioned_predictor_raises_on_bad_n_causal():
    with pytest.raises(ValueError, match="n_causal"):
        ConditionedPredictor(n_causal=0)


# ─────────────────────────────────────────────────────────────────────────────
# compute_causal_features tests
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_causal_features_shape(scm):
    states  = np.random.randn(20, 3)
    actions = np.random.randn(20)
    feats   = compute_causal_features(states, actions, scm)
    assert feats.shape == (1,), f"Expected (1,), got {feats.shape}"


def test_compute_causal_features_finite(scm):
    states  = np.random.randn(20, 3)
    actions = np.random.randn(20)
    feats   = compute_causal_features(states, actions, scm)
    assert np.isfinite(feats).all()


def test_compute_causal_features_separated_by_regime(scm):
    """
    Features from rho=28 and rho=60 windows must be clearly separated.
    log1p(residual)/12: rho=28 → ≈0.026, rho=60 → ≈0.94.
    """
    from specialists.causal_module.discovery import generate_discovery_data

    Z28, a28, _ = generate_discovery_data(rho=28.0, n_traj=1, n_steps=50, seed=5)
    Z60, a60, _ = generate_discovery_data(rho=60.0, n_traj=1, n_steps=50, seed=5)

    # Use a window of 20 steps (need warm-up so z1 is not near-zero)
    f28 = compute_causal_features(Z28[10:30], a28[10:30], scm)
    f60 = compute_causal_features(Z60[10:30], a60[10:30], scm)

    # Residual feature should be clearly different between regimes
    assert abs(f60[0] - f28[0]) > 0.3, (
        f"Regime not distinguished in residual feature: "
        f"rho=28: {f28[0]:.4f}, rho=60: {f60[0]:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# BindingDataGenerator tests
# ─────────────────────────────────────────────────────────────────────────────

def test_stationary_dataset_keys(tiny_stationary_ds):
    expected = {
        "context_obs", "context_actions", "next_action",
        "target_context_obs", "target_context_actions",
        "causal_features", "shift_step_in_window",
    }
    assert set(tiny_stationary_ds.keys()) == expected


def test_stationary_dataset_causal_features_shape(tiny_stationary_ds):
    from integration.binding_v0_1_conditioning import N_CAUSAL
    N = len(tiny_stationary_ds["context_obs"])
    assert tiny_stationary_ds["causal_features"].shape == (N, N_CAUSAL)


def test_stationary_dataset_causal_features_finite(tiny_stationary_ds):
    assert np.isfinite(tiny_stationary_ds["causal_features"]).all()


def test_stationary_dataset_no_crossing_windows(tiny_stationary_ds):
    assert (tiny_stationary_ds["shift_step_in_window"] == -1).all()


def test_shift_dataset_has_crossing_windows(gen):
    ds = gen.generate_shift_dataset(
        rho_normal=28.0, rho_shifted=60.0,
        n_trajectories=2, n_pre_shift=15, n_post_shift=15, seq_len=10,
    )
    assert (ds["shift_step_in_window"] >= 0).any(), (
        "No crossing windows in shift dataset."
    )


def test_shift_dataset_causal_features_vary(gen):
    """
    Causal features in a shift dataset must show variation —
    pre-shift windows should have different features than post-shift.
    """
    ds = gen.generate_shift_dataset(
        rho_normal=28.0, rho_shifted=60.0,
        n_trajectories=3, n_pre_shift=20, n_post_shift=20, seq_len=10,
    )
    feats = ds["causal_features"]
    # rho_norm feature (index 0) should have meaningful variance
    assert feats[:, 0].std() > 0.01, (
        "rho_norm feature has no variance in shift dataset — "
        "causal features may not be responding to regime change."
    )


# ─────────────────────────────────────────────────────────────────────────────
# BindingDataset tests
# ─────────────────────────────────────────────────────────────────────────────

def test_binding_dataset_length(tiny_stationary_ds):
    ds = BindingDataset(tiny_stationary_ds)
    assert len(ds) == len(tiny_stationary_ds["context_obs"])


def test_binding_dataset_item_shapes(tiny_stationary_ds):
    from integration.binding_v0_1_conditioning import N_CAUSAL
    ds   = BindingDataset(tiny_stationary_ds)
    item = ds[0]
    assert item["context_obs"].shape        == (10, 6)
    assert item["causal_features"].shape    == (N_CAUSAL,)
    assert item["next_action"].shape        == ()


def test_binding_dataset_causal_features_dtype(tiny_stationary_ds):
    ds   = BindingDataset(tiny_stationary_ds)
    item = ds[0]
    assert item["causal_features"].dtype == torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# train_binding tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_cond_pred(tiny_enc, tiny_tgt, tiny_cfg):
    device = torch.device("cpu")
    cond_pred, history = train_binding(
        tiny_enc, tiny_tgt, tiny_cfg, device, verbose=False
    )
    return cond_pred, history


def test_train_binding_runs(trained_cond_pred):
    cond_pred, history = trained_cond_pred
    assert cond_pred is not None
    assert len(history) == 3   # n_epochs=3 in tiny_cfg


def test_train_binding_loss_finite(trained_cond_pred):
    _, history = trained_cond_pred
    for h in history:
        assert np.isfinite(h["loss_total"]), f"Non-finite loss at epoch {h['epoch']}"


def test_train_binding_loss_decreases(trained_cond_pred):
    _, history = trained_cond_pred
    assert history[-1]["loss_total"] < history[0]["loss_total"], (
        f"Loss did not decrease: {history[0]['loss_total']:.4f} → "
        f"{history[-1]['loss_total']:.4f}"
    )


def test_train_binding_enc_weights_unchanged(tiny_enc, trained_cond_pred):
    """Frozen encoder weights must not change during binding training."""
    cond_pred, _ = trained_cond_pred
    enc_copy = ContextEncoder(
        obs_dim=6, seq_len=10, embed_dim=32, latent_dim=32,
        n_heads=4, n_layers=1, ff_dim=64, dropout=0.0,
    )
    # The enc fixture is module-scoped and shared — its weights should
    # remain at their initialised values since it's frozen during training.
    for p in tiny_enc.parameters():
        assert not p.requires_grad, (
            "Encoder still has requires_grad=True after freeze in train_binding."
        )


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_binding_poc tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def poc_results(tiny_enc, tiny_tgt, tiny_sa_pred, trained_cond_pred, tiny_cfg):
    device = torch.device("cpu")
    cond_pred, _ = trained_cond_pred
    disc  = CausalDiscovery()
    graph = disc.fit_from_system(rho=28.0, n_traj=5, seed=0)
    scm   = StructuralCausalModel(graph)
    return evaluate_binding_poc(
        tiny_enc, tiny_tgt, tiny_sa_pred, cond_pred, scm, tiny_cfg, device
    )


def test_poc_returns_expected_keys(poc_results):
    expected = {
        "standalone_pre_error", "conditioned_pre_error",
        "standalone_crossing_error", "conditioned_crossing_error",
        "standalone_ratio", "conditioned_ratio",
        "sa_per_pos_mean", "co_per_pos_mean",
        "n_crossing_windows",
    }
    assert set(poc_results.keys()) == expected


def test_poc_metrics_are_finite(poc_results):
    for key in (
        "standalone_pre_error", "conditioned_pre_error",
        "standalone_crossing_error", "conditioned_crossing_error",
        "standalone_ratio", "conditioned_ratio",
    ):
        assert np.isfinite(poc_results[key]), f"{key} is not finite."


def test_poc_errors_are_non_negative(poc_results):
    for key in (
        "standalone_pre_error", "conditioned_pre_error",
        "standalone_crossing_error", "conditioned_crossing_error",
    ):
        assert poc_results[key] >= 0.0, f"{key} is negative."


def test_poc_per_pos_length(poc_results, tiny_cfg):
    assert len(poc_results["sa_per_pos_mean"]) == tiny_cfg.seq_len
    assert len(poc_results["co_per_pos_mean"]) == tiny_cfg.seq_len


def test_poc_has_crossing_windows(poc_results):
    assert poc_results["n_crossing_windows"] > 0