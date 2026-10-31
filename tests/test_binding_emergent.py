"""
Test suite for integration/binding_emergent.py

Covers:
- EmergentBindingModule: output shape, causal features used, gradient flows
- EmergentBindingModule: hidden_activations returns correct shape
- EmergentBindingModule: 1D and 2D action inputs both accepted
- EmergentBindingModule: no LayerNorm in architecture (confirms unstructured)
- train_emergent: runs, loss decreases, enc frozen
- analyse_emergent_structure: returns expected keys, values in valid ranges
- evaluate_emergent_poc: returns expected keys, metrics finite and non-negative
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from integration.binding_emergent import (
    EmergentBindingModule,
    analyse_emergent_structure,
    evaluate_emergent_poc,
    train_emergent,
)
from integration.binding_v0_1_conditioning import BindingConfig, N_CAUSAL
from specialists.causal_module.discovery import CausalDiscovery
from specialists.causal_module.scm import StructuralCausalModel
from specialists.world_model.encoder import ContextEncoder
from specialists.world_model.predictor import Predictor
from specialists.world_model.target_encoder import TargetEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

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
def tiny_b_cfg():
    return BindingConfig(
        n_traj_per_rho=2, n_steps=30, seq_len=10,
        n_eval_traj=2, n_pre_shift=15, n_post_shift=15,
        n_epochs=3, batch_size=16, scm_n_traj=5,
        data_seed=9, model_seed=0,
    )


@pytest.fixture(scope="module")
def tiny_ebm():
    return EmergentBindingModule(
        latent_dim=32, hidden_dim=64, n_layers=2, n_causal=N_CAUSAL
    )


@pytest.fixture(scope="module")
def scm():
    disc  = CausalDiscovery()
    graph = disc.fit_from_system(rho=28.0, n_traj=5, seed=0)
    return StructuralCausalModel(graph)


@pytest.fixture(scope="module")
def trained_ebm(tiny_enc, tiny_tgt, tiny_b_cfg):
    device = torch.device("cpu")
    ebm, history = train_emergent(
        tiny_enc, tiny_tgt, tiny_b_cfg, device,
        n_epochs=3, verbose=False,
    )
    return ebm, history


# ─────────────────────────────────────────────────────────────────────────────
# EmergentBindingModule — structural tests
# ─────────────────────────────────────────────────────────────────────────────

def test_output_shape(tiny_ebm):
    tiny_ebm.eval()
    z_ctx    = torch.randn(4, 32)
    a_next   = torch.randn(4)
    causal_f = torch.randn(4, N_CAUSAL)
    with torch.no_grad():
        out = tiny_ebm(z_ctx, a_next, causal_f)
    assert out.shape == (4, 32)


def test_1d_and_2d_action_give_same_output(tiny_ebm):
    tiny_ebm.eval()
    z_ctx    = torch.randn(3, 32)
    a_next   = torch.randn(3)
    causal_f = torch.randn(3, N_CAUSAL)
    with torch.no_grad():
        o1 = tiny_ebm(z_ctx, a_next, causal_f)
        o2 = tiny_ebm(z_ctx, a_next.unsqueeze(-1), causal_f)
    assert torch.allclose(o1, o2)


def test_causal_features_affect_output(tiny_ebm):
    """Different causal features must produce different output."""
    tiny_ebm.eval()
    torch.manual_seed(0)
    z_ctx = torch.randn(2, 32)
    a     = torch.randn(2)
    cf1   = torch.tensor([[0.026]] * 2)
    cf2   = torch.tensor([[0.941]] * 2)
    with torch.no_grad():
        o1 = tiny_ebm(z_ctx, a, cf1)
        o2 = tiny_ebm(z_ctx, a, cf2)
    assert not torch.allclose(o1, o2), (
        "Same z_ctx + action but different causal features produced identical "
        "output — causal feature not being used."
    )


def test_output_finite(tiny_ebm):
    tiny_ebm.eval()
    z_ctx    = torch.randn(4, 32) * 10
    a_next   = torch.randn(4)
    causal_f = torch.tensor([[0.5]] * 4)
    with torch.no_grad():
        out = tiny_ebm(z_ctx, a_next, causal_f)
    assert out.isfinite().all()


def test_gradients_flow(tiny_ebm):
    ebm = EmergentBindingModule(latent_dim=32, hidden_dim=64, n_layers=2, n_causal=N_CAUSAL)
    z_ctx    = torch.randn(4, 32)
    a_next   = torch.randn(4)
    causal_f = torch.randn(4, N_CAUSAL)
    out = ebm(z_ctx, a_next, causal_f)
    out.sum().backward()
    for name, p in ebm.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert p.grad.isfinite().all()


def test_no_layernorm_in_architecture():
    """
    The EmergentBindingModule must not use LayerNorm.
    LayerNorm imposes normalisation structure that could bias emergence.
    The module must be architecturally blank — only Linear + GELU.
    """
    ebm = EmergentBindingModule(latent_dim=32, hidden_dim=128, n_layers=3)
    for name, module in ebm.named_modules():
        assert not isinstance(module, nn.LayerNorm), (
            f"Found LayerNorm at {name} — EmergentBindingModule must not "
            f"use LayerNorm (it prescribes normalisation structure)."
        )


def test_no_attention_in_architecture():
    """The module must not use MultiheadAttention (that's v0.2)."""
    ebm = EmergentBindingModule(latent_dim=32, hidden_dim=128, n_layers=3)
    for name, module in ebm.named_modules():
        assert not isinstance(module, nn.MultiheadAttention), (
            f"Found MultiheadAttention at {name} — emergent module must not "
            f"use prescribed attention mechanisms."
        )


def test_hidden_activations_shape(tiny_ebm):
    """hidden_activations must return penultimate hidden layer output."""
    tiny_ebm.eval()
    z_ctx    = torch.randn(4, 32)
    a_next   = torch.randn(4)
    causal_f = torch.randn(4, N_CAUSAL)
    with torch.no_grad():
        h = tiny_ebm.hidden_activations(z_ctx, a_next, causal_f)
    # Shape: (B, hidden_dim)
    assert h.shape[0] == 4
    assert h.shape[1] == 64   # tiny_ebm has hidden_dim=64


def test_jacobian_causal_is_finite(tiny_ebm):
    """Jacobian d(output)/d(causal_feature) must be finite."""
    z_ctx = torch.randn(1, 32)
    a     = torch.randn(1)
    cf    = torch.randn(1, N_CAUSAL).requires_grad_(True)
    out   = tiny_ebm(z_ctx, a, cf)
    grad  = torch.autograd.grad(out.sum(), cf)[0]
    assert grad.isfinite().all()


def test_params_are_learnable(tiny_ebm):
    for name, p in tiny_ebm.named_parameters():
        assert p.requires_grad, f"{name} has requires_grad=False"


# ─────────────────────────────────────────────────────────────────────────────
# train_emergent tests
# ─────────────────────────────────────────────────────────────────────────────

def test_train_emergent_runs(trained_ebm):
    ebm, history = trained_ebm
    assert ebm is not None
    assert len(history) == 3


def test_train_emergent_loss_finite(trained_ebm):
    _, history = trained_ebm
    for h in history:
        assert np.isfinite(h["loss_total"])


def test_train_emergent_loss_decreases(trained_ebm):
    _, history = trained_ebm
    assert history[-1]["loss_total"] < history[0]["loss_total"], (
        f"Loss did not decrease: {history[0]['loss_total']:.4f} → "
        f"{history[-1]['loss_total']:.4f}"
    )


def test_train_emergent_enc_frozen(tiny_enc, trained_ebm):
    for p in tiny_enc.parameters():
        assert not p.requires_grad


# ─────────────────────────────────────────────────────────────────────────────
# analyse_emergent_structure tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def structure_results(trained_ebm, tiny_enc, tiny_b_cfg, scm):
    ebm, _ = trained_ebm
    return analyse_emergent_structure(
        ebm, tiny_enc, tiny_b_cfg, scm,
        torch.device("cpu"), n_probe_samples=30,
    )


def test_structure_returns_expected_keys(structure_results):
    expected = {
        "jacobian_pre_mean", "jacobian_cross_mean", "jacobian_post_mean",
        "jacobian_cross_rise",
        "probe_accuracy", "probe_chance", "probe_above_chance",
        "activation_pre_mean", "activation_cross_mean", "activation_post_mean",
        "adaptive_influence", "emergent_differentiation",
    }
    assert set(structure_results.keys()) == expected


def test_structure_values_finite(structure_results):
    for key in ("jacobian_pre_mean", "jacobian_cross_mean",
                "probe_accuracy", "activation_pre_mean"):
        assert np.isfinite(structure_results[key]), f"{key} not finite"


def test_jacobian_norms_positive(structure_results):
    """Jacobian norms must be positive (causal signal has some influence)."""
    assert structure_results["jacobian_pre_mean"]   > 0
    assert structure_results["jacobian_cross_mean"] > 0


def test_probe_accuracy_in_valid_range(structure_results):
    acc = structure_results["probe_accuracy"]
    assert 0.0 <= acc <= 1.0, f"Probe accuracy {acc} out of [0,1]"


def test_probe_chance_is_one_third(structure_results):
    assert abs(structure_results["probe_chance"] - 1/3) < 1e-6


def test_activation_norms_positive(structure_results):
    assert structure_results["activation_pre_mean"]   > 0
    assert structure_results["activation_cross_mean"] > 0
    assert structure_results["activation_post_mean"]  > 0


def test_boolean_flags_are_bool(structure_results):
    assert isinstance(structure_results["adaptive_influence"],     bool)
    assert isinstance(structure_results["emergent_differentiation"], bool)
    assert isinstance(structure_results["probe_above_chance"],     bool)


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_emergent_poc tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def poc_results(tiny_enc, tiny_tgt, tiny_sa_pred, trained_ebm, tiny_b_cfg, scm):
    ebm, _ = trained_ebm
    return evaluate_emergent_poc(
        tiny_enc, tiny_tgt, tiny_sa_pred, ebm, scm, tiny_b_cfg,
        torch.device("cpu"),
    )


def test_poc_returns_expected_keys(poc_results):
    expected = {
        "standalone_pre_error", "emergent_pre_error",
        "standalone_crossing_error", "emergent_crossing_error",
        "standalone_norm_ratio", "emergent_norm_ratio",
        "abs_improvement", "n_crossing_windows",
    }
    assert set(poc_results.keys()) == expected


def test_poc_metrics_finite(poc_results):
    for key in ("standalone_pre_error", "emergent_pre_error",
                "standalone_crossing_error", "emergent_crossing_error",
                "standalone_norm_ratio", "emergent_norm_ratio"):
        assert np.isfinite(poc_results[key]), f"{key} not finite"


def test_poc_errors_non_negative(poc_results):
    for key in ("standalone_pre_error", "emergent_pre_error",
                "standalone_crossing_error", "emergent_crossing_error"):
        assert poc_results[key] >= 0.0


def test_poc_has_crossing_windows(poc_results):
    assert poc_results["n_crossing_windows"] > 0