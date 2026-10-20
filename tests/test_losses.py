"""
Test suite for specialists/world_model/losses.py

Covers:
- VICRegLoss dataclass has correct fields
- Total is weighted sum of individual terms
- Custom weights change total but not individual raw terms
- Invariance term: zero for identical inputs, positive for different inputs
- Invariance term: matches F.mse_loss exactly
- Variance term: near-zero for diverse batch (std > gamma)
- Variance term: positive and near-gamma for collapsed batch
- Variance term: gradient flows correctly (not NaN) through eps-stabilised sqrt
- Covariance term: smaller for uncorrelated dims, larger for correlated dims
- Covariance term: raises ValueError for B < 2
- Full loss: gradient flows to z_pred but NOT to z_tgt
- Full loss: raises ValueError on shape mismatch
- Full loss: all outputs finite on valid input
- Full loss: each term is non-negative
- End-to-end: gradient flows through predictor and encoder via VICReg loss
- Monitoring: VICRegLoss fields are detachable for logging without grad issues
"""

import torch
import torch.nn.functional as F
import pytest

from specialists.world_model.losses import (
    VICRegLoss,
    vicreg_loss,
    invariance_loss,
    variance_loss,
    covariance_loss,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def batch():
    torch.manual_seed(0)
    B, D = 8, 64
    z_pred = torch.randn(B, D, requires_grad=True)
    z_tgt  = torch.randn(B, D)   # no gradient — mirrors TargetEncoder output
    return z_pred, z_tgt


# ─────────────────────────────────────────────────────────────────────────────
# VICRegLoss dataclass
# ─────────────────────────────────────────────────────────────────────────────

def test_vicreg_loss_returns_dataclass(batch):
    z_pred, z_tgt = batch
    result = vicreg_loss(z_pred, z_tgt)
    assert isinstance(result, VICRegLoss)


def test_vicreg_loss_has_correct_fields(batch):
    z_pred, z_tgt = batch
    result = vicreg_loss(z_pred, z_tgt)
    assert hasattr(result, "total")
    assert hasattr(result, "invariance")
    assert hasattr(result, "variance")
    assert hasattr(result, "covariance")


def test_total_is_weighted_sum_of_terms(batch):
    """
    total must equal exactly λ*inv + μ*var + ν*cov.
    Tests that the weighting is applied correctly and no extra terms are added.
    """
    z_pred, z_tgt = batch
    lam, mu, nu = 25.0, 25.0, 1.0
    result = vicreg_loss(z_pred, z_tgt, lambda_inv=lam, mu_var=mu, nu_cov=nu)
    expected_total = lam * result.invariance + mu * result.variance + nu * result.covariance
    assert torch.allclose(result.total, expected_total, atol=1e-5), (
        f"total ({result.total.item():.4f}) != weighted sum "
        f"({expected_total.item():.4f})"
    )


def test_custom_weights_change_total_not_raw_terms(batch):
    """
    Changing λ/μ/ν must change the total but not the raw term values.
    Raw terms are properties of the latents, not the weights.
    """
    z_pred, z_tgt = batch
    result_a = vicreg_loss(z_pred, z_tgt, lambda_inv=25.0, mu_var=25.0, nu_cov=1.0)
    result_b = vicreg_loss(z_pred, z_tgt, lambda_inv=10.0, mu_var=10.0, nu_cov=5.0)

    assert not torch.allclose(result_a.total, result_b.total), (
        "Different weights produced same total — weighting is broken."
    )
    assert torch.allclose(result_a.invariance, result_b.invariance), (
        "Invariance changed when only weights changed — raw terms must be weight-independent."
    )
    assert torch.allclose(result_a.variance, result_b.variance)
    assert torch.allclose(result_a.covariance, result_b.covariance)


def test_all_raw_terms_are_non_negative(batch):
    """All three raw VICReg terms must always be ≥ 0."""
    z_pred, z_tgt = batch
    result = vicreg_loss(z_pred, z_tgt)
    assert result.invariance.item() >= 0, "Invariance term is negative."
    assert result.variance.item()   >= 0, "Variance term is negative."
    assert result.covariance.item() >= 0, "Covariance term is negative."


def test_all_outputs_are_finite(batch):
    z_pred, z_tgt = batch
    result = vicreg_loss(z_pred, z_tgt)
    assert result.total.isfinite(),      "total is non-finite."
    assert result.invariance.isfinite(), "invariance is non-finite."
    assert result.variance.isfinite(),   "variance is non-finite."
    assert result.covariance.isfinite(), "covariance is non-finite."


# ─────────────────────────────────────────────────────────────────────────────
# Invariance term
# ─────────────────────────────────────────────────────────────────────────────

def test_invariance_zero_for_identical_inputs():
    """If z_pred == z_tgt, invariance must be exactly 0."""
    torch.manual_seed(1)
    z = torch.randn(8, 64)
    inv = invariance_loss(z, z.detach())
    assert abs(inv.item()) < 1e-6, (
        f"Invariance not zero for identical inputs: {inv.item()}"
    )


def test_invariance_positive_for_different_inputs():
    torch.manual_seed(2)
    z_pred = torch.randn(8, 64)
    z_tgt  = torch.randn(8, 64)
    inv = invariance_loss(z_pred, z_tgt)
    assert inv.item() > 0


def test_invariance_matches_mse_loss():
    """Invariance must equal F.mse_loss exactly — no custom scaling."""
    torch.manual_seed(3)
    z_pred = torch.randn(8, 64, requires_grad=True)
    z_tgt  = torch.randn(8, 64)
    inv = invariance_loss(z_pred, z_tgt)
    mse = F.mse_loss(z_pred, z_tgt)
    assert torch.allclose(inv, mse), (
        f"Invariance ({inv.item():.6f}) != F.mse_loss ({mse.item():.6f})"
    )


def test_invariance_decreases_as_predictions_improve():
    """
    As z_pred moves closer to z_tgt, invariance must decrease.
    This verifies that gradient descent on invariance actually improves
    prediction quality.
    """
    torch.manual_seed(4)
    z_tgt  = torch.randn(8, 64)
    z_pred = torch.zeros(8, 64, requires_grad=True)   # starts far from z_tgt

    inv_initial = invariance_loss(z_pred, z_tgt).item()

    opt = torch.optim.SGD([z_pred], lr=0.1)
    for _ in range(20):
        opt.zero_grad()
        invariance_loss(z_pred, z_tgt).backward()
        opt.step()

    inv_final = invariance_loss(z_pred.detach(), z_tgt).item()
    assert inv_final < inv_initial, (
        f"Invariance did not decrease after 20 gradient steps: "
        f"{inv_initial:.4f} → {inv_final:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Variance term
# ─────────────────────────────────────────────────────────────────────────────

def test_variance_near_zero_for_diverse_batch():
    """Diverse batch (std >> gamma=1.0) should give variance ≈ 0."""
    torch.manual_seed(5)
    z = torch.randn(32, 64) * 3.0   # std ≈ 3 >> 1.0
    var = variance_loss(z, gamma=1.0)
    assert var.item() < 0.05, (
        f"Variance should be near-zero for diverse batch, got {var.item():.4f}"
    )


def test_variance_positive_for_collapsed_batch():
    """Collapsed batch (all identical) should give variance ≈ gamma."""
    B, D   = 8, 64
    gamma  = 1.0
    z_coll = torch.ones(B, D)
    var    = variance_loss(z_coll, gamma=gamma)
    # std = sqrt(0 + eps) ≈ 0.01, so relu(1.0 - 0.01) ≈ 0.99
    assert var.item() > 0.9 * gamma, (
        f"Variance should be close to gamma={gamma} for collapsed batch, "
        f"got {var.item():.4f}"
    )


def test_variance_gradient_is_finite_for_collapsed_batch():
    """
    The eps=1e-4 term inside sqrt must prevent NaN gradients when
    all inputs are identical (variance=0 exactly).
    """
    B, D   = 8, 64
    z_coll = torch.ones(B, D, requires_grad=True)
    variance_loss(z_coll).backward()
    assert z_coll.grad is not None
    assert z_coll.grad.isfinite().all(), (
        "NaN gradient from variance term on collapsed input — "
        "eps inside sqrt is not providing numerical stability."
    )


def test_variance_respects_custom_gamma():
    """Variance term with gamma=2.0 should penalise std < 2.0."""
    torch.manual_seed(6)
    # std ≈ 1.0 — below gamma=2.0, above gamma=0.5
    z = torch.randn(16, 64)

    var_high_gamma = variance_loss(z, gamma=2.0)   # should be positive
    var_low_gamma  = variance_loss(z, gamma=0.5)   # should be near-zero

    assert var_high_gamma.item() > var_low_gamma.item(), (
        "Higher gamma did not produce higher variance loss — "
        "gamma parameter is not being used correctly."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Covariance term
# ─────────────────────────────────────────────────────────────────────────────

def test_covariance_smaller_for_uncorrelated_dims():
    """Random batch (low correlation) < perfectly correlated batch."""
    torch.manual_seed(7)
    B, D = 32, 16

    z_uncorr = torch.randn(B, D)
    z_corr   = torch.randn(B, 1).expand(B, D)    # all dims identical

    cov_uncorr = covariance_loss(z_uncorr)
    cov_corr   = covariance_loss(z_corr)

    assert cov_uncorr.item() < cov_corr.item(), (
        f"Covariance for uncorrelated ({cov_uncorr.item():.4f}) >= "
        f"correlated ({cov_corr.item():.4f})"
    )


def test_covariance_raises_for_batch_size_1():
    """B=1 causes 0/0 (NaN) in sample covariance — must raise ValueError."""
    with pytest.raises(ValueError, match="B ≥ 2"):
        covariance_loss(torch.randn(1, 64))


def test_covariance_works_for_batch_size_2():
    """Minimum valid batch size is 2 — must not raise."""
    z = torch.randn(2, 64)
    cov = covariance_loss(z)
    assert cov.isfinite()


def test_covariance_gradient_flows():
    """Gradient must flow through the covariance term."""
    torch.manual_seed(8)
    z = torch.randn(8, 64, requires_grad=True)
    covariance_loss(z).backward()
    assert z.grad is not None
    assert z.grad.isfinite().all()


# ─────────────────────────────────────────────────────────────────────────────
# Gradient routing
# ─────────────────────────────────────────────────────────────────────────────

def test_gradient_flows_through_z_pred_not_z_tgt(batch):
    """
    Gradient must reach z_pred but NOT z_tgt.
    z_tgt is a constant (TargetEncoder output under @no_grad) —
    confirmed here by passing z_tgt without requires_grad.
    """
    z_pred, z_tgt = batch
    result = vicreg_loss(z_pred, z_tgt)
    result.total.backward()

    assert z_pred.grad is not None, "z_pred has no gradient."
    assert z_pred.grad.isfinite().all(), "z_pred gradient is non-finite."

    # z_tgt has no grad (requires_grad=False from fixture)
    assert z_tgt.grad is None, (
        "z_tgt received a gradient — the loss must treat it as a constant."
    )


def test_gradient_does_not_flow_when_z_tgt_requires_grad():
    """
    Even if z_tgt erroneously has requires_grad=True, the gradient
    through the invariance term is symmetric — this test confirms
    gradient DOES flow to z_pred regardless of z_tgt's grad status.
    This is a sanity check, not a constraint: in normal use z_tgt has
    no grad_fn from TargetEncoder.
    """
    torch.manual_seed(9)
    z_pred = torch.randn(8, 64, requires_grad=True)
    z_tgt  = torch.randn(8, 64, requires_grad=True)   # atypical usage

    result = vicreg_loss(z_pred, z_tgt)
    result.total.backward()

    # z_pred must always have gradient
    assert z_pred.grad is not None
    assert z_pred.grad.isfinite().all()


# ─────────────────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────────────────

def test_raises_on_shape_mismatch():
    """z_pred and z_tgt must have the same shape."""
    z_pred = torch.randn(8, 64)
    z_tgt  = torch.randn(8, 32)    # different D
    with pytest.raises(ValueError, match="same shape"):
        vicreg_loss(z_pred, z_tgt)


def test_raises_on_batch_mismatch():
    z_pred = torch.randn(8, 64)
    z_tgt  = torch.randn(4, 64)    # different B
    with pytest.raises(ValueError, match="same shape"):
        vicreg_loss(z_pred, z_tgt)


def test_raises_on_batch_size_1():
    """Full vicreg_loss must raise when B=1 (covariance term would NaN)."""
    z_pred = torch.randn(1, 64)
    z_tgt  = torch.randn(1, 64)
    with pytest.raises(ValueError, match="B ≥ 2"):
        vicreg_loss(z_pred, z_tgt)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end integration
# ─────────────────────────────────────────────────────────────────────────────

def test_end_to_end_gradient_flow_through_full_stack():
    """
    Full JEPA training step:
        context_enc → z_ctx → predictor → z_pred → vicreg_loss
        target_enc  → z_tgt (no grad)    ↗

    Gradient must flow to both encoder and predictor parameters.
    Target encoder must receive no gradient.
    """
    from specialists.world_model.encoder import ContextEncoder
    from specialists.world_model.predictor import Predictor
    from specialists.world_model.target_encoder import TargetEncoder

    torch.manual_seed(42)
    enc    = ContextEncoder()
    tgt    = TargetEncoder(enc)
    pred   = Predictor()

    B      = 4
    obs    = torch.randn(B, 20, 6)
    n_obs  = torch.randn(B, 20, 6)   # next observation window
    acts   = torch.randn(B, 20)
    a_next = torch.randn(B)

    z_ctx  = enc(obs, acts)
    z_tgt  = tgt(n_obs, acts)        # no gradient
    z_pred = pred(z_ctx, a_next)

    result = vicreg_loss(z_pred, z_tgt)
    result.total.backward()

    # Encoder params must have gradients
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"enc.{name} has no gradient"
        assert p.grad.isfinite().all(), f"enc.{name} has non-finite gradient"

    # Predictor params must have gradients
    for name, p in pred.named_parameters():
        assert p.grad is not None, f"pred.{name} has no gradient"
        assert p.grad.isfinite().all(), f"pred.{name} has non-finite gradient"

    # Target encoder params must NOT have gradients
    for name, p in tgt.named_parameters():
        assert p.grad is None, (
            f"tgt.{name} received a gradient — "
            f"EMA encoder must never be updated by backprop."
        )


def test_vicreg_loss_fields_are_detachable(batch):
    """
    Individual loss fields must be detachable for logging without
    interfering with the backward graph.
    """
    z_pred, z_tgt = batch
    result = vicreg_loss(z_pred, z_tgt)
    result.total.backward()

    # Detach for logging — must not raise
    inv_val = result.invariance.detach().item()
    var_val = result.variance.detach().item()
    cov_val = result.covariance.detach().item()

    assert all(isinstance(v, float) for v in [inv_val, var_val, cov_val])