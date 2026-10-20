"""
Test suite for specialists/world_model/predictor.py

Covers:
- Output shape for 1D and 2D action inputs
- Both action shapes produce identical output (normalisation)
- Action is actually used (same z_ctx, different actions → different predictions)
- Context latent is used (different z_ctx, same action → different predictions)
- Output is finite (no NaN / Inf) on valid input
- Gradient flows to all parameters
- All parameters are learnable
- Parameters update after a gradient step
- Batch consistency (batch vs individual gives same result)
- Predictor parameter count is smaller than encoder (asymmetry check)
- Deterministic in eval mode
- Dropout active in train mode (when dropout > 0)
- Invalid constructor arguments raise ValueError
- Chaining: full JEPA forward pass (encoder → predictor) end-to-end
"""

import torch
import pytest

from specialists.world_model.encoder import ContextEncoder
from specialists.world_model.predictor import Predictor


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def pred():
    torch.manual_seed(0)
    p = Predictor(latent_dim=64, hidden_dim=128, dropout=0.0)
    p.eval()
    return p


@pytest.fixture
def batch():
    torch.manual_seed(1)
    B = 4
    return torch.randn(B, 64), torch.randn(B)   # z_ctx, a_next


# ─────────────────────────────────────────────────────────────────────────────
# Shape tests
# ─────────────────────────────────────────────────────────────────────────────

def test_output_shape_1d_action(pred, batch):
    z_ctx, a_next = batch
    out = pred(z_ctx, a_next)
    assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"


def test_output_shape_2d_action(pred, batch):
    z_ctx, a_next = batch
    out = pred(z_ctx, a_next.unsqueeze(-1))
    assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"


def test_1d_and_2d_action_give_same_output(pred, batch):
    """
    Predictor must accept both (B,) and (B, 1) action shapes and produce
    identical output. Callers should not have to reshape manually.
    """
    z_ctx, a_next = batch
    with torch.no_grad():
        out_1d = pred(z_ctx, a_next)
        out_2d = pred(z_ctx, a_next.unsqueeze(-1))
    assert torch.allclose(out_1d, out_2d), (
        "1D and 2D action inputs produced different outputs — "
        "shape normalisation in forward() is broken."
    )


def test_output_shape_single_sample(pred):
    z_ctx  = torch.randn(1, 64)
    a_next = torch.randn(1)
    out    = pred(z_ctx, a_next)
    assert out.shape == (1, 64)


# ─────────────────────────────────────────────────────────────────────────────
# Input sensitivity — proves both inputs are used
# ─────────────────────────────────────────────────────────────────────────────

def test_action_is_used(pred, batch):
    """
    Same context latent, different actions must produce different predictions.
    If the predictor ignores the action, it cannot satisfy the action-conditioned
    prediction role — the core reason g_φ exists.
    """
    torch.manual_seed(2)
    z_ctx = torch.randn(1, 64)
    a1    = torch.randn(1)
    a2    = torch.randn(1)

    with torch.no_grad():
        p1 = pred(z_ctx, a1)
        p2 = pred(z_ctx, a2)

    assert not torch.allclose(p1, p2), (
        "Predictor produced same output for different actions on the same "
        "context latent — action input is being ignored."
    )


def test_context_latent_is_used(pred, batch):
    """Different context latents with the same action must give different outputs."""
    torch.manual_seed(3)
    z1     = torch.randn(1, 64)
    z2     = torch.randn(1, 64)
    a_next = torch.randn(1)

    with torch.no_grad():
        p1 = pred(z1, a_next)
        p2 = pred(z2, a_next)

    assert not torch.allclose(p1, p2)


# ─────────────────────────────────────────────────────────────────────────────
# Output validity
# ─────────────────────────────────────────────────────────────────────────────

def test_output_is_finite(pred, batch):
    z_ctx, a_next = batch
    with torch.no_grad():
        out = pred(z_ctx, a_next)
    assert out.isfinite().all(), "Predictor output contains NaN or Inf."


def test_output_finite_for_large_input(pred):
    """Output must be finite even with extreme input magnitudes."""
    z_ctx  = torch.randn(2, 64) * 500
    a_next = torch.randn(2) * 500
    with torch.no_grad():
        out = pred(z_ctx, a_next)
    assert out.isfinite().all(), (
        "Non-finite output for large inputs — "
        "LayerNorm should have stabilised this."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gradient and parameter tests
# ─────────────────────────────────────────────────────────────────────────────

def test_all_parameters_receive_finite_gradients(batch):
    pred = Predictor()
    pred.train()
    z_ctx, a_next = batch

    out = pred(z_ctx, a_next)
    out.sum().backward()

    for name, param in pred.named_parameters():
        assert param.grad is not None, f"No gradient for: {name}"
        assert param.grad.isfinite().all(), f"Non-finite gradient for: {name}"


def test_all_parameters_are_learnable(pred):
    for name, param in pred.named_parameters():
        assert param.requires_grad, f"{name} has requires_grad=False"


def test_parameters_update_after_gradient_step(batch):
    pred = Predictor()
    pred.train()
    z_ctx, a_next = batch

    params_before = {
        name: p.clone().detach() for name, p in pred.named_parameters()
    }

    opt = torch.optim.Adam(pred.parameters(), lr=1e-3)
    pred(z_ctx, a_next).sum().backward()
    opt.step()

    changed = any(
        not torch.allclose(p, params_before[name])
        for name, p in pred.named_parameters()
    )
    assert changed, "No parameters changed after a gradient step."


# ─────────────────────────────────────────────────────────────────────────────
# Batch consistency
# ─────────────────────────────────────────────────────────────────────────────

def test_batch_vs_individual_consistency(pred):
    """
    Encoding samples in a batch must give identical results to encoding
    them individually. Verifies no cross-sample contamination.
    """
    torch.manual_seed(9)
    z_ctx  = torch.randn(3, 64)
    a_next = torch.randn(3)

    with torch.no_grad():
        batch_out = pred(z_ctx, a_next)
        individual_out = torch.stack([
            pred(z_ctx[i].unsqueeze(0), a_next[i].unsqueeze(0)).squeeze(0)
            for i in range(3)
        ])

    assert torch.allclose(batch_out, individual_out, atol=1e-5), (
        "Batch and individual outputs differ — cross-sample contamination."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Asymmetry check — predictor must be simpler than encoder
# ─────────────────────────────────────────────────────────────────────────────

def test_predictor_has_fewer_params_than_encoder():
    """
    Predictor parameter count must be strictly less than context encoder.
    This asymmetry is load-bearing for JEPA training stability — a predictor
    as complex as the encoder makes the trivial collapse solution too easy.
    """
    pred    = Predictor(latent_dim=64, hidden_dim=128)
    enc     = ContextEncoder(obs_dim=6, seq_len=20, embed_dim=64, latent_dim=64)
    n_pred  = sum(p.numel() for p in pred.parameters())
    n_enc   = sum(p.numel() for p in enc.parameters())
    assert n_pred < n_enc, (
        f"Predictor ({n_pred:,} params) >= encoder ({n_enc:,} params). "
        f"Predictor must be simpler than the encoder."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────

def test_deterministic_in_eval_mode(pred, batch):
    z_ctx, a_next = batch
    with torch.no_grad():
        results = [pred(z_ctx, a_next) for _ in range(5)]
    for r in results[1:]:
        assert torch.allclose(results[0], r), (
            "Non-deterministic output in eval mode — dropout may be active."
        )


def test_dropout_active_in_train_mode(batch):
    """
    With high dropout, repeated forward passes in train mode should
    (with very high probability) produce different outputs.
    """
    pred = Predictor(dropout=0.5)
    pred.train()
    z_ctx, a_next = batch
    results = [pred(z_ctx, a_next) for _ in range(5)]
    any_different = any(
        not torch.allclose(results[0], results[i])
        for i in range(1, 5)
    )
    assert any_different, (
        "All forward passes identical in train mode with dropout=0.5 — "
        "dropout may not be active."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────────────────

def test_raises_on_latent_dim_zero():
    with pytest.raises(ValueError, match="latent_dim"):
        Predictor(latent_dim=0)


def test_raises_on_hidden_dim_zero():
    with pytest.raises(ValueError, match="hidden_dim"):
        Predictor(hidden_dim=0)


def test_raises_on_dropout_above_1():
    with pytest.raises(ValueError, match="dropout"):
        Predictor(dropout=1.5)


def test_raises_on_dropout_below_0():
    with pytest.raises(ValueError, match="dropout"):
        Predictor(dropout=-0.1)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end chain: encoder → predictor
# ─────────────────────────────────────────────────────────────────────────────

def test_encoder_to_predictor_chain():
    """
    Full JEPA forward: ContextEncoder → Predictor.
    Verifies the two modules connect cleanly at their interface
    (latent_dim must match between encoder output and predictor input).
    """
    torch.manual_seed(42)
    enc  = ContextEncoder(obs_dim=6, seq_len=20, embed_dim=64, latent_dim=64)
    pred = Predictor(latent_dim=64, hidden_dim=128)

    B       = 4
    obs     = torch.randn(B, 20, 6)
    actions = torch.randn(B, 20)
    a_next  = torch.randn(B)

    z_ctx  = enc(obs, actions)           # (B, 64)
    z_pred = pred(z_ctx, a_next)         # (B, 64)

    assert z_pred.shape == (B, 64)
    assert z_pred.isfinite().all()

    # Gradient flows all the way back through both modules
    z_pred.sum().backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"encoder param {name} has no gradient"
    for name, p in pred.named_parameters():
        assert p.grad is not None, f"predictor param {name} has no gradient"


def test_chain_gradient_does_not_flow_through_target_encoder():
    """
    In the real JEPA training setup:
        z_ctx  = context_enc(obs, actions)     # has gradient
        z_tgt  = target_enc(next_obs, actions) # no gradient (EMA, no_grad)
        z_pred = predictor(z_ctx, a_next)      # has gradient
        loss   = (z_pred - z_tgt).pow(2).sum()

    Gradient must flow through predictor and context_enc, but NOT
    through target_enc. This test confirms the full chain.
    """
    from specialists.world_model.target_encoder import TargetEncoder

    torch.manual_seed(99)
    context_enc = ContextEncoder()
    target_enc  = TargetEncoder(context_enc)
    pred        = Predictor()

    B         = 2
    obs       = torch.randn(B, 20, 6)
    next_obs  = torch.randn(B, 20, 6)
    actions   = torch.randn(B, 20)
    a_next    = torch.randn(B)

    z_ctx  = context_enc(obs, actions)
    z_tgt  = target_enc(next_obs, actions)     # no grad
    z_pred = pred(z_ctx, a_next)

    loss = (z_pred - z_tgt).pow(2).sum()
    loss.backward()

    # context_enc and predictor must have gradients
    for name, p in context_enc.named_parameters():
        assert p.grad is not None, f"context_enc.{name} has no gradient"
    for name, p in pred.named_parameters():
        assert p.grad is not None, f"predictor.{name} has no gradient"

    # target_enc must NOT have gradients
    for name, p in target_enc.named_parameters():
        assert p.grad is None, (
            f"Gradient leaked into target_enc.{name} — "
            f"EMA encoder must never receive gradients."
        )