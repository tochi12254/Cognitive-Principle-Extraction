"""
Test suite for specialists/world_model/encoder.py

Covers:
- Output shapes (standard forward and with token_reprs)
- Actions are actually used (different actions → different latents)
- Observations are used (different obs → different latents)
- Determinism in eval mode (same input → same output)
- Gradient flow: all parameters receive finite gradients
- All parameters are learnable (requires_grad=True)
- Invalid embed_dim/n_heads raises ValueError
- Effective rank method returns a value in valid range [1, latent_dim]
- Effective rank is higher for diverse inputs than collapsed inputs
- Output is finite (no NaN / Inf) on valid input
- Batch consistency: encoding in batch vs. individually gives same result
- token_reprs shape when return_token_reprs=True
- Dropout is active in train mode, inactive in eval mode
"""

import torch
import torch.nn as nn
import pytest

from specialists.world_model.encoder import ContextEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def enc():
    """Default encoder in eval mode for deterministic tests."""
    model = ContextEncoder(
        obs_dim=6, seq_len=20, embed_dim=64,
        latent_dim=64, n_heads=4, n_layers=2,
    )
    model.eval()
    return model


@pytest.fixture
def batch():
    """Standard input batch."""
    torch.manual_seed(0)
    B, T, D = 4, 20, 6
    return torch.randn(B, T, D), torch.randn(B, T)


# ─────────────────────────────────────────────────────────────────────────────
# Shape tests
# ─────────────────────────────────────────────────────────────────────────────

def test_output_shape(enc, batch):
    obs, actions = batch
    latent = enc(obs, actions)
    assert latent.shape == (4, 64), f"Expected (4, 64), got {latent.shape}"


def test_output_shape_single_sample(enc):
    obs     = torch.randn(1, 20, 6)
    actions = torch.randn(1, 20)
    latent  = enc(obs, actions)
    assert latent.shape == (1, 64)


def test_token_reprs_shape(enc, batch):
    obs, actions = batch
    latent, token_reprs = enc(obs, actions, return_token_reprs=True)
    assert latent.shape      == (4, 64),      f"latent: {latent.shape}"
    assert token_reprs.shape == (4, 20, 64),  f"token_reprs: {token_reprs.shape}"


def test_token_reprs_latent_consistent(enc, batch):
    """Latent from return_token_reprs=True must match latent from standard forward."""
    obs, actions = batch
    latent_standard          = enc(obs, actions)
    latent_with_tokens, _    = enc(obs, actions, return_token_reprs=True)
    assert torch.allclose(latent_standard, latent_with_tokens), (
        "Latent differs depending on whether token_reprs are returned — "
        "return_token_reprs flag must not affect the forward computation."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Input sensitivity tests — proves both obs and actions are used
# ─────────────────────────────────────────────────────────────────────────────

def test_different_actions_give_different_latents(enc):
    """
    Same observations, different actions must produce different latents.
    This proves the encoder actually uses the action input rather than
    ignoring it — critical for the action-conditioned prediction role.
    """
    torch.manual_seed(1)
    obs = torch.randn(1, 20, 6)
    a1  = torch.randn(1, 20)
    a2  = torch.randn(1, 20)
    with torch.no_grad():
        l1 = enc(obs, a1)
        l2 = enc(obs, a2)
    assert not torch.allclose(l1, l2), (
        "Same obs + different actions produced identical latents. "
        "The encoder is ignoring the action input."
    )


def test_different_obs_give_different_latents(enc):
    """Different observations must produce different latents."""
    torch.manual_seed(2)
    actions = torch.randn(1, 20)
    obs1    = torch.randn(1, 20, 6)
    obs2    = torch.randn(1, 20, 6)
    with torch.no_grad():
        l1 = enc(obs1, actions)
        l2 = enc(obs2, actions)
    assert not torch.allclose(l1, l2)


# ─────────────────────────────────────────────────────────────────────────────
# Determinism and reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def test_deterministic_in_eval_mode(enc, batch):
    """Same input in eval mode must produce identical output on every call."""
    obs, actions = batch
    with torch.no_grad():
        la = enc(obs, actions)
        lb = enc(obs, actions)
    assert torch.allclose(la, lb), (
        "Encoder is non-deterministic in eval mode — dropout may still be active."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gradient and parameter tests
# ─────────────────────────────────────────────────────────────────────────────

def test_all_parameters_receive_gradients(batch):
    """Every learnable parameter must receive a finite gradient on backward."""
    enc = ContextEncoder()
    enc.train()
    obs, actions = batch

    latent = enc(obs, actions)
    loss   = latent.sum()
    loss.backward()

    for name, param in enc.named_parameters():
        assert param.grad is not None, f"No gradient for parameter: {name}"
        assert param.grad.isfinite().all(), (
            f"Non-finite gradient for parameter: {name}"
        )


def test_all_parameters_are_learnable(enc):
    """All parameters must have requires_grad=True."""
    for name, param in enc.named_parameters():
        assert param.requires_grad, f"Parameter {name} has requires_grad=False"


def test_parameters_update_after_gradient_step(batch):
    """A gradient step must change the encoder's parameters."""
    enc = ContextEncoder()
    enc.train()
    obs, actions = batch

    params_before = {
        name: p.clone().detach()
        for name, p in enc.named_parameters()
    }

    optimiser = torch.optim.Adam(enc.parameters(), lr=1e-3)
    latent = enc(obs, actions)
    loss   = latent.sum()
    loss.backward()
    optimiser.step()

    changed = False
    for name, p in enc.named_parameters():
        if not torch.allclose(p, params_before[name]):
            changed = True
            break

    assert changed, "No parameters changed after a gradient step."


# ─────────────────────────────────────────────────────────────────────────────
# Output validity
# ─────────────────────────────────────────────────────────────────────────────

def test_output_is_finite(enc, batch):
    """Encoder must not produce NaN or Inf on normal input."""
    obs, actions = batch
    with torch.no_grad():
        latent = enc(obs, actions)
    assert latent.isfinite().all(), "Encoder output contains NaN or Inf."


def test_output_finite_after_large_input(enc):
    """Output must remain finite even with large-magnitude input."""
    obs     = torch.randn(2, 20, 6) * 100
    actions = torch.randn(2, 20) * 100
    with torch.no_grad():
        latent = enc(obs, actions)
    assert latent.isfinite().all(), (
        "Encoder output is non-finite for large input — "
        "LayerNorm should have stabilised this."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Batch consistency test
# ─────────────────────────────────────────────────────────────────────────────

def test_batch_vs_individual_consistency(enc):
    """
    Encoding samples individually vs. in a batch must give the same result.
    A failure here would indicate a bug in batch normalisation or attention
    masking that introduces cross-sample contamination.
    """
    torch.manual_seed(5)
    obs     = torch.randn(3, 20, 6)
    actions = torch.randn(3, 20)

    with torch.no_grad():
        latent_batch = enc(obs, actions)
        latent_individual = torch.stack([
            enc(obs[i].unsqueeze(0), actions[i].unsqueeze(0)).squeeze(0)
            for i in range(3)
        ])

    assert torch.allclose(latent_batch, latent_individual, atol=1e-5), (
        "Batch encoding differs from individual encoding — "
        "possible cross-sample contamination."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────────────────

def test_raises_on_incompatible_embed_n_heads():
    """embed_dim not divisible by n_heads must raise ValueError."""
    with pytest.raises(ValueError, match="divisible"):
        ContextEncoder(embed_dim=65, n_heads=4)


def test_raises_on_zero_n_heads():
    with pytest.raises((ValueError, AssertionError)):
        ContextEncoder(embed_dim=64, n_heads=0)


# ─────────────────────────────────────────────────────────────────────────────
# Effective rank tests — collapse monitoring
# ─────────────────────────────────────────────────────────────────────────────

def test_effective_rank_is_in_valid_range(enc, batch):
    """Effective rank must be in [1, latent_dim]."""
    obs, actions = batch
    with torch.no_grad():
        latent = enc(obs, actions)
    er = ContextEncoder.effective_rank(latent)
    assert 1.0 <= er <= 64.0, f"Effective rank {er} out of valid range [1, 64]."


def test_effective_rank_higher_for_diverse_latents(enc):
    """
    A batch of diverse latents (varied inputs) should have higher effective
    rank than a batch of near-identical latents (constant input repeated).
    This validates that effective_rank correctly distinguishes collapsed from
    non-collapsed representations.
    """
    torch.manual_seed(7)
    # Diverse: many different inputs
    obs_diverse     = torch.randn(32, 20, 6)
    actions_diverse = torch.randn(32, 20)

    # Collapsed: same input repeated
    obs_single      = torch.randn(1, 20, 6).expand(32, -1, -1)
    actions_single  = torch.randn(1, 20).expand(32, -1)

    with torch.no_grad():
        latent_diverse   = enc(obs_diverse, actions_diverse)
        latent_collapsed = enc(obs_single, actions_single)

    er_diverse   = ContextEncoder.effective_rank(latent_diverse)
    er_collapsed = ContextEncoder.effective_rank(latent_collapsed)

    assert er_diverse > er_collapsed, (
        f"Diverse latents (er={er_diverse:.2f}) should have higher effective rank "
        f"than collapsed latents (er={er_collapsed:.2f})."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dropout behaviour
# ─────────────────────────────────────────────────────────────────────────────

def test_dropout_inactive_in_eval_mode(enc, batch):
    """eval() mode must produce identical results across calls (dropout off)."""
    obs, actions = batch
    with torch.no_grad():
        results = [enc(obs, actions) for _ in range(5)]
    for r in results[1:]:
        assert torch.allclose(results[0], r), "Dropout appears active in eval mode."


def test_dropout_active_in_train_mode(batch):
    """
    In train mode with dropout > 0, repeated forward passes on the same input
    should (with very high probability) produce different outputs.
    """
    enc = ContextEncoder(dropout=0.5)
    enc.train()
    obs, actions = batch
    results = [enc(obs, actions) for _ in range(5)]
    # At least one pair should differ (dropout makes this virtually certain)
    any_different = any(
        not torch.allclose(results[0], results[i]) for i in range(1, 5)
    )
    assert any_different, (
        "All forward passes produced identical output in train mode — "
        "dropout may not be active."
    )