"""
Test suite for specialists/world_model/target_encoder.py

Covers:
- Output shape matches ContextEncoder
- All parameters have requires_grad=False
- Always in eval mode — train() call is a no-op
- At construction, identical weights → identical output to source
- forward() runs under no_grad (no computation graph built)
- No gradient flows through TargetEncoder during backward
- EMA update moves xi toward theta
- EMA momentum=1.0 is a no-op (xi unchanged)
- EMA momentum=0.0 makes xi exact copy of theta
- EMA update only affects xi, not theta (source unaffected)
- Sync makes xi exact copy of theta
- Consecutive EMA updates progressively track theta
- Invalid momentum raises ValueError
- Convenience properties (latent_dim, obs_dim, seq_len) match source
- return_token_reprs passthrough works correctly
- Training step on source (theta) does not affect xi
"""

import torch
import pytest

from specialists.world_model.encoder import ContextEncoder
from specialists.world_model.target_encoder import TargetEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def context_enc():
    torch.manual_seed(0)
    return ContextEncoder(obs_dim=6, seq_len=20, embed_dim=64, latent_dim=64)


@pytest.fixture
def target_enc(context_enc):
    return TargetEncoder(context_enc)


@pytest.fixture
def batch():
    torch.manual_seed(1)
    return torch.randn(4, 20, 6), torch.randn(4, 20)


# ─────────────────────────────────────────────────────────────────────────────
# Shape tests
# ─────────────────────────────────────────────────────────────────────────────

def test_output_shape(target_enc, batch):
    obs, actions = batch
    latent = target_enc(obs, actions)
    assert latent.shape == (4, 64)


def test_token_reprs_shape(target_enc, batch):
    obs, actions = batch
    latent, tokens = target_enc(obs, actions, return_token_reprs=True)
    assert latent.shape  == (4, 64)
    assert tokens.shape  == (4, 20, 64)


# ─────────────────────────────────────────────────────────────────────────────
# Gradient and parameter tests
# ─────────────────────────────────────────────────────────────────────────────

def test_all_params_have_requires_grad_false(target_enc):
    """All target encoder parameters must have requires_grad=False."""
    for name, param in target_enc.named_parameters():
        assert not param.requires_grad, (
            f"Parameter {name} has requires_grad=True — "
            f"target encoder must never accumulate gradients."
        )


def test_no_gradient_flows_through_target_encoder(context_enc, batch):
    """
    Backward through a realistic loss (context latent vs target latent)
    must assign gradients to f_θ parameters but NOT to f_ξ parameters.

    The correct test pattern mirrors the real training loop:
        loss = (predictor(z_ctx) - z_tgt).pow(2).sum()
    z_tgt is a constant (no grad_fn from @no_grad), so gradients flow
    through z_ctx (context_enc) but cannot reach z_tgt (target_enc).
    Calling backward directly on z_tgt.sum() would fail because there
    is no grad_fn — that would test a broken usage, not the real one.
    """
    target_enc = TargetEncoder(context_enc)
    obs, actions = batch

    # Simulate real training: context latent has gradient, target is constant
    latent_ctx = context_enc(obs, actions)           # grad_fn present
    latent_tgt = target_enc(obs, actions)            # no grad_fn (no_grad)

    loss = (latent_ctx - latent_tgt).pow(2).sum()    # grad flows through ctx only
    loss.backward()

    # f_θ must have gradients
    for name, p in context_enc.named_parameters():
        assert p.grad is not None, f"f_θ parameter {name} has no gradient."

    # f_ξ must NOT have gradients
    for name, param in target_enc.named_parameters():
        assert param.grad is None, (
            f"Gradient found on f_ξ parameter {name} — "
            f"gradient must never reach the target encoder."
        )


def test_forward_does_not_build_computation_graph(target_enc, batch):
    """
    Output of target encoder must not require gradient,
    confirming no computation graph was built.
    """
    obs, actions = batch
    latent = target_enc(obs, actions)
    assert not latent.requires_grad, (
        "Target encoder output has requires_grad=True — "
        "a computation graph was built when it should not have been."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Eval mode enforcement
# ─────────────────────────────────────────────────────────────────────────────

def test_target_encoder_starts_in_eval_mode(target_enc):
    assert not target_enc.training
    assert not target_enc.encoder.training


def test_train_call_is_no_op(target_enc):
    """Calling .train() on target encoder must not enable training mode."""
    target_enc.train()
    assert not target_enc.training, (
        "target_enc.train() enabled training mode — "
        "this allows stochastic dropout in targets, which must never happen."
    )
    assert not target_enc.encoder.training


def test_train_true_is_no_op(target_enc):
    target_enc.train(True)
    assert not target_enc.training


def test_deterministic_output(target_enc, batch):
    """Same input must always produce same output (dropout never active)."""
    obs, actions = batch
    results = [target_enc(obs, actions) for _ in range(5)]
    for r in results[1:]:
        assert torch.allclose(results[0], r), (
            "Non-deterministic output — train mode may have been enabled."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation tests
# ─────────────────────────────────────────────────────────────────────────────

def test_identical_weights_at_init(context_enc, target_enc):
    """
    At construction, f_ξ must be an exact copy of f_θ.
    If they differ, the first training step predicts toward an
    unrelated representation.
    """
    for (name_θ, p_θ), (name_ξ, p_ξ) in zip(
        context_enc.named_parameters(),
        target_enc.encoder.named_parameters(),
    ):
        assert torch.allclose(p_θ, p_ξ), (
            f"Weight mismatch at init for {name_θ}: "
            f"target encoder was not correctly copied from source."
        )


def test_identical_output_at_init(context_enc, target_enc, batch):
    """Identical weights → identical forward output at construction."""
    obs, actions = batch
    context_enc.eval()
    with torch.no_grad():
        l_ctx = context_enc(obs, actions)
    l_tgt = target_enc(obs, actions)
    assert torch.allclose(l_ctx, l_tgt), (
        "Output differs at init despite identical weights."
    )


def test_source_modification_after_init_does_not_affect_target(context_enc, target_enc, batch):
    """
    Modifying source (f_θ) weights after construction must not affect
    target encoder (f_ξ) until .update() is called.
    This confirms deepcopy gave f_ξ its own independent weight tensors.
    """
    obs, actions = batch
    latent_before = target_enc(obs, actions).clone()

    with torch.no_grad():
        for p in context_enc.parameters():
            p.data.fill_(99.0)

    latent_after = target_enc(obs, actions)
    assert torch.allclose(latent_before, latent_after), (
        "Target encoder output changed when source weights were modified — "
        "deepcopy did not correctly isolate f_ξ from f_θ."
    )


# ─────────────────────────────────────────────────────────────────────────────
# EMA update tests
# ─────────────────────────────────────────────────────────────────────────────

def test_ema_update_changes_xi(context_enc, target_enc):
    """EMA update (default momentum) must change f_ξ weights."""
    p_before = next(iter(target_enc.encoder.parameters())).data.clone()

    with torch.no_grad():
        for p in context_enc.parameters():
            p.data.fill_(5.0)

    target_enc.update(context_enc, momentum=0.99)
    p_after = next(iter(target_enc.encoder.parameters())).data.clone()
    assert not torch.allclose(p_before, p_after), (
        "EMA update did not change f_ξ weights."
    )


def test_ema_update_moves_xi_toward_theta(context_enc, target_enc):
    """After EMA update, f_ξ should be closer to f_θ than it was before."""
    with torch.no_grad():
        for p in context_enc.parameters():
            p.data.fill_(10.0)

    params_before = [p.data.clone() for p in target_enc.encoder.parameters()]
    target_enc.update(context_enc, momentum=0.99)
    params_after = [p.data.clone() for p in target_enc.encoder.parameters()]

    theta_params = [p.data.clone() for p in context_enc.parameters()]

    dist_before = sum((b - t).abs().mean() for b, t in zip(params_before, theta_params))
    dist_after  = sum((a - t).abs().mean() for a, t in zip(params_after, theta_params))
    assert dist_after < dist_before, (
        "EMA update did not move f_ξ closer to f_θ."
    )


def test_ema_momentum_1_is_no_op(context_enc, target_enc):
    """momentum=1.0: f_ξ must not change."""
    p_before = [p.data.clone() for p in target_enc.encoder.parameters()]

    with torch.no_grad():
        for p in context_enc.parameters():
            p.data.fill_(99.0)

    target_enc.update(context_enc, momentum=1.0)
    for p_b, p_xi in zip(p_before, target_enc.encoder.parameters()):
        assert torch.allclose(p_b, p_xi.data), (
            "momentum=1.0 changed f_ξ weights — should be a no-op."
        )


def test_ema_momentum_0_is_exact_copy(context_enc, target_enc):
    """momentum=0.0: f_ξ must become exact copy of f_θ."""
    with torch.no_grad():
        for p in context_enc.parameters():
            p.data.fill_(7.0)

    target_enc.update(context_enc, momentum=0.0)

    for p_xi, p_theta in zip(
        target_enc.encoder.parameters(), context_enc.parameters()
    ):
        assert torch.allclose(p_xi.data, p_theta.data), (
            "momentum=0.0 did not make f_ξ an exact copy of f_θ."
        )


def test_ema_update_does_not_modify_source(context_enc, target_enc):
    """EMA update must only modify f_ξ — f_θ (source) must be unchanged."""
    params_before = [p.data.clone() for p in context_enc.parameters()]
    target_enc.update(context_enc, momentum=0.99)
    for p_b, p_theta in zip(params_before, context_enc.parameters()):
        assert torch.allclose(p_b, p_theta.data), (
            "EMA update modified source encoder (f_θ) — must only modify f_ξ."
        )


def test_consecutive_ema_updates_progressively_track_theta(context_enc, target_enc):
    """
    After many EMA updates with momentum < 1.0, f_ξ should converge
    toward f_θ. Distance should decrease monotonically.
    """
    with torch.no_grad():
        for p in context_enc.parameters():
            p.data.fill_(50.0)

    distances = []
    for _ in range(10):
        target_enc.update(context_enc, momentum=0.9)
        dist = sum(
            (p_xi.data - p_theta.data).abs().mean()
            for p_xi, p_theta in zip(
                target_enc.encoder.parameters(), context_enc.parameters()
            )
        ).item() # type: ignore
        distances.append(dist)

    assert distances[-1] < distances[0], (
        "f_ξ did not converge toward f_θ over 10 EMA updates."
    )
    # Also check monotonically decreasing
    for i in range(len(distances) - 1):
        assert distances[i + 1] <= distances[i], (
            f"Distance increased at step {i + 1}: "
            f"{distances[i]:.4f} → {distances[i + 1]:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sync test
# ─────────────────────────────────────────────────────────────────────────────

def test_sync_makes_exact_copy(context_enc, target_enc):
    """sync() must make f_ξ an exact copy of f_θ, regardless of drift."""
    # Drift xi away from theta
    with torch.no_grad():
        for p in target_enc.encoder.parameters():
            p.data.fill_(0.0)
        for p in context_enc.parameters():
            p.data.fill_(3.0)

    target_enc.sync(context_enc)

    for p_xi, p_theta in zip(
        target_enc.encoder.parameters(), context_enc.parameters()
    ):
        assert torch.allclose(p_xi.data, p_theta.data), (
            "sync() did not make f_ξ an exact copy of f_θ."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Training step isolation
# ─────────────────────────────────────────────────────────────────────────────

def test_training_step_on_source_does_not_affect_xi(context_enc, target_enc, batch):
    """
    A full backward + optimiser.step() on f_θ must not change f_ξ weights.
    f_ξ should only change when .update() is explicitly called.
    """
    obs, actions = batch
    xi_params_before = [p.data.clone() for p in target_enc.encoder.parameters()]

    optimiser = torch.optim.Adam(context_enc.parameters(), lr=1e-3)
    latent = context_enc(obs, actions)
    loss   = latent.sum()
    loss.backward()
    optimiser.step()

    for p_b, p_xi in zip(xi_params_before, target_enc.encoder.parameters()):
        assert torch.allclose(p_b, p_xi.data), (
            "f_ξ weights changed after optimiser.step() on f_θ — "
            "f_ξ must only change via explicit .update() calls."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────────────────

def test_invalid_momentum_above_1(context_enc, target_enc):
    with pytest.raises(ValueError, match="momentum"):
        target_enc.update(context_enc, momentum=1.5)


def test_invalid_momentum_below_0(context_enc, target_enc):
    with pytest.raises(ValueError, match="momentum"):
        target_enc.update(context_enc, momentum=-0.1)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience properties
# ─────────────────────────────────────────────────────────────────────────────

def test_convenience_properties_match_source(context_enc, target_enc):
    assert target_enc.latent_dim == context_enc.latent_dim
    assert target_enc.obs_dim    == context_enc.obs_dim
    assert target_enc.seq_len    == context_enc.seq_len