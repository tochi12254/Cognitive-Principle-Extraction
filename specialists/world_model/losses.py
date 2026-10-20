"""
VICReg loss — training objective for Specialist A (world-model).

VICReg (Variance-Invariance-Covariance Regularization)
Bardes, Ponce, LeCun — Meta/FAIR, 2022
https://arxiv.org/abs/2105.04906

Ties the three trainable components together:
    ContextEncoder f_θ  → z_ctx  (B, latent_dim)
    TargetEncoder  f_ξ  → z_tgt  (B, latent_dim)  [no gradient]
    Predictor      g_φ  → z_pred (B, latent_dim)

Loss:
    L = λ · s(z_pred, z_tgt)   [invariance  — predictive signal]
      + μ · v(z_pred)           [variance    — prevents dimensional collapse]
      + ν · c(z_pred)           [covariance  — prevents redundant dimensions]

Default weights (from original paper): λ=25, μ=25, ν=1
The covariance weight ν is lower because the covariance matrix has O(D²)
terms vs. O(D) for the other two — without this the covariance term dominates.

Why each term is necessary (full reasoning in docs/research-log):

INVARIANCE: MSE between z_pred and z_tgt. This is the actual predictive signal.
Without it, the system has no reason to predict anything. Alone, it collapses
everything to zero (z_pred = z_tgt = 0 satisfies it trivially).

VARIANCE: Hinge loss keeping per-dimension std ≥ γ (γ=1.0) across the batch.
Prevents dimensional collapse (all latents mapping to a single point or line).
Note: gradient is zero at exact collapse (all identical inputs → variance=0 →
d(var)/d(z_i) = 2(z_i - mean)/(B-1) = 0). This is a known property of VICReg.
The variance term catches near-collapses before they become exact, and the
invariance term provides gradient even when variance gradient is zero.

COVARIANCE: Penalises off-diagonal entries of the latent covariance matrix.
Prevents dimensional redundancy (different dimensions encoding identical info).
Normalised by D (latent dimension) to keep scale consistent across architectures.

Scope: variance and covariance are applied to z_pred only. z_tgt comes from
TargetEncoder whose forward() runs under @torch.no_grad() — no gradient flows
through it, so applying regularisation terms to z_tgt contributes no gradient
signal and only inflates reported values. z_tgt collapse monitoring is handled
separately via ContextEncoder.effective_rank(z_tgt) in the training loop.

Numerical stability:
    eps=1e-4 inside sqrt of variance term prevents NaN gradient when var→0.
    Covariance requires B ≥ 2 (computes sample covariance with N-1 denominator).
    B=1 raises ValueError explicitly rather than silently returning NaN.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Return type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VICRegLoss:
    """
    Container for VICReg loss components.

    Fields:
        total:      Weighted sum — the scalar to call .backward() on.
        invariance: Unweighted MSE(z_pred, z_tgt). Pure predictive error.
        variance:   Unweighted mean hinge-std across latent dimensions.
        covariance: Unweighted off-diagonal covariance energy / D.

    Logging discipline:
        Watch all four. A rising invariance with falling variance/covariance
        usually means the system is learning but struggling with prediction.
        A falling invariance with zero variance/covariance usually means
        near-collapse (variance term not yet catching it — increase mu_var).
        A flat total with high covariance means redundant dimensions — the
        system is encoding the same information in multiple dimensions
        instead of spreading information across the latent space.
    """
    total:      Tensor
    invariance: Tensor
    variance:   Tensor
    covariance: Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Individual loss terms — exposed for unit testing and ablation
# ─────────────────────────────────────────────────────────────────────────────

def invariance_loss(z_pred: Tensor, z_tgt: Tensor) -> Tensor:
    """
    Invariance term: mean squared error between predicted and target latents.

        s(Z, Z') = (1 / B*D) * Σ_i Σ_j (z_pred_ij - z_tgt_ij)²

    Uses F.mse_loss (mean over all elements), consistent with the official
    VICReg codebase from Meta/FAIR.

    z_tgt has no grad_fn (TargetEncoder.forward() runs under @no_grad),
    so gradient flows through z_pred only — correct behaviour.

    Args:
        z_pred: (B, D) — predictor output
        z_tgt:  (B, D) — target encoder output (no gradient)

    Returns:
        Scalar tensor.
    """
    return F.mse_loss(z_pred, z_tgt)


def variance_loss(z: Tensor, gamma: float = 1.0, eps: float = 1e-4) -> Tensor:
    """
    Variance term: hinge loss keeping per-dimension std ≥ γ.

        v(Z) = (1/D) * Σ_j max(0, γ - std(Z_j))

    where std(Z_j) = sqrt(Var(Z_j) + ε) and ε prevents NaN gradients
    when variance approaches zero exactly.

    Args:
        z:     (B, D) — latent batch
        gamma: Minimum desired std per dimension. Default 1.0 (from paper).
        eps:   Numerical stability term inside sqrt. Default 1e-4 (from paper).

    Returns:
        Scalar tensor. Zero when all dimensions have std ≥ γ.
        Positive (up to γ) when dimensions have collapsed.

    Known limitation:
        Gradient is zero at exact collapse (all inputs identical → all
        deviations from mean = 0 → d(var)/d(z_i) = 0). The eps term
        stabilises the sqrt gradient but cannot fix the upstream var gradient.
        In practice, training never reaches exact collapse because the
        invariance term and random initialisation keep z moving.
    """
    # std per dimension across the batch
    std = torch.sqrt(z.var(dim=0) + eps)      # (D,)
    return F.relu(gamma - std).mean()


def covariance_loss(z: Tensor) -> Tensor:
    """
    Covariance term: sum of squared off-diagonal entries of the sample
    covariance matrix, normalised by D.

        c(Z) = (1/D) * Σ_{i≠j} [Cov(Z)_{ij}]²

    Penalises redundant information shared between dimensions.
    Requires B ≥ 2 (sample covariance uses B-1 denominator).

    Args:
        z: (B, D) — latent batch, B ≥ 2.

    Returns:
        Scalar tensor. Zero when all dimensions are uncorrelated.
        Larger when dimensions encode redundant information.

    Raises:
        ValueError: if B < 2.
    """
    B, D = z.shape
    if B < 2:
        raise ValueError(
            f"covariance_loss requires batch size B ≥ 2, got B={B}. "
            f"With B=1, the sample covariance denominator (B-1) is zero → NaN."
        )

    # Centre latents — required for covariance computation
    z = z - z.mean(dim=0)

    # Sample covariance matrix: (D, D)
    cov = (z.T @ z) / (B - 1)

    # Off-diagonal squared entries, normalised by D.
    # fill_diagonal_(0) in-place on a clone to avoid modifying the original.
    off = cov.pow(2).clone()
    off.fill_diagonal_(0.0)

    return off.sum() / D


# ─────────────────────────────────────────────────────────────────────────────
# Combined VICReg loss
# ─────────────────────────────────────────────────────────────────────────────

def vicreg_loss(
    z_pred:     Tensor,
    z_tgt:      Tensor,
    lambda_inv: float = 25.0,
    mu_var:     float = 25.0,
    nu_cov:     float = 1.0,
    gamma:      float = 1.0,
    eps:        float = 1e-4,
) -> VICRegLoss:
    """
    Full VICReg loss combining all three terms.

        L = λ * s(z_pred, z_tgt)  +  μ * v(z_pred)  +  ν * c(z_pred)

    Call .backward() on the returned .total field.
    Log .invariance, .variance, .covariance individually each training step.

    Args:
        z_pred:     (B, D) — predictor output g_φ(z_ctx, a_next)
        z_tgt:      (B, D) — target encoder output f_ξ(next_obs, next_actions)
                    Expected to have no grad_fn (from @torch.no_grad() in
                    TargetEncoder.forward). If it does have a grad_fn,
                    the loss will still work but gradients will flow through
                    both branches — unintended, so ensure z_tgt is detached.
        lambda_inv: Weight on invariance term. Default 25 (from paper).
        mu_var:     Weight on variance term. Default 25 (from paper).
        nu_cov:     Weight on covariance term. Default 1 (from paper).
        gamma:      Variance threshold. Default 1.0 (from paper).
        eps:        Numerical stability for variance sqrt. Default 1e-4.

    Returns:
        VICRegLoss dataclass with .total, .invariance, .variance, .covariance.

    Raises:
        ValueError: if z_pred and z_tgt have different shapes.
        ValueError: if batch size B < 2.
    """
    if z_pred.shape != z_tgt.shape:
        raise ValueError(
            f"z_pred and z_tgt must have the same shape. "
            f"Got z_pred: {z_pred.shape}, z_tgt: {z_tgt.shape}."
        )

    B = z_pred.shape[0]
    if B < 2:
        raise ValueError(
            f"vicreg_loss requires batch size B ≥ 2, got B={B}. "
            f"The variance and covariance terms are undefined for B=1."
        )

    inv = invariance_loss(z_pred, z_tgt)
    var = variance_loss(z_pred, gamma=gamma, eps=eps)
    cov = covariance_loss(z_pred)

    total = lambda_inv * inv + mu_var * var + nu_cov * cov

    return VICRegLoss(
        total=total,
        invariance=inv,
        variance=var,
        covariance=cov,
    )