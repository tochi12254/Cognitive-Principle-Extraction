"""
JEPA predictor  g_φ  (Specialist A — world-model, action-conditioned half).

Implements the action-conditioned prediction half of predictive world-modeling
from the CPE thesis:
    "JEPA-style latent-state prediction is built specifically for that second
     half [of the capacity]: anticipating the consequences of a candidate
     action before taking it."

Given the context latent z_ctx (from f_θ) and the next action a_next, g_φ
predicts the target latent z_pred that f_ξ would produce for the next state.

    g_φ(z_ctx, a_next) → z_pred ≈ f_ξ(obs_{t+1}, actions_{t+1})

The loss (VICReg, in losses.py) then penalises the distance between z_pred
and z_tgt, plus variance and covariance terms that prevent collapse.

Architecture decisions (full reasoning in docs/research-log):

1. Simpler and narrower than the encoders (by design):
   The predictor being architecturally weaker than the encoder is load-bearing
   for training stability. If predictor capacity == encoder capacity, the trivial
   solution (encoder maps everything to a constant → predictor trivially matches)
   becomes easy to find. A simpler predictor forces the encoder to learn rich,
   structured representations that are predictable only through genuine world
   structure, not shortcut memorisation.

   Context encoder: ~110K params
   Predictor:        ~34K params  (~3x smaller — correct asymmetry)

2. Action conditioning via concatenation, not a learned embedding:
   At toy scale with a single continuous action, concatenating [z_ctx ∥ a_next]
   is the right call. A learned embedding of a scalar adds parameters and
   complexity with no benefit. The first linear layer learns to weight the action
   relative to the context latent.

3. LayerNorm over BatchNorm:
   BatchNorm behaves differently in train vs. eval mode (batch statistics vs.
   running statistics). The target encoder f_ξ is always in eval mode; the
   predictor trains in train mode. Using BatchNorm would create an inconsistency
   between how the predictor sees its own training distribution and how f_ξ
   generates targets. LayerNorm normalises per-sample across features — no
   batch-size dependence, consistent between train and eval, correct here.

4. No normalisation on output:
   VICReg's variance term prevents the predictor output from collapsing to
   a constant. L2 normalisation of z_pred is redundant given VICReg's
   explicit collapse prevention and would constrain the geometry of the latent
   space unnecessarily.

5. Three-layer MLP with hidden_dim = 2 × latent_dim:
   Two hidden layers give the predictor capacity to apply a nonlinear transform
   rather than a linear projection (which would be too weak) but not so deep
   that it becomes hard to train or approaches encoder capacity.
"""

import torch
import torch.nn as nn


class Predictor(nn.Module):
    """
    JEPA action-conditioned predictor  g_φ.

    Args:
        latent_dim: Dimension of context and target latent vectors.
                    Must match ContextEncoder.latent_dim and
                    TargetEncoder.latent_dim.
        hidden_dim: MLP hidden dimension. Default 128 (2 × latent_dim).
                    Should be kept < encoder embed_dim to preserve asymmetry.
        dropout:    Dropout rate. Default 0.0 (off). VICReg's variance term
                    already provides regularisation; dropout is an optional
                    ablation.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        dropout:    float = 0.0,
    ):
        super().__init__()

        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}.")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}.")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0], got {dropout}.")

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Input: [z_ctx ∥ a_next] — latent_dim + 1
        input_dim = latent_dim + 1

        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            # Output projection — no activation, no normalisation.
            # VICReg's variance term handles collapse prevention on the output.
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        z_ctx:  torch.Tensor,  # (B, latent_dim) — context encoder output
        a_next: torch.Tensor,  # (B,) or (B, 1) — action taken at prediction step
    ) -> torch.Tensor:
        """
        Predict the target latent from context latent and next action.

        Args:
            z_ctx:  (B, latent_dim) — output of context encoder f_θ
            a_next: (B,) or (B, 1) — the action taken after the context window.
                    Accepts both shapes for convenience.

        Returns:
            z_pred: (B, latent_dim) — predicted target latent
        """
        # Normalise action shape to (B, 1) for concatenation
        if a_next.dim() == 1:
            a_next = a_next.unsqueeze(-1)   # (B,) → (B, 1)

        # Concatenate context latent and action along feature dimension
        x = torch.cat([z_ctx, a_next], dim=-1)   # (B, latent_dim + 1)

        return self.net(x)                         # (B, latent_dim)