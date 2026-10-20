"""
Target encoder  f_ξ  (Specialist A — world-model, EMA target provider).

f_ξ is an Exponential Moving Average (EMA) copy of the context encoder f_θ.
It provides stable latent targets for the JEPA predictor g_φ to predict toward.

Why EMA instead of using f_θ directly as the target encoder:
    If we encoded both context and target with the same f_θ, the system would
    be chasing its own tail — predicting toward a target that's moving at the
    same rate as the predictor. The trivial solution (mapping everything to a
    constant vector) always satisfies this, causing representational collapse.

    The EMA solution: f_ξ's weights are a slow-moving average of f_θ's weights.
        f_ξ ← τ * f_ξ + (1 - τ) * f_θ    (τ = momentum, typically 0.99)
    f_ξ changes slowly, providing a stable target. f_θ changes quickly via
    gradient descent. The asymmetry between them is what prevents collapse.

Design decisions (full reasoning in docs/research-log):
    - Wrapper pattern: TargetEncoder holds a ContextEncoder internally and
      delegates forward() to it. ContextEncoder knows nothing about EMA.
    - deepcopy on __init__: guarantees f_ξ starts as an exact copy of f_θ,
      not a random initialization. If they started with different weights,
      the first training step would predict toward an unrelated representation.
    - requires_grad=False on all params: EMA update uses .data directly,
      not the gradient mechanism. Belt-and-suspenders: also wrapped in no_grad.
    - train() overridden to always enforce eval mode: dropout in f_ξ would
      make targets stochastic and noisy. This override makes it impossible to
      accidentally enable train mode.
    - forward() decorated with @torch.no_grad(): targets are treated as
      constants — no computation graph built, no gradient flows back through f_ξ.

Update rule:
    call target_encoder.update(context_encoder, momentum=0.99)
    after every optimiser.step() in the training loop.

Sync:
    call target_encoder.sync(context_encoder)
    to hard-reset f_ξ to exact copy of f_θ (used if resuming from checkpoint
    where the EMA may have drifted from what f_θ currently is).
"""

import copy

import torch
import torch.nn as nn

from specialists.world_model.encoder import ContextEncoder


class TargetEncoder(nn.Module):
    """
    EMA target encoder  f_ξ.

    Args:
        source: The context encoder f_θ to copy architecture and initial
                weights from. TargetEncoder deepcopies source — changes to
                source after construction do not affect TargetEncoder until
                .update() is called.

    Usage:
        # Construction — copy f_θ exactly:
        target_enc = TargetEncoder(context_enc)

        # Training loop:
        latent_ctx = context_enc(obs, actions)          # gradient flows
        with torch.no_grad():
            latent_tgt = target_enc(next_obs, actions)  # no gradient

        loss = vicreg_loss(predictor(latent_ctx, next_action), latent_tgt)
        loss.backward()
        optimiser.step()
        target_enc.update(context_enc, momentum=0.99)   # EMA update
    """

    def __init__(self, source: ContextEncoder):
        super().__init__()

        # Deep copy: f_ξ is architecturally identical to f_θ and starts
        # with exactly the same weights. Changes to source after this point
        # do not affect self.encoder until .update() is called.
        self.encoder: ContextEncoder = copy.deepcopy(source)

        # Disable gradient accumulation on all parameters.
        # EMA update uses .data directly, bypassing autograd entirely.
        for param in self.encoder.parameters():
            param.requires_grad_(False)

        # Always eval: no dropout, no BatchNorm running-stat updates.
        # train() is overridden below to enforce this permanently.
        # super().train(False) sets both self.training and self.encoder.training
        # to False. Must be called here because nn.Module.__init__ sets
        # self.training = True by default — the override on train() prevents
        # future calls from enabling training mode, but doesn't retroactively
        # set the initial state.
        self.encoder.eval()
        super().train(False)   # set outer TargetEncoder wrapper to eval too

    # ── Eval mode enforcement ─────────────────────────────────────────────

    def train(self, mode: bool = True) -> "TargetEncoder":
        """
        Override: target encoder is permanently in eval mode.
        Calling .train() has no effect — it always stays in eval.

        Rationale: stochastic dropout in f_ξ would introduce noise into the
        prediction targets, destabilising training. This cannot be an
        accident-prone convention — it is enforced structurally.
        """
        # Always call super().train(False) == super().eval()
        return super().train(False)

    # ── Forward pass ─────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(
        self,
        obs: torch.Tensor,      # (B, seq_len, obs_dim)
        actions: torch.Tensor,  # (B, seq_len)
        return_token_reprs: bool = False,
    ) -> torch.Tensor:
        """
        Encode a target window into a latent vector.

        Identical signature to ContextEncoder.forward().
        Always runs under torch.no_grad() — targets are treated as constants,
        no computation graph is built, no gradient flows back through f_ξ.

        Args:
            obs:                (B, seq_len, obs_dim)
            actions:            (B, seq_len)
            return_token_reprs: passed through to ContextEncoder.forward()

        Returns:
            latent:             (B, latent_dim)
            token_reprs:        (B, seq_len, embed_dim)  only if return_token_reprs=True
        """
        return self.encoder(obs, actions, return_token_reprs=return_token_reprs)

    # ── EMA update ───────────────────────────────────────────────────────

    def update(self, source: ContextEncoder, momentum: float = 0.99) -> None:
        """
        Update f_ξ weights via EMA from f_θ (source).

            f_ξ ← momentum * f_ξ + (1 - momentum) * f_θ

        Call this once per training step, after optimiser.step().

        Args:
            source:   The context encoder f_θ whose weights to track.
            momentum: τ in the EMA formula. Typical range: [0.99, 0.999].
                      Higher = f_ξ changes more slowly (more stable targets).
                      Lower  = f_ξ tracks f_θ more closely (less stable).

        Edge cases:
            momentum=1.0  → no update (f_ξ stays exactly as it is)
            momentum=0.0  → hard copy (f_ξ becomes exact copy of f_θ)
        """
        if not (0.0 <= momentum <= 1.0):
            raise ValueError(
                f"momentum must be in [0.0, 1.0], got {momentum}."
            )

        with torch.no_grad():
            for param_xi, param_theta in zip(
                self.encoder.parameters(), source.parameters()
            ):
                # In-place EMA: ξ = τ * ξ + (1 - τ) * θ
                param_xi.data.mul_(momentum).add_(
                    param_theta.data, alpha=1.0 - momentum
                )

    # ── Hard sync ─────────────────────────────────────────────────────────

    def sync(self, source: ContextEncoder) -> None:
        """
        Hard reset: copy all weights from source exactly into f_ξ.
        Equivalent to update(source, momentum=0.0).

        Use case: resuming from a checkpoint where you want to reset f_ξ
        to exactly match the saved f_θ before continuing training, rather
        than trusting that the checkpoint already has correct EMA state.
        """
        with torch.no_grad():
            for param_xi, param_theta in zip(
                self.encoder.parameters(), source.parameters()
            ):
                param_xi.data.copy_(param_theta.data)

    # ── Convenience properties ─────────────────────────────────────────────

    @property
    def latent_dim(self) -> int:
        return self.encoder.latent_dim

    @property
    def obs_dim(self) -> int:
        return self.encoder.obs_dim

    @property
    def seq_len(self) -> int:
        return self.encoder.seq_len