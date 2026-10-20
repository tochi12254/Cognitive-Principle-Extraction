"""
Transformer context encoder  f_θ  (Specialist A — world-model, compression half).

Implements the compression half of predictive world-modeling from the CPE thesis:
    "A transformer trained on next-token prediction compresses and predicts
     sequence continuations, satisfying the compression half of the capacity."

Architecture (full reasoning in docs/research-log):
    Input tokens:   [obs_t ∥ action_t]  per timestep  → shape (B, seq_len, obs_dim+1)
    Input proj:     Linear(obs_dim+1, embed_dim)
    Pos embedding:  Learned, shape (seq_len, embed_dim)
    Transformer:    N × (pre-LayerNorm + multi-head self-attn + pre-LayerNorm + FF/GELU)
    Pooling:        Mean across sequence dimension  → (B, embed_dim)
    Proj head:      MLP(embed_dim → embed_dim → latent_dim)
    Output:         latent z_ctx  shape (B, latent_dim)

Optionally returns pre-pooling token representations for later use by the
integration/binding layer (which may prefer to attend over all tokens rather
than the pooled summary alone).

Instrumentation:
    effective_rank(cov) is exposed as a static method so the training loop
    can monitor representational collapse without importing additional tooling.
    Collapse = effective rank drops toward 1 = all latents map to a line.
"""

import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    """
    Transformer context encoder  f_θ.

    Args:
        obs_dim:    Dimension of a single observation vector.
                    Must match SyntheticSystem.obs_dim (default 6).
        seq_len:    Number of timesteps in the context window.
                    Must match TrajectoryGenerator seq_len.
        embed_dim:  Transformer internal dimension. Must be divisible by n_heads.
        latent_dim: Output latent dimension (fed to predictor g_φ).
        n_heads:    Number of attention heads.
        n_layers:   Number of transformer encoder layers.
        ff_dim:     Feedforward hidden dimension inside each layer.
                    Convention: 4 × embed_dim.
        dropout:    Dropout rate applied inside attention and feedforward.
    """

    def __init__(
        self,
        obs_dim:    int = 6,
        seq_len:    int = 20,
        embed_dim:  int = 32,
        latent_dim: int = 32,
        n_heads:    int = 4,
        n_layers:   int = 2,
        ff_dim:     int = 256,
        dropout:    float = 0.1,
    ):
        super().__init__()

        if n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}.")
        if embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads}). "
                f"head_dim would be {embed_dim / n_heads:.1f}."
            )

        self.obs_dim    = obs_dim
        self.seq_len    = seq_len
        self.embed_dim  = embed_dim
        self.latent_dim = latent_dim

        # ── Input projection ──────────────────────────────────────────────
        # Each token is [obs_t ∥ action_t]: dimension obs_dim + 1.
        # Project to embed_dim before the transformer sees it.
        self.input_proj = nn.Linear(obs_dim + 1, embed_dim)

        # ── Learned positional embeddings ─────────────────────────────────
        # One embedding vector per position in the context window.
        # Learned rather than sinusoidal: simpler, transparent, and at
        # seq_len=20 there is no generalisation-to-longer-sequences concern.
        self.pos_embedding = nn.Embedding(seq_len, embed_dim)
        # Register a fixed position index buffer — no gradient, not a parameter.
        self.register_buffer(
            "positions",
            torch.arange(seq_len).unsqueeze(0),  # shape (1, seq_len)
        )

        # ── Transformer encoder ───────────────────────────────────────────
        # Pre-norm (norm_first=True): LayerNorm applied before attention
        # and before feedforward rather than after. More stable training —
        # gradient scale is controlled before it enters the attention weights,
        # not after. Standard in all post-GPT-2 architectures.
        #
        # enable_nested_tensor=False: suppresses a UserWarning that PyTorch
        # raises when norm_first=True disables its nested-tensor optimisation.
        # We don't need that optimisation at toy scale.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,       # (B, seq_len, embed_dim) convention
            norm_first=True,        # pre-norm
            activation="gelu",      # GELU: smoother than ReLU, standard in modern transformers
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # Final LayerNorm on the pooled representation before projection.
        # Stabilises the scale of the vector entering the projection head.
        self.norm = nn.LayerNorm(embed_dim)

        # ── Projection head ───────────────────────────────────────────────
        # Two-layer MLP: embed_dim → embed_dim → latent_dim.
        # The intermediate layer adds a nonlinear transform between the
        # transformer's output space and the VICReg latent space, so the
        # transformer's internal representations aren't forced to directly
        # match the latent geometry VICReg optimises. Standard practice
        # from SimCLR / VICReg / BYOL — all use a projector after the backbone.
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, latent_dim),
        )

    def forward(
        self,
        obs: torch.Tensor,      # (B, seq_len, obs_dim)
        actions: torch.Tensor,  # (B, seq_len)
        return_token_reprs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a context window into a latent vector.

        Args:
            obs:                (B, seq_len, obs_dim)  — observation sequence
            actions:            (B, seq_len)            — action sequence
            return_token_reprs: If True, also return the pre-pooling token
                                representations (B, seq_len, embed_dim).
                                Used by the integration/binding layer later.

        Returns:
            latent:             (B, latent_dim)         — compressed representation
            token_reprs:        (B, seq_len, embed_dim) — only if return_token_reprs=True
        """
        B, T, _ = obs.shape

        # Concatenate obs and action along the feature dimension.
        # action goes from (B, T) to (B, T, 1) for concatenation.
        tokens = torch.cat([obs, actions.unsqueeze(-1)], dim=-1)  # (B, T, obs_dim+1)

        # Project to embed_dim and add positional embeddings.
        x = self.input_proj(tokens)                  # (B, T, embed_dim)
        x = x + self.pos_embedding(self.positions)   # broadcast over batch

        # Transformer encoder: bidirectional self-attention over the window.
        # No causal mask — the encoder compresses the full context at once,
        # not autoregressively. All positions attend to all other positions.
        token_reprs = self.transformer(x)             # (B, T, embed_dim)

        # Mean pool: compress sequence into a single vector.
        # Uses all timesteps equally — preferable to last-token only (which
        # throws away most of the context) or CLS token (which wastes a
        # position and adds a parameter).
        pooled = token_reprs.mean(dim=1)              # (B, embed_dim)
        pooled = self.norm(pooled)

        # Project to latent space.
        latent = self.proj_head(pooled)               # (B, latent_dim)

        if return_token_reprs:
            return latent, token_reprs
        return latent

    # ── Collapse monitoring ───────────────────────────────────────────────

    @staticmethod
    def effective_rank(latents: torch.Tensor) -> float:
        """
        Compute the effective rank of the latent batch covariance matrix.

        Effective rank = exp(entropy of normalised singular values).
        Range: [1, latent_dim]. Values near 1 indicate collapse (all latents
        lie on a line). Values near latent_dim indicate full-rank, diverse
        representations.

        Call this in the training loop every N steps and log it.
        A sudden drop is an early warning of representational collapse
        before VICReg's variance term has fully caught it.

        Args:
            latents: (B, latent_dim) — a batch of encoder outputs.

        Returns:
            Scalar float: the effective rank.
        """
        with torch.no_grad():
            # Centre the latents before SVD.
            z = latents - latents.mean(dim=0, keepdim=True)
            # SVD of the centred latent matrix.
            _, S, _ = torch.linalg.svd(z, full_matrices=False)
            # Normalise singular values to a probability distribution.
            S = S / (S.sum() + 1e-8)
            # Shannon entropy of the normalised singular values.
            entropy = -(S * torch.log(S + 1e-8)).sum()
            return entropy.exp().item()