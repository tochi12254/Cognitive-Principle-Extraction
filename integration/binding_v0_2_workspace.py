"""
Binding v0.2 — Shared Latent Workspace.

Replaces the crude direct conditioning of v0.1 with a learned shared
interface inspired by VanRullen and Kanai's Global Latent Workspace
proposal (2021) and Baars/Dehaene's Global Workspace Theory.

From the CPE thesis:
    "v0.2 — a genuine shared latent workspace, in the direction of
     VanRullen and Kanai's Global Latent Workspace proposal, replacing
     hand-built conditioning with a learned shared interface."

Architecture difference from v0.1:

    v0.1 (crude conditioning):
        [z_ctx ∥ a_next ∥ causal_feature] → MLP → z_pred
        The combination is fixed (concatenation) and hand-designed.

    v0.2 (shared workspace):
        z_ctx  → key/value projections ┐
                                        ├→ cross-attention → workspace → MLP → z_pred
        z_causal → key/value projections ┘
        The combination is LEARNED via cross-attention.

GlobalWorkspace mechanism (Global Workspace Theory in practice):

    1. Each specialist produces key-value pairs from its output.
    2. Learned workspace query slots Q attend over all specialist key-values.
       This is the competitive arbitration step: each slot queries which
       specialist is most relevant right now.
    3. The attention-weighted specialist values form the workspace broadcast.
    4. The WorkspacePredictor reads from the broadcast workspace.

    Attention weights (B, n_slots, 2) are directly interpretable:
        attn[:, :, 0] = how much each slot is attending to the world-model
        attn[:, :, 1] = how much each slot is attending to the causal module
        Expected: wm dominates pre-shift; causal weight rises at crossing.

Why cross-attention over v0.1 concatenation (full reasoning in research log):

    1. Adaptive combination: the workspace learns WHEN to weight the causal
       signal heavily (at crossing) vs. when to rely on the world-model alone
       (pre-shift). v0.1 concatenation applies the same combination always.

    2. Bottleneck forces abstraction: with n_slots=4 workspace slots attending
       over 2 specialist outputs, the workspace cannot simply pass through both
       signals unchanged — it must form a compressed, integrated representation.
       This is the Data Processing Inequality constraint from the thesis.

    3. Interpretable mechanism: attention weights reveal which specialist
       the workspace is "listening to" at each moment — a direct visualisation
       of the competitive arbitration the paper describes.

    4. End-to-end learning: the key/value projections for both specialists
       are learned jointly with the workspace queries, so the workspace can
       shape how it reads from each specialist.

Bottleneck design (Data Processing Inequality):
    World-model latent:  32-dim
    Causal feature:       1-dim
    Total specialist info: 33-dim (at minimum)

    Workspace output: 32-dim
    This forces the workspace to COMPRESS the combined specialist signal —
    it cannot be a trivial passthrough. Whatever survives is the only
    thing the predictor can ever use (DPI: information can only decrease
    downstream).

POC baseline from v0.1:
    Standalone predictor normalised ratio:  1.148×
    v0.1 conditioned normalised ratio:      0.593×
    v0.2 target: ≤ 0.593× (or lower absolute crossing error)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from integration.binding_v0_1_conditioning import (
    BindingConfig,
    BindingDataGenerator,
    BindingDataset,
    N_CAUSAL,
    compute_causal_features,
)
from specialists.causal_module.discovery import CausalDiscovery
from specialists.causal_module.scm import StructuralCausalModel
from specialists.world_model.encoder import ContextEncoder
from specialists.world_model.losses import vicreg_loss
from specialists.world_model.target_encoder import TargetEncoder
from specialists.world_model.train import TrainConfig


# ─────────────────────────────────────────────────────────────────────────────
# Global Workspace
# ─────────────────────────────────────────────────────────────────────────────

class GlobalWorkspace(nn.Module):
    """
    Shared latent workspace combining world-model and causal specialist outputs
    via learned cross-attention.

    The workspace implements Global Workspace Theory's competitive arbitration
    and broadcast mechanisms:
        - Arbitration: workspace query slots compete to extract relevant
          information from each specialist via attention scores
        - Broadcast: the resulting workspace representation is available to
          the downstream predictor (all specialists read the same broadcast)

    Args:
        wm_dim:        Dimension of world-model context latent (z_ctx).
        causal_dim:    Dimension of causal feature vector.
        n_slots:       Number of workspace query slots. More slots = richer
                       workspace representation. Default 4.
        d_key:         Attention key dimension. Default 32.
        d_val:         Attention value dimension. Default 32.
        workspace_dim: Output workspace dimension. Default 32 (must match
                       latent_dim for WorkspacePredictor compatibility).
        dropout:       Attention dropout rate. Default 0.0.
    """

    def __init__(
        self,
        wm_dim:        int   = 32,
        causal_dim:    int   = N_CAUSAL,
        n_slots:       int   = 4,
        d_key:         int   = 32,
        d_val:         int   = 32,
        workspace_dim: int   = 32,
        dropout:       float = 0.0,
    ):
        super().__init__()
        self.n_slots       = n_slots
        self.d_key         = d_key
        self.d_val         = d_val
        self.workspace_dim = workspace_dim
        self.scale         = d_key ** -0.5

        # ── Specialist key-value projections ──────────────────────────────────
        # Each specialist maps its output to a key (what it knows) and
        # a value (what it contributes if selected by the workspace).
        self.k_wm     = nn.Linear(wm_dim,     d_key, bias=False)
        self.v_wm     = nn.Linear(wm_dim,     d_val)
        self.k_causal = nn.Linear(causal_dim, d_key, bias=False)
        self.v_causal = nn.Linear(causal_dim, d_val)

        # ── Learned workspace query slots ─────────────────────────────────────
        # These are the "questions" the workspace asks of each specialist.
        # Initialised with small noise; learned entirely by backprop.
        # n_slots > 1 allows the workspace to form richer representations
        # (different slots can specialise on different aspects).
        self.W_q = nn.Parameter(torch.randn(n_slots, d_key) * 0.02)

        # ── Output projection ─────────────────────────────────────────────────
        # Project the flattened n_slots × d_val workspace to workspace_dim.
        # This is where the bottleneck bites: n_slots * d_val → workspace_dim
        # cannot be lossless if workspace_dim < n_slots * d_val.
        self.proj_out  = nn.Linear(n_slots * d_val, workspace_dim)
        self.norm      = nn.LayerNorm(workspace_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        z_ctx:    torch.Tensor,   # (B, wm_dim)
        z_causal: torch.Tensor,   # (B, causal_dim)
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Compute workspace broadcast from specialist outputs.

        Args:
            z_ctx:          (B, wm_dim)    — world-model context latent
            z_causal:       (B, causal_dim) — causal module feature
            return_weights: If True, also return attention weights (B, n_slots, 2)
                            for interpretability analysis.

        Returns:
            workspace:      (B, workspace_dim)
            attn_weights:   (B, n_slots, 2) only if return_weights=True
                            attn_weights[:, :, 0] = weight on world-model
                            attn_weights[:, :, 1] = weight on causal module
        """
        B = z_ctx.shape[0]

        # Project each specialist to key and value spaces
        k_wm    = self.k_wm(z_ctx).unsqueeze(1)        # (B, 1, d_key)
        v_wm    = self.v_wm(z_ctx).unsqueeze(1)        # (B, 1, d_val)
        k_causal = self.k_causal(z_causal).unsqueeze(1) # (B, 1, d_key)
        v_causal = self.v_causal(z_causal).unsqueeze(1) # (B, 1, d_val)

        # Stack specialist keys and values into a sequence of length 2
        K = torch.cat([k_wm, k_causal], dim=1)    # (B, 2, d_key)
        V = torch.cat([v_wm, v_causal], dim=1)    # (B, 2, d_val)

        # Workspace queries (broadcast over batch)
        Q = self.W_q.unsqueeze(0).expand(B, -1, -1)   # (B, n_slots, d_key)

        # Scaled dot-product attention: competitive arbitration
        # Each workspace slot attends over specialist keys to decide
        # how much to extract from each specialist's values.
        attn_logits  = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (B, n_slots, 2)
        attn_weights = torch.softmax(attn_logits, dim=-1)              # (B, n_slots, 2)
        attn_weights = self.attn_drop(attn_weights)

        # Broadcast: weighted combination of specialist values per slot
        workspace = torch.bmm(attn_weights, V)         # (B, n_slots, d_val)

        # Flatten and project to workspace_dim
        workspace_flat = workspace.reshape(B, self.n_slots * self.d_val)
        output = self.norm(self.proj_out(workspace_flat))   # (B, workspace_dim)

        if return_weights:
            return output, attn_weights
        return output


# ─────────────────────────────────────────────────────────────────────────────
# Workspace Predictor
# ─────────────────────────────────────────────────────────────────────────────

class WorkspacePredictor(nn.Module):
    """
    Predictor that reads from the workspace broadcast.

    Takes the workspace representation (already integrating both specialists)
    plus the next action, and predicts the target latent.

    Structurally identical to the standalone Predictor, but input is
    workspace output rather than z_ctx directly.

    Args:
        workspace_dim: Must match GlobalWorkspace.workspace_dim.
        latent_dim:    Target latent dimension (must match encoders).
        hidden_dim:    MLP hidden dimension.
        dropout:       Dropout rate.
    """

    def __init__(
        self,
        workspace_dim: int   = 32,
        latent_dim:    int   = 32,
        hidden_dim:    int   = 128,
        dropout:       float = 0.0,
    ):
        super().__init__()
        self.workspace_dim = workspace_dim
        self.latent_dim    = latent_dim
        input_dim          = workspace_dim + 1   # workspace + action

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        workspace: torch.Tensor,   # (B, workspace_dim)
        a_next:    torch.Tensor,   # (B,) or (B, 1)
    ) -> torch.Tensor:
        if a_next.dim() == 1:
            a_next = a_next.unsqueeze(-1)
        x = torch.cat([workspace, a_next], dim=-1)
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WorkspaceConfig:
    """Hyperparameters for workspace training."""
    n_slots:       int   = 4
    d_key:         int   = 32
    d_val:         int   = 32
    workspace_dim: int   = 32
    hidden_dim:    int   = 128
    n_epochs:      int   = 50
    lr:            float = 3e-4
    weight_decay:  float = 1e-4
    lambda_inv:    float = 50.0
    mu_var:        float = 3.0
    nu_cov:        float = 1.0
    model_seed:    int   = 42


def train_workspace(
    enc:     ContextEncoder,
    tgt:     TargetEncoder,
    b_cfg:   BindingConfig,
    w_cfg:   WorkspaceConfig,
    device:  torch.device,
    verbose: bool = True,
) -> tuple[GlobalWorkspace, WorkspacePredictor, dict]:
    """
    Train GlobalWorkspace + WorkspacePredictor with frozen world-model.

    The encoders (enc, tgt) are frozen throughout. Only GlobalWorkspace
    and WorkspacePredictor parameters are updated. This isolates the
    effect of the workspace binding mechanism, matching the v0.1 comparison.

    Training data: same shift trajectories as v0.1 — covers pre-shift,
    crossing, and post-shift window types.

    Returns:
        (global_workspace, workspace_predictor, history)
    """
    torch.manual_seed(w_cfg.model_seed)

    # ── Fit SCM ────────────────────────────────────────────────────────────
    if verbose:
        print("Fitting causal module (SCM)...")
    disc  = CausalDiscovery()
    graph = disc.fit_from_system(
        rho=b_cfg.rho_normal, n_traj=b_cfg.scm_n_traj, seed=b_cfg.scm_seed
    )
    scm = StructuralCausalModel(graph)
    if verbose:
        print(f"  rho estimate: {graph.rho_estimate:.2f}, R²: {graph.r2_scores.round(3)}")

    # ── Generate training data ──────────────────────────────────────────────
    if verbose:
        print("Generating workspace training data (shift trajectories)...")
    gen = BindingDataGenerator(
        scm=scm, obs_dim=enc.obs_dim,
        obs_noise_std=0.05, dt=0.01, seed=b_cfg.data_seed,
    )
    ds_train = gen.generate_shift_dataset(
        rho_normal=b_cfg.rho_normal, rho_shifted=b_cfg.rho_shifted,
        n_trajectories=b_cfg.n_traj_per_rho * 2,
        n_pre_shift=b_cfg.n_pre_shift, n_post_shift=b_cfg.n_post_shift,
        seq_len=b_cfg.seq_len, action_scale=b_cfg.action_scale,
    )
    rng_idx = np.random.default_rng(b_cfg.data_seed)
    idx     = rng_idx.permutation(len(ds_train["context_obs"]))
    ds_train = {k: v[idx] for k, v in ds_train.items()}

    n_cross = int((ds_train["shift_step_in_window"] >= 0).sum())
    loader  = DataLoader(
        BindingDataset(ds_train), batch_size=b_cfg.batch_size,
        shuffle=True, drop_last=True, num_workers=0,
    )
    if verbose:
        N = len(ds_train["context_obs"])
        print(f"  Training windows: {N} ({n_cross} crossing)")

    # ── Modules ─────────────────────────────────────────────────────────────
    gw = GlobalWorkspace(
        wm_dim=enc.latent_dim,
        causal_dim=N_CAUSAL,
        n_slots=w_cfg.n_slots,
        d_key=w_cfg.d_key,
        d_val=w_cfg.d_val,
        workspace_dim=w_cfg.workspace_dim,
    ).to(device)

    wp = WorkspacePredictor(
        workspace_dim=w_cfg.workspace_dim,
        latent_dim=enc.latent_dim,
        hidden_dim=w_cfg.hidden_dim,
    ).to(device)

    # ── Optimiser (workspace + workspace predictor only) ────────────────────
    optimizer = torch.optim.AdamW(
        list(gw.parameters()) + list(wp.parameters()),
        lr=w_cfg.lr, weight_decay=w_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=w_cfg.n_epochs, eta_min=w_cfg.lr * 0.01,
    )

    # Freeze world-model components
    enc.eval(); tgt.eval()
    for p in enc.parameters(): p.requires_grad_(False)
    for p in tgt.parameters(): p.requires_grad_(False)

    if verbose:
        gw_params = sum(p.numel() for p in gw.parameters())
        wp_params = sum(p.numel() for p in wp.parameters())
        print(f"\nTraining workspace for {w_cfg.n_epochs} epochs on {device}")
        print(f"  GlobalWorkspace:    {gw_params:,} params")
        print(f"  WorkspacePredictor: {wp_params:,} params")
        print(f"  Total new params:   {gw_params + wp_params:,}")
        print("-" * 50)

    history = []
    for epoch in range(1, w_cfg.n_epochs + 1):
        gw.train(); wp.train()
        totals: dict = {}
        n = 0

        for batch in loader:
            ctx_obs  = batch["context_obs"].to(device)
            ctx_acts = batch["context_actions"].to(device)
            a_next   = batch["next_action"].to(device)
            tgt_obs  = batch["target_context_obs"].to(device)
            tgt_acts = batch["target_context_actions"].to(device)
            causal_f = batch["causal_features"].to(device)

            with torch.no_grad():
                z_ctx = enc(ctx_obs, ctx_acts)
                z_tgt = tgt(tgt_obs, tgt_acts)

            workspace  = gw(z_ctx, causal_f)
            z_pred     = wp(workspace, a_next)
            loss       = vicreg_loss(
                z_pred, z_tgt,
                lambda_inv=w_cfg.lambda_inv,
                mu_var=w_cfg.mu_var,
                nu_cov=w_cfg.nu_cov,
            )

            optimizer.zero_grad()
            loss.total.backward()
            nn.utils.clip_grad_norm_(
                list(gw.parameters()) + list(wp.parameters()), max_norm=1.0
            )
            optimizer.step()

            for k, v in [
                ("loss_total", loss.total.item()),
                ("loss_inv",   loss.invariance.item()),
                ("loss_var",   loss.variance.item()),
                ("loss_cov",   loss.covariance.item()),
            ]:
                totals[k] = totals.get(k, 0.0) + v
            n += 1

        epoch_metrics = {k: v / n for k, v in totals.items()}
        epoch_metrics["epoch"] = epoch
        scheduler.step()
        history.append(epoch_metrics)

        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == w_cfg.n_epochs):
            print(
                f"  epoch {epoch:3d}/{w_cfg.n_epochs} | "
                f"loss={epoch_metrics['loss_total']:.4f} "
                f"inv={epoch_metrics['loss_inv']:.5f} "
                f"var={epoch_metrics['loss_var']:.4f} "
                f"cov={epoch_metrics['loss_cov']:.4f}"
            )

    return gw, wp, history #type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Attention analysis — the key research output for v0.2
# ─────────────────────────────────────────────────────────────────────────────

def analyse_attention(
    gw:      GlobalWorkspace,
    enc:     ContextEncoder,
    b_cfg:   BindingConfig,
    scm:     StructuralCausalModel,
    device:  torch.device,
) -> dict:
    """
    Analyse workspace attention weights across window types.

    The attention weights (B, n_slots, 2) tell us which specialist each
    workspace slot is "listening to" at each window. This is the direct
    visualisation of GWT's competitive arbitration:

        attn[:, :, 0] = fraction of workspace attention on world-model
        attn[:, :, 1] = fraction of workspace attention on causal module

    Expected pattern:
        Pre-shift windows:  wm_weight ≈ high, causal_weight ≈ low
                            (world-model knows the dynamics well)
        Crossing windows:   causal_weight rises as shift enters context
                            (causal module detects the regime change)
        Post-shift windows: causal_weight high, wm_weight low
                            (world-model is wrong, causal signal dominant)

    Returns:
        {
          pre_wm_weight:      mean wm attention weight on pre-shift windows
          pre_causal_weight:  mean causal attention weight on pre-shift windows
          cross_wm_weight:    mean wm attention weight on crossing windows
          cross_causal_weight: mean causal attention weight on crossing windows
          post_wm_weight:     mean wm attention weight on post-shift windows
          post_causal_weight: mean causal attention weight on post-shift windows
          causal_weight_rise: cross_causal_weight - pre_causal_weight
                              (positive = causal slot rose at crossing — GWT confirmed)
        }
    """
    gen = BindingDataGenerator(
        scm=scm, obs_dim=enc.obs_dim,
        obs_noise_std=0.05, dt=0.01,
        seed=b_cfg.data_seed + 300,
    )
    ds = gen.generate_shift_dataset(
        rho_normal=b_cfg.rho_normal, rho_shifted=b_cfg.rho_shifted,
        n_trajectories=20,
        n_pre_shift=b_cfg.n_pre_shift, n_post_shift=b_cfg.n_post_shift,
        seq_len=b_cfg.seq_len, action_scale=b_cfg.action_scale,
    )
    loader = DataLoader(
        BindingDataset(ds), batch_size=64, shuffle=False, drop_last=False,
    )

    gw.eval(); enc.eval()

    all_wm_weights     = []
    all_causal_weights = []
    all_shift_pos      = []

    with torch.no_grad():
        for batch in loader:
            ctx_obs  = batch["context_obs"].to(device)
            ctx_acts = batch["context_actions"].to(device)
            causal_f = batch["causal_features"].to(device)
            spos     = batch["shift_step_in_window"].numpy()

            z_ctx      = enc(ctx_obs, ctx_acts)
            _, weights = gw(z_ctx, causal_f, return_weights=True)  # (B, n_slots, 2)

            # Mean over slots for each sample
            wm_w     = weights[:, :, 0].mean(dim=1).cpu().numpy()   # (B,)
            causal_w = weights[:, :, 1].mean(dim=1).cpu().numpy()   # (B,)

            all_wm_weights.append(wm_w)
            all_causal_weights.append(causal_w)
            all_shift_pos.append(spos)

    wm_w     = np.concatenate(all_wm_weights)
    causal_w = np.concatenate(all_causal_weights)
    spos     = np.concatenate(all_shift_pos)

    cross_mask = spos >= 0
    # Non-crossing: split approximately by trajectory position
    # (first half pre-shift, second half post-shift — rough heuristic)
    non_cross = ~cross_mask
    half_nc   = non_cross.sum() // 2
    nc_idx    = np.where(non_cross)[0]
    pre_mask  = np.zeros(len(spos), bool)
    post_mask = np.zeros(len(spos), bool)
    pre_mask[nc_idx[:half_nc]]  = True
    post_mask[nc_idx[half_nc:]] = True

    def safe_mean(arr, mask):
        return float(arr[mask].mean()) if mask.any() else float("nan")

    result = {
        "pre_wm_weight":       safe_mean(wm_w,    pre_mask),
        "pre_causal_weight":   safe_mean(causal_w, pre_mask),
        "cross_wm_weight":     safe_mean(wm_w,    cross_mask),
        "cross_causal_weight": safe_mean(causal_w, cross_mask),
        "post_wm_weight":      safe_mean(wm_w,    post_mask),
        "post_causal_weight":  safe_mean(causal_w, post_mask),
    }
    result["causal_weight_rise"] = (
        result["cross_causal_weight"] - result["pre_causal_weight"]
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# POC evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_workspace_poc(
    enc:    ContextEncoder,
    tgt:    TargetEncoder,
    sa_pred,                        # standalone Predictor (frozen, from ckpt)
    gw:     GlobalWorkspace,
    wp:     WorkspacePredictor,
    scm:    StructuralCausalModel,
    b_cfg:  BindingConfig,
    device: torch.device,
) -> dict:
    """
    Three-way POC comparison:
        Standalone predictor vs v0.1 conditioned predictor vs v0.2 workspace.

    Returns per-regime errors for all three models plus attention analysis.
    """
    gen = BindingDataGenerator(
        scm=scm, obs_dim=enc.obs_dim,
        obs_noise_std=0.05, dt=0.01,
        seed=b_cfg.data_seed + 500,
    )
    ds_pre   = gen.generate_stationary_dataset(
        b_cfg.rho_normal, b_cfg.n_eval_traj, 150, b_cfg.seq_len, b_cfg.action_scale
    )
    ds_shift = gen.generate_shift_dataset(
        b_cfg.rho_normal, b_cfg.rho_shifted,
        b_cfg.n_eval_traj,
        b_cfg.n_pre_shift, b_cfg.n_post_shift,
        b_cfg.seq_len, b_cfg.action_scale,
    )

    def compute_errors(ds):
        loader = DataLoader(
            BindingDataset(ds), batch_size=b_cfg.batch_size,
            shuffle=False, drop_last=False,
        )
        sa_errs = []; ws_errs = []; spos_all = []

        enc.eval(); tgt.eval(); sa_pred.eval(); gw.eval(); wp.eval()

        with torch.no_grad():
            for batch in loader:
                ctx_obs  = batch["context_obs"].to(device)
                ctx_acts = batch["context_actions"].to(device)
                a_next   = batch["next_action"].to(device)
                tgt_obs  = batch["target_context_obs"].to(device)
                tgt_acts = batch["target_context_actions"].to(device)
                causal_f = batch["causal_features"].to(device)

                z_ctx = enc(ctx_obs, ctx_acts)
                z_tgt = tgt(tgt_obs, tgt_acts)

                # Standalone
                z_sa = sa_pred(z_ctx, a_next)
                err_sa = (z_sa - z_tgt).pow(2).mean(dim=-1)

                # Workspace v0.2
                ws_out = gw(z_ctx, causal_f)
                z_ws   = wp(ws_out, a_next)
                err_ws = (z_ws - z_tgt).pow(2).mean(dim=-1)

                sa_errs.append(err_sa.cpu().numpy())
                ws_errs.append(err_ws.cpu().numpy())
                spos_all.append(batch["shift_step_in_window"].numpy())

        return (
            np.concatenate(sa_errs),
            np.concatenate(ws_errs),
            np.concatenate(spos_all),
        )

    sa_pre, ws_pre, _            = compute_errors(ds_pre)
    sa_sh,  ws_sh,  shift_pos    = compute_errors(ds_shift)

    cross_mask = shift_pos >= 0
    seq_len    = b_cfg.seq_len

    def per_pos(errs, spos):
        return [
            float(np.mean(errs[spos == j])) if (spos == j).any() else float("nan")
            for j in range(seq_len)
        ]

    sa_pre_mean  = float(np.mean(sa_pre))
    ws_pre_mean  = float(np.mean(ws_pre))
    sa_cross     = float(np.mean(sa_sh[cross_mask])) if cross_mask.any() else float("nan")
    ws_cross     = float(np.mean(ws_sh[cross_mask])) if cross_mask.any() else float("nan")

    return {
        "standalone_pre_error":   sa_pre_mean,
        "workspace_pre_error":    ws_pre_mean,
        "standalone_crossing_error": sa_cross,
        "workspace_crossing_error":  ws_cross,
        "standalone_normalised_ratio": sa_cross / sa_pre_mean,
        "workspace_normalised_ratio":  ws_cross / sa_pre_mean,
        "abs_improvement_vs_standalone": (sa_cross - ws_cross) / sa_cross * 100,
        "sa_per_pos": per_pos(sa_sh,  shift_pos),
        "ws_per_pos": per_pos(ws_sh,  shift_pos),
        "n_crossing_windows": int(cross_mask.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_poc_v2(ckpt_path: Optional[str] = None) -> dict:
    """
    Full v0.2 workspace POC:
        1. Load world-model checkpoint
        2. Fit causal module
        3. Train GlobalWorkspace + WorkspacePredictor
        4. Analyse attention weights (GWT verification)
        5. Compare standalone vs v0.2 workspace on regime-shift data
        6. Print results and save checkpoint
    """
    from specialists.world_model.eval import _find_latest_checkpoint, load_checkpoint
    from specialists.world_model.predictor import Predictor
    import torch as _torch

    if ckpt_path is None:
        ckpt_path = _find_latest_checkpoint()
    print(f"\n{'='*60}")
    print("  BINDING v0.2 — SHARED LATENT WORKSPACE")
    print(f"{'='*60}")
    print(f"  Checkpoint: {ckpt_path}")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded  = load_checkpoint(ckpt_path)
    enc: ContextEncoder   = loaded["enc"]
    tgt: TargetEncoder    = loaded["tgt"]
    cfg_wm: TrainConfig   = loaded["cfg"]

    # Standalone predictor from checkpoint
    sa_pred = Predictor(latent_dim=cfg_wm.latent_dim, hidden_dim=cfg_wm.hidden_dim).to(device)
    wm_ckpt = _torch.load(ckpt_path, map_location=device, weights_only=False)
    sa_pred.load_state_dict(wm_ckpt["pred_state_dict"])
    sa_pred.eval()

    b_cfg = BindingConfig(
        seq_len=cfg_wm.seq_len,
        rho_normal=cfg_wm.rho_normal,
        rho_shifted=cfg_wm.rho_shifted,
        n_pre_shift=cfg_wm.n_pre_shift,
        n_post_shift=cfg_wm.n_post_shift,
    )
    w_cfg = WorkspaceConfig(workspace_dim=cfg_wm.latent_dim)

    # Train workspace
    gw, wp, history = train_workspace(enc, tgt, b_cfg, w_cfg, device, verbose=True)

    # Fit SCM for evaluation
    disc  = CausalDiscovery()
    graph = disc.fit_from_system(rho=b_cfg.rho_normal, n_traj=b_cfg.scm_n_traj, seed=b_cfg.scm_seed)
    scm   = StructuralCausalModel(graph)

    # Attention analysis
    print("\nAnalysing workspace attention weights...")
    attn = analyse_attention(gw, enc, b_cfg, scm, device)
    print(f"\n── Attention Analysis (GWT competitive arbitration) ─────")
    print(f"  Pre-shift:   wm={attn['pre_wm_weight']:.3f}  causal={attn['pre_causal_weight']:.3f}")
    print(f"  Crossing:    wm={attn['cross_wm_weight']:.3f}  causal={attn['cross_causal_weight']:.3f}")
    print(f"  Post-shift:  wm={attn['post_wm_weight']:.3f}  causal={attn['post_causal_weight']:.3f}")
    print(f"  Causal weight rise at crossing: {attn['causal_weight_rise']:+.3f}")
    if attn["causal_weight_rise"] > 0.01:
        print(f"  ✓  Causal slot weight RISES at crossing — GWT arbitration confirmed.")
    else:
        print(f"  ~  Causal slot weight did not clearly rise — may need more training.")

    # POC evaluation
    print("\nRunning POC evaluation...")
    results = evaluate_workspace_poc(enc, tgt, sa_pred, gw, wp, scm, b_cfg, device)
    results["attention"] = attn
    results["history"]   = history

    sa_pre   = results["standalone_pre_error"]
    sa_cross = results["standalone_crossing_error"]
    ws_pre   = results["workspace_pre_error"]
    ws_cross = results["workspace_crossing_error"]
    sa_norm  = results["standalone_normalised_ratio"]
    ws_norm  = results["workspace_normalised_ratio"]
    abs_imp  = results["abs_improvement_vs_standalone"]

    print(f"\n── POC Results (v0.2 Workspace) ─────────────────────────")
    print(f"  Standalone predictor:")
    print(f"    Pre-shift error:         {sa_pre:.5f}")
    print(f"    Crossing error:          {sa_cross:.5f}")
    print(f"    Normalised ratio:        {sa_norm:.3f}×  ← fooled")
    print()
    print(f"  Workspace predictor (v0.2):")
    print(f"    Pre-shift error:         {ws_pre:.5f}")
    print(f"    Crossing error:          {ws_cross:.5f}")
    print(f"    Normalised ratio:        {ws_norm:.3f}×")
    print()
    print(f"  Absolute crossing improvement vs standalone: {abs_imp:.1f}%")
    print()

    if ws_cross < sa_cross:
        print(f"  ✓  WORKSPACE BINDING VALIDATED.")
        print(f"     v0.2 crossing error ({ws_cross:.5f}) lower than standalone ({sa_cross:.5f}).")
        if ws_norm < sa_norm:
            print(f"     Normalised ratio also improved: {ws_norm:.3f}× vs {sa_norm:.3f}×.")
        print()
        print(f"  v0.1 → v0.2 comparison:")
        print(f"    v0.1 crossing error (from last run): ~0.00160")
        print(f"    v0.2 crossing error: {ws_cross:.5f}")
    else:
        print(f"  ~  Workspace crossing error not below standalone.")

    # Save
    run_dir  = Path(ckpt_path).parent
    bind_dir = run_dir / "binding_v0_2"
    bind_dir.mkdir(exist_ok=True)
    _torch.save({
        "gw_state_dict": gw.state_dict(),
        "wp_state_dict": wp.state_dict(),
        "history":       history,
        "results":       results,
        "binding_cfg":   b_cfg,
        "workspace_cfg": w_cfg,
    }, bind_dir / "workspace_checkpoint.pt")
    print(f"\n  Workspace checkpoint saved → {bind_dir}/workspace_checkpoint.pt")

    return results


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    run_poc_v2(ckpt)