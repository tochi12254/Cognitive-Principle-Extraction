"""
Emergent Binding Experiment.

From the CPE thesis:
    "An alternative path if binding is emergent rather than engineered:
     construct a task where specialist A and specialist B receive genuinely
     conflicting partial signals, and reward flows only to one integrated
     decision rather than to either specialist individually. This scarcity
     structure might produce integration because it arguably does biologically,
     and the J-space finding suggests it may be sufficient to produce it
     artificially without a hand-built conditioning mechanism at all."

The question this experiment answers:
    Does integration structure emerge from training pressure alone, without
    any architectural prescription about HOW to combine the specialists?

Design:
    - EmergentBindingModule: a deliberately unstructured MLP.
      No attention, no gating, no prescribed combination mechanism.
      Input: [z_ctx ∥ a_next ∥ causal_features]. Architecture: wide + deep
      feedforward (hidden_dim=256, 4 layers, GELU only, no LayerNorm).
      Whatever internal structure develops comes purely from gradient pressure.

    - Conflicting-signal training: shift trajectories where Specialist A's
      z_ctx (from the world-model encoder, trained on rho=28) says "predict
      rho=28 dynamics" while Specialist B's causal signal says "rho has
      changed to 60." Only the final integrated prediction gets a loss signal.

    - Reward flows only to integrated decision: no per-specialist loss —
      only VICReg on the final output. The module must learn which specialist
      to trust at each window type purely from whether its integrated output
      matches z_tgt.

Interpretability analysis (the key research output):
    After training, we probe whether the EmergentBindingModule developed
    internal structure — specifically, whether it learned to up-weight the
    causal signal at crossing windows where it conflicts with z_ctx.

    Three probes (all applied to the TRAINED module):

    1. Jacobian analysis: ||d(output)/d(causal_feature)|| per window type.
       If this norm is higher at crossing than pre-shift, the module learned
       to make its output more sensitive to the causal signal precisely when
       z_ctx conflicts with it. This is the adaptive influence probe.

    2. Linear probe: train a logistic classifier on hidden activations to
       predict window type (pre/crossing/post). Accuracy > 33% (chance) →
       the module developed internally differentiated representations by
       window type. This is the emergent structure probe.

    3. Activation analysis: mean hidden activation magnitude by window type.
       Systematic differences → the module responds differently to each regime.

Connection to J-space finding (Anthropic, July 2026):
    The J-space finding showed workspace-like structure emerging unprompted
    inside a single language model during ordinary next-token prediction.
    We ask the analogous question: does integration-like structure emerge
    inside a blank MLP trained on a two-specialist conflicting-signal task?

    "Whether integration-like structure emerges under that pressure is a
     result worth having either way." — source thesis

Expected result:
    Functional binding (≈48% crossing improvement) is likely, since even a
    simple MLP with access to both inputs can learn to use the causal signal.
    The question is whether the INTERNAL MECHANISM matches what we'd expect
    from competitive arbitration: rising causal influence at crossing time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from integration.binding_v0_1_conditioning import (
    BindingConfig,
    BindingDataGenerator,
    BindingDataset,
    N_CAUSAL,
)
from specialists.causal_module.discovery import CausalDiscovery
from specialists.causal_module.scm import StructuralCausalModel
from specialists.world_model.encoder import ContextEncoder
from specialists.world_model.losses import vicreg_loss
from specialists.world_model.target_encoder import TargetEncoder
from specialists.world_model.train import TrainConfig


# ─────────────────────────────────────────────────────────────────────────────
# Emergent binding module — deliberately unstructured
# ─────────────────────────────────────────────────────────────────────────────

class EmergentBindingModule(nn.Module):
    """
    Deliberately unstructured integration network.

    No attention, no gating, no prescribed combination mechanism.
    Architecture: wide feedforward (hidden_dim=256, n_layers=4, GELU).
    No LayerNorm — LayerNorm imposes normalisation structure.
    No residual connections — residuals bias toward identity mappings.

    Whatever internal structure develops emerges purely from gradient
    pressure on the conflicting-signal training task.

    Why wider than v0.1/v0.2 (148K vs 15K params):
        The v0.1 ConditionedPredictor is prescribed (MLP with LayerNorm).
        The v0.2 GlobalWorkspace is prescribed (cross-attention).
        The EmergentBindingModule is given MORE capacity and LESS structure,
        so any internal organisation that emerges is entirely gradient-driven,
        not architecturally incentivised.

    Args:
        latent_dim:  Must match the frozen ContextEncoder latent_dim.
        hidden_dim:  MLP hidden dimension. Default 256 (deliberately larger
                     than v0.1/v0.2 to give structure room to emerge).
        n_layers:    Number of hidden layers. Default 4.
        n_causal:    Causal feature dimension (default 1).
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        n_layers:   int = 4,
        n_causal:   int = N_CAUSAL,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_causal   = n_causal

        # Input: [z_ctx ∥ a_next ∥ causal_features]
        input_dim = latent_dim + 1 + n_causal

        # Build flat MLP — no LayerNorm, no residuals
        dims   = [input_dim] + [hidden_dim] * n_layers + [latent_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

        # Store layer indices for hidden activation extraction
        # (every other layer after the first is a GELU, every other is Linear)
        self._n_hidden = n_layers

    def forward(
        self,
        z_ctx:           torch.Tensor,   # (B, latent_dim)
        a_next:          torch.Tensor,   # (B,) or (B, 1)
        causal_features: torch.Tensor,   # (B, n_causal)
    ) -> torch.Tensor:
        if a_next.dim() == 1:
            a_next = a_next.unsqueeze(-1)
        x = torch.cat([z_ctx, a_next, causal_features], dim=-1)
        return self.net(x)

    def hidden_activations(
        self,
        z_ctx:           torch.Tensor,
        a_next:          torch.Tensor,
        causal_features: torch.Tensor,
        layer_idx:       int = -2,       # -2 = penultimate hidden layer
    ) -> torch.Tensor:
        """
        Return hidden activations from a specified layer index
        (for linear probe and activation analysis).

        layer_idx=-2 returns the penultimate hidden layer output.
        This is the most informative layer for probing emergent structure —
        it's the last internal representation before the output projection.
        """
        if a_next.dim() == 1:
            a_next = a_next.unsqueeze(-1)
        x = torch.cat([z_ctx, a_next, causal_features], dim=-1)

        # Collect outputs of each Linear layer's GELU output
        activations = []
        h = x
        for module in self.net:
            h = module(h)
            if isinstance(module, nn.GELU):
                activations.append(h)

        idx = layer_idx if layer_idx >= 0 else len(activations) + layer_idx
        return activations[max(0, min(idx, len(activations) - 1))]


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_emergent(
    enc:    ContextEncoder,
    tgt:    TargetEncoder,
    b_cfg:  BindingConfig,
    device: torch.device,
    n_epochs: int = 50,
    lr:       float = 3e-4,
    verbose:  bool = True,
) -> tuple[EmergentBindingModule, dict]:
    """
    Train EmergentBindingModule with frozen world-model on shift trajectories.

    The only loss is VICReg on the final integrated prediction — no
    per-specialist loss. The module must learn which input to trust at each
    window type from the gradient signal alone.

    Returns: (module, history)
    """
    import torch.optim as optim
    from torch.optim.lr_scheduler import CosineAnnealingLR

    torch.manual_seed(0)

    # ── SCM ──────────────────────────────────────────────────────────────────
    if verbose:
        print("Fitting causal module (SCM)...")
    disc  = CausalDiscovery()
    graph = disc.fit_from_system(
        rho=b_cfg.rho_normal, n_traj=b_cfg.scm_n_traj, seed=b_cfg.scm_seed
    )
    scm = StructuralCausalModel(graph)
    if verbose:
        print(f"  rho estimate: {graph.rho_estimate:.2f}, R²: {graph.r2_scores.round(3)}")

    # ── Data (shift trajectories — all three window types) ─────────────────
    if verbose:
        print("Generating training data (shift trajectories)...")
    gen = BindingDataGenerator(
        scm=scm, obs_dim=enc.obs_dim,
        obs_noise_std=0.05, dt=0.01, seed=b_cfg.data_seed + 200,
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

    # ── Module ────────────────────────────────────────────────────────────────
    ebm = EmergentBindingModule(
        latent_dim=enc.latent_dim,
        hidden_dim=256,
        n_layers=4,
        n_causal=N_CAUSAL,
    ).to(device)

    optimizer = optim.AdamW(ebm.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    # Freeze world-model
    enc.eval(); tgt.eval()
    for p in enc.parameters(): p.requires_grad_(False)
    for p in tgt.parameters(): p.requires_grad_(False)

    if verbose:
        n_params = sum(p.numel() for p in ebm.parameters())
        print(f"\nTraining EmergentBindingModule for {n_epochs} epochs on {device}")
        print(f"  Parameters: {n_params:,}  (deliberately larger — no structural prescription)")
        print(f"  Architecture: 4-layer MLP, hidden_dim=256, GELU only, no LayerNorm")
        print("-" * 55)

    history = []
    for epoch in range(1, n_epochs + 1):
        ebm.train()
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

            z_pred = ebm(z_ctx, a_next, causal_f)
            loss   = vicreg_loss(z_pred, z_tgt,
                                 lambda_inv=50.0, mu_var=3.0, nu_cov=1.0)

            optimizer.zero_grad()
            loss.total.backward()
            nn.utils.clip_grad_norm_(ebm.parameters(), max_norm=1.0)
            optimizer.step()

            for k, v in [("loss_total", loss.total.item()),
                          ("loss_inv", loss.invariance.item()),
                          ("loss_var", loss.variance.item()),
                          ("loss_cov", loss.covariance.item())]:
                totals[k] = totals.get(k, 0.0) + v
            n += 1

        epoch_metrics = {k: v / n for k, v in totals.items()}
        epoch_metrics["epoch"] = epoch
        scheduler.step()
        history.append(epoch_metrics)

        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == n_epochs):
            print(
                f"  epoch {epoch:3d}/{n_epochs} | "
                f"loss={epoch_metrics['loss_total']:.4f} "
                f"inv={epoch_metrics['loss_inv']:.5f} "
                f"var={epoch_metrics['loss_var']:.4f}"
            )

    return ebm, history #type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Interpretability analysis — the key research output
# ─────────────────────────────────────────────────────────────────────────────

def analyse_emergent_structure(
    ebm:    EmergentBindingModule,
    enc:    ContextEncoder,
    b_cfg:  BindingConfig,
    scm:    StructuralCausalModel,
    device: torch.device,
    n_probe_samples: int = 200,
) -> dict:
    """
    Probe whether the trained EmergentBindingModule developed internal
    structure that responds to specialist conflict.

    Three analyses:

    1. Jacobian analysis — adaptive causal influence:
       For each window, compute ||d(output)/d(causal_feature)||.
       If this is systematically higher at crossing than pre-shift, the
       module learned to be MORE sensitive to the causal signal precisely
       when it conflicts with z_ctx. This is adaptive influence — the
       closest analogue to competitive arbitration in an unstructured module.

    2. Linear probe — emergent differentiation:
       Train a logistic classifier on hidden activations to predict window
       type (pre=0 / crossing=1 / post=2). Accuracy > 33% (chance baseline)
       → the module developed internally differentiated representations
       by window type. This is the emergent structure test.

    3. Activation analysis — response magnitude:
       Mean ||hidden_activation|| by window type. Systematic differences
       → the module responds to each window type with different activation
       energy, showing it "knows" which regime it is in.

    Returns:
        {
          # Jacobian analysis
          jacobian_pre_mean:      mean ||d(out)/d(causal)|| on pre-shift
          jacobian_cross_mean:    mean ||d(out)/d(causal)|| on crossing
          jacobian_post_mean:     mean ||d(out)/d(causal)|| on post-shift
          jacobian_cross_rise:    cross - pre (positive = adaptive influence)

          # Linear probe
          probe_accuracy:         accuracy of window-type classifier (0-1)
          probe_chance:           1/3 (chance baseline)
          probe_above_chance:     bool — probe_accuracy > probe_chance + 0.05

          # Activation analysis
          activation_pre_mean:    mean hidden activation norm on pre-shift
          activation_cross_mean:  mean hidden activation norm on crossing
          activation_post_mean:   mean hidden activation norm on post-shift

          # Emergent structure verdict
          adaptive_influence:     bool — jacobian_cross_rise > 0.001
          emergent_differentiation: bool — probe_above_chance
        }
    """
    # Generate eval data
    gen = BindingDataGenerator(
        scm=scm, obs_dim=enc.obs_dim,
        obs_noise_std=0.05, dt=0.01,
        seed=b_cfg.data_seed + 400,
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

    ebm.eval(); enc.eval()

    # Collect data for analysis
    all_z_ctx     = []
    all_a_next    = []
    all_causal    = []
    all_shift_pos = []
    all_z_tgt     = []
    all_tgt_obs   = []
    all_tgt_acts  = []

    with torch.no_grad():
        for batch in loader:
            ctx_obs  = batch["context_obs"].to(device)
            ctx_acts = batch["context_actions"].to(device)
            a_next   = batch["next_action"].to(device)
            causal_f = batch["causal_features"].to(device)
            tgt_obs  = batch["target_context_obs"]
            tgt_acts = batch["target_context_actions"]

            z_ctx = enc(ctx_obs, ctx_acts)
            all_z_ctx.append(z_ctx.cpu())
            all_a_next.append(a_next.cpu())
            all_causal.append(causal_f.cpu())
            all_shift_pos.append(batch["shift_step_in_window"])
            all_tgt_obs.append(tgt_obs)
            all_tgt_acts.append(tgt_acts)

    z_ctx_all  = torch.cat(all_z_ctx)
    a_all      = torch.cat(all_a_next)
    causal_all = torch.cat(all_causal)
    spos_all   = torch.cat(all_shift_pos).numpy()

    cross_mask = spos_all >= 0
    nc_idx     = np.where(~cross_mask)[0]
    half_nc    = len(nc_idx) // 2
    pre_mask   = np.zeros(len(spos_all), bool)
    post_mask  = np.zeros(len(spos_all), bool)
    pre_mask[nc_idx[:half_nc]]  = True
    post_mask[nc_idx[half_nc:]] = True

    # Subsample for Jacobian (expensive)
    def subsample(mask, n=n_probe_samples):
        idx = np.where(mask)[0]
        if len(idx) > n:
            idx = np.random.default_rng(0).choice(idx, n, replace=False)
        return idx

    pre_idx   = subsample(pre_mask)
    cross_idx = subsample(cross_mask)
    post_idx  = subsample(post_mask)

    # ── 1. Jacobian analysis ──────────────────────────────────────────────────
    def jacobian_causal_norms(indices):
        norms = []
        for i in indices:
            z  = z_ctx_all[i:i+1].to(device)
            a  = a_all[i:i+1].to(device)
            cf = causal_all[i:i+1].to(device).requires_grad_(True)
            out  = ebm(z, a, cf)
            grad = torch.autograd.grad(out.sum(), cf)[0]
            norms.append(grad.norm().item())
        return np.array(norms)

    jac_pre   = jacobian_causal_norms(pre_idx)
    jac_cross = jacobian_causal_norms(cross_idx)
    jac_post  = jacobian_causal_norms(post_idx)

    # ── 2. Linear probe ────────────────────────────────────────────────────────
    # Collect hidden activations and labels
    hidden_acts = []
    labels      = []

    with torch.no_grad():
        for i, (mask, label) in enumerate(
            [(pre_mask, 0), (cross_mask, 1), (post_mask, 2)]
        ):
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            for j in range(0, len(idx), 64):
                batch_idx = idx[j:j+64]
                z  = z_ctx_all[batch_idx].to(device)
                a  = a_all[batch_idx].to(device)
                cf = causal_all[batch_idx].to(device)
                acts = ebm.hidden_activations(z, a, cf)
                hidden_acts.append(acts.cpu().numpy())
                labels.extend([label] * len(batch_idx))

    H  = np.concatenate(hidden_acts)
    L  = np.array(labels)
    sc = StandardScaler()
    H_scaled = sc.fit_transform(H)

    # Train linear probe with cross-validation (80/20 split)
    split = int(0.8 * len(H_scaled))
    perm  = np.random.default_rng(42).permutation(len(H_scaled))
    tr_idx, va_idx = perm[:split], perm[split:]

    probe = LogisticRegression(max_iter=500, C=0.1)
    probe.fit(H_scaled[tr_idx], L[tr_idx])
    probe_accuracy = probe.score(H_scaled[va_idx], L[va_idx])

    # ── 3. Activation analysis ────────────────────────────────────────────────
    def act_norms(indices):
        norms = []
        with torch.no_grad():
            for j in range(0, len(indices), 64):
                bi = indices[j:j+64]
                z  = z_ctx_all[bi].to(device)
                a  = a_all[bi].to(device)
                cf = causal_all[bi].to(device)
                acts = ebm.hidden_activations(z, a, cf)
                norms.append(acts.norm(dim=-1).cpu().numpy())
        return np.concatenate(norms) if norms else np.array([float("nan")])

    act_pre   = act_norms(np.where(pre_mask)[0])
    act_cross = act_norms(np.where(cross_mask)[0])
    act_post  = act_norms(np.where(post_mask)[0])

    # ── Results ───────────────────────────────────────────────────────────────
    jac_cross_rise  = float(jac_cross.mean()) - float(jac_pre.mean())
    probe_chance    = 1.0 / 3.0
    above_chance    = probe_accuracy > probe_chance + 0.05

    return {
        "jacobian_pre_mean":         float(jac_pre.mean()),
        "jacobian_cross_mean":       float(jac_cross.mean()),
        "jacobian_post_mean":        float(jac_post.mean()),
        "jacobian_cross_rise":       jac_cross_rise,
        "probe_accuracy":            float(probe_accuracy),
        "probe_chance":              probe_chance,
        "probe_above_chance":        above_chance,
        "activation_pre_mean":       float(act_pre.mean()),
        "activation_cross_mean":     float(act_cross.mean()),
        "activation_post_mean":      float(act_post.mean()),
        "adaptive_influence":        jac_cross_rise > 0.001,
        "emergent_differentiation":  above_chance,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POC evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_emergent_poc(
    enc:    ContextEncoder,
    tgt:    TargetEncoder,
    sa_pred,
    ebm:    EmergentBindingModule,
    scm:    StructuralCausalModel,
    b_cfg:  BindingConfig,
    device: torch.device,
) -> dict:
    """Four-way comparison: standalone / v0.1 / v0.2 / emergent."""
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
        sa_e = []; em_e = []; spos = []
        enc.eval(); tgt.eval(); sa_pred.eval(); ebm.eval()

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

                z_sa  = sa_pred(z_ctx, a_next)
                z_em  = ebm(z_ctx, a_next, causal_f)

                sa_e.append((z_sa - z_tgt).pow(2).mean(dim=-1).cpu().numpy())
                em_e.append((z_em - z_tgt).pow(2).mean(dim=-1).cpu().numpy())
                spos.append(batch["shift_step_in_window"].numpy())

        return (np.concatenate(sa_e), np.concatenate(em_e), np.concatenate(spos))

    sa_pre, em_pre, _         = compute_errors(ds_pre)
    sa_sh,  em_sh,  shift_pos = compute_errors(ds_shift)
    cross = shift_pos >= 0

    sa_pre_m  = float(np.mean(sa_pre))
    em_pre_m  = float(np.mean(em_pre))
    sa_cross  = float(np.mean(sa_sh[cross])) if cross.any() else float("nan")
    em_cross  = float(np.mean(em_sh[cross])) if cross.any() else float("nan")

    return {
        "standalone_pre_error":      sa_pre_m,
        "emergent_pre_error":        em_pre_m,
        "standalone_crossing_error": sa_cross,
        "emergent_crossing_error":   em_cross,
        "standalone_norm_ratio":     sa_cross / sa_pre_m,
        "emergent_norm_ratio":       em_cross / sa_pre_m,
        "abs_improvement":           (sa_cross - em_cross) / sa_cross * 100,
        "n_crossing_windows":        int(cross.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_emergent_poc(ckpt_path: Optional[str] = None) -> dict:
    from specialists.world_model.eval import _find_latest_checkpoint, load_checkpoint
    from specialists.world_model.predictor import Predictor
    import torch as _torch

    if ckpt_path is None:
        ckpt_path = _find_latest_checkpoint()
    print(f"\n{'='*60}")
    print("  EMERGENT BINDING EXPERIMENT")
    print(f"{'='*60}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Question: does integration structure emerge from gradient")
    print(f"  pressure alone, without architectural prescription?")

    device  = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    loaded  = load_checkpoint(ckpt_path)
    enc: ContextEncoder = loaded["enc"]
    tgt: TargetEncoder = loaded["tgt"]
    cfg_wm: TrainConfig  = loaded["cfg"]

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

    # Train emergent module
    ebm, history = train_emergent(enc, tgt, b_cfg, device, verbose=True)

    # SCM for eval
    disc  = CausalDiscovery()
    graph = disc.fit_from_system(rho=b_cfg.rho_normal, n_traj=b_cfg.scm_n_traj, seed=b_cfg.scm_seed)
    scm   = StructuralCausalModel(graph)

    # Interpretability analysis
    print("\nRunning interpretability analysis...")
    structure = analyse_emergent_structure(ebm, enc, b_cfg, scm, device)

    print(f"\n── Emergent Structure Analysis ──────────────────────────────")
    print(f"  Jacobian (causal influence ||d_out/d_causal||):")
    print(f"    Pre-shift:   {structure['jacobian_pre_mean']:.5f}")
    print(f"    Crossing:    {structure['jacobian_cross_mean']:.5f}  ← conflict point")
    print(f"    Post-shift:  {structure['jacobian_post_mean']:.5f}")
    print(f"    Rise at crossing: {structure['jacobian_cross_rise']:+.5f}")
    if structure["adaptive_influence"]:
        print(f"    ✓  Causal influence rises at crossing — adaptive weighting emerged.")
    else:
        print(f"    ~  Causal influence does not clearly rise at crossing.")
    print()
    print(f"  Linear probe (window type from hidden activations):")
    print(f"    Accuracy: {structure['probe_accuracy']:.3f}  (chance: {structure['probe_chance']:.3f})")
    if structure["emergent_differentiation"]:
        print(f"    ✓  Probe above chance — module differentiated window types internally.")
    else:
        print(f"    ~  Probe near chance — limited internal differentiation.")
    print()
    print(f"  Activation norms by window type:")
    print(f"    Pre-shift:   {structure['activation_pre_mean']:.3f}")
    print(f"    Crossing:    {structure['activation_cross_mean']:.3f}")
    print(f"    Post-shift:  {structure['activation_post_mean']:.3f}")

    # POC evaluation
    print("\nRunning POC evaluation...")
    results = evaluate_emergent_poc(enc, tgt, sa_pred, ebm, scm, b_cfg, device)
    results["structure"] = structure
    results["history"]   = history

    sa_pre   = results["standalone_pre_error"]
    sa_cross = results["standalone_crossing_error"]
    em_cross = results["emergent_crossing_error"]
    sa_norm  = results["standalone_norm_ratio"]
    em_norm  = results["emergent_norm_ratio"]
    abs_imp  = results["abs_improvement"]

    print(f"\n── POC Results (Emergent Binding) ───────────────────────────")
    print(f"  Standalone predictor:           crossing={sa_cross:.5f}  ratio={sa_norm:.3f}×")
    print(f"  Emergent binding module:         crossing={em_cross:.5f}  ratio={em_norm:.3f}×")
    print(f"  Absolute crossing improvement:   {abs_imp:.1f}%")
    print()

    print(f"── Emergent Binding Verdict ─────────────────────────────────")
    adaptive   = structure["adaptive_influence"]
    differentiated = structure["emergent_differentiation"]
    functional = em_cross < sa_cross

    if functional and adaptive and differentiated:
        print("  ✓  FULL EMERGENT BINDING CONFIRMED.")
        print("     Functional: crossing error reduced.")
        print("     Adaptive influence: causal weight rose at crossing.")
        print("     Emergent differentiation: hidden reps differ by window type.")
        print("     Integration-like structure emerged from gradient pressure alone.")
    elif functional and (adaptive or differentiated):
        print("  ✓  PARTIAL EMERGENT BINDING CONFIRMED.")
        print(f"     Functional: {'✓' if functional else '✗'}")
        print(f"     Adaptive influence: {'✓' if adaptive else '✗'}")
        print(f"     Emergent differentiation: {'✓' if differentiated else '✗'}")
    elif functional:
        print("  ~  FUNCTIONAL BINDING ONLY.")
        print("     Crossing error is reduced but no evidence of emergent")
        print("     internal structure differentiating window types.")
        print("     The module uses the causal signal but without detected arbitration.")
    else:
        print("  ✗  Binding not validated — crossing error not reduced.")

    # Save
    run_dir  = Path(ckpt_path).parent
    bind_dir = run_dir / "binding_emergent"
    bind_dir.mkdir(exist_ok=True)
    _torch.save({
        "ebm_state_dict": ebm.state_dict(),
        "history":        history,
        "results":        results,
        "binding_cfg":    b_cfg,
    }, bind_dir / "emergent_checkpoint.pt")
    print(f"\n  Checkpoint saved → {bind_dir}/emergent_checkpoint.pt")

    return results


if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    run_emergent_poc(ckpt)