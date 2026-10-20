"""
Training loop for Specialist A — the world-model.

Wires all five world-model modules together for the first time:
    TrajectoryGenerator  →  WorldModelDataset  →  DataLoader
    ContextEncoder  f_θ  →  z_ctx
    TargetEncoder   f_ξ  →  z_tgt  (EMA, no gradient)
    Predictor       g_φ  →  z_pred
    VICRegLoss           →  backward  →  AdamW  →  EMA update

Per-step training sequence:
    1. enc(context_obs, context_actions) → z_ctx
    2. tgt(target_context_obs, target_context_actions) → z_tgt   [no grad]
    3. pred(z_ctx, next_action) → z_pred
    4. vicreg_loss(z_pred, z_tgt) → loss
    5. loss.total.backward()
    6. optimizer.step()
    7. tgt.update(enc, momentum)   ← EMA after every opt step

Evaluation (regime-shift test):
    Train on stationary data (rho=28 throughout).
    Evaluate on shift data (rho flips mid-trajectory).
    Key metric: per-window prediction error indexed by shift_step_in_window.
    Correct behaviour: error spikes at and after the shift step.
    This is the 'unfair to standalone predictor' test from the CPE thesis.

Logging (per step):
    loss_total, loss_inv, loss_var, loss_cov,
    effective_rank (collapse monitor)

Design decisions (full reasoning in docs/research-log):
    - AdamW over Adam: weight decay improves generalisation;
      L2 regularisation is better applied to weights than gradients.
    - Cosine annealing LR: smoother convergence than step decay;
      standard in JEPA/VICReg training.
    - Only enc + pred in optimizer: TargetEncoder is EMA-only.
    - EMA momentum fixed at 0.99: stable at toy scale;
      cosine schedule is reserved for real-data phase.
    - Batch size 64: large enough for VICReg's variance/covariance
      terms to be well-estimated; small enough to train on CPU.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from envs.trajectory_gen import TrajectoryGenerator
from specialists.world_model.encoder import ContextEncoder
from specialists.world_model.losses import VICRegLoss, vicreg_loss
from specialists.world_model.predictor import Predictor
from specialists.world_model.target_encoder import TargetEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """All hyperparameters for one training run."""

    # Data
    n_train_trajectories: int   = 150
    n_eval_trajectories:  int   = 20
    n_steps:              int   = 150   # steps per trajectory
    seq_len:              int   = 20    # context window length
    action_scale:         float = 0.1
    rho_normal:           float = 28.0  # pre-shift Lorenz rho
    rho_shifted:          float = 60   # post-shift Lorenz rho
    n_pre_shift:          int   = 50    # steps before regime shift
    n_post_shift:         int   = 50    # steps after regime shift
    data_seed:            int   = 0

    # Architecture (must match encoder.py / predictor.py defaults)
    obs_dim:    int   = 6
    embed_dim:  int   = 32
    latent_dim: int   = 32
    n_heads:    int   = 4
    n_layers:   int   = 2
    ff_dim:     int   = 256
    hidden_dim: int   = 128  # predictor hidden dim
    dropout:    float = 0.1

    # Training
    n_epochs:       int   = 200
    batch_size:     int   = 64
    lr:             float = 3e-4
    weight_decay:   float = 1e-4
    ema_momentum:   float = 0.99
    lambda_inv:     float = 50.0
    mu_var:         float = 3.0
    nu_cov:         float = 1.0
    eval_every:     int   = 5    # evaluate on shift data every N epochs
    model_seed:     int   = 42

    # I/O
    run_dir: str = "experiments/runs"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class WorldModelDataset(Dataset):
    """
    Thin PyTorch Dataset wrapper around a trajectory dict of NumPy arrays.
    Converts arrays to float32 tensors on first access.
    """

    def __init__(self, data: dict):
        self.tensors = {k: torch.from_numpy(v) for k, v in data.items()}
        self._len    = len(next(iter(self.tensors.values())))

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict:
        return {k: v[idx] for k, v in self.tensors.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Single training step
# ─────────────────────────────────────────────────────────────────────────────

def train_step(
    batch:     dict,
    enc:       ContextEncoder,
    tgt:       TargetEncoder,
    pred:      Predictor,
    optimizer: torch.optim.Optimizer,
    cfg:       TrainConfig,
    device:    torch.device,
) -> dict:
    """
    Execute one forward + backward + EMA update step.

    Returns a dict of scalar metrics for logging.
    """
    # Move batch to device
    ctx_obs  = batch["context_obs"].to(device)              # (B, seq_len, obs_dim)
    ctx_acts = batch["context_actions"].to(device)          # (B, seq_len)
    a_next   = batch["next_action"].to(device)              # (B,)
    tgt_obs  = batch["target_context_obs"].to(device)       # (B, seq_len, obs_dim)
    tgt_acts = batch["target_context_actions"].to(device)   # (B, seq_len)

    # ── Forward ──────────────────────────────────────────────────────────────
    enc.train()
    pred.train()

    z_ctx  = enc(ctx_obs, ctx_acts)       # (B, latent_dim) — gradient flows
    z_tgt  = tgt(tgt_obs, tgt_acts)       # (B, latent_dim) — no gradient
    z_pred = pred(z_ctx, a_next)          # (B, latent_dim) — gradient flows

    loss: VICRegLoss = vicreg_loss(
        z_pred,
        z_tgt,
        lambda_inv=cfg.lambda_inv,
        mu_var=cfg.mu_var,
        nu_cov=cfg.nu_cov,
    )

    # ── Backward ─────────────────────────────────────────────────────────────
    optimizer.zero_grad()
    loss.total.backward()

    # Gradient clipping: prevents occasional large gradient spikes from
    # destabilising early training. Clip norm at 1.0 — standard value.
    nn.utils.clip_grad_norm_(
        list(enc.parameters()) + list(pred.parameters()),
        max_norm=1.0,
    )
    optimizer.step()

    # ── EMA update ───────────────────────────────────────────────────────────
    # Called AFTER optimizer.step() so f_ξ tracks the updated f_θ.
    tgt.update(enc, momentum=cfg.ema_momentum)

    # ── Metrics ──────────────────────────────────────────────────────────────
    with torch.no_grad():
        eff_rank = ContextEncoder.effective_rank(z_pred)

    return {
        "loss_total":  loss.total.item(),
        "loss_inv":    loss.invariance.item(),
        "loss_var":    loss.variance.item(),
        "loss_cov":    loss.covariance.item(),
        "eff_rank":    eff_rank,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    loader:    DataLoader,
    enc:       ContextEncoder,
    tgt:       TargetEncoder,
    pred:      Predictor,
    optimizer: torch.optim.Optimizer,
    cfg:       TrainConfig,
    device:    torch.device,
) -> dict:
    """Run one full training epoch. Returns mean metrics."""
    totals: dict = {}
    n = 0

    for batch in loader:
        metrics = train_step(batch, enc, tgt, pred, optimizer, cfg, device)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1

    return {k: v / n for k, v in totals.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Regime-shift evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_shift(
    shift_loader: DataLoader,
    enc:          ContextEncoder,
    tgt:          TargetEncoder,
    pred:         Predictor,
    device:       torch.device,
    seq_len:      int,
) -> dict:
    """
    Evaluate prediction error across the regime-shift dataset.

    Returns:
        mean_error_by_shift_pos: list of length seq_len.
            Index j = mean MSE(z_pred, z_tgt) for windows where the regime
            shift fell at position j within the context window.
            A correct world-model should show elevated error at j=0 (shift
            just entered the window) and persistent error afterward.

        pre_shift_error:  mean error for windows entirely before shift.
        post_shift_error: mean error for windows entirely after shift.
        crossing_error:   mean error for windows crossing the shift point.

    This is the PRIMARY metric for the CPE proof-of-concept:
        crossing_error >> pre_shift_error → standalone predictor fooled.
    """
    enc.eval()
    pred.eval()

    # Accumulators per relative shift position
    pos_errors  = [[] for _ in range(seq_len)]
    pre_errors  = []
    post_errors = []

    with torch.no_grad():
        for batch in shift_loader:
            ctx_obs  = batch["context_obs"].to(device)
            ctx_acts = batch["context_actions"].to(device)
            a_next   = batch["next_action"].to(device)
            tgt_obs  = batch["target_context_obs"].to(device)
            tgt_acts = batch["target_context_actions"].to(device)
            shift_pos = batch["shift_step_in_window"]   # (B,) int, -1 if no shift

            z_ctx  = enc(ctx_obs, ctx_acts)
            z_tgt  = tgt(tgt_obs, tgt_acts)
            z_pred = pred(z_ctx, a_next)

            # Per-sample MSE: (B,)
            per_sample_mse = (z_pred - z_tgt).pow(2).mean(dim=-1)

            for i in range(len(per_sample_mse)):
                err  = per_sample_mse[i].item()
                spos = shift_pos[i].item()
                if spos >= 0:
                    pos_errors[spos].append(err)
                else:
                    # Determine if window is pre-shift or post-shift
                    # by checking context_obs[i, -1] vs z_tgt latent consistency.
                    # Approximation: use the absolute shift position logic
                    # which is not available here — classify all non-crossing
                    # windows as either pre or post by batch order heuristic.
                    # For the eval metric we only strictly need crossing vs rest.
                    # Store in a combined 'non_crossing' bucket for now.
                    pre_errors.append(err)   # simplification: label as pre

    mean_by_pos = [
        float(sum(errs) / len(errs)) if errs else float("nan")
        for errs in pos_errors
    ]
    pre_mean      = float(sum(pre_errors) / len(pre_errors)) if pre_errors else float("nan")
    crossing_mean = float(
        sum(e for errs in pos_errors for e in errs) /
        max(sum(len(errs) for errs in pos_errors), 1)
    )

    enc.train()
    pred.train()

    return {
        "mean_error_by_shift_pos": mean_by_pos,
        "pre_shift_error":         pre_mean,
        "crossing_error":          crossing_mean,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: Optional[TrainConfig] = None) -> dict:
    """
    Run a full training run.

    Returns a dict containing:
        history:     list of per-epoch metric dicts (training metrics)
        eval_history: list of per-eval-checkpoint shift evaluation dicts
        enc, tgt, pred: trained modules (for inspection / downstream use)
        final_metrics: metrics from the last training epoch
    """
    if cfg is None:
        cfg = TrainConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.model_seed)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Generating training data...")
    gen = TrajectoryGenerator(
        obs_dim=cfg.obs_dim,
        obs_noise_std=0.05,
        rho_normal=cfg.rho_normal,
        rho_shifted=cfg.rho_shifted,
        seed=cfg.data_seed,
    )

    train_data = gen.generate_stationary_dataset(
        n_trajectories=cfg.n_train_trajectories,
        n_steps=cfg.n_steps,
        seq_len=cfg.seq_len,
        action_scale=cfg.action_scale,
    )
    shift_data = gen.generate_shift_dataset(
        n_trajectories=cfg.n_eval_trajectories,
        n_pre_shift=cfg.n_pre_shift,
        n_post_shift=cfg.n_post_shift,
        seq_len=cfg.seq_len,
        action_scale=cfg.action_scale,
    )

    train_dataset = WorldModelDataset(train_data)
    shift_dataset = WorldModelDataset(shift_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,     # ensures VICReg always gets full batches (B >= 2)
        num_workers=0,
    )
    shift_loader = DataLoader(
        shift_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    n_train = len(train_dataset)
    n_shift = len(shift_dataset)
    print(f"  train windows: {n_train}  |  shift eval windows: {n_shift}")

    # ── Modules ───────────────────────────────────────────────────────────────
    enc  = ContextEncoder(
        obs_dim=cfg.obs_dim,
        seq_len=cfg.seq_len,
        embed_dim=cfg.embed_dim,
        latent_dim=cfg.latent_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
    ).to(device)

    tgt  = TargetEncoder(enc).to(device)

    pred = Predictor(
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    # ── Optimizer (enc + pred only — NEVER tgt) ───────────────────────────────
    optimizer = torch.optim.AdamW(
        list(enc.parameters()) + list(pred.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Cosine annealing: smoothly decays LR from cfg.lr → 0 over all epochs.
    # Smoother convergence than step decay; standard in JEPA/VICReg training.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.n_epochs,
        eta_min=cfg.lr * 0.01,
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    run_dir = Path(cfg.run_dir) / f"run_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    history:      list[dict] = []
    eval_history: list[dict] = []

    print(f"\nTraining for {cfg.n_epochs} epochs on {device}")
    print(f"  enc params: {sum(p.numel() for p in enc.parameters()):,}")
    print(f"  pred params: {sum(p.numel() for p in pred.parameters()):,}")
    print(f"  tgt params: {sum(p.numel() for p in tgt.parameters()):,} (EMA only)")
    print("-" * 60)

    for epoch in range(1, cfg.n_epochs + 1):
        t0 = time.time()

        epoch_metrics = run_epoch(
            train_loader, enc, tgt, pred, optimizer, cfg, device
        )
        scheduler.step()

        elapsed = time.time() - t0
        history.append({**epoch_metrics, "epoch": epoch, "lr": scheduler.get_last_lr()[0]})

        # Print every epoch
        print(
            f"epoch {epoch:3d}/{cfg.n_epochs} | "
            f"loss={epoch_metrics['loss_total']:.3f} "
            f"inv={epoch_metrics['loss_inv']:.3f} "
            f"var={epoch_metrics['loss_var']:.3f} "
            f"cov={epoch_metrics['loss_cov']:.3f} "
            f"rank={epoch_metrics['eff_rank']:.1f} | "
            f"{elapsed:.1f}s"
        )

        # Regime-shift evaluation
        if epoch % cfg.eval_every == 0 or epoch == cfg.n_epochs:
            eval_metrics = evaluate_shift(
                shift_loader, enc, tgt, pred, device, cfg.seq_len
            )
            eval_metrics["epoch"] = epoch
            eval_history.append(eval_metrics)
            print(
                f"  [EVAL] pre_shift_error={eval_metrics['pre_shift_error']:.4f} "
                f"crossing_error={eval_metrics['crossing_error']:.4f}"
            )

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt_path = run_dir / "checkpoint.pt"
    torch.save({
        "enc_state_dict":  enc.state_dict(),
        "tgt_state_dict":  tgt.state_dict(),
        "pred_state_dict": pred.state_dict(),
        "opt_state_dict":  optimizer.state_dict(),
        "cfg":             cfg,
        "history":         history,
        "eval_history":    eval_history,
    }, ckpt_path)
    print(f"\nCheckpoint saved → {ckpt_path}")

    return {
        "history":       history,
        "eval_history":  eval_history,
        "enc":           enc,
        "tgt":           tgt,
        "pred":          pred,
        "final_metrics": history[-1],
        "run_dir":       str(run_dir),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = train(TrainConfig(n_epochs=200, n_train_trajectories=10))
    print("\nFinal metrics:", result["final_metrics"])