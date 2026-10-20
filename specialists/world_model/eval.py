"""
Formal evaluation of Specialist A — the world-model.

Produces the primary result the CPE thesis defines as the proof-of-concept
baseline for the world-model stage:

    "The primary metric: whether the bound pair correctly adjusts at the
     engineered novel condition while the standalone predictor continues to
     be fooled by it."

At this stage (Specialist A only, before binding) we measure:
    1. Pre-shift error   — baseline MSE on stationary dynamics (rho=28)
    2. Crossing error    — MSE when regime shift falls within context window
    3. Post-shift error  — MSE after model has adapted to new dynamics (rho=0.5)
    4. Per-position error curve — MSE indexed by where in the window the
       shift occurred (rel=seq_len-1: shift just entered → expect peak error)

The expected standalone-predictor result (correct at this stage):
    crossing_error >> pre_shift_error
    per-position curve peaks at high rel values, decreases toward rel=0

This elevated crossing error IS the baseline that binding with Specialist B
will later be measured against. The causal module should help the bound pair
adjust at the shift point — reducing crossing error — while the standalone
predictor (this module alone) continues to be fooled.

Plots saved to: <run_dir>/eval/
    training_curves.png      — loss, invariance, variance, covariance, eff_rank
    regime_shift_result.png  — per-position error curve + pre/crossing/post bars

Usage:
    python -m specialists.world_model.eval <checkpoint_path>
    python -m specialists.world_model.eval  # uses the most recent run
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from envs.trajectory_gen import TrajectoryGenerator
from specialists.world_model.encoder import ContextEncoder
from specialists.world_model.predictor import Predictor
from specialists.world_model.target_encoder import TargetEncoder
from specialists.world_model.train import TrainConfig, WorldModelDataset

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend — works everywhere
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[eval] matplotlib not found — plots will be skipped, numbers only.")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path: str) -> dict:
    """
    Load a training checkpoint and restore all modules.

    Returns:
        {
          enc, tgt, pred: restored modules in eval mode
          cfg:            TrainConfig used for training
          history:        per-epoch training metrics
          eval_history:   per-eval-checkpoint shift metrics
          device:         torch.device
        }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: TrainConfig = ckpt["cfg"]

    enc = ContextEncoder(
        obs_dim=cfg.obs_dim, seq_len=cfg.seq_len,
        embed_dim=cfg.embed_dim, latent_dim=cfg.latent_dim,
        n_heads=cfg.n_heads, n_layers=cfg.n_layers,
        ff_dim=cfg.ff_dim, dropout=cfg.dropout,
    ).to(device)

    tgt = TargetEncoder(enc).to(device)

    pred = Predictor(
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    enc.load_state_dict(ckpt["enc_state_dict"])
    tgt.load_state_dict(ckpt["tgt_state_dict"])
    pred.load_state_dict(ckpt["pred_state_dict"])

    enc.eval()
    pred.eval()

    return {
        "enc":          enc,
        "tgt":          tgt,
        "pred":         pred,
        "cfg":          cfg,
        "history":      ckpt["history"],
        "eval_history": ckpt["eval_history"],
        "device":       device,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample prediction error
# ─────────────────────────────────────────────────────────────────────────────

def _compute_errors(
    loader: DataLoader,
    enc:    ContextEncoder,
    tgt:    TargetEncoder,
    pred:   Predictor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-window prediction MSE and shift_step_in_window for a loader.

    Returns:
        errors:     (N,) float32 — per-window MSE(z_pred, z_tgt)
        shift_pos:  (N,) int32   — shift_step_in_window (-1 if no shift)
    """
    all_errors    = []
    all_shift_pos = []

    with torch.no_grad():
        for batch in loader:
            ctx_obs  = batch["context_obs"].to(device)
            ctx_acts = batch["context_actions"].to(device)
            a_next   = batch["next_action"].to(device)
            tgt_obs  = batch["target_context_obs"].to(device)
            tgt_acts = batch["target_context_actions"].to(device)

            z_ctx  = enc(ctx_obs, ctx_acts)
            z_tgt  = tgt(tgt_obs, tgt_acts)
            z_pred = pred(z_ctx, a_next)

            per_sample = (z_pred - z_tgt).pow(2).mean(dim=-1)  # (B,)
            all_errors.append(per_sample.cpu().numpy())
            all_shift_pos.append(batch["shift_step_in_window"].numpy())

    return (
        np.concatenate(all_errors).astype(np.float32),
        np.concatenate(all_shift_pos).astype(np.int32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    ckpt_path:          str,
    n_eval_trajectories: int = 30,
    batch_size:          int = 64,
) -> dict:
    """
    Full evaluation of a trained world-model checkpoint.

    Three error regimes are computed on separate datasets:
        pre_shift  — stationary dynamics (rho_normal throughout)
        crossing   — regime shift within context window
        post_shift — stationary dynamics after shift (rho_shifted throughout)

    The crossing/pre_shift ratio is the primary POC metric.

    Args:
        ckpt_path:           Path to a .pt checkpoint saved by train.py.
        n_eval_trajectories: How many trajectories to evaluate over.
        batch_size:          Inference batch size.

    Returns:
        Full results dict (also printed and saved as plots).
    """
    print("=" * 60)
    print("  WORLD-MODEL EVALUATION")
    print("=" * 60)
    print(f"  Checkpoint: {ckpt_path}\n")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    loaded = load_checkpoint(ckpt_path)
    enc, tgt, pred = loaded["enc"], loaded["tgt"], loaded["pred"]
    cfg:   TrainConfig = loaded["cfg"]
    history            = loaded["history"]
    device             = loaded["device"]

    # ── Data generation ───────────────────────────────────────────────────────
    # Use a different seed than training to ensure true out-of-sample eval.
    eval_seed = cfg.data_seed + 1000

    gen = TrajectoryGenerator(
        obs_dim=cfg.obs_dim,
        obs_noise_std=0.05,
        rho_normal=cfg.rho_normal,
        rho_shifted=cfg.rho_shifted,
        seed=eval_seed,
    )

    # Pre-shift: stationary at rho_normal
    pre_data   = gen.generate_stationary_dataset(
        n_trajectories=n_eval_trajectories,
        n_steps=cfg.n_steps,
        seq_len=cfg.seq_len,
    )
    # Shift: regime changes mid-trajectory
    shift_data = gen.generate_shift_dataset(
        n_trajectories=n_eval_trajectories,
        n_pre_shift=cfg.n_pre_shift,
        n_post_shift=cfg.n_post_shift,
        seq_len=cfg.seq_len,
    )
    # Post-shift: stationary at rho_shifted
    post_gen  = TrajectoryGenerator(
        obs_dim=cfg.obs_dim,
        obs_noise_std=0.05,
        rho_normal=cfg.rho_shifted,   # rho_shifted as the "normal" here
        rho_shifted=cfg.rho_normal,
        seed=eval_seed + 1,
    )
    post_data = post_gen.generate_stationary_dataset(
        n_trajectories=n_eval_trajectories,
        n_steps=cfg.n_steps,
        seq_len=cfg.seq_len,
    )

    def make_loader(data):
        return DataLoader(
            WorldModelDataset(data),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

    pre_loader   = make_loader(pre_data)
    shift_loader = make_loader(shift_data)
    post_loader  = make_loader(post_data)

    # ── Compute errors ─────────────────────────────────────────────────────────
    pre_errors,   _           = _compute_errors(pre_loader,   enc, tgt, pred, device)
    shift_errors, shift_pos   = _compute_errors(shift_loader, enc, tgt, pred, device)
    post_errors,  _           = _compute_errors(post_loader,  enc, tgt, pred, device)

    # Per-position error: only crossing windows (shift_pos >= 0)
    seq_len = cfg.seq_len
    pos_errors: list[list] = [[] for _ in range(seq_len)]
    for err, spos in zip(shift_errors, shift_pos):
        if spos >= 0:
            pos_errors[spos].append(float(err))

    per_pos_mean = np.array([
        float(np.mean(errs)) if errs else float("nan")
        for errs in pos_errors
    ])
    per_pos_std  = np.array([
        float(np.std(errs))  if len(errs) > 1 else 0.0
        for errs in pos_errors
    ])

    crossing_mask  = shift_pos >= 0
    pre_mean       = float(np.mean(pre_errors))
    crossing_mean  = float(np.mean(shift_errors[crossing_mask])) if crossing_mask.any() else float("nan")
    post_mean      = float(np.mean(post_errors))
    ratio          = crossing_mean / pre_mean if pre_mean > 0 else float("nan")

    # ── Untrained baseline ────────────────────────────────────────────────────
    print("  Computing untrained baseline...")
    torch.manual_seed(999)
    rand_enc  = ContextEncoder(
        obs_dim=cfg.obs_dim, seq_len=cfg.seq_len,
        embed_dim=cfg.embed_dim, latent_dim=cfg.latent_dim,
        n_heads=cfg.n_heads, n_layers=cfg.n_layers,
        ff_dim=cfg.ff_dim, dropout=0.0,
    ).to(device)
    rand_tgt  = TargetEncoder(rand_enc).to(device)
    rand_pred = Predictor(
        latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim
    ).to(device)
    rand_enc.eval(); rand_pred.eval()

    rand_pre_errors, _         = _compute_errors(pre_loader,   rand_enc, rand_tgt, rand_pred, device)
    rand_shift_errors, _       = _compute_errors(shift_loader, rand_enc, rand_tgt, rand_pred, device)
    rand_pre_mean    = float(np.mean(rand_pre_errors))
    rand_cross_mean  = float(np.mean(rand_shift_errors[crossing_mask])) if crossing_mask.any() else float("nan")

    # ── Build results dict ────────────────────────────────────────────────────
    results = {
        # Primary POC metrics
        "pre_shift_error":        pre_mean,
        "crossing_error":         crossing_mean,
        "post_shift_error":       post_mean,
        "crossing_to_pre_ratio":  ratio,

        # Per-position breakdown (the key curve)
        "per_pos_mean":           per_pos_mean.tolist(),
        "per_pos_std":            per_pos_std.tolist(),
        "n_crossing_windows":     int(crossing_mask.sum()),
        "n_pre_windows":          len(pre_errors),
        "n_post_windows":         len(post_errors),

        # Untrained baseline
        "baseline_pre_error":     rand_pre_mean,
        "baseline_crossing_error":rand_cross_mean,

        # Training summary
        "epochs_trained":     len(history),
        "final_loss":         history[-1]["loss_total"],
        "final_inv":          history[-1]["loss_inv"],
        "final_var":          history[-1]["loss_var"],
        "final_cov":          history[-1]["loss_cov"],
        "final_eff_rank":     history[-1]["eff_rank"],

        # Raw data for further analysis
        "history":            history,
        "cfg":                cfg,
        "ckpt_path":          ckpt_path,
    }

    # ── Print report ──────────────────────────────────────────────────────────
    _print_report(results)

    # ── Save plots ────────────────────────────────────────────────────────────
    run_dir  = Path(ckpt_path).parent
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(exist_ok=True)

    if MATPLOTLIB_AVAILABLE:
        _plot_training_curves(history, eval_dir / "training_curves.png")
        _plot_regime_shift(results, eval_dir / "regime_shift_result.png")
        print(f"\n  Plots saved → {eval_dir}/")
    else:
        print("\n  [plots skipped — install matplotlib to enable]")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(r: dict) -> None:
    cfg: TrainConfig = r["cfg"]

    print("\n── Architecture ──────────────────────────────────────────────")
    enc_dummy = ContextEncoder(
        obs_dim=cfg.obs_dim, seq_len=cfg.seq_len,
        embed_dim=cfg.embed_dim, latent_dim=cfg.latent_dim,
        n_heads=cfg.n_heads, n_layers=cfg.n_layers, ff_dim=cfg.ff_dim,
    )
    pred_dummy = Predictor(latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim)
    print(f"  Context encoder:   {sum(p.numel() for p in enc_dummy.parameters()):>10,} params")
    print(f"  Predictor:         {sum(p.numel() for p in pred_dummy.parameters()):>10,} params")
    print(f"  Latent dim:        {cfg.latent_dim}")
    print(f"  Sequence length:   {cfg.seq_len}")

    print("\n── Training summary ──────────────────────────────────────────")
    print(f"  Epochs trained:        {r['epochs_trained']}")
    print(f"  Final total loss:      {r['final_loss']:.4f}")
    print(f"  Final invariance:      {r['final_inv']:.4f}")
    print(f"  Final variance:        {r['final_var']:.4f}")
    print(f"  Final covariance:      {r['final_cov']:.4f}")
    print(f"  Final effective rank:  {r['final_eff_rank']:.1f}  (collapse if → 1)")

    print("\n── Regime-shift evaluation ──────────────────────────────────")
    print(f"  Pre-shift windows:   {r['n_pre_windows']:>5}  → mean MSE: {r['pre_shift_error']:.5f}")
    print(f"  Crossing windows:    {r['n_crossing_windows']:>5}  → mean MSE: {r['crossing_error']:.5f}  ← shift here")
    print(f"  Post-shift windows:  {r['n_post_windows']:>5}  → mean MSE: {r['post_shift_error']:.5f}")

    print("\n── Per-position error (shift_pos = position of shift in window) ──")
    print("  pos | n_windows | mean_MSE")
    print("  ----+-----------+---------")
    per_pos = r["per_pos_mean"]
    for pos, err in enumerate(per_pos):
        bar = "█" * int(err * 200) if not np.isnan(err) else "-"
        print(f"  {pos:3d} |           | {err:.5f}  {bar}")

    print("\n── Key result ────────────────────────────────────────────────")
    ratio = r["crossing_to_pre_ratio"]
    print(f"  crossing_error / pre_shift_error = {ratio:.3f}×")
    if ratio > 1.05:
        print("  ✓  Prediction error elevated at regime-shift boundary.")
        print("     Standalone predictor IS being fooled. (Expected at this stage.)")
        print("     This is the baseline. Binding with Specialist B will be")
        print("     measured against this crossing error.")
    elif ratio > 0.95:
        print("  ~  Crossing error close to pre-shift error.")
        print("     Shift may not be clearly visible yet — try more training epochs.")
    else:
        print("  ?  Crossing error lower than pre-shift. Unexpected.")
        print("     Investigate: model may have collapsed or shift is too subtle.")

    print("\n── Untrained baseline comparison ────────────────────────────")
    print(f"  Pre-shift error  (random weights): {r['baseline_pre_error']:.5f}")
    print(f"  Crossing error   (random weights): {r['baseline_crossing_error']:.5f}")
    improvement = r["baseline_pre_error"] / r["pre_shift_error"] if r["pre_shift_error"] > 0 else float("nan")
    print(f"  Training reduced pre-shift error by {improvement:.2f}× vs random weights.")


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────


def _set_research_style() -> None:
    """Consistent, publication-ready Matplotlib styling."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.labelcolor": "#374151",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": "#4B5563",
        "ytick.color": "#4B5563",
        "text.color": "#111827",
        "axes.edgecolor": "#D1D5DB",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E5E7EB",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.85,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


COLORS = {
    "navy": "#355C7D",
    "blue": "#4C78A8",
    "red": "#C65D5D",
    "green": "#6A9F78",
    "orange": "#E39A59",
    "purple": "#8064A2",
    "text": "#374151",
    "muted": "#6B7280",
    "grid": "#E5E7EB",
}



def _plot_training_curves(history: list[dict], save_path: Path) -> None:
    """Publication-style training curves."""
    _set_research_style()

    epochs = np.asarray([h["epoch"] for h in history])
    total = np.asarray([h["loss_total"] for h in history])
    inv = np.asarray([25 * h["loss_inv"] for h in history])
    var = np.asarray([25 * h["loss_var"] for h in history])
    cov = np.asarray([h["loss_cov"] for h in history])
    eff_rank = np.asarray([h["eff_rank"] for h in history])

    fig, axes = plt.subplots(
        2, 1,
        figsize=(8.2, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 0.85]},
        layout="constrained",
    )
    fig.suptitle(
        "World-model training",
        fontsize=15,
        fontweight="semibold",
        x=0.06,
        ha="left",
    )

    # Panel A: VICReg losses
    ax = axes[0]
    ax.plot(
        epochs, total, color=COLORS["navy"],
        lw=2.4, label="Total loss", zorder=4,
    )
    ax.plot(
        epochs, inv, color=COLORS["blue"],
        lw=1.7, ls="--", label="25 × invariance",
    )
    ax.plot(
        epochs, var, color=COLORS["orange"],
        lw=1.7, ls="--", label="25 × variance",
    )
    ax.plot(
        epochs, cov, color=COLORS["green"],
        lw=1.7, ls="--", label="Covariance",
    )

    ax.set_title("VICReg loss components", loc="left", pad=9)
    ax.set_ylabel("Loss")
    ax.legend(
        loc="upper right",
        ncol=2,
        columnspacing=1.2,
        handlelength=2.2,
    )
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.set_xlim(epochs.min(), epochs.max())

    # Panel B: latent representation health
    ax = axes[1]
    ax.plot(
        epochs, eff_rank,
        color=COLORS["purple"], lw=2.3,
        label="Effective rank",
    )
    ax.axhline(
        1.0,
        color=COLORS["red"],
        lw=1.3,
        ls=":",
        label="Collapse threshold",
    )

    ax.set_title("Latent effective rank", loc="left", pad=9)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Effective rank")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.12,
    )

    fig.savefig(
        save_path.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)




def _plot_regime_shift(results: dict, save_path: Path) -> None:
    """Publication-style regime-shift evaluation."""
    _set_research_style()

    means = np.asarray(results["per_pos_mean"], dtype=float)
    stds = np.asarray(results["per_pos_std"], dtype=float)
    seq_len = len(means)
    positions = np.arange(seq_len)

    pre = results["pre_shift_error"]
    cross = results["crossing_error"]
    post = results["post_shift_error"]
    ratio = results["crossing_to_pre_ratio"]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(11.0, 4.7),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
        layout="constrained",
    )
    fig.suptitle(
        "Regime-shift evaluation",
        fontsize=15,
        fontweight="semibold",
        x=0.055,
        ha="left",
    )

    # Panel A: prediction error by shift position
    ax = axes[0]
    valid = np.isfinite(means) & np.isfinite(stds)

    ax.fill_between(
        positions[valid],
        np.maximum(means[valid] - stds[valid], 0),
        means[valid] + stds[valid],
        color=COLORS["red"],
        alpha=0.15,
        linewidth=0,
        label="±1 SD",
        zorder=1,
    )
    ax.plot(
        positions[valid],
        means[valid],
        color=COLORS["red"],
        lw=2.3,
        marker="o",
        markersize=4,
        markerfacecolor="white",
        markeredgewidth=1.4,
        label="Crossing windows",
        zorder=3,
    )

    ax.axhline(
        pre,
        color=COLORS["blue"],
        lw=1.6,
        ls="--",
        label=f"Pre-shift ({pre:.4f})",
    )
    ax.axhline(
        post,
        color=COLORS["green"],
        lw=1.6,
        ls=":",
        label=f"Post-shift ({post:.4f})",
    )

    ax.set_title("Prediction error by shift position", loc="left", pad=10)
    ax.set_xlabel("Shift position within context window")
    ax.set_ylabel("Mean prediction MSE")
    ax.set_xlim(-0.4, seq_len - 0.6)
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", ncol=2, handlelength=2.0)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    # Explain the endpoint interpretation without arrow collisions.
    ax.text(
        0.01, -0.22,
        "Left: shift fully absorbed     ·     Right: shift just entered",
        transform=ax.transAxes,
        fontsize=8,
        color=COLORS["muted"],
        ha="left",
        va="top",
    )

    # Panel B: comparison across regimes
    ax = axes[1]
    labels = ["Pre-shift", "Crossing", "Post-shift"]
    values = [pre, cross, post]
    colors = [COLORS["blue"], COLORS["red"], COLORS["green"]]

    bars = ax.bar(
        labels,
        values,
        color=colors,
        width=0.64,
        alpha=0.92,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    offset = max(values) * 0.025
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="semibold",
            color=COLORS["text"],
        )

    ax.set_title("Mean error by regime", loc="left", pad=10)
    ax.set_ylabel("Mean prediction MSE")
    ax.set_ylim(0, max(values) * 1.20)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=7)

    ax.text(
        0.5, 0.96,
        f"Crossing / pre-shift: {ratio:.2f}×",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        fontweight="semibold",
        color=COLORS["red"],
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.15,
    )
    fig.savefig(
        save_path.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _find_latest_checkpoint() -> str:
    runs = sorted(Path("experiments/runs").glob("run_*/checkpoint.pt"))
    if not runs:
        raise FileNotFoundError(
            "No checkpoints found in experiments/runs/. "
            "Run training first: python -m specialists.world_model.train"
        )
    return str(runs[-1])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    else:
        ckpt = _find_latest_checkpoint()
        print(f"  No checkpoint specified — using latest: {ckpt}\n")

    evaluate(ckpt_path=ckpt)