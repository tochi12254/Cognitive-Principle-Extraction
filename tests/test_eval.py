"""
Test suite for specialists/world_model/eval.py

Covers:
- load_checkpoint restores modules with correct architecture
- load_checkpoint puts modules in eval mode
- load_checkpoint preserves target encoder parameter values
- _compute_errors returns correct array shapes and dtypes
- _compute_errors: all errors are non-negative
- evaluate returns dict with all expected keys
- evaluate: all scalar metrics are finite
- evaluate: per_pos_mean has length == seq_len
- evaluate: n_crossing_windows > 0 (shift data contains crossing windows)
- evaluate: baseline errors computed (untrained model gives different values)
- evaluate: trained model outperforms random baseline on pre-shift data
- evaluate: plots are saved when matplotlib is available
- _find_latest_checkpoint: raises FileNotFoundError when no runs exist
"""

import torch
import pytest
import numpy as np
from pathlib import Path

from specialists.world_model.train import train, TrainConfig
from specialists.world_model.eval import (
    load_checkpoint,
    evaluate,
    _compute_errors,
    _find_latest_checkpoint,
)
from specialists.world_model.encoder import ContextEncoder
from specialists.world_model.target_encoder import TargetEncoder
from specialists.world_model.predictor import Predictor
from specialists.world_model.train import WorldModelDataset
from envs.trajectory_gen import TrajectoryGenerator
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: run a tiny training run and use the checkpoint
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    """Train a tiny model and return the checkpoint path. Module-scoped so
    it only trains once across all tests in this file."""
    tmp = tmp_path_factory.mktemp("runs")
    cfg = TrainConfig(
        n_train_trajectories=3,
        n_eval_trajectories=2,
        n_steps=40,
        seq_len=10,
        n_pre_shift=20,
        n_post_shift=20,
        n_epochs=3,
        batch_size=16,
        eval_every=3,
        run_dir=str(tmp),
        data_seed=0,
        model_seed=0,
        embed_dim=32,
        latent_dim=32,
        n_heads=4,
        n_layers=1,
        ff_dim=64,
        hidden_dim=64,
        dropout=0.0,
    )
    result = train(cfg)
    ckpt_path = str(Path(result["run_dir"]) / "checkpoint.pt")
    return ckpt_path, cfg


# ─────────────────────────────────────────────────────────────────────────────
# load_checkpoint tests
# ─────────────────────────────────────────────────────────────────────────────

def test_load_checkpoint_returns_expected_keys(tiny_checkpoint):
    ckpt_path, _ = tiny_checkpoint
    loaded = load_checkpoint(ckpt_path)
    expected = {"enc", "tgt", "pred", "cfg", "history", "eval_history", "device"}
    assert set(loaded.keys()) == expected


def test_load_checkpoint_enc_in_eval_mode(tiny_checkpoint):
    ckpt_path, _ = tiny_checkpoint
    loaded = load_checkpoint(ckpt_path)
    assert not loaded["enc"].training, "Encoder should be in eval mode after load."


def test_load_checkpoint_pred_in_eval_mode(tiny_checkpoint):
    ckpt_path, _ = tiny_checkpoint
    loaded = load_checkpoint(ckpt_path)
    assert not loaded["pred"].training, "Predictor should be in eval mode after load."


def test_load_checkpoint_tgt_in_eval_mode(tiny_checkpoint):
    ckpt_path, _ = tiny_checkpoint
    loaded = load_checkpoint(ckpt_path)
    assert not loaded["tgt"].training, "Target encoder should be in eval mode after load."


def test_load_checkpoint_restores_enc_weights(tiny_checkpoint):
    """
    Loading the same checkpoint twice must give identical encoder weights.
    Confirms the state dict is correctly restored, not randomly reinitialised.
    """
    ckpt_path, _ = tiny_checkpoint
    loaded_a = load_checkpoint(ckpt_path)
    loaded_b = load_checkpoint(ckpt_path)

    for (na, pa), (nb, pb) in zip(
        loaded_a["enc"].named_parameters(),
        loaded_b["enc"].named_parameters(),
    ):
        assert torch.allclose(pa, pb), (
            f"Encoder weight {na} differs between two loads of the same checkpoint."
        )


def test_load_checkpoint_tgt_params_no_grad(tiny_checkpoint):
    """Target encoder parameters must have requires_grad=False after loading."""
    ckpt_path, _ = tiny_checkpoint
    loaded = load_checkpoint(ckpt_path)
    for name, p in loaded["tgt"].named_parameters():
        assert not p.requires_grad, f"tgt.{name} has requires_grad=True after load."


def test_load_checkpoint_history_not_empty(tiny_checkpoint):
    ckpt_path, _ = tiny_checkpoint
    loaded = load_checkpoint(ckpt_path)
    assert len(loaded["history"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# _compute_errors tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def eval_loader(tiny_checkpoint):
    _, cfg = tiny_checkpoint
    gen = TrajectoryGenerator(
        obs_dim=cfg.obs_dim,
        rho_normal=cfg.rho_normal,
        rho_shifted=cfg.rho_shifted,
        seed=42,
    )
    data = gen.generate_shift_dataset(
        n_trajectories=2,
        n_pre_shift=cfg.n_pre_shift,
        n_post_shift=cfg.n_post_shift,
        seq_len=cfg.seq_len,
    )
    return DataLoader(
        WorldModelDataset(data),
        batch_size=16, shuffle=False, drop_last=False,
    )


def test_compute_errors_returns_correct_shapes(tiny_checkpoint, eval_loader):
    ckpt_path, cfg = tiny_checkpoint
    loaded = load_checkpoint(ckpt_path)
    enc, tgt, pred, device = loaded["enc"], loaded["tgt"], loaded["pred"], loaded["device"]

    errors, shift_pos = _compute_errors(eval_loader, enc, tgt, pred, device)
    N = sum(len(b["context_obs"]) for b in eval_loader)

    assert errors.shape    == (N,), f"Expected errors shape ({N},), got {errors.shape}"
    assert shift_pos.shape == (N,), f"Expected shift_pos shape ({N},), got {shift_pos.shape}"


def test_compute_errors_all_non_negative(tiny_checkpoint, eval_loader):
    """MSE is always >= 0."""
    ckpt_path, _ = tiny_checkpoint
    loaded = load_checkpoint(ckpt_path)
    errors, _ = _compute_errors(
        eval_loader, loaded["enc"], loaded["tgt"], loaded["pred"], loaded["device"]
    )
    assert (errors >= 0).all(), "Negative prediction MSE found."


def test_compute_errors_all_finite(tiny_checkpoint, eval_loader):
    ckpt_path, _ = tiny_checkpoint
    loaded = load_checkpoint(ckpt_path)
    errors, _ = _compute_errors(
        eval_loader, loaded["enc"], loaded["tgt"], loaded["pred"], loaded["device"]
    )
    assert np.isfinite(errors).all(), "Non-finite prediction MSE found."


def test_compute_errors_shift_pos_values(tiny_checkpoint, eval_loader):
    """shift_pos values must be either -1 or in [0, seq_len)."""
    ckpt_path, cfg = tiny_checkpoint
    loaded = load_checkpoint(ckpt_path)
    _, shift_pos = _compute_errors(
        eval_loader, loaded["enc"], loaded["tgt"], loaded["pred"], loaded["device"]
    )
    valid = (shift_pos == -1) | ((shift_pos >= 0) & (shift_pos < cfg.seq_len))
    assert valid.all(), "Invalid shift_pos values found."


# ─────────────────────────────────────────────────────────────────────────────
# evaluate() tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def eval_results(tiny_checkpoint):
    """Run evaluate() once and reuse results across tests."""
    ckpt_path, _ = tiny_checkpoint
    return evaluate(ckpt_path=ckpt_path, n_eval_trajectories=5)


def test_evaluate_returns_expected_keys(eval_results):
    expected = {
        "pre_shift_error", "crossing_error", "post_shift_error",
        "crossing_to_pre_ratio",
        "per_pos_mean", "per_pos_std",
        "n_crossing_windows", "n_pre_windows", "n_post_windows",
        "baseline_pre_error", "baseline_crossing_error",
        "epochs_trained", "final_loss", "final_inv", "final_var",
        "final_cov", "final_eff_rank",
        "history", "cfg", "ckpt_path",
    }
    assert set(eval_results.keys()) == expected


def test_evaluate_scalar_metrics_are_finite(eval_results):
    scalars = [
        "pre_shift_error", "crossing_error", "post_shift_error",
        "crossing_to_pre_ratio", "baseline_pre_error",
        "baseline_crossing_error", "final_loss", "final_eff_rank",
    ]
    for key in scalars:
        val = eval_results[key]
        assert np.isfinite(val), f"{key} is not finite: {val}"


def test_evaluate_errors_are_non_negative(eval_results):
    for key in ("pre_shift_error", "crossing_error", "post_shift_error"):
        assert eval_results[key] >= 0, f"{key} is negative."


def test_evaluate_per_pos_mean_has_correct_length(eval_results, tiny_checkpoint):
    _, cfg = tiny_checkpoint
    assert len(eval_results["per_pos_mean"]) == cfg.seq_len


def test_evaluate_crossing_windows_exist(eval_results):
    """Must have at least some crossing windows in the shift dataset."""
    assert eval_results["n_crossing_windows"] > 0, (
        "No crossing windows found — shift dataset may not contain shift trajectories."
    )


def test_evaluate_epochs_trained_matches_config(eval_results, tiny_checkpoint):
    _, cfg = tiny_checkpoint
    assert eval_results["epochs_trained"] == cfg.n_epochs


def test_evaluate_baseline_differs_from_trained(eval_results):
    """
    Random-weight baseline must produce a different pre-shift error than
    the trained model. Confirms evaluation is not accidentally comparing
    the model against itself.
    """
    trained  = eval_results["pre_shift_error"]
    baseline = eval_results["baseline_pre_error"]
    assert not np.isclose(trained, baseline, rtol=0.01), (
        "Trained model and random baseline have nearly identical pre-shift error — "
        "evaluation may be using the same weights for both."
    )


def test_evaluate_trained_outperforms_random_baseline(eval_results):
    """
    After training, the model should predict pre-shift dynamics better
    (lower MSE) than random weights. Confirms training is actually learning.
    """
    assert eval_results["pre_shift_error"] < eval_results["baseline_pre_error"], (
        f"Trained model ({eval_results['pre_shift_error']:.5f}) has higher "
        f"pre-shift error than random baseline ({eval_results['baseline_pre_error']:.5f}). "
        f"Training did not help."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plot tests
# ─────────────────────────────────────────────────────────────────────────────

def test_plots_are_saved(tiny_checkpoint):
    """Both PNG plots must exist after evaluate() runs."""
    ckpt_path, _ = tiny_checkpoint
    evaluate(ckpt_path=ckpt_path, n_eval_trajectories=3)

    run_dir  = Path(ckpt_path).parent
    eval_dir = run_dir / "eval"

    try:
        import matplotlib
        assert (eval_dir / "training_curves.png").exists(), (
            "training_curves.png not found after evaluate()."
        )
        assert (eval_dir / "regime_shift_result.png").exists(), (
            "regime_shift_result.png not found after evaluate()."
        )
    except ImportError:
        pytest.skip("matplotlib not installed — plot test skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# _find_latest_checkpoint test
# ─────────────────────────────────────────────────────────────────────────────

def test_find_latest_checkpoint_raises_when_no_runs(tmp_path, monkeypatch):
    """
    _find_latest_checkpoint must raise FileNotFoundError if no checkpoints
    exist in the expected directory.
    """
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="No checkpoints found"):
        _find_latest_checkpoint()