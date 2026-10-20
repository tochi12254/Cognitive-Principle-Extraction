"""
Test suite for specialists/world_model/train.py

Covers:
- WorldModelDataset: correct length, correct item shapes, all keys present
- WorldModelDataset: tensors are float32 (not float64)
- DataLoader with drop_last=True never produces batch of size 1
- train_step: runs without error, returns expected metric keys
- train_step: loss_total is finite and positive
- train_step: target encoder NOT in optimizer param groups
- train_step: target encoder weights change only via EMA (not backprop)
- train_step: context encoder + predictor weights change after step
- train_step: effective_rank is logged and in valid range
- run_epoch: returns mean metrics with correct keys
- evaluate_shift: returns expected keys
- evaluate_shift: pre_shift_error and crossing_error are finite
- Loss decreases over a short training run (functional test)
- Checkpoint saves with correct keys
- Checkpoint restores and produces same output (reproducibility)
"""

import torch
import pytest
from pathlib import Path

from envs.trajectory_gen import TrajectoryGenerator
from specialists.world_model.encoder import ContextEncoder
from specialists.world_model.target_encoder import TargetEncoder
from specialists.world_model.predictor import Predictor
from specialists.world_model.train import (
    TrainConfig,
    WorldModelDataset,
    train_step,
    run_epoch,
    evaluate_shift,
    train,
)
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Minimal config for fast tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_cfg(tmp_path):
    return TrainConfig(
        n_train_trajectories=3,
        n_eval_trajectories=2,
        n_steps=40,
        seq_len=10,
        n_pre_shift=20,
        n_post_shift=20,
        n_epochs=2,
        batch_size=16,
        eval_every=1,
        run_dir=str(tmp_path),
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


@pytest.fixture
def tiny_modules(tiny_cfg):
    enc  = ContextEncoder(
        obs_dim=tiny_cfg.obs_dim, seq_len=tiny_cfg.seq_len,
        embed_dim=tiny_cfg.embed_dim, latent_dim=tiny_cfg.latent_dim,
        n_heads=tiny_cfg.n_heads, n_layers=tiny_cfg.n_layers,
        ff_dim=tiny_cfg.ff_dim, dropout=tiny_cfg.dropout,
    )
    tgt  = TargetEncoder(enc)
    pred = Predictor(latent_dim=tiny_cfg.latent_dim, hidden_dim=tiny_cfg.hidden_dim)
    return enc, tgt, pred


@pytest.fixture
def tiny_optimizer(tiny_modules):
    enc, _, pred = tiny_modules
    return torch.optim.AdamW(
        list(enc.parameters()) + list(pred.parameters()), lr=3e-4
    )


@pytest.fixture
def tiny_dataset(tiny_cfg):
    gen = TrajectoryGenerator(
        obs_dim=tiny_cfg.obs_dim,
        rho_normal=tiny_cfg.rho_normal,
        rho_shifted=tiny_cfg.rho_shifted,
        seed=tiny_cfg.data_seed,
    )
    data = gen.generate_stationary_dataset(
        n_trajectories=tiny_cfg.n_train_trajectories,
        n_steps=tiny_cfg.n_steps,
        seq_len=tiny_cfg.seq_len,
    )
    return WorldModelDataset(data)


@pytest.fixture
def tiny_batch(tiny_dataset, tiny_cfg):
    loader = DataLoader(tiny_dataset, batch_size=tiny_cfg.batch_size,
                        shuffle=False, drop_last=True)
    return next(iter(loader))


@pytest.fixture
def device():
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# WorldModelDataset tests
# ─────────────────────────────────────────────────────────────────────────────

def test_dataset_length(tiny_dataset, tiny_cfg):
    expected = tiny_cfg.n_train_trajectories * (tiny_cfg.n_steps - tiny_cfg.seq_len)
    assert len(tiny_dataset) == expected


def test_dataset_item_keys(tiny_dataset):
    item = tiny_dataset[0]
    expected_keys = {
        "context_obs", "context_actions", "next_action",
        "target_obs", "target_context_obs", "target_context_actions",
        "shift_step_in_window",
    }
    assert set(item.keys()) == expected_keys


def test_dataset_item_shapes(tiny_dataset, tiny_cfg):
    item = tiny_dataset[0]
    seq, obs = tiny_cfg.seq_len, tiny_cfg.obs_dim
    assert item["context_obs"].shape            == (seq, obs)
    assert item["context_actions"].shape        == (seq,)
    assert item["next_action"].shape            == ()
    assert item["target_context_obs"].shape     == (seq, obs)
    assert item["target_context_actions"].shape == (seq,)


def test_dataset_tensors_are_float32(tiny_dataset):
    item = tiny_dataset[0]
    for key in ("context_obs", "context_actions", "next_action",
                "target_context_obs", "target_context_actions"):
        assert item[key].dtype == torch.float32, (
            f"{key} is {item[key].dtype}, expected float32"
        )


def test_dataloader_drop_last_prevents_tiny_batches(tiny_dataset, tiny_cfg):
    """
    drop_last=True must ensure every batch has batch_size samples.
    VICReg requires B >= 2; a batch of 1 would raise ValueError.
    """
    loader = DataLoader(tiny_dataset, batch_size=tiny_cfg.batch_size,
                        shuffle=True, drop_last=True)
    for batch in loader:
        assert len(batch["context_obs"]) == tiny_cfg.batch_size


# ─────────────────────────────────────────────────────────────────────────────
# train_step tests
# ─────────────────────────────────────────────────────────────────────────────

def test_train_step_runs_without_error(
    tiny_batch, tiny_modules, tiny_optimizer, tiny_cfg, device
):
    enc, tgt, pred = tiny_modules
    metrics = train_step(tiny_batch, enc, tgt, pred, tiny_optimizer, tiny_cfg, device)
    assert metrics is not None


def test_train_step_returns_expected_keys(
    tiny_batch, tiny_modules, tiny_optimizer, tiny_cfg, device
):
    enc, tgt, pred = tiny_modules
    metrics = train_step(tiny_batch, enc, tgt, pred, tiny_optimizer, tiny_cfg, device)
    expected = {"loss_total", "loss_inv", "loss_var", "loss_cov", "eff_rank"}
    assert set(metrics.keys()) == expected


def test_train_step_loss_is_finite_and_positive(
    tiny_batch, tiny_modules, tiny_optimizer, tiny_cfg, device
):
    enc, tgt, pred = tiny_modules
    metrics = train_step(tiny_batch, enc, tgt, pred, tiny_optimizer, tiny_cfg, device)
    assert metrics["loss_total"] > 0
    assert all(
        torch.isfinite(torch.tensor(v))
        for v in metrics.values()
    )


def test_target_encoder_not_in_optimizer(tiny_modules, tiny_optimizer):
    _, tgt, _ = tiny_modules
    tgt_param_ids = {id(p) for p in tgt.parameters()}
    opt_param_ids = {id(p) for pg in tiny_optimizer.param_groups
                     for p in pg["params"]}
    assert len(tgt_param_ids & opt_param_ids) == 0, (
        "Target encoder parameters found in optimizer — "
        "EMA encoder must never be updated by gradient descent."
    )


def test_encoder_and_predictor_weights_update_after_step(
    tiny_batch, tiny_modules, tiny_optimizer, tiny_cfg, device
):
    enc, tgt, pred = tiny_modules

    enc_before  = {n: p.clone() for n, p in enc.named_parameters()}
    pred_before = {n: p.clone() for n, p in pred.named_parameters()}

    train_step(tiny_batch, enc, tgt, pred, tiny_optimizer, tiny_cfg, device)

    enc_changed  = any(not torch.allclose(p, enc_before[n])
                       for n, p in enc.named_parameters())
    pred_changed = any(not torch.allclose(p, pred_before[n])
                       for n, p in pred.named_parameters())

    assert enc_changed,  "Context encoder weights did not change after train_step."
    assert pred_changed, "Predictor weights did not change after train_step."


def test_target_encoder_changes_only_via_ema(
    tiny_batch, tiny_modules, tiny_optimizer, tiny_cfg, device
):
    """
    Target encoder weights must change (due to EMA update) but not via
    gradient — they must not be identical to what they were before,
    since train_step calls tgt.update(enc) at the end.
    """
    enc, tgt, pred = tiny_modules
    tgt_before = {n: p.clone() for n, p in tgt.named_parameters()}

    train_step(tiny_batch, enc, tgt, pred, tiny_optimizer, tiny_cfg, device)

    tgt_after = {n: p.data for n, p in tgt.named_parameters()}
    changed   = any(not torch.allclose(tgt_before[n], tgt_after[n])
                    for n in tgt_before)
    assert changed, (
        "Target encoder weights unchanged after train_step — "
        "EMA update may not be running."
    )

    # Target encoder params must still have no grad
    for name, p in tgt.named_parameters():
        assert p.grad is None, f"tgt.{name} has gradient after train_step."


def test_effective_rank_in_valid_range(
    tiny_batch, tiny_modules, tiny_optimizer, tiny_cfg, device
):
    enc, tgt, pred = tiny_modules
    metrics = train_step(tiny_batch, enc, tgt, pred, tiny_optimizer, tiny_cfg, device)
    latent_dim = tiny_cfg.latent_dim
    assert 1.0 <= metrics["eff_rank"] <= latent_dim, (
        f"effective_rank {metrics['eff_rank']:.2f} outside [1, {latent_dim}]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# run_epoch tests
# ─────────────────────────────────────────────────────────────────────────────

def test_run_epoch_returns_mean_metrics(
    tiny_dataset, tiny_modules, tiny_optimizer, tiny_cfg, device
):
    enc, tgt, pred = tiny_modules
    loader = DataLoader(tiny_dataset, batch_size=tiny_cfg.batch_size,
                        shuffle=True, drop_last=True)
    metrics = run_epoch(loader, enc, tgt, pred, tiny_optimizer, tiny_cfg, device)
    expected = {"loss_total", "loss_inv", "loss_var", "loss_cov", "eff_rank"}
    assert set(metrics.keys()) == expected
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_shift tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def shift_loader(tiny_cfg):
    gen = TrajectoryGenerator(
        obs_dim=tiny_cfg.obs_dim,
        rho_normal=tiny_cfg.rho_normal,
        rho_shifted=tiny_cfg.rho_shifted,
        seed=99,
    )
    data = gen.generate_shift_dataset(
        n_trajectories=tiny_cfg.n_eval_trajectories,
        n_pre_shift=tiny_cfg.n_pre_shift,
        n_post_shift=tiny_cfg.n_post_shift,
        seq_len=tiny_cfg.seq_len,
    )
    ds = WorldModelDataset(data)
    return DataLoader(ds, batch_size=tiny_cfg.batch_size, shuffle=False, drop_last=False)


def test_evaluate_shift_returns_expected_keys(
    shift_loader, tiny_modules, device, tiny_cfg
):
    enc, tgt, pred = tiny_modules
    result = evaluate_shift(shift_loader, enc, tgt, pred, device, tiny_cfg.seq_len)
    assert "mean_error_by_shift_pos" in result
    assert "pre_shift_error" in result
    assert "crossing_error" in result


def test_evaluate_shift_errors_are_finite(
    shift_loader, tiny_modules, device, tiny_cfg
):
    enc, tgt, pred = tiny_modules
    result = evaluate_shift(shift_loader, enc, tgt, pred, device, tiny_cfg.seq_len)
    assert torch.isfinite(torch.tensor(result["pre_shift_error"]))
    assert torch.isfinite(torch.tensor(result["crossing_error"]))


def test_evaluate_shift_mean_error_has_correct_length(
    shift_loader, tiny_modules, device, tiny_cfg
):
    enc, tgt, pred = tiny_modules
    result = evaluate_shift(shift_loader, enc, tgt, pred, device, tiny_cfg.seq_len)
    assert len(result["mean_error_by_shift_pos"]) == tiny_cfg.seq_len


# ─────────────────────────────────────────────────────────────────────────────
# Full training run tests
# ─────────────────────────────────────────────────────────────────────────────

def test_loss_decreases_over_short_run(tiny_cfg):
    """
    Loss must decrease over a short training run.
    This is a functional test — it confirms the training loop is
    actually learning, not just running without error.
    """
    result = train(tiny_cfg)
    history = result["history"]
    first_loss = history[0]["loss_total"]
    last_loss  = history[-1]["loss_total"]
    assert last_loss < first_loss, (
        f"Loss did not decrease: {first_loss:.4f} → {last_loss:.4f}"
    )


def test_full_train_returns_expected_keys(tiny_cfg):
    result = train(tiny_cfg)
    assert "history"       in result
    assert "eval_history"  in result
    assert "enc"           in result
    assert "tgt"           in result
    assert "pred"          in result
    assert "final_metrics" in result
    assert "run_dir"       in result


def test_checkpoint_saved_with_correct_keys(tiny_cfg, tmp_path):
    tiny_cfg.run_dir = str(tmp_path)
    result  = train(tiny_cfg)
    run_dir = Path(result["run_dir"])
    ckpt    = torch.load(run_dir / "checkpoint.pt", weights_only=False)

    expected_keys = {
        "enc_state_dict", "tgt_state_dict", "pred_state_dict",
        "opt_state_dict", "cfg", "history", "eval_history",
    }
    assert set(ckpt.keys()) == expected_keys


def test_checkpoint_restores_encoder_output(tiny_cfg, tmp_path):
    """
    After saving and loading a checkpoint, the restored encoder must
    produce identical output to the trained encoder.
    """
    tiny_cfg.run_dir = str(tmp_path)
    result  = train(tiny_cfg)
    run_dir = Path(result["run_dir"])
    enc     = result["enc"]

    ckpt   = torch.load(run_dir / "checkpoint.pt", weights_only=False)
    enc2   = ContextEncoder(
        obs_dim=tiny_cfg.obs_dim, seq_len=tiny_cfg.seq_len,
        embed_dim=tiny_cfg.embed_dim, latent_dim=tiny_cfg.latent_dim,
        n_heads=tiny_cfg.n_heads, n_layers=tiny_cfg.n_layers,
        ff_dim=tiny_cfg.ff_dim, dropout=tiny_cfg.dropout,
    )
    enc2.load_state_dict(ckpt["enc_state_dict"])
    enc2.eval()
    enc.eval()

    torch.manual_seed(77)
    obs  = torch.randn(2, tiny_cfg.seq_len, tiny_cfg.obs_dim)
    acts = torch.randn(2, tiny_cfg.seq_len)

    with torch.no_grad():
        out1 = enc(obs, acts)
        out2 = enc2(obs, acts)

    assert torch.allclose(out1, out2, atol=1e-6), (
        "Restored encoder produces different output — checkpoint is corrupted."
    )