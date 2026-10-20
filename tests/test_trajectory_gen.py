"""
Test suite for envs/trajectory_gen.py

Covers:
- Output shapes for stationary and shift datasets
- Window indexing correctness (target is one step ahead of context)
- Shift detection: crossing windows have correct shift_step_in_window values
- Action distribution (Gaussian with correct scale)
- Train/val/test split sizes and no data loss
- Reproducibility with same seed
- Edge case: n_steps too small raises ValueError
- Stationary dataset has no crossing windows (all shift_step_in_window = -1)
- Shift dataset contains the expected number of crossing windows
"""

import numpy as np
import pytest

from envs.trajectory_gen import TrajectoryGenerator


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def gen():
    return TrajectoryGenerator(seed=0)


@pytest.fixture
def stationary_ds(gen):
    return gen.generate_stationary_dataset(
        n_trajectories=3, n_steps=50, seq_len=10
    )


@pytest.fixture
def shift_ds():
    g = TrajectoryGenerator(seed=7)
    return g.generate_shift_dataset(
        n_trajectories=3, n_pre_shift=40, n_post_shift=40, seq_len=10
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shape tests
# ─────────────────────────────────────────────────────────────────────────────

def test_stationary_dataset_keys(stationary_ds):
    expected_keys = {
        "context_obs",
        "context_actions",
        "next_action",
        "target_obs",
        "target_context_obs",
        "target_context_actions",
        "shift_step_in_window",
    }
    assert set(stationary_ds.keys()) == expected_keys


def test_stationary_dataset_shapes(stationary_ds):
    N = len(stationary_ds["context_obs"])
    seq_len = 10
    obs_dim = 6

    assert stationary_ds["context_obs"].shape    == (N, seq_len, obs_dim)
    assert stationary_ds["context_actions"].shape == (N, seq_len)
    assert stationary_ds["next_action"].shape     == (N,)
    assert stationary_ds["target_obs"].shape      == (N, obs_dim)
    assert stationary_ds["shift_step_in_window"].shape == (N,)


def test_shift_dataset_shapes(shift_ds):
    N = len(shift_ds["context_obs"])
    seq_len = 10
    obs_dim = 6

    assert shift_ds["context_obs"].shape    == (N, seq_len, obs_dim)
    assert shift_ds["context_actions"].shape == (N, seq_len)
    assert shift_ds["next_action"].shape     == (N,)
    assert shift_ds["target_obs"].shape      == (N, obs_dim)
    assert shift_ds["shift_step_in_window"].shape == (N,)


def test_stationary_dataset_dtypes(stationary_ds):
    assert stationary_ds["context_obs"].dtype             == np.float32
    assert stationary_ds["context_actions"].dtype          == np.float32
    assert stationary_ds["next_action"].dtype              == np.float32
    assert stationary_ds["target_obs"].dtype               == np.float32
    assert stationary_ds["target_context_obs"].dtype       == np.float32
    assert stationary_ds["target_context_actions"].dtype   == np.float32
    assert stationary_ds["shift_step_in_window"].dtype     == np.int32


def test_target_context_obs_shape(stationary_ds):
    """target_context_obs must have same shape as context_obs."""
    assert stationary_ds["target_context_obs"].shape == stationary_ds["context_obs"].shape


def test_target_context_actions_shape(stationary_ds):
    """target_context_actions must have same shape as context_actions."""
    assert stationary_ds["target_context_actions"].shape == stationary_ds["context_actions"].shape


def test_target_context_is_shifted_by_one_step():
    """
    target_context_obs[i] must equal context_obs for window i+1
    (from a single trajectory with no shuffle).
    Both represent the same observations, just shifted one step forward.
    """
    gen = TrajectoryGenerator(seed=99, obs_noise_std=0.0)
    ds  = gen.generate_stationary_dataset(
        n_trajectories=1, n_steps=30, seq_len=5, action_scale=0.0
    )
    # target_context_obs[i] = obs[i+1 : i+seq_len+1]
    # context_obs[i+1]      = obs[i+1 : i+seq_len+1]  (same slice)
    for i in range(len(ds["context_obs"]) - 1):
        np.testing.assert_allclose(
            ds["target_context_obs"][i],
            ds["context_obs"][i + 1],
            err_msg=f"target_context_obs[{i}] != context_obs[{i+1}]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Window count tests
# ─────────────────────────────────────────────────────────────────────────────

def test_stationary_window_count():
    """Each trajectory of n_steps yields exactly n_steps - seq_len windows."""
    n_traj, n_steps, seq_len = 4, 50, 10
    expected_windows = n_traj * (n_steps - seq_len)
    gen = TrajectoryGenerator(seed=1)
    ds = gen.generate_stationary_dataset(
        n_trajectories=n_traj, n_steps=n_steps, seq_len=seq_len
    )
    assert len(ds["context_obs"]) == expected_windows


def test_shift_window_count():
    """Shift trajectory of (pre + post) steps yields (pre + post - seq_len) windows."""
    n_traj, n_pre, n_post, seq_len = 2, 40, 40, 10
    expected_windows = n_traj * (n_pre + n_post - seq_len)
    gen = TrajectoryGenerator(seed=2)
    ds = gen.generate_shift_dataset(
        n_trajectories=n_traj,
        n_pre_shift=n_pre,
        n_post_shift=n_post,
        seq_len=seq_len,
    )
    assert len(ds["context_obs"]) == expected_windows


# ─────────────────────────────────────────────────────────────────────────────
# Window indexing correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_target_is_one_step_ahead_of_context():
    """
    target_obs[i] must be a different observation than the last step
    in context_obs[i]. They come from consecutive timesteps, not the
    same one. Verified with zero-noise system so stochasticity doesn't
    mask indexing bugs.
    """
    # Use a non-zero action scale so consecutive obs are guaranteed
    # to differ even with noise; checking they are NOT the same step.
    gen = TrajectoryGenerator(seed=42, obs_noise_std=0.0)
    ds = gen.generate_stationary_dataset(
        n_trajectories=1, n_steps=30, seq_len=5, action_scale=0.0
    )
    for i in range(min(10, len(ds["context_obs"]))):
        last_ctx = ds["context_obs"][i, -1]  # last step of context
        tgt      = ds["target_obs"][i]        # target to predict
        # They should be different observations (one Euler step apart)
        assert not np.allclose(last_ctx, tgt), (
            f"Window {i}: target_obs equals last context obs — "
            f"indexing is off by one or target is not one step ahead."
        )


def test_context_windows_are_sequential():
    """
    In a single-trajectory dataset, consecutive windows should overlap
    by (seq_len - 1) steps: context_obs[i+1, :-1] == context_obs[i, 1:].
    """
    gen = TrajectoryGenerator(seed=10, obs_noise_std=0.0)
    ds  = gen.generate_stationary_dataset(
        n_trajectories=1, n_steps=30, seq_len=5, action_scale=0.0
    )
    for i in range(len(ds["context_obs"]) - 1):
        overlap_next = ds["context_obs"][i + 1, :-1]  # all but last of next window
        overlap_curr = ds["context_obs"][i, 1:]        # all but first of current window
        np.testing.assert_allclose(
            overlap_curr, overlap_next,
            err_msg=f"Windows {i} and {i+1} do not share the expected overlap."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Shift detection tests
# ─────────────────────────────────────────────────────────────────────────────

def test_stationary_dataset_has_no_crossing_windows(stationary_ds):
    """Stationary data has no regime shifts — all shift_step_in_window must be -1."""
    assert np.all(stationary_ds["shift_step_in_window"] == -1)


def test_shift_dataset_has_crossing_windows(shift_ds):
    """Shift dataset must contain windows where the shift falls mid-window."""
    n_crossing = (shift_ds["shift_step_in_window"] >= 0).sum()
    assert n_crossing > 0, "No windows crossing the shift found — shift detection broken."


def test_shift_step_in_window_values_are_valid(shift_ds):
    """All recorded shift_step_in_window values must be in [0, seq_len)."""
    seq_len = 10
    crossing = shift_ds["shift_step_in_window"]
    cross_vals = crossing[crossing >= 0]
    assert (cross_vals >= 0).all()
    assert (cross_vals < seq_len).all()


def test_crossing_window_count_is_exactly_seq_len():
    """
    For one trajectory, exactly seq_len windows span the shift point.
    Window i crosses the shift (at step n_pre_shift) iff:
        0 <= (n_pre_shift - i) < seq_len
    which gives exactly seq_len values of i.
    """
    seq_len   = 10
    n_pre     = 40
    n_post    = 40
    gen       = TrajectoryGenerator(seed=99)
    ds        = gen.generate_shift_dataset(
        n_trajectories=1,
        n_pre_shift=n_pre,
        n_post_shift=n_post,
        seq_len=seq_len,
    )
    n_crossing = (ds["shift_step_in_window"] >= 0).sum()
    assert n_crossing == seq_len, (
        f"Expected exactly {seq_len} crossing windows, got {n_crossing}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Action distribution tests
# ─────────────────────────────────────────────────────────────────────────────

def test_action_distribution_mean_and_std():
    """Actions should be approximately Gaussian(0, action_scale)."""
    action_scale = 0.2
    gen = TrajectoryGenerator(seed=3)
    ds  = gen.generate_stationary_dataset(
        n_trajectories=20, n_steps=100, seq_len=10, action_scale=action_scale
    )
    acts = ds["context_actions"].flatten()
    assert abs(acts.mean()) < 0.02,  f"Action mean too far from 0: {acts.mean():.4f}"
    assert abs(acts.std() - action_scale) < 0.02, (
        f"Action std {acts.std():.4f} too far from {action_scale}"
    )


def test_zero_action_scale_gives_zero_actions():
    """action_scale=0.0 must produce all-zero actions."""
    gen = TrajectoryGenerator(seed=4)
    ds  = gen.generate_stationary_dataset(
        n_trajectories=2, n_steps=30, seq_len=5, action_scale=0.0
    )
    assert np.allclose(ds["context_actions"], 0.0)
    assert np.allclose(ds["next_action"], 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Train/val/test split tests
# ─────────────────────────────────────────────────────────────────────────────

def test_split_sizes_sum_to_total():
    gen = TrajectoryGenerator(seed=5)
    ds  = gen.generate_stationary_dataset(
        n_trajectories=5, n_steps=50, seq_len=10
    )
    N = len(ds["context_obs"])
    train, val, test = gen.split_dataset(ds, val_frac=0.1, test_frac=0.1)

    total = (
        len(train["context_obs"])
        + len(val["context_obs"])
        + len(test["context_obs"])
    )
    assert total == N, f"Split dropped samples: {total} != {N}"


def test_split_produces_correct_keys():
    gen = TrajectoryGenerator(seed=6)
    ds  = gen.generate_stationary_dataset(
        n_trajectories=3, n_steps=30, seq_len=5
    )
    train, val, test = gen.split_dataset(ds)
    for split in (train, val, test):
        assert set(split.keys()) == set(ds.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility tests
# ─────────────────────────────────────────────────────────────────────────────

def test_stationary_dataset_is_reproducible():
    kwargs = dict(n_trajectories=3, n_steps=30, seq_len=5)
    gen_a = TrajectoryGenerator(seed=77)
    gen_b = TrajectoryGenerator(seed=77)
    ds_a  = gen_a.generate_stationary_dataset(**kwargs)
    ds_b  = gen_b.generate_stationary_dataset(**kwargs)
    for k in ds_a:
        np.testing.assert_array_equal(ds_a[k], ds_b[k], err_msg=f"Mismatch in {k}")


def test_shift_dataset_is_reproducible():
    kwargs = dict(n_trajectories=2, n_pre_shift=30, n_post_shift=30, seq_len=5)
    gen_a = TrajectoryGenerator(seed=88)
    gen_b = TrajectoryGenerator(seed=88)
    ds_a  = gen_a.generate_shift_dataset(**kwargs)
    ds_b  = gen_b.generate_shift_dataset(**kwargs)
    for k in ds_a:
        np.testing.assert_array_equal(ds_a[k], ds_b[k], err_msg=f"Mismatch in {k}")


# ─────────────────────────────────────────────────────────────────────────────
# Edge case tests
# ─────────────────────────────────────────────────────────────────────────────

def test_raises_when_n_steps_equals_seq_len():
    """n_steps must be strictly greater than seq_len."""
    gen = TrajectoryGenerator(seed=0)
    with pytest.raises(ValueError, match="Need at least seq_len"):
        gen.generate_stationary_dataset(
            n_trajectories=1, n_steps=10, seq_len=10
        )


def test_raises_when_n_steps_less_than_seq_len():
    gen = TrajectoryGenerator(seed=0)
    with pytest.raises(ValueError):
        gen.generate_stationary_dataset(
            n_trajectories=1, n_steps=5, seq_len=10
        )


def test_single_trajectory_single_window():
    """Minimum valid case: one trajectory, n_steps = seq_len + 1 → 1 window."""
    gen = TrajectoryGenerator(seed=0)
    ds  = gen.generate_stationary_dataset(
        n_trajectories=1, n_steps=11, seq_len=10
    )
    assert len(ds["context_obs"]) == 1