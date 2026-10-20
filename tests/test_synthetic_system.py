import numpy as np
import pytest

from envs.synthetic_system import SyntheticSystem


def test_reset_returns_correct_obs_dim():
    sys = SyntheticSystem(obs_dim=6, seed=0)
    obs = sys.reset()
    assert obs.shape == (6,)


def test_step_returns_correct_obs_dim():
    sys = SyntheticSystem(obs_dim=6, seed=0)
    sys.reset()
    obs = sys.step(action=0.0)
    assert obs.shape == (6,)


def test_long_rollout_is_numerically_stable():
    """Euler integration on a chaotic system can diverge with a bad
    dt/parameter combination. Confirm it doesn't over a long run."""
    sys = SyntheticSystem(seed=1)
    sys.reset()
    states = []
    for _ in range(5000):
        sys.step(action=0.0)
        states.append(sys.true_state().copy())
    states = np.array(states)
    assert not np.isnan(states).any()
    assert not np.isinf(states).any()
    assert np.abs(states).max() < 200


def test_intervene_overwrites_latent_state():
    sys = SyntheticSystem(seed=2)
    sys.reset()
    sys.intervene(index=0, value=100.0)
    assert sys.true_state()[0] == pytest.approx(100.0)


def test_intervene_rejects_invalid_index():
    sys = SyntheticSystem(seed=2)
    sys.reset()
    with pytest.raises(ValueError):
        sys.intervene(index=5, value=1.0)


def test_regime_shift_changes_qualitative_behavior():
    """rho=28 keeps the system on the chaotic attractor (away from
    origin). rho=0.5 collapses dynamics to a fixed point near origin.
    This is the regime-shift lever the causal module will later be
    tested against."""
    sys_chaotic = SyntheticSystem(seed=3, rho=28.0)
    sys_chaotic.reset(z0=[1.0, 1.0, 1.0]) # type: ignore
    for _ in range(200):
        sys_chaotic.step()

    sys_fixed_point = SyntheticSystem(seed=3, rho=0.5)
    sys_fixed_point.reset(z0=[1.0, 1.0, 1.0]) # type: ignore
    for _ in range(200):
        sys_fixed_point.step()

    dist_chaotic = np.linalg.norm(sys_chaotic.true_state())
    dist_fixed = np.linalg.norm(sys_fixed_point.true_state())

    assert dist_chaotic > 5.0
    assert dist_fixed < 1.0


def test_observation_hides_true_state():
    """Sanity check that obs is not just a linear passthrough of z —
    the world-model must learn through the nonlinearity + noise, not
    trivially invert it."""
    sys = SyntheticSystem(seed=4, obs_noise_std=0.0)
    sys.reset(z0=[1.0, 1.0, 1.0]) # type: ignore
    obs = sys.step()
    assert np.all(np.abs(obs) < 1.0)


def test_seed_reproducibility():
    sys_a = SyntheticSystem(seed=42)
    sys_a.reset(z0=[0.5, 0.5, 0.5]) # type: ignore
    traj_a = [sys_a.step().copy() for _ in range(20)]

    sys_b = SyntheticSystem(seed=42)
    sys_b.reset(z0=[0.5, 0.5, 0.5]) # type: ignore
    traj_b = [sys_b.step().copy() for _ in range(20)]

    for a, b in zip(traj_a, traj_b):
        np.testing.assert_allclose(a, b)