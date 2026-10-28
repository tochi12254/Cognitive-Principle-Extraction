"""
Trajectory generation for the world-model training pipeline.

Generates sliding windows of (context_obs, context_actions, next_action,
target_obs) from SyntheticSystem trajectories, including regime-shift
trajectories for the 'unfair to standalone predictor' evaluation condition.

Design decisions (full reasoning in docs/research-log/RESEARCH_LOG.docx):

1. Full trajectory first, then sliding window extraction:
   One simulation run yields (T - seq_len) overlapping training windows.
   More data-efficient than one simulation call per window, and lets us
   track exactly where a regime shift falls relative to each window.

2. Separate stationary vs. shift datasets:
   Train on stationary (pre-shift only), evaluate on shift trajectories.
   Clean separation between training distribution and test condition.

3. Actions are small Gaussian perturbations:
   At toy scale, actions don't need semantic meaning. They exist so the
   world-model learns action-conditioned prediction rather than purely
   passive sequence continuation. action_scale=0.0 gives zero-action
   trajectories for ablation if needed.

4. NumPy arrays throughout:
   Framework-agnostic data layer. PyTorch DataLoader wraps NumPy natively.
   Tensors belong in the training loop, not the data pipeline.

5. shift_step recorded per window:
   shift_step_in_window is the position within a context window where the
   regime shift occurred. -1 means the window is entirely pre- or post-shift.
   This gives the evaluation loop an exact reference point for checking
   whether prediction error spikes at the right step.
"""

import numpy as np
from typing import Optional

from envs.synthetic_system import SyntheticSystem


class TrajectoryGenerator:
    """
    Generates trajectory datasets from SyntheticSystem.

    Args:
        obs_dim:        Observation dimension passed to SyntheticSystem.
        obs_noise_std:  Observation noise std passed to SyntheticSystem.
        dt:             Integration timestep passed to SyntheticSystem.
        rho_normal:     Lorenz rho for stationary (pre-shift) regime.
        rho_shifted:    Lorenz rho after regime shift.
        seed:           Master seed. Each trajectory uses a derived seed
                        so datasets are reproducible but trajectories are
                        diverse.
    """

    def __init__(
        self,
        obs_dim: int = 6,
        obs_noise_std: float = 0.05,
        dt: float = 0.01,
        rho_normal: float = 28.0,
        rho_shifted: float = 60.0,
        seed: int = 0,
    ):
        self.obs_dim = obs_dim
        self.obs_noise_std = obs_noise_std
        self.dt = dt
        self.rho_normal = rho_normal
        self.rho_shifted = rho_shifted
        self.rng = np.random.default_rng(seed)

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _make_system(self, seed: int, rho: float) -> SyntheticSystem:
        return SyntheticSystem(
            dt=self.dt,
            rho=rho,
            obs_dim=self.obs_dim,
            obs_noise_std=self.obs_noise_std,
            seed=seed,
        )

    def _run_trajectory(
        self,
        system: SyntheticSystem,
        n_steps: int,
        action_scale: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Roll out `system` for `n_steps` steps.

        Returns:
            obs:     shape (n_steps + 1, obs_dim)  — includes initial obs
            actions: shape (n_steps + 1,)           — action[t] was taken at step t
        """
        actions = self.rng.normal(
            scale=action_scale, size=n_steps + 1
        ).astype(np.float32)

        obs = np.empty((n_steps + 1, self.obs_dim), dtype=np.float32)
        obs[0] = system.reset()

        for t in range(n_steps):
            obs[t + 1] = system.step(action=float(actions[t]))

        return obs, actions

    def _extract_windows(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        seq_len: int,
        shift_step: int = -1,
    ) -> dict[str, np.ndarray]:
        """
        Extract all sliding windows of length seq_len from a trajectory.

        Args:
            obs:        shape (T+1, obs_dim)
            actions:    shape (T+1,)
            seq_len:    context window length
            shift_step: absolute step in trajectory where regime shifted.
                        -1 means no shift in this trajectory.

        Returns dict with keys:
            context_obs:        (N, seq_len, obs_dim)
            context_actions:    (N, seq_len)
            next_action:        (N,)
            target_obs:         (N, obs_dim)
            shift_step_in_window: (N,) — position of shift within window,
                                  or -1 if window doesn't cross the shift.
        """
        T = len(obs) - 1  # number of steps (obs has T+1 entries)
        n_windows = T - seq_len

        if n_windows <= 0:
            raise ValueError(
                f"Trajectory has {T} steps but seq_len={seq_len}. "
                f"Need at least seq_len + 1 steps to extract one window."
            )

        context_obs        = np.empty((n_windows, seq_len, self.obs_dim), dtype=np.float32)
        context_actions    = np.empty((n_windows, seq_len), dtype=np.float32)
        next_action        = np.empty((n_windows,), dtype=np.float32)
        target_obs         = np.empty((n_windows, self.obs_dim), dtype=np.float32)
        # Target window: context shifted forward by one step.
        # target_context_obs[i]  = obs[i+1 : i+seq_len+1]
        # target_context_actions[i] = actions[i+1 : i+seq_len+1]
        # This is what the TargetEncoder (a full ContextEncoder) receives.
        # Bounds: for last window i = T-seq_len-1, last index = i+seq_len+1 = T
        # which is the last valid obs index → fits exactly within trajectory.
        target_context_obs     = np.empty((n_windows, seq_len, self.obs_dim), dtype=np.float32)
        target_context_actions = np.empty((n_windows, seq_len), dtype=np.float32)
        shift_in_win           = np.full((n_windows,), -1, dtype=np.int32)

        for i in range(n_windows):
            context_obs[i]            = obs[i : i + seq_len]
            context_actions[i]        = actions[i : i + seq_len]
            next_action[i]            = actions[i + seq_len]
            target_obs[i]             = obs[i + seq_len]
            target_context_obs[i]     = obs[i + 1 : i + seq_len + 1]
            target_context_actions[i] = actions[i + 1 : i + seq_len + 1]

            if shift_step >= 0:
                rel = shift_step - i
                if 0 <= rel < seq_len:
                    shift_in_win[i] = rel

        return {
            "context_obs":            context_obs,
            "context_actions":        context_actions,
            "next_action":            next_action,
            "target_obs":             target_obs,
            "target_context_obs":     target_context_obs,
            "target_context_actions": target_context_actions,
            "shift_step_in_window":   shift_in_win,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def generate_stationary_dataset(
        self,
        n_trajectories: int,
        n_steps: int,
        seq_len: int,
        action_scale: float = 0.1,
    ) -> dict[str, np.ndarray]:
        """
        Generate a dataset of windows from stationary (pre-shift) trajectories.
        This is the training distribution: rho stays at rho_normal throughout.

        Args:
            n_trajectories: number of independent trajectories to simulate.
            n_steps:        steps per trajectory (must be > seq_len).
            seq_len:        context window length fed to the world-model.
            action_scale:   std of Gaussian action perturbations.

        Returns:
            Concatenated dict of arrays across all trajectories.
            shift_step_in_window is all -1 (no shift in stationary data).
        """
        all_windows = []
        for i in range(n_trajectories):
            traj_seed = int(self.rng.integers(0, 2**31))
            system = self._make_system(seed=traj_seed, rho=self.rho_normal)
            obs, actions = self._run_trajectory(system, n_steps, action_scale)
            windows = self._extract_windows(obs, actions, seq_len, shift_step=-1)
            all_windows.append(windows)

        return self._concat_windows(all_windows)

    def generate_shift_dataset(
        self,
        n_trajectories: int,
        n_pre_shift: int,
        n_post_shift: int,
        seq_len: int,
        action_scale: float = 0.1,
    ) -> dict[str, np.ndarray]:
        """
        Generate a dataset of windows from regime-shift trajectories.
        This is the evaluation distribution: rho changes mid-trajectory.

        The shift happens at step n_pre_shift. Windows that span this
        step have shift_step_in_window >= 0; all others have -1.

        Args:
            n_trajectories: number of independent shift trajectories.
            n_pre_shift:    steps before regime shift (rho_normal).
            n_post_shift:   steps after regime shift (rho_shifted).
            seq_len:        context window length.
            action_scale:   std of Gaussian action perturbations.

        Returns:
            Concatenated dict including shift_step_in_window markers.
        """
        all_windows = []
        for i in range(n_trajectories):
            traj_seed = int(self.rng.integers(0, 2**31))
            system = self._make_system(seed=traj_seed, rho=self.rho_normal)

            # Phase 1: pre-shift
            obs_pre, actions_pre = self._run_trajectory(
                system, n_pre_shift, action_scale
            )

            # Regime shift: change the causal driver
            system.set_regime(self.rho_shifted)

            # Phase 2: post-shift — continue from current state
            # We don't reset: the model must adapt to the shift,
            # not detect a teleport to a new initial condition.
            obs_post = np.empty((n_post_shift, self.obs_dim), dtype=np.float32)
            actions_post = self.rng.normal(
                scale=action_scale, size=n_post_shift
            ).astype(np.float32)

            for t in range(n_post_shift):
                obs_post[t] = system.step(action=float(actions_post[t]))

            # Stitch into one continuous trajectory
            # obs_pre has shape (n_pre_shift+1, obs_dim); last entry is
            # the obs at step n_pre_shift (before shift dynamics kick in).
            # obs_post starts from step n_pre_shift+1 onwards.
            obs_full = np.concatenate([obs_pre, obs_post], axis=0)
            actions_full = np.concatenate(
                [actions_pre, actions_post, [0.0]]  # pad final action
            )

            windows = self._extract_windows(
                obs_full,
                actions_full,
                seq_len,
                shift_step=n_pre_shift,
            )
            all_windows.append(windows)

        return self._concat_windows(all_windows)

    def split_dataset(
        self,
        dataset: dict[str, np.ndarray],
        val_frac: float = 0.1,
        test_frac: float = 0.1,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ]:
        """
        Shuffle and split a dataset into train / val / test.

        Returns: (train_dataset, val_dataset, test_dataset)
        """
        N = len(next(iter(dataset.values())))
        idx = self.rng.permutation(N)

        n_test = int(N * test_frac)
        n_val  = int(N * val_frac)
        n_train = N - n_val - n_test

        train_idx = idx[:n_train]
        val_idx   = idx[n_train : n_train + n_val]
        test_idx  = idx[n_train + n_val :]

        def subset(d, indices):
            return {k: v[indices] for k, v in d.items()}

        return subset(dataset, train_idx), subset(dataset, val_idx), subset(dataset, test_idx)

    # ─────────────────────────────────────────────────────────────────────
    # Internal utility
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _concat_windows(windows_list: list[dict]) -> dict[str, np.ndarray]:
        keys = windows_list[0].keys()
        return {k: np.concatenate([w[k] for w in windows_list], axis=0) for k in keys}