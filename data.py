"""WaypointSubtrajectoryDataset: variable-stride waypoint sampler over HDF5.

HWM paper App. B.3 (Push-T):
  > "To construct training sequences, we subsample trajectory segments
  > with lengths uniformly drawn between 25 and 70 timesteps. From each
  > segment, we sample N = 5 waypoint states, which define the high-level
  > transitions."

Each __getitem__:
  1. picks a random episode,
  2. samples a segment length L in LeWM blocks ~ Uniform(min_blocks, max_blocks),
  3. samples a start offset s ~ Uniform(0, T_blocks - L),
  4. anchors t_1 = s, t_N = s + L (so the sampled-segment length is exact),
  5. samples N-2 middle waypoints uniformly from (s+1, ..., s+L-1) without
     replacement (matches the paper's Franka recipe of "middle waypoint
     sampled uniformly", generalised to N=5 for Push-T).

The {min,max}_blocks bounds are LeWM blocks; multiply by frameskip to get
env-step lengths (Push-T: 5..14 blocks = 25..70 env steps).

Wraps `swm.data.HDF5Dataset` -- we use its `_load_slice` to read raw env
steps and reshape actions into LeWM blocks (one block = `frameskip` env
steps, matching `train.py:92` effective_act_dim).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class WaypointSubtrajectoryDataset(Dataset):
    """Per __getitem__, sample N waypoint indices from one episode.

    Args:
        base: an `swm.data.HDF5Dataset` (or any subclass exposing `lengths`,
            `offsets`, `frameskip`, and `_load_slice(ep, start, end)`).
        n_target: target number of waypoints per item (HWM Push-T = 5).
        min_blocks, max_blocks: SEGMENT-length bounds in LeWM blocks
            (Push-T: 5..14 blocks = 25..70 env steps at frameskip 5).
            The N waypoints are placed *inside* a sampled segment of this
            length, so the average inter-waypoint gap is L/(N-1).
        mode: 'variable' (paper recipe) or 'fixed' (HWM_PLDM-style sanity).
        stride: required when mode='fixed'.
        samples_per_episode: how many distinct (random-segment) items to
            emit per episode per epoch. Default 1. Bumping to e.g. 4
            gives 4x more random-segment coverage of each trajectory per
            epoch (each call samples a fresh random segment, so the 4 items
            are not duplicates). Same trajectories, more variety per epoch;
            wall-clock per epoch grows roughly linearly.
        episode_indices: optional list/array of episode indices to restrict
            the dataset to. When set, valid_episodes is built from this
            subset (intersected with the length filter). Used by
            train_highlevel.py to split TRAIN and VAL at episode
            granularity -- a flat random_split AFTER samples_per_episode
            expansion would leak segments of the same trajectory across
            both splits.
        action_normalizer: optional callable applied to each action chunk
            (raw (n_envsteps, action_dim) -> normalised same shape) before
            reshape into LeWM blocks. Mirrors the `get_column_normalizer`
            pattern from train.py:62.
        pixel_transform: optional callable applied to a (N, C, H, W) tensor
            of waypoint pixels. Typically `get_img_preprocessor` from utils.
        seed: optional seed for reproducible per-call rng.
    """

    def __init__(
        self,
        base,
        n_target: int,
        min_blocks: int,
        max_blocks: int,
        mode: str = 'variable',
        stride: int | None = None,
        samples_per_episode: int = 1,
        episode_indices=None,
        action_normalizer=None,
        pixel_transform=None,
        seed: int | None = None,
    ):
        super().__init__()
        self.base = base
        self.frameskip = base.frameskip
        self.lengths = base.lengths
        self.n_target = int(n_target)
        self.min_blocks = int(min_blocks)
        self.max_blocks = int(max_blocks)
        self.mode = mode
        self.stride = int(stride) if stride is not None else None
        self.samples_per_episode = max(1, int(samples_per_episode))
        self.action_normalizer = action_normalizer
        self.pixel_transform = pixel_transform

        # Episode must accommodate (a) a segment of length L >= min_blocks,
        # which spans L+1 distinct block positions [s, s+L] and so requires
        # T_blocks >= min_blocks + 1, AND (b) N distinct waypoint indices,
        # which requires T_blocks >= N. Take the larger of the two bounds.
        # Without the +1 on min_blocks, T_blocks = min_blocks episodes hit
        # the sampler's fallback path and silently violate the user-stated
        # min_blocks lower bound (e.g. Push-T config user says min L=5 but
        # the sampler returns L=4 for T_blocks=5 episodes).
        if mode == 'variable':
            min_blocks_required = max(self.min_blocks + 1, self.n_target)
        elif mode == 'fixed':
            assert self.stride is not None, "fixed mode needs stride"
            min_blocks_required = (self.n_target - 1) * self.stride + 1
        else:
            raise ValueError(f'unknown mode {mode}')
        min_envsteps_required = min_blocks_required * self.frameskip

        if episode_indices is not None:
            # Restrict to the caller-provided subset; still apply the length
            # filter to be safe (caller may pass dirty indices).
            requested = {int(e) for e in episode_indices}
            self.valid_episodes = np.array(
                sorted(
                    ep for ep in requested
                    if int(self.lengths[ep]) >= min_envsteps_required
                ),
                dtype=np.int64,
            )
        else:
            self.valid_episodes = np.array(
                [ep for ep, L in enumerate(self.lengths) if L >= min_envsteps_required],
                dtype=np.int64,
            )
        if len(self.valid_episodes) == 0:
            raise ValueError(
                f'No episodes long enough for {n_target} waypoints '
                f'with {min_blocks_required} blocks min'
            )

        # Per-worker rng. Lazy-initialised on first __getitem__ call so
        # each DataLoader worker has its own persistent rng that advances
        # per call (avoids the seed-reuse bug under persistent_workers=True).
        # If self._init_seed is set, the rng is seeded deterministically
        # (mixed with worker_id); otherwise OS entropy is used at creation
        # but the stream advances naturally per call.
        self._init_seed = seed
        self._rng = None

    def __len__(self):
        return int(len(self.valid_episodes)) * self.samples_per_episode

    def _get_rng(self):
        """Return the per-worker persistent rng, lazy-creating on first use.

        Each DataLoader worker has its own copy of `self`, so this dict is
        worker-local. The rng is created once per worker and advances
        naturally as samples are drawn -- no per-call reseeding, so
        successive epochs see fresh waypoints even with persistent_workers.
        """
        if self._rng is None:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else -1
            if self._init_seed is not None:
                # Mix the init seed with worker_id (×7919, a prime, avoids
                # alignment between workers) so workers see different streams.
                seed = (self._init_seed + worker_id * 7919) % (2 ** 31)
                self._rng = np.random.default_rng(seed)
            else:
                self._rng = np.random.default_rng()
        return self._rng

    def _sample_waypoints(self, T_blocks: int, rng) -> list[int]:
        """Sample exactly n_target LeWM-block waypoint indices [t_1, ..., t_N].

        HWM paper App. B.3 (Push-T) recipe:
          1. L ~ Uniform(min_blocks, max_blocks)   -- segment length in blocks
          2. s ~ Uniform(0, T_blocks - 1 - L)      -- start offset in episode
          3. anchor t_1 = s, t_N = s + L           -- so segment span = L exactly
          4. sample N-2 middle waypoints uniformly from (s+1, ..., s+L-1)
             without replacement
        """
        N = self.n_target

        if self.mode == 'fixed':
            t = [k * self.stride for k in range(N)]
            assert t[-1] < T_blocks, (
                f'fixed-stride {self.stride} requires T_blocks > '
                f'{(N - 1) * self.stride}; episode has {T_blocks}'
            )
            return [int(x) for x in t]

        # 1. Sample segment length L in LeWM blocks.
        #    Cap by (a) the episode's available range and (b) the minimum
        #    required to fit N distinct waypoint indices.
        max_eff = min(self.max_blocks, T_blocks - 1)
        min_eff = max(self.min_blocks, N - 1)
        if max_eff < min_eff:
            # Pathological short-episode fallback. Use whatever fits.
            L = max(N - 1, min(T_blocks - 1, self.min_blocks))
        else:
            L = int(rng.integers(min_eff, max_eff + 1))

        # 2. Sample start offset s within the episode.
        max_s = T_blocks - 1 - L
        s = int(rng.integers(0, max_s + 1)) if max_s > 0 else 0

        # 3-4. Anchor first/last waypoint at segment endpoints; sample the
        # N-2 middle indices from the open interior (s+1, ..., s+L-1).
        if N - 2 > 0:
            interior = np.arange(s + 1, s + L, dtype=np.int64)
            middle = sorted(rng.choice(interior, size=N - 2, replace=False).tolist())
        else:
            middle = []
        return [s] + [int(x) for x in middle] + [s + L]

    def __getitem__(self, idx: int):
        # idx is in [0, len(valid_episodes) * samples_per_episode).
        # Map every samples_per_episode consecutive idx values to one episode;
        # each one samples a fresh random segment (rng is per-call, so the
        # samples_per_episode items from the same episode are different).
        ep = int(self.valid_episodes[idx // self.samples_per_episode])
        T_blocks = int(self.lengths[ep]) // self.frameskip

        # Persistent per-worker rng -- advances naturally per call so
        # different epochs see different waypoint samples even when
        # persistent_workers=True (no seed-reuse bug).
        rng = self._get_rng()
        t = self._sample_waypoints(T_blocks, rng)
        N = len(t)
        # _sample_waypoints is built to always return exactly n_target indices
        # given a non-trivial config + the validity filter; assert that loudly.
        assert N == self.n_target, (
            f'_sample_waypoints returned {N} waypoints, expected {self.n_target}'
        )

        s_env = t[0] * self.frameskip
        e_env = (t[-1] + 1) * self.frameskip
        steps = self.base._load_slice(ep, s_env, e_env)

        # steps['pixels']: (t[-1] - t[0] + 1, C, H, W) -- already frameskipped.
        # HDF5Dataset._load_slice always returns torch tensors (hdf5.py:93).
        pixels = steps['pixels']
        local_idx = [tk - t[0] for tk in t]
        waypoint_pixels = pixels[local_idx]

        # steps['action']: (e_env - s_env, action_dim) -- raw env-step actions.
        raw_action = steps['action']
        if self.action_normalizer is not None:
            raw_action = self.action_normalizer(raw_action)

        action_dim = raw_action.shape[-1]
        a_block = self.frameskip * action_dim
        L_max = self.max_blocks

        actions_chunk = torch.zeros(N - 1, L_max, a_block, dtype=raw_action.dtype)
        actions_mask = torch.zeros(N - 1, L_max, dtype=torch.bool)
        for k in range(N - 1):
            n_blocks = t[k + 1] - t[k]
            cs = (t[k] - t[0]) * self.frameskip
            ce = (t[k + 1] - t[0]) * self.frameskip
            chunk = raw_action[cs:ce].reshape(n_blocks, a_block)
            actions_chunk[k, :n_blocks] = chunk
            actions_mask[k, :n_blocks] = True

        if self.pixel_transform is not None:
            waypoint_pixels = self.pixel_transform(waypoint_pixels)

        return {
            # Use 'pixels' key so existing pixel transforms (e.g.
            # get_img_preprocessor) work without modification.
            'pixels': waypoint_pixels,
            'actions_chunk': actions_chunk,
            'actions_mask': actions_mask,
            'episode_idx': torch.tensor(ep, dtype=torch.long),
            'n_waypoints': torch.tensor(N, dtype=torch.long),
        }
