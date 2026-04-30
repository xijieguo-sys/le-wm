"""WaypointSubtrajectoryDataset: variable-stride waypoint sampler over HDF5.

HWM paper Sec. 3.1 -- the high-level world model trains on
(z_{t_k}, a_{t_k:t_{k+1}}, z_{t_{k+1}}) triples sampled at variable stride.
Each __getitem__ samples N waypoint LeWM-block indices from one episode
and returns padded action chunks for the inter-waypoint transitions.

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
        min_blocks, max_blocks: variable-stride bounds in LeWM blocks
            (Push-T: 5--14, matching paper's 25--70 env steps at frameskip 5).
        mode: 'variable' (paper recipe) or 'fixed' (HWM_PLDM-style sanity).
        stride: required when mode='fixed'.
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
        self.action_normalizer = action_normalizer
        self.pixel_transform = pixel_transform

        # Episode is valid if it can fit at least N waypoints with the
        # *minimum* stride between each (gives a lower bound on length).
        if mode == 'variable':
            min_blocks_required = (self.n_target - 1) * self.min_blocks + 1
        elif mode == 'fixed':
            assert self.stride is not None, "fixed mode needs stride"
            min_blocks_required = (self.n_target - 1) * self.stride + 1
        else:
            raise ValueError(f'unknown mode {mode}')
        min_envsteps_required = min_blocks_required * self.frameskip

        self.valid_episodes = np.array(
            [ep for ep, L in enumerate(self.lengths) if L >= min_envsteps_required],
            dtype=np.int64,
        )
        if len(self.valid_episodes) == 0:
            raise ValueError(
                f'No episodes long enough for {n_target} waypoints '
                f'with {min_blocks_required} blocks min'
            )

        # Note: we do NOT seed a single rng -- DataLoader workers each get
        # their own rng via torch's worker_init_fn / numpy seeding.
        self._init_seed = seed

    def __len__(self):
        return int(len(self.valid_episodes))

    def _sample_waypoints(self, T_blocks: int, rng) -> list[int]:
        """Sample exactly n_target LeWM-block waypoint indices [t_1, ..., t_N].

        Guarantees N waypoints by clamping the per-episode max gap so the
        full sequence always fits: max_eff = min(max_blocks, (T-1) // (N-1)).
        For short episodes this narrows the stride distribution; for long
        ones it leaves the user-specified [min, max] range intact.
        """
        N = self.n_target

        if self.mode == 'fixed':
            t = [k * self.stride for k in range(N)]
            assert t[-1] < T_blocks, (
                f'fixed-stride {self.stride} requires T_blocks > '
                f'{(N - 1) * self.stride}; episode has {T_blocks}'
            )
            return [int(x) for x in t]

        # variable stride; HWM paper recipe (architecture proposal §1.5)
        max_eff = min(self.max_blocks, (T_blocks - 1) // (N - 1))
        if max_eff < self.min_blocks:
            # Episode too short for [min,max] -- collapse to fixed min spacing.
            return [k * self.min_blocks for k in range(N)]

        t = [0]
        for _ in range(N - 1):
            gap = int(rng.integers(self.min_blocks, max_eff + 1))
            t.append(t[-1] + gap)
        return t

    def __getitem__(self, idx: int):
        ep = int(self.valid_episodes[idx])
        T_blocks = int(self.lengths[ep]) // self.frameskip

        # Per-call rng. Use torch.initial_seed() inside workers (Lightning
        # advances this per-epoch and per-worker) so successive epochs sample
        # different waypoints. Reproducible given the same Lightning seed.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            base = torch.initial_seed()
        elif self._init_seed is not None:
            base = self._init_seed
        else:
            base = int(np.random.SeedSequence().entropy)
        seed = (base + idx) % (2 ** 31)
        rng = np.random.default_rng(seed)
        t = self._sample_waypoints(T_blocks, rng)
        N = len(t)
        if N < 2:
            # Pathological: degrade to a fixed-stride fallback so the loader
            # doesn't crash. Should be rare given valid_episodes filter.
            t = list(range(min(self.n_target, T_blocks)))
            N = len(t)

        s_env = t[0] * self.frameskip
        e_env = (t[-1] + 1) * self.frameskip
        steps = self.base._load_slice(ep, s_env, e_env)

        # steps['pixels']: (t[-1] - t[0] + 1, C, H, W) -- already frameskipped.
        pixels = steps['pixels']
        local_idx = [tk - t[0] for tk in t]
        if torch.is_tensor(pixels):
            waypoint_pixels = pixels[local_idx]
        else:
            waypoint_pixels = torch.as_tensor(np.asarray(pixels)[local_idx])

        # steps['action']: (e_env - s_env, action_dim) -- raw env-step actions.
        raw_action = steps['action']
        if not torch.is_tensor(raw_action):
            raw_action = torch.as_tensor(np.asarray(raw_action))
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
