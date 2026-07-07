"""
MultiMazeGoalDataset: horizon windows pooled across several pointmaze maps,
each tagged with its source map's padded occupancy grid.

Mirrors the repo's GoalDataset (trajectories = [normed_action | normed_obs],
conditions inpaint start+goal) but:
  * uses ONE shared GlobalNormalizer across maps, and
  * returns the maze grid alongside each window so the model can condition on it.

Mujoco-free: episodes come straight from the on-disk .npy files via data_utils;
base grids come from ast-parsing maze.py via maze_grids, and giant variant
grids can come from explicit JSON-backed map specs. No gym.make / simulator.
"""
from collections import namedtuple
import numpy as np
import torch

from . import data_utils as DU
from . import maze_grids as MG
from .global_normalizer import GlobalNormalizer, fit_global_normalizer

# Batch carries the maze grid as a third field (GoalDataset's Batch has two).
MapBatch = namedtuple("MapBatch", "trajectories conditions maze")


class MultiMazeGoalDataset(torch.utils.data.Dataset):
    def __init__(self, maze_types=None, horizon=256, normalizer=None,
                 canvas_hw=None, pad_anchor="corner", min_episode_len=None,
                 max_episodes_per_map=None, map_specs=None):
        if map_specs is None:
            if maze_types is None:
                raise ValueError("provide maze_types or map_specs")
            map_specs = DU.base_map_specs(list(maze_types))
            self.maze_types = list(maze_types)
        else:
            if maze_types is not None:
                raise ValueError("pass either maze_types or map_specs, not both")
            self.maze_types = None

        self.map_specs = [self._coerce_map_spec(s) for s in map_specs]
        self.map_ids = [s["map_id"] for s in self.map_specs]
        if self.maze_types is None:
            self.maze_types = list(self.map_ids)
        self.horizon = horizon
        self.pad_anchor = pad_anchor
        self.min_episode_len = min_episode_len

        # shared normalizer (fit if not supplied)
        self.normalizer = normalizer or fit_global_normalizer(self.map_specs)

        # grids -> common canvas (giant defines 12x16 by default)
        raw_grids = {s["map_id"]: s["grid"] for s in self.map_specs}
        self.canvas_hw = canvas_hw or MG.canvas_size(raw_grids)
        self.grids = {}          # map_id -> padded float32 canvas grid
        for map_id, g in raw_grids.items():
            padded, _ = MG.pad_to_canvas(g, self.canvas_hw, pad_value=1,
                                         anchor=self.pad_anchor)
            self.grids[map_id] = padded.astype(np.float32)

        # Load + normalize episodes. Instead of materializing one Python tuple
        # per window, keep cumulative window counts; full variant training has
        # tens of millions of valid windows.
        self.episodes = {}       # map_id -> list of {normed obs/act} or None
        self._episode_refs = []  # slot -> (map_index, ep_idx)
        self._cum_windows = []   # exclusive cumsum of windows per slot
        self._first_index_by_map = {}
        total_windows = 0
        for map_index, spec in enumerate(self.map_specs):
            map_id = spec["map_id"]
            eps = DU.load_episodes_from_spec(spec)
            if max_episodes_per_map is not None:
                eps = eps[:max_episodes_per_map]
            normed_eps = []
            for ei, e in enumerate(eps):
                obs = e["observations"]; act = e["actions"]
                L = obs.shape[0]
                required_len = horizon if min_episode_len is None else max(horizon, min_episode_len)
                if L < required_len:
                    normed_eps.append(None)
                    continue
                nobs = self.normalizer.normalize(obs.astype(np.float32), "observations")
                nact = self.normalizer.normalize(act.astype(np.float32), "actions")
                normed_eps.append({"obs": nobs, "act": nact})
                # non-padded windows only (use_padding=False in the repo config)
                max_start = L - horizon
                if max_start <= 0:
                    continue
                if map_id not in self._first_index_by_map:
                    self._first_index_by_map[map_id] = total_windows
                total_windows += max_start
                self._episode_refs.append((map_index, ei))
                self._cum_windows.append(total_windows)
            self.episodes[map_id] = normed_eps

        self._cum_windows = np.asarray(self._cum_windows, dtype=np.int64)
        self.n_windows = int(total_windows)

        self.observation_dim = 2
        self.action_dim = 2
        self.transition_dim = 4

    def _coerce_map_spec(self, spec):
        spec = dict(spec)
        if "map_id" not in spec:
            if "maze_type" in spec:
                spec["map_id"] = str(spec["maze_type"])
            else:
                raise KeyError("map spec needs map_id")
        spec["map_id"] = str(spec["map_id"])

        if "grid" not in spec or spec["grid"] is None:
            if "maze_type" in spec and "npz_path" not in spec:
                spec["grid"] = MG.parse_grid(spec["maze_type"])
            else:
                raise KeyError("map spec %s needs an explicit grid" % spec["map_id"])

        grid = np.asarray(spec["grid"], dtype=np.int64)
        values = set(np.unique(grid).tolist())
        if grid.ndim != 2 or not values.issubset({0, 1}):
            raise ValueError("map spec %s grid must be HxW 0/1, got shape=%r values=%r" %
                             (spec["map_id"], grid.shape, sorted(values)))
        spec["grid"] = grid
        return spec

    def __len__(self):
        return self.n_windows

    def _index_to_ref(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        slot = int(np.searchsorted(self._cum_windows, idx, side="right"))
        prev = 0 if slot == 0 else int(self._cum_windows[slot - 1])
        start = int(idx - prev)
        map_index, ep_idx = self._episode_refs[slot]
        map_id = self.map_ids[map_index]
        return map_id, ep_idx, start

    def source_map_id(self, idx):
        """Return the map_id backing a global dataset index."""
        map_id, _, _ = self._index_to_ref(idx)
        return map_id

    def first_index_for_map(self, map_id):
        """Return the first global sample index for a map with at least one window."""
        return self._first_index_by_map[str(map_id)]

    def serializable_map_specs(self):
        """Checkpoint-safe map specs, including grids but excluding callables."""
        keep = ("map_id", "kind", "maze_type", "task", "var",
                "validation", "npz_path", "json_path")
        out = []
        for spec in self.map_specs:
            d = {k: spec[k] for k in keep if k in spec}
            d["grid"] = np.asarray(spec["grid"], dtype=np.int64).tolist()
            out.append(d)
        return out

    def get_conditions(self, observations):
        # inpaint current + goal observation, exactly as GoalDataset
        return {0: observations[0], self.horizon - 1: observations[-1]}

    def __getitem__(self, idx):
        map_id, ei, start = self._index_to_ref(idx)
        ep = self.episodes[map_id][ei]
        end = start + self.horizon
        obs = ep["obs"][start:end]
        act = ep["act"][start:end]
        conditions = self.get_conditions(obs)
        trajectories = np.concatenate([act, obs], axis=-1).astype(np.float32)
        maze = self.grids[map_id]  # [Hc, Wc] float32
        return MapBatch(trajectories, conditions, maze)


def map_batch_collate(samples):
    """
    Collate MapBatch samples into batched tensors:
      trajectories: [B, H, transition_dim]
      conditions:   {t: [B, obs_dim]}
      maze:         [B, Hc, Wc]
    """
    trajs = torch.as_tensor(np.stack([s.trajectories for s in samples]))
    mazes = torch.as_tensor(np.stack([s.maze for s in samples]))
    cond_keys = samples[0].conditions.keys()
    conditions = {
        t: torch.as_tensor(np.stack([s.conditions[t] for s in samples]))
        for t in cond_keys
    }
    return MapBatch(trajs, conditions, mazes)
