"""
Mujoco-free loading of the OGBench pointmaze offline datasets.

Each map's data lives in pointmaze/ogbench/data/pointmaze-<maze>-navigate-v0/
as observations.npy / actions.npy / terminals.npy (a `terminals` flag marks
the last transition of each fixed-length episode). We segment into per-episode
(observations, actions) arrays -- no simulator, no gym.make -- which is all the
diffusion trainer needs (it predicts [action, observation] sequences and
inpaints start/goal from the trajectory itself).

The giant variant data is stored as single .npz files outside this repo, with
per-variant grids in JSON. The helpers below expose both sources through the
same "map spec" surface so base maps and variants can be pooled in one dataset.
"""
import os
import json
import numpy as np

from . import maze_grids as MG

_HERE = os.path.dirname(os.path.abspath(__file__))
_PM = os.path.dirname(_HERE)
_REPO = os.path.dirname(_PM)
DATA_ROOT = os.path.join(_PM, "ogbench", "data")
# DEFAULT_VARIANT_DIR = "/home/yufei/research/diffusion/ogbench/data_gen_scripts/newdata_mazev1"
DEFAULT_VARIANT_DIR = "/home/yufei/research/diffusion/ogbench/data_gen_scripts/newdata_mazev2"
# DEFAULT_VARIANT_JSON_DIR = os.path.join(_REPO, "maze_update", "maze_variants")
DEFAULT_VARIANT_JSON_DIR = os.path.join(_REPO, "maze_update", "maze_variants_v2")


def dataset_dir(maze_type):
    return os.path.join(DATA_ROOT, f"pointmaze-{maze_type}-navigate-v0")


def load_raw(maze_type):
    """Return (observations, actions, terminals) as float32/bool arrays."""
    d = dataset_dir(maze_type)
    obs = np.load(os.path.join(d, "observations.npy")).astype(np.float32)
    act = np.load(os.path.join(d, "actions.npy")).astype(np.float32)
    term = np.load(os.path.join(d, "terminals.npy"))
    return obs, act, term.astype(bool)


def load_raw_npz(path):
    """
    Return (observations, actions, terminals) from one variant .npz file.

    The file also contains qpos/qvel, which are simulator-side extras and are
    intentionally ignored by the diffusion trainer.
    """
    with np.load(path) as data:
        obs = data["observations"].astype(np.float32)
        act = data["actions"].astype(np.float32)
        term = data["terminals"].astype(bool)
    return obs, act, term


def segment_episodes(obs, act, term):
    """
    Split flat (N, .) arrays into a list of per-episode dicts using the
    terminal flag as an inclusive episode end. Returns
    [{'observations': (L,2), 'actions': (L,2)}, ...].
    """
    ends = np.where(term)[0]
    episodes = []
    start = 0
    for e in ends:
        sl = slice(start, e + 1)
        episodes.append({"observations": obs[sl], "actions": act[sl]})
        start = e + 1
    # tail without a terminal (defensive; OGBench data ends cleanly on a terminal)
    if start < len(obs):
        episodes.append({"observations": obs[start:], "actions": act[start:]})
    return episodes


def load_episodes(maze_type):
    """Convenience: load_raw + segment_episodes for one map."""
    return segment_episodes(*load_raw(maze_type))


def load_episodes_npz(path):
    """Convenience: load_raw_npz + segment_episodes for one variant file."""
    return segment_episodes(*load_raw_npz(path))


def _task_label(task):
    if isinstance(task, str):
        return task if task.startswith("task") else "task" + task
    return "task%d" % int(task)


def _task_number(task):
    if isinstance(task, str):
        return int(task.replace("task", ""))
    return int(task)


def variant_npz_path(variant_dir, task, var, validation=False):
    """Path to pointmaze-giant-navigate-v0-taskX-varY(.|-val.)npz."""
    suffix = "-val" if validation else ""
    name = "pointmaze-giant-navigate-v0-%s-var%d%s.npz" % (
        _task_label(task), int(var), suffix)
    return os.path.join(variant_dir, name)


def variant_json_path(variant_json_dir, task):
    """Path to the JSON containing all 12 variants for one giant task."""
    return os.path.join(variant_json_dir, "giant_task%d.json" % _task_number(task))


def load_variant_grid(variant_json_dir, task, var):
    """Return the 12x16 0/1 occupancy grid for one giant task/variant."""
    path = variant_json_path(variant_json_dir, task)
    with open(path, "r") as f:
        data = json.load(f)
    grid = np.asarray(data["variants"][int(var)]["maze_map"], dtype=np.int64)
    values = set(np.unique(grid).tolist())
    if grid.ndim != 2 or not values.issubset({0, 1}):
        raise ValueError("invalid variant grid in %s var%d: shape=%r values=%r" %
                         (path, int(var), grid.shape, sorted(values)))
    return grid


def load_variant_raw(variant_dir, task, var, validation=False):
    """Return raw arrays for one giant variant .npz."""
    return load_raw_npz(variant_npz_path(variant_dir, task, var, validation))


def load_variant_episodes(variant_dir, task, var, validation=False):
    """Return segmented episodes for one giant variant .npz."""
    return segment_episodes(*load_variant_raw(variant_dir, task, var, validation))


def base_map_specs(maze_types):
    """Serializable map specs for the original .npy-backed OGBench maps."""
    raw_grids = MG.all_grids(maze_types)
    specs = []
    for maze_type in maze_types:
        specs.append({
            "map_id": str(maze_type),
            "kind": "base",
            "maze_type": str(maze_type),
            "grid": raw_grids[maze_type],
        })
    return specs


def giant_variant_map_specs(variant_dir=DEFAULT_VARIANT_DIR,
                            variant_json_dir=DEFAULT_VARIANT_JSON_DIR,
                            tasks=(1, 2, 3, 4),
                            vars=tuple(range(12)),
                            validation=False):
    """Serializable map specs for giant task/variant .npz files."""
    specs = []
    for task in tasks:
        task_num = _task_number(task)
        for var in vars:
            var = int(var)
            specs.append({
                "map_id": "giant_task%d_var%d%s" % (
                    task_num, var, "_val" if validation else ""),
                "kind": "giant_variant",
                "maze_type": "giant",
                "task": task_num,
                "var": var,
                "validation": bool(validation),
                "npz_path": variant_npz_path(
                    variant_dir, task_num, var, validation=validation),
                "json_path": variant_json_path(variant_json_dir, task_num),
                "grid": load_variant_grid(variant_json_dir, task_num, var),
            })
    return specs


def load_raw_from_spec(spec):
    """
    Load raw arrays from a maze name or map spec.

    A spec may provide a raw_loader callable, an npz_path, or a maze_type.
    npz_path wins over maze_type so giant variants do not accidentally load the
    base giant directory.
    """
    if isinstance(spec, str):
        return load_raw(spec)
    if callable(spec):
        return spec()
    if "raw_loader" in spec and spec["raw_loader"] is not None:
        return spec["raw_loader"]()
    if "npz_path" in spec and spec["npz_path"] is not None:
        return load_raw_npz(spec["npz_path"])
    if "maze_type" in spec and spec["maze_type"] is not None:
        return load_raw(spec["maze_type"])
    raise KeyError("map spec needs raw_loader, npz_path, or maze_type: %r" % spec)


def load_episodes_from_spec(spec):
    """
    Load segmented episodes from a maze name or map spec.

    A spec may provide an episodes_loader callable. Otherwise it is segmented
    from load_raw_from_spec(), which reuses the existing terminal logic.
    """
    if isinstance(spec, str):
        return load_episodes(spec)
    if "episodes_loader" in spec and spec["episodes_loader"] is not None:
        return spec["episodes_loader"]()
    return segment_episodes(*load_raw_from_spec(spec))


def coordinate_bounds(data_sources):
    """
    Per-key (observations, actions) global min/max across the given sources,
    for fitting a single shared normalizer. Returns
    {'observations': (min(2,), max(2,)), 'actions': (min(2,), max(2,))}.

    `data_sources` can be the old list of maze names or the newer explicit map
    specs used by MultiMazeGoalDataset.
    """
    o_min = a_min = None
    o_max = a_max = None
    for source in data_sources:
        if (isinstance(source, dict) and
                "episodes_loader" in source and
                source.get("raw_loader") is None and
                source.get("npz_path") is None and
                source.get("maze_type") is None):
            episodes = source["episodes_loader"]()
            pairs = ((e["observations"], e["actions"]) for e in episodes)
        else:
            obs, act, _ = load_raw_from_spec(source)
            pairs = ((obs, act),)

        for obs, act in pairs:
            om, oM = obs.min(0), obs.max(0)
            am, aM = act.min(0), act.max(0)
            o_min = om if o_min is None else np.minimum(o_min, om)
            o_max = oM if o_max is None else np.maximum(o_max, oM)
            a_min = am if a_min is None else np.minimum(a_min, am)
            a_max = aM if a_max is None else np.maximum(a_max, aM)

    if o_min is None:
        raise ValueError("cannot fit coordinate bounds over zero data sources")
    return {
        "observations": (o_min.astype(np.float32), o_max.astype(np.float32)),
        "actions": (a_min.astype(np.float32), a_max.astype(np.float32)),
    }
