"""
plot_demo_trajectories.py
=========================

Overlay the real training demonstration trajectories from an OGBench navigate
dataset on top of the maze map.

The continuous (x, y) observations are converted to grid-fractional coordinates
that align exactly with the maze cells as drawn by imshow (origin='upper'):

    col = (x + offset + 0.5*unit)/unit - 0.5
    row = (y + offset + 0.5*unit)/unit - 0.5

with maze_unit=4.0, offset=4.0 (from ogbench/locomaze/maze.py).

Usage
-----
    python plot_demo_trajectories.py
or import and call:
    plot_trajectories(data_dir, maze_type="giant", n_traj=40, out="demo.png")
"""

from __future__ import annotations
import os
import numpy as np

from maze_utils import MAZE_MAPS, MAZE_TASKS

MAZE_UNIT = 4.0
OFFSET = 4.0


def xy_to_grid(x, y):
    """Continuous xy -> fractional (col, row) aligned to imshow cell centers."""
    col = (x + OFFSET + 0.5 * MAZE_UNIT) / MAZE_UNIT - 0.5
    row = (y + OFFSET + 0.5 * MAZE_UNIT) / MAZE_UNIT - 0.5
    return col, row


def load_episodes(data_dir):
    """Return a list of (T, 2) arrays, one per episode."""
    obs = np.load(os.path.join(data_dir, "observations.npy"))
    term = np.load(os.path.join(data_dir, "terminals.npy"))
    ep_end = np.where(term)[0]
    ep_start = np.concatenate([[0], ep_end[:-1] + 1])
    return [obs[s:e + 1] for s, e in zip(ep_start, ep_end)]


def plot_trajectories(data_dir, maze_type="giant", n_traj=40, out="demo_trajectories.png",
                      seed=0, show_heat=False, task=None, alpha=0.25, linewidth=0.8):
    """
    Draw the maze and overlay `n_traj` randomly sampled demonstration
    trajectories.

    show_heat: if True, shade cells by visitation density behind the maze walls.
    task: optionally pass a task name (e.g. "task3") to mark its start/goal.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    base = np.array(MAZE_MAPS[maze_type])
    episodes = load_episodes(data_dir)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(episodes), size=min(n_traj, len(episodes)), replace=False)

    fig, ax = plt.subplots(figsize=(9, 7))

    # optional density heat behind walls
    if show_heat:
        H, W = base.shape
        heat = np.zeros((H, W))
        for ep in episodes:
            for x, y in ep[::10]:
                c, r = xy_to_grid(x, y)
                ci, ri = int(round(c)), int(round(r))
                if 0 <= ri < H and 0 <= ci < W:
                    heat[ri, ci] += 1
        heat = np.log1p(heat)
        ax.imshow(heat, cmap="YlOrRd", origin="upper", alpha=0.6)
        # draw walls on top as black where base==1
        wall_overlay = np.ma.masked_where(base == 0, base)
        ax.imshow(wall_overlay, cmap="binary", origin="upper", vmin=0, vmax=1)
    else:
        ax.imshow(base, cmap="binary", origin="upper")

    # overlay trajectories, each a different color
    colors = cm.viridis(np.linspace(0, 1, len(idx)))
    for k, ei in enumerate(idx):
        ep = episodes[ei]
        cols, rows = [], []
        for x, y in ep:
            c, r = xy_to_grid(x, y)
            cols.append(c); rows.append(r)
        ax.plot(cols, rows, "-", color=colors[k], lw=linewidth, alpha=alpha)
        # mark each trajectory's own start/end faintly
        ax.scatter([cols[0]], [rows[0]], color=colors[k], s=10, alpha=0.6, zorder=3)

    # optionally mark a task's canonical start/goal
    if task and task in MAZE_TASKS[maze_type]:
        (si, sj), (gi, gj) = MAZE_TASKS[maze_type][task]
        ax.scatter([sj], [si], c="limegreen", s=260, marker="*",
                   edgecolors="k", zorder=6, label=f"{task} start")
        ax.scatter([gj], [gi], c="gold", s=260, marker="X",
                   edgecolors="k", zorder=6, label=f"{task} goal")
        ax.legend(loc="upper right", fontsize=9)

    ax.set_title(f"pointmaze-{maze_type}: {len(idx)} demonstration trajectories"
                 + (f" (task {task} start/goal marked)" if task else ""),
                 fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, base.shape[1] - 0.5)
    ax.set_ylim(base.shape[0] - 0.5, -0.5)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_single_trajectory(data_dir, maze_type="giant", episode_index=0,
                           out="demo_single.png"):
    """Draw one full demonstration trajectory with a start->end color gradient."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    base = np.array(MAZE_MAPS[maze_type])
    episodes = load_episodes(data_dir)
    ep = episodes[episode_index]

    pts = np.array([xy_to_grid(x, y) for x, y in ep])  # (T, 2) = (col, row)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(base, cmap="binary", origin="upper")
    lc = LineCollection(segs, cmap="plasma",
                        array=np.linspace(0, 1, len(segs)), linewidth=1.8)
    ax.add_collection(lc)
    ax.scatter([pts[0, 0]], [pts[0, 1]], c="limegreen", s=200, marker="*",
               edgecolors="k", zorder=5, label="start")
    ax.scatter([pts[-1, 0]], [pts[-1, 1]], c="red", s=160, marker="o",
               edgecolors="k", zorder=5, label="end")
    ax.set_title(f"pointmaze-{maze_type}: demonstration episode {episode_index} "
                 f"({len(ep)} steps; color = time)", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, base.shape[1] - 0.5)
    ax.set_ylim(base.shape[0] - 0.5, -0.5)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    DATA = "../pointmaze/ogbench/data/pointmaze-giant-navigate-v0"
    OUT = "./trajectory_out"
    os.makedirs(OUT, exist_ok=True)
    # plot_trajectories(DATA, "giant", n_traj=40,
    #                   out=os.path.join(OUT, "giant_demo_40.png"), task="task3")
    plot_trajectories(DATA, "giant", n_traj=200,
                      out=os.path.join(OUT, "giant_demo_40.png"), task="task3")
    # plot_trajectories(DATA, "giant", n_traj=200, alpha=0.12, linewidth=0.5,
    #                   out=os.path.join(OUT, "giant_demo_200_heat.png"),
    #                   show_heat=True)
    # plot_single_trajectory(DATA, "giant", episode_index=0,
    #                        out=os.path.join(OUT, "giant_demo_single.png"))
    print("saved trajectory plots to", OUT)
