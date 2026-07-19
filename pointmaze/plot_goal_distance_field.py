"""
Plot goal-distance heatmaps for the original PointMaze giant maze.

This visualizes the same BFS distance-to-goal field used by the distance-field
soft steering guidance: free cells get their shortest-path distance to the
goal, walls are masked, and optional Gaussian smoothing can be applied to match
the verifier setting.

Examples:
  python3 plot_goal_distance_field.py --task 1
  python3 plot_goal_distance_field.py --task 1 2 3 4 5
  python3 plot_goal_distance_field.py --all-tasks --smooth-sigma 0.5
  python3 plot_goal_distance_field.py --goal-cell 10 14 --start-cell 1 1
  python3 plot_goal_distance_field.py --goal-xy 52 36 --start-xy 0 0
"""

import argparse
import copy
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mapcond import maze_grids as MG


HERE = Path(__file__).resolve().parent

# Mirrors ogbench/ogbench/locomaze/maze.py::set_tasks for maze_type == "giant".
GIANT_TASKS = {
    1: ((1, 1), (10, 14)),
    2: ((1, 14), (10, 1)),
    3: ((8, 14), (1, 1)),
    4: ((8, 3), (5, 12)),
    5: ((5, 9), (3, 8)),
}


def compute_distance_field(maze_grid, goal_cell, connectivity=4):
    """
    BFS from goal_cell over free cells.

    This mirrors search/distance_field.py without importing the search package,
    whose __init__ pulls in the full diffusion stack.
    """
    H, W = maze_grid.shape
    distances = np.full((H, W), np.inf, dtype=np.float32)
    gi, gj = goal_cell

    if maze_grid[gi, gj] == 1:
        return distances

    distances[gi, gj] = 0.0
    queue = deque([(gi, gj)])
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]

    while queue:
        i, j = queue.popleft()
        for di, dj in deltas:
            ni, nj = i + di, j + dj
            if (
                0 <= ni < H
                and 0 <= nj < W
                and maze_grid[ni, nj] == 0
                and distances[ni, nj] == np.inf
            ):
                distances[ni, nj] = distances[i, j] + 1.0
                queue.append((ni, nj))

    return distances


def smooth_distance_field(distances, sigma=0.5, inf_replace=1000.0):
    """Match the verifier's finite replacement plus Gaussian smoothing."""
    out = distances.copy()
    out[np.isinf(out)] = inf_replace
    if sigma > 0:
        from scipy.ndimage import gaussian_filter

        out = gaussian_filter(out.astype(np.float64), sigma=sigma).astype(np.float32)
    return out


def _slug_float(value):
    text = ("%g" % float(value)).replace("-", "m").replace(".", "p")
    return text


def _cell_arg(values, name):
    if values is None:
        return None
    if len(values) != 2:
        raise SystemExit(f"{name} expects exactly two integers: row col")
    return int(values[0]), int(values[1])


def _xy_arg(values, name):
    if values is None:
        return None
    if len(values) != 2:
        raise SystemExit(f"{name} expects exactly two numbers: x y")
    return float(values[0]), float(values[1])


def _xy_to_ij(xy):
    """World (x, y) to cell (i, j), matching maze.py xy_to_ij."""
    x, y = xy
    i = int((y + MG.OFFSET_Y + 0.5 * MG.MAZE_UNIT) / MG.MAZE_UNIT)
    j = int((x + MG.OFFSET_X + 0.5 * MG.MAZE_UNIT) / MG.MAZE_UNIT)
    return i, j


def _validate_cell(maze_grid, cell, name, require_free=True):
    i, j = cell
    H, W = maze_grid.shape
    if not (0 <= i < H and 0 <= j < W):
        raise SystemExit(f"{name} cell {cell} is outside maze shape {(H, W)}.")
    if require_free and maze_grid[i, j] == 1:
        raise SystemExit(f"{name} cell {cell} is a wall.")


def _task_cells(task):
    task = int(task)
    if task not in GIANT_TASKS:
        raise SystemExit(
            f"Unknown giant task {task}. Available tasks: "
            f"{', '.join(str(t) for t in sorted(GIANT_TASKS))}"
        )
    return GIANT_TASKS[task]


def _default_output(task, goal_cell, smooth_sigma):
    suffix = ""
    if smooth_sigma > 0:
        suffix = f"_smooth{_slug_float(smooth_sigma)}"
    if task is not None:
        return HERE / f"goal_distance_field_giant_task{task}{suffix}.png"
    gi, gj = goal_cell
    return HERE / f"goal_distance_field_giant_goal{gi}-{gj}{suffix}.png"


def _output_for_task(output_arg, task, multi_output):
    output = Path(output_arg)
    if not multi_output:
        return output
    return output.with_name(f"{output.stem}_task{int(task)}{output.suffix}")


def _distance_for_plot(maze_grid, goal_cell, connectivity, smooth_sigma, inf_replace):
    raw = compute_distance_field(
        maze_grid, goal_cell, connectivity=connectivity,
    )
    if np.isinf(raw[goal_cell]):
        raise SystemExit(f"Goal cell {goal_cell} is not reachable.")

    if smooth_sigma > 0:
        values = smooth_distance_field(
            raw, sigma=smooth_sigma, inf_replace=inf_replace,
        )
    else:
        values = raw.copy()

    values = values.astype(float)
    values[maze_grid == 1] = np.nan
    values[np.isinf(values)] = np.nan
    return raw, values


def plot_distance_field(
    maze_grid,
    goal_cell,
    output_path,
    start_cell=None,
    task=None,
    connectivity=4,
    smooth_sigma=0.0,
    inf_replace=1000.0,
    cmap_name="viridis_r",
    annotate=False,
):
    raw, values = _distance_for_plot(
        maze_grid, goal_cell, connectivity, smooth_sigma, inf_replace,
    )

    H, W = maze_grid.shape
    finite = values[np.isfinite(values)]
    vmax = float(finite.max()) if finite.size else 1.0

    cmap = copy.copy(plt.get_cmap(cmap_name))
    cmap.set_bad("#222222")

    fig_width = max(7.0, W * 0.42)
    fig_height = max(5.2, H * 0.42)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(
        values,
        origin="upper",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(W))
    ax.set_yticks(np.arange(H))
    ax.tick_params(axis="both", labelsize=8, length=0)

    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.45, alpha=0.35)

    gi, gj = goal_cell
    ax.scatter(
        [gj], [gi],
        marker="*",
        s=230,
        facecolor="#ffdd57",
        edgecolor="#111111",
        linewidth=0.9,
        zorder=4,
        label="Goal",
    )
    ax.text(
        gj, gi, "G",
        ha="center", va="center",
        fontsize=9, weight="bold", color="#111111", zorder=5,
    )

    if start_cell is not None:
        si, sj = start_cell
        ax.scatter(
            [sj], [si],
            marker="o",
            s=120,
            facecolor="white",
            edgecolor="#111111",
            linewidth=0.9,
            zorder=4,
            label="Start",
        )
        ax.text(
            sj, si, "S",
            ha="center", va="center",
            fontsize=8, weight="bold", color="#111111", zorder=5,
        )

    if annotate:
        for i in range(H):
            for j in range(W):
                if maze_grid[i, j] == 1 or np.isinf(raw[i, j]):
                    continue
                label = str(int(raw[i, j]))
                ax.text(
                    j, i, label,
                    ha="center", va="center",
                    fontsize=6.5,
                    color="white" if values[i, j] > 0.55 * vmax else "#111111",
                    alpha=0.9,
                    zorder=3,
                )

    title_prefix = "Original giant maze"
    if task is not None:
        title_prefix += f", Task {int(task)}"
    if smooth_sigma > 0:
        title = f"{title_prefix}: smoothed distance to goal"
        cbar_label = f"Smoothed BFS distance cost (sigma={smooth_sigma:g})"
    else:
        title = f"{title_prefix}: BFS distance to goal"
        cbar_label = "BFS distance to goal (cells)"
    ax.set_title(title, fontsize=12, pad=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(labelsize=8)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(
            handles=handles,
            labels=labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=len(handles),
            fontsize=9,
            frameon=True,
            framealpha=0.92,
            edgecolor="#cccccc",
            fancybox=False,
            columnspacing=1.2,
            handletextpad=0.45,
        )
        leg.get_frame().set_linewidth(0.6)

    plt.tight_layout(pad=0.45)
    plt.savefig(output_path, dpi=240, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", nargs="+", type=int, default=[1],
                        help="Giant task id(s) whose built-in goal cells to plot. "
                             "Default: 1.")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Plot all built-in giant tasks.")
    parser.add_argument("--goal-cell", nargs=2, type=int, metavar=("I", "J"),
                        help="Custom goal cell as row i, column j. Overrides --task.")
    parser.add_argument("--start-cell", nargs=2, type=int, metavar=("I", "J"),
                        help="Optional custom start cell marker as row i, column j.")
    parser.add_argument("--goal-xy", nargs=2, type=float, metavar=("X", "Y"),
                        help="Custom goal world coordinates. Overrides --task.")
    parser.add_argument("--start-xy", nargs=2, type=float, metavar=("X", "Y"),
                        help="Optional custom start world-coordinate marker.")
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=4,
                        help="BFS connectivity. Guidance default is 4.")
    parser.add_argument("--smooth-sigma", type=float, default=0.0,
                        help="Gaussian smoothing sigma. Use 0.5 to match the "
                             "DistanceFieldVerifier default.")
    parser.add_argument("--inf-replace", type=float, default=1000.0,
                        help="Finite value used for inf before smoothing.")
    parser.add_argument("--cmap", default="viridis_r",
                        help="Matplotlib colormap for free-cell distances.")
    parser.add_argument("--annotate", action="store_true",
                        help="Write raw integer BFS distance inside each free cell.")
    parser.add_argument("--output", default=None,
                        help="Output path. With multiple tasks, _taskN is inserted.")
    args = parser.parse_args()

    maze_grid = MG.parse_grid("giant").astype(np.int64)
    goal_cell = _cell_arg(args.goal_cell, "--goal-cell")
    start_cell = _cell_arg(args.start_cell, "--start-cell")
    goal_xy = _xy_arg(args.goal_xy, "--goal-xy")
    start_xy = _xy_arg(args.start_xy, "--start-xy")

    if goal_cell is not None and goal_xy is not None:
        raise SystemExit("Use only one of --goal-cell or --goal-xy.")
    if start_cell is not None and start_xy is not None:
        raise SystemExit("Use only one of --start-cell or --start-xy.")
    if goal_xy is not None:
        goal_cell = _xy_to_ij(goal_xy)
    if start_xy is not None:
        start_cell = _xy_to_ij(start_xy)

    if goal_cell is not None:
        _validate_cell(maze_grid, goal_cell, "Goal")
        if start_cell is not None:
            _validate_cell(maze_grid, start_cell, "Start", require_free=False)
        output = Path(args.output) if args.output else _default_output(
            task=None, goal_cell=goal_cell, smooth_sigma=args.smooth_sigma,
        )
        plot_distance_field(
            maze_grid, goal_cell, str(output),
            start_cell=start_cell,
            task=None,
            connectivity=args.connectivity,
            smooth_sigma=args.smooth_sigma,
            inf_replace=args.inf_replace,
            cmap_name=args.cmap,
            annotate=args.annotate,
        )
        return

    tasks = sorted(GIANT_TASKS) if args.all_tasks else list(dict.fromkeys(args.task))
    multi_output = len(tasks) > 1
    for task in tasks:
        default_start, default_goal = _task_cells(task)
        goal = default_goal
        start = start_cell if start_cell is not None else default_start
        _validate_cell(maze_grid, goal, "Goal")
        if start is not None:
            _validate_cell(maze_grid, start, "Start", require_free=False)

        if args.output is None:
            output = _default_output(
                task=task, goal_cell=goal, smooth_sigma=args.smooth_sigma,
            )
        else:
            output = _output_for_task(args.output, task, multi_output)

        plot_distance_field(
            maze_grid, goal, str(output),
            start_cell=start,
            task=task,
            connectivity=args.connectivity,
            smooth_sigma=args.smooth_sigma,
            inf_replace=args.inf_replace,
            cmap_name=args.cmap,
            annotate=args.annotate,
        )


if __name__ == "__main__":
    main()
