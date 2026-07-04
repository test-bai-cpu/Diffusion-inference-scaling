#!/usr/bin/env python3
"""
Build side, by, side comparison grids of plan images for the map, guidance vs DFS runs.

For each (task, level, variant) combo found under the two log roots, this script
creates two grid images (one per method) showing all 40 runs in a single picture,
so you can scan visually and see where map, guidance helps.

Default layout: 40 runs arranged as 5 rows x 8 cols. Each tile is labeled with
its run index. Output goes to ./comparison_grids/ by default.

Usage:
    python compare_plans.py
    python compare_plans.py --logs-root logs --out comparison_grids
    python compare_plans.py --runs 40 --rows 5 --cols 8
    python compare_plans.py --combined   # also produce a single map+dfs combined grid per combo
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Path templates. The leaf is always {run}/0.png
MAP_GUIDANCE_ROOT_NAME = "pointmaze-giant-newvar-navigate-v0"
DFS_ROOT_NAME = "pointmaze-giant-newvar-navigate-v0-dfs"
INFERENCE_SUBPATH = "inference/plans/release_H400_T256_LimitsNormalizer_b1_condFalse"

# Folder name patterns inside the inference/plans dir
# map, guidance variant uses the "dfs, df, " prefix, plain dfs uses "dfs, " prefix
MAP_FOLDER_RE = re.compile(r"^dfs-df-task(\d+)-level(\d+)-variant(\d+)$")
DFS_FOLDER_RE = re.compile(r"^dfs-task(\d+)-level(\d+)-variant(\d+)$")


def find_combos(root: Path, regex: re.Pattern):
    """
    Return dict mapping (task, level, variant) -> Path to the combo folder.
    """
    base = root / INFERENCE_SUBPATH
    combos = {}
    if not base.is_dir():
        return combos
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        m = regex.match(entry.name)
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        combos[key] = entry
    return combos


def collect_run_images(combo_dir: Path, runs: int):
    """
    Return a list of length `runs` with Path or None per run index.
    Runs are looked up at combo_dir/{i}/0.png for i in 0..runs-1.
    If your runs are 1, indexed, the function also tries that as a fallback.
    """
    paths = []
    # Try 0, indexed first
    zero_indexed = all((combo_dir / str(i) / "0.png").exists() for i in range(min(runs, 3)))
    one_indexed = all((combo_dir / str(i) / "0.png").exists() for i in range(1, min(runs, 3) + 1))

    if zero_indexed and not one_indexed:
        start = 0
    elif one_indexed and not zero_indexed:
        start = 1
    else:
        # Fall back to whichever yields more hits
        zero_hits = sum((combo_dir / str(i) / "0.png").exists() for i in range(runs))
        one_hits = sum((combo_dir / str(i) / "0.png").exists() for i in range(1, runs + 1))
        start = 0 if zero_hits >= one_hits else 1

    for i in range(start, start + runs):
        p = combo_dir / str(i) / "0.png"
        paths.append(p if p.exists() else None)
    return paths, start


def load_font(size: int):
    """Try a few common font paths, fall back to default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def build_grid(image_paths, rows, cols, tile_w, tile_h, title, start_index):
    """
    Compose a grid image with title bar at top and per, tile run labels.
    Missing images render as gray placeholders.
    """
    label_h = max(18, tile_h // 14)
    title_h = max(40, tile_h // 6)
    pad = 6

    cell_w = tile_w + 2 * pad
    cell_h = tile_h + label_h + 2 * pad

    grid_w = cell_w * cols
    grid_h = title_h + cell_h * rows

    canvas = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(max(20, title_h // 2))
    label_font = load_font(max(12, label_h - 4))

    # Title bar
    draw.rectangle([0, 0, grid_w, title_h], fill=(30, 30, 30))
    # Center the title text
    try:
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except AttributeError:
        tw, th = draw.textsize(title, font=title_font)
    draw.text(((grid_w - tw) / 2, (title_h - th) / 2), title, fill="white", font=title_font)

    for idx, img_path in enumerate(image_paths):
        r = idx // cols
        c = idx % cols
        x0 = c * cell_w
        y0 = title_h + r * cell_h

        # Tile background
        draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], outline=(200, 200, 200))

        # Image area
        img_x = x0 + pad
        img_y = y0 + pad
        if img_path is not None and img_path.exists():
            try:
                im = Image.open(img_path).convert("RGB")
                im.thumbnail((tile_w, tile_h), Image.LANCZOS)
                # Center within tile slot
                off_x = img_x + (tile_w - im.width) // 2
                off_y = img_y + (tile_h - im.height) // 2
                canvas.paste(im, (off_x, off_y))
            except Exception as e:
                draw.rectangle([img_x, img_y, img_x + tile_w, img_y + tile_h], fill=(240, 200, 200))
                draw.text((img_x + 4, img_y + 4), f"err: {e}", fill="black", font=label_font)
        else:
            draw.rectangle([img_x, img_y, img_x + tile_w, img_y + tile_h], fill=(230, 230, 230))
            draw.text((img_x + 4, img_y + 4), "missing", fill=(120, 120, 120), font=label_font)

        # Label
        label = f"run {start_index + idx}"
        label_y = img_y + tile_h + 2
        try:
            lbbox = draw.textbbox((0, 0), label, font=label_font)
            lw = lbbox[2] - lbbox[0]
        except AttributeError:
            lw, _ = draw.textsize(label, font=label_font)
        draw.text((x0 + (cell_w - lw) / 2, label_y), label, fill="black", font=label_font)

    return canvas


def build_combined(map_img, dfs_img, header_h=50):
    """Stack two grids side by side with a shared header."""
    w = map_img.width + dfs_img.width
    h = max(map_img.height, dfs_img.height)
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(map_img, (0, 0))
    canvas.paste(dfs_img, (map_img.width, 0))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-root", default="logs", help="root dir containing both log folders")
    ap.add_argument("--out", default="comparison_grids", help="output directory")
    ap.add_argument("--runs", type=int, default=40, help="number of runs per combo")
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--tile-width", type=int, default=240)
    ap.add_argument("--tile-height", type=int, default=240)
    ap.add_argument("--combined", action="store_true", help="also save a map+dfs side, by, side image per combo")
    args = ap.parse_args()

    if args.rows * args.cols < args.runs:
        print(f"WARNING: rows*cols ({args.rows*args.cols}) < runs ({args.runs}); some runs will be omitted.",
              file=sys.stderr)

    logs_root = Path(args.logs_root)
    map_root = logs_root / MAP_GUIDANCE_ROOT_NAME
    dfs_root = logs_root / DFS_ROOT_NAME

    map_combos = find_combos(map_root, MAP_FOLDER_RE)
    dfs_combos = find_combos(dfs_root, DFS_FOLDER_RE)

    all_keys = sorted(set(map_combos) | set(dfs_combos))
    if not all_keys:
        print("No combos found. Check, logs, root path and folder naming.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(all_keys)} (task, level, variant) combos.")
    print(f"  Map, guidance combos: {len(map_combos)}")
    print(f"  DFS combos:          {len(dfs_combos)}")
    print(f"Writing grids to: {out_dir.resolve()}")

    total_cells = args.rows * args.cols

    for key in all_keys:
        task, level, variant = key
        tag = f"task{task}-level{level}-variant{variant}"

        # Map, guidance
        if key in map_combos:
            paths, start = collect_run_images(map_combos[key], args.runs)
            paths = paths[:total_cells]
            title = f"MAP, GUIDANCE   {tag}   ({sum(p is not None for p in paths)}/{args.runs} runs)"
            grid = build_grid(paths, args.rows, args.cols, args.tile_width, args.tile_height,
                              title, start)
            map_path = out_dir / f"{tag}__map_guidance.png"
            grid.save(map_path, optimize=True)
            print(f"  wrote {map_path.name}")
            map_img = grid
        else:
            print(f"  [skip map, guidance] no folder for {tag}")
            map_img = None

        # DFS
        if key in dfs_combos:
            paths, start = collect_run_images(dfs_combos[key], args.runs)
            paths = paths[:total_cells]
            title = f"DFS   {tag}   ({sum(p is not None for p in paths)}/{args.runs} runs)"
            grid = build_grid(paths, args.rows, args.cols, args.tile_width, args.tile_height,
                              title, start)
            dfs_path = out_dir / f"{tag}__dfs.png"
            grid.save(dfs_path, optimize=True)
            print(f"  wrote {dfs_path.name}")
            dfs_img = grid
        else:
            print(f"  [skip dfs] no folder for {tag}")
            dfs_img = None

        # Optional combined side, by, side
        if args.combined and map_img is not None and dfs_img is not None:
            combo = build_combined(map_img, dfs_img)
            combo_path = out_dir / f"{tag}__combined.png"
            combo.save(combo_path, optimize=True)
            print(f"  wrote {combo_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
