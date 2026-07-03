# 迷宫变体生成流程 (maze_update)

为 OGBench giant pointmaze 生成分级难度的迷宫变体阶梯，用于评测 diffusion planner。
每个任务产出 4 档难度 × 3 变体 = 12 张图。

## 文件分工

| 文件 | 角色 | 状态 |
|---|---|---|
| `generate_variants.py` | 主入口/流程编排：造候选、打分、分级、挑选、画图、写 JSON | 活跃 |
| `difficulty_metric.py` | 难度度量核心：特征计算、`rank_composite` 合成 difficulty_score、分位分级 | 活跃 |
| `route_space.py` | 路线枚举：k-shortest / waypoint / diverse routes / `route_signature` | 活跃 |
| `maze_utils.py` | 底层工具：从 maze.py 解析 base 迷宫、`shortest_path`、`_neighbors`、`ij_to_xy` | 活跃 |
| `calibrate_difficulty.py` | 事后校准：DFS 日志算 difficulty_score vs 成功率相关性（不参与生成） | 活跃 |
| `reroute_disruption.py` | 旧 RDI 度量 + 老画图套件 | 死代码（route_space 参考移植过） |
| `data_disruption.py` | 旧 traffic 模型 | 死代码 |
| `maze_variants_v2/giant_task{1..5}.json` | 产物：每任务 12 张图 | 输出 |

## 生成流程（数据流顺序）

一句话：**随机造一大堆候选 → 用 5 特征打分分级 → 每档挑几条互异路线**。

### 阶段 0 — 准备输入 (`generate_variants.py::main`)
1. `load_base_maze()` 从 `pointmaze/ogbench/ogbench/locomaze/maze.py` 用 `ast` 解析
   12×16 的 giant base 迷宫（不 import，避免拉起 mujoco）。
2. `load_traffic()` 从训练轨迹 `observations.npy` 统计每格通行频次（喂方向性特征）。
3. `GIANT_TASKS[tid]` 给出每任务的 start/goal。

### 阶段 1 — 造候选池（不看特征，纯随机 + 图论约束）
4. `generate_task` 构造预算网格 `budgets` = n_block(0–8) × n_open(0–2)，共 26 组。
5. `build_pool` 对每组预算调 `make_candidate` 最多采样 40 次：
   - 先随机开几面内墙 (`_interior_walls`，种备选走廊)，
   - 再在当前最短路走廊上随机堵格子 (`_corridor_cells`，逼迫绕路)，每堵一格重算路；
   - 硬性拒绝：不可达 / 变捷径(更短=更简单) / 和原路完全相同(无变化)。
6. **去重键 = top-3 route-set 签名** (`rs.route_signature`)：同一套"最优路 + 前 3 条
   互异备选走廊"只保留编辑量最少的代表。池子最终 298–517 个独特候选/任务
   （相比旧的单最短路去重 ~50–80 个，宽 1.5–2×）。

### 阶段 2 — 打分、分级、挑选（5 特征在这里起作用）
7. 每候选 `dm.compute_variant_features` 算特征，`dm.rank_composite` 用
   `GOOD_FEATURES = (spectral, sp_ratio, net_shift, deadend_delta, n_blocked)`
   合成 **difficulty_score**（每特征跨全池取百分位排名 → 等权平均，0–1，越高越难）。
8. `dm.quantile_levels` 按分数分 4 档 (L1–L4)。
9. 每档内：按 route-set 塌成一路一代表 + 跨全梯 `used_paths` 排除已用最优路 →
   `pick_distinct` 用路线 Jaccard 贪心挑 3 条最互异的。
10. 写 `giant_task{tid}.json`，自审：难度层间单调 + 12/12 最优路互异。
11. `_plot_ladder` 出接触图 `variant_ladder.png`（`origin="lower"`，row 0 在底，
    与原始 reroute_disruption.plot_suite 一致）。

## 产物 JSON 结构

顶层：`base_maze`、`good_features`、`n_levels`/`n_variants`。
`variants[]` 每项：`maze_map`(改后网格)、`level_index`、`difficulty_score`、
`broken_cells`(加的墙)、`opened_cells`(开的墙)、`path_len`、各特征值。

## 关键函数速查

- 造墙：`generate_variants.make_candidate`（随机）
- 判重/判异：`route_space.route_signature`（前 3 条路）
- 定难度：`difficulty_metric.rank_composite`（5 特征）
- 分档挑选：`generate_variants.generate_task` + `pick_distinct`

## 运行

```bash
# 重新生成全部（seed 0，约 10 分钟——route-set 签名是主要开销）
python maze_update/generate_variants.py --seed 0

# 事后校准（需先在 v2 地图上跑 DFS eval 得到日志）
python maze_update/calibrate_difficulty.py --log pointmaze/<new_log>
```

可调旋钮：`--pool-per-budget`（每预算采样数）、`--max-block`/`--max-open`（预算网格）；
`route_space` 里的 `k`（取几条路）、`overlap_tol`（判异重叠阈值）。
