# 项目交接文档 — 面向 diffusion pointmaze planner 的 map-aware inference-time search

**分支：** `guidance-claude` · **仓库：** `Diffusion-inference-scaling` · **最新提交：** `8b14797`
**基线论文：** arXiv 2505.23614v2 · **基准环境：** OGBench `pointmaze-giant-navigate-v0`

---

## 0. 先看这一条 —— 最重要的一句话

所有 **geometry、metric、side model** 相关的部分，都是在 sandbox 里（CPU）用**真实迷宫数据**跑出来的，可以信任。
唯一**没有跑**的，是需要 diffusion planner 真正 rollout 的部分 —— 也就是端到端的 success / collision
随难度退化的曲线，因为 sandbox 里**没有 GPU、也没有 `maze` conda 环境**。这部分是以「已测试、可直接运行的代码」
形式交付的。唯一一张展示 planner 对比数字的图（`degradation_demo_curves.png`）是**合成的、仅作示意（illustrative）**，
并已明确打上标注。

所以你接手要做的事，本质上就是：**跑一遍代码已经准备好的 GPU 实验，把那张示意图替换成真实结果图。**

---

## 1. 一段话讲清目标

拿一个**冻结的（frozen）、已预训练好的** diffusion trajectory planner，只在 **inference time**
（外加一个小的 learned side model）把它变得更好，专门针对 baseline planner 在 giant maze 里表现出的
两种 failure mode：

- **dead-end trapping（死胡同陷入）** —— planner 开进一个死角后再也出不来。
- **diagonal corner-cutting（对角穿墙）** —— 路径从两个仅在角点相接的 wall block 之间斜穿过去；
  论文用的 **verifier** 以为那个缝是通的，但 **simulator** 当它是堵死的，于是这一幕会「悄无声息」地失败。

三个目标（aim）：
- **(a)** 在 planner 已知的地图上，提高 success rate、降低 collision rate。
- **(b)** 让 planner 具备 **map-aware（地图感知）** 能力，从而能泛化到**全新的、没见过的**地图。
- **(c)** 造一个 **map-variant generator**，让它的 difficulty「等级（level）」带来**单调（monotone）**的
  success 下降 —— 也就是让「level 3 比 level 1 难」这句话在实测中真的成立。

---

## 2. 当前进度 —— 计划的 8 步全部完成（以代码形式）

| # | 步骤 | 提交 | sandbox 内有真实结果？ |
|---|------|------|--------------------|
| 1 | 结果指标 instrumentation（把 success / collision / corner-cut / dead-end / stall 分开统计） | `c7bd8c1` | ✅ 在真实 giant maze 上验证 |
| 2 | 校准后的 difficulty metric | `1809fff`, `0539e69` | ✅ 60 次真实 DFS run 上 Spearman −0.53 |
| 3 | 带 monotone difficulty ladder 的 variant generator | `ade4159` | ✅ ladder 已生成并校准 |
| 4 | corner-safe clearance verifier | `517079c` | ✅ 修复 4 个真实 pinch 的对角缝 bug |
| 5 | 修正的 distance-to-goal field + dead-end detector | `67bdc16`, `bf43e64` | ✅ field RMS 0.28→0.03，越墙违规 0% |
| 6 | **MAFGS** —— Map-Aware Feasibility-Guided Search | `b5fc881` | ✅ 逻辑在真实 geometry 上测试（GPU 依赖已 mock） |
| 7 | **MCGN** —— learned map-conditioned guidance side model | `addc043` | ✅ 在 CPU 上可训练、可迁移 |
| 8 | Evaluation protocol + graceful-degradation report | `dd2c986` | ⚠️ evaluation 框架已测试；**planner sweep 未跑** |

跨 6 个 test suite 共 **67 项 CPU 检查全部通过**。

---

## 3. 每个模块是什么、在哪里

以下路径均相对仓库根目录。

### Instrumentation —— `pointmaze/search/episode_metrics.py`
`score_episode()` 把一次 rollout 变成一组**分离的指标**，而不是单个 success 位：
`success`（total_reward>0）、`collision_rate`（精确 box-SDF 接触，ball_radius=0.5，contact_margin=0.15）、
`cornercut_rate`（到对角 pinch 的接近度，半径 0.9）、`deadend_frac`、`stalled_prog`、`final_gap`、`steps`。
已接入 `base_pipeline.sample()`（用 `'pointmaze' in dataset` 保护）；`run.py` 写出 `results_detailed.csv`；
`plot_metrics.py` 画出分解图。**这正是让你看清「为什么失败」而不只是「失败了」的东西。**

### Difficulty metric —— `maze_update/difficulty_metric.py`
6 个 feature 的组合（`spectral, sp_ratio, net_shift, deadend_delta, novelty_x_elong, n_blocked`）。
对 60 次实测 planner success 做校准：**rank correlation −0.53**，而朴素的 RDI（relative difficulty index）
只有 −0.30（它把 20% 的 level pair 搞反了）。`calibrate_difficulty.py` + `difficulty_calibration.png` 可复现。

### Variant generator —— `maze_update/generate_variants.py` → `maze_update/maze_variants_v2/giant_task{1..5}.json`
只做「加 wall block」编辑（绝不打开墙、绝不生成非法迷宫），分成 **4 level × 3 variant**，
difficulty 沿 level 非递减。即插即用的 OGBench schema。`variant_ladder.png` 是总览图。

### Corner-safe verifier —— `pointmaze/search/maze_verifier_clearance.py`
`CornerSafeMazeVerifier`：精确 box signed-distance + clearance margin + segment supersampling +
显式的 **diagonal-pinch penalty**。这就是 corner-cutting 的修复：旧 verifier 给标准对角缝打 0.00（以为通），
现在打 3.69（惩罚）。4 个真实 giant pinch 全部修复。由 `test_verifier_geometry.py` 证明。

### Distance field v2 —— `pointmaze/search/distance_field_v2.py` + `distance_field_verifier_v2.py`
4-connected BFS 到 goal + 尊重墙体的 smoothing（不会**穿墙**模糊）+ 严格上坡的 wall fill，
外加 **dead-end pocket detector**（leaf-pruning）。相对真实 geodesic，field 的保真度 RMS 从 0.28 → 0.03，
且**没有任何越墙的假捷径（false shortcut）**（原来 8%）。这是引导 search、并检测 trap 的
「哪边真的朝向 goal」信号。

### MAFGS —— `pointmaze/search/methods/mafgs.py`（`--method mafgs`）
目标 (a) 的主贡献。在 diffusion sampling 过程中它会：
- **guide** 每一步 denoising：用「clearance loss + goal-field descent（+ 可选的 MCGN 项）」的加权和。
- **gate**（feasibility 门控，纯解析）：如果某个候选子轨迹出现 collision、corner-cut、stall、
  或 dead-end occupancy 超过容差，就 **backtrack** 并重新采样。
- 关键旋钮默认值（在 `configs.py`）：`clear_weight=1.0, field_weight=0.5, margin=0.7,
  corner_radius=0.9, collision_tol=0.05, stall_tol=0.10, deadend_tol=0.15`。
  搜索预算：`recur_depth=12, budget=20, threshold=6, start_step=12`。
- **一个值得知道的设计选择：** learned side model 只进入 *guidance*，绝不进入 *gate*。
  神经网络的坏预测可以「轻推」，但永远无法覆盖 analytic 的 safety check。

### MCGN —— `pointmaze/sidemodel/`（目标 (b)，map-awareness 的来源）
小型 CNN（`model.py`，patch=15 的 occupancy crop + goal → 2D direction），训练目标是预测任一格子上的
**geodesic descent direction**。采用 **multi-map + D4 augmentation** 训练（`train.py`, `dataset.py`）——
D4 augmentation 正是它能泛化的关键。`sidemodel_verifier.py` 把它封装成 MAFGS 的 MCGN guidance 项。
- **真实迁移结果（CPU）：** 在 task {1,2,4,5} 的 variant layout 上训练，hold out task 3
  （layout *和* goal 都没见过）：holdout 方向 cosine **0.46 对比 0.33**（goal-direction baseline）；
  在「must-detour（必须绕路）」的格子上（朴素指向 goal 是错的），从 **−0.22 → +0.22**。
  这就是 map-awareness 在从未见过的地图上起作用。

### Evaluation 框架 —— `pointmaze/run_eval.py`, `plot_degradation.py`, `EVALUATION.md`
- `run_eval.py`：扫描 **4 config × 5 task × 4 level × N seed**，写出整洁、可续跑（resumable）的
  `eval_results.csv`。4 个 config 共享同一 frozen backbone：`dfs`（论文 baseline）、
  `field`（dfs + distance field）、`mafgs`、`mafgs+mcgn`。
- `plot_degradation.py`：产出 `degradation_curves.png`（success / collision / dead-end 随 level 变化，
  带 95% CI + 一个「相对 dfs baseline 的 margin」面板）和 `degradation_report.md`
  （每个方法的 monotonicity Spearman、汇总表、最难 level 的 margin、transfer 段落）。
- `EVALUATION.md`：标准 protocol —— 精确命令、graceful degradation 应长什么样、以及完整测试清单。

---

## 4. 真实 vs 合成 —— 诚实的账本

**真实（sandbox 内用真实数据算出，可信）：**
- corner-cutting 修复：4 个真实 giant pinch 上，对角缝 penalty 0.00 → 3.69。
- distance field：相对真实 geodesic，RMS 0.28 → 0.03；越墙 shortcut 0%（原 8%）。
- difficulty 校准：60 次真实 DFS 上 Spearman −0.53（composite）对比 −0.30（RDI）。
- MCGN transfer：holdout cosine 0.46 对比 0.33；must-detour 格子 −0.22 → +0.22。
- metric 行为：干净路径 → 全零 failure + success；贴墙 → collision=1.0；穿 pinch → corner-cut>0；
  死胡同乱走 → dead-end 0.21 / stall 0.49。

**合成（示意用占位，切勿当作结果引用）：**
- `degradation_demo_curves.png` —— planner 的 success / collision / dead-end 曲线。
  **已打上红色「ILLUSTRATIVE — synthetic numbers」标注。** 仅展示好结果*应有的形状*。
- `degradation_report_example.md` + `eval_results_demo.csv` —— 同一合成来源。

这些合成文件存在的唯一目的，是让你能提前看到 `plot_degradation.py` 会输出什么样子。
一旦你跑完真实实验，它们就会被同样形状的真实文件替换。

---

## 5. 如何接手 —— 环境 + 精确命令

在 `pointmaze/` 目录下，用你的 GPU `maze` conda 环境（baseline planner 已经在用的那个）运行。

**第 A 步 —— 训练 MCGN side model**（GPU 上几分钟；仅 `mafgs+mcgn` config 需要）：
```
python sidemodel/train.py \
  --maze_json ../maze_update/maze_variants_v2/giant_task1.json \
              ../maze_update/maze_variants_v2/giant_task2.json \
              ../maze_update/maze_variants_v2/giant_task4.json \
              ../maze_update/maze_variants_v2/giant_task5.json \
  --holdout task3 --epochs 40 --patch 15 --out sidemodel/mcgn_giant.pt
```
它会打印 holdout cosine + 45° 以内 accuracy + goal-direction baseline，方便你在花时间跑实验前先确认
transfer 数字。

**第 B 步 —— 跑 evaluation sweep**（这是真正的实验；它会驱动 diffusion planner）：
```
python run_eval.py \
  --methods dfs field mafgs mafgs+mcgn \
  --tasks 1 2 3 4 5 --levels 0 1 2 3 --seeds 0 1 2 \
  --maze_json_dir ../maze_update/maze_variants_v2 \
  --mcgn_ckpt sidemodel/mcgn_giant.pt \
  --device cuda:0 --out eval_results.csv
```
`eval_results.csv` 是**增量写入、可续跑（resumable）**的 —— 可以放心 Ctrl-C 后重启，或一次只跑一个 method。

**第 C 步 —— 生成真实图与报告：**
```
python plot_degradation.py eval_results.csv degradation
# -> degradation_curves.png  +  degradation_report.md  （替换掉那些 demo 文件）
```

**大规模 sweep 前的 smoke test：** 先用一个 method、一个 seed 起步
（`--methods dfs mafgs --tasks 1 --seeds 0`），确认 planner 能加载、CSV 有行写出，再放大规模。

---

## 6. 该跑哪些实验（按优先级）

1. **主对比（headline comparison）** —— 完整的第 B 步 sweep。它回答目标 (a)：在已知地图上，MAFGS
   相对 `dfs` 论文 baseline 是否提高 success、降低 collision / corner-cut？看汇总表和「相对 dfs 的 margin」面板。
2. **graceful degradation / monotonicity** —— 用同一个 CSV，`plot_degradation.py` 报告每个 method 的
   Spearman(level, success)。这验证目标 (c)：对每个 method，success 应随 level 升高单调下降。
3. **map transfer / map-awareness** —— 对比 `mafgs` 与 `mafgs+mcgn`，重点看*更难*的 level 和 task 3
   （MCGN 的 holdout task）。如果 MCGN 恰好在「朴素指向 goal 会失败」的地方帮助最大，
   那就端到端确认了目标 (b)。
4. **ablation（消融，有了框架很便宜）：** `field` 对 `dfs` 隔离 distance field 的贡献；
   `mafgs` 对 `mafgs+mcgn` 隔离 learned side model；调 `mafgs_corner_weight` / `mafgs_deadend_tol`
   隔离各 failure-mode gate。

---

## 7. 开始前如何确认没有东西坏掉

全部可在 CPU 上跑，从 `pointmaze/` 目录：
```
python search/test_episode_metrics.py     # 4/4   结果 metric
python search/test_verifier_geometry.py   # all   corner-safe verifier（对角缝修复）
python search/test_distance_field.py      # 5/5   field 保真度 + dead-end detector
python search/test_mafgs.py               # 8/8   MAFGS guide+gate 逻辑（GPU 依赖已 mock）
python sidemodel/test_mcgn.py             # 31/31 MCGN model/dataset/transfer/verifier
python test_run_eval.py                   # 23/23 sweep 记账 + plotter + report
```
最近一次全跑：**全绿（67 项检查）。**

---

## 8. 注意事项与需要留心的坑

- **唯一没跑的就是那个 evaluation sweep。** 如果你 `maze` 环境里的 planner API 与 `run_eval.py` 的假设不同
  （它通过 `search.script_utils` 调 `get_pipe(args).experiment()`），那这就是第一个要检查的对接点。
  框架是针对*被 mock 的* pipe 测试的，所以这个契约有文档、但没跟真实 planner 对跑过。
- **MCGN checkpoint 路径** 必须在第 A 步 `--out` 和第 B 步 `--mcgn_ckpt` 之间一致
  （默认 `sidemodel/mcgn_giant.pt`）。若 `mcgn_ckpt` 为空，`mafgs+mcgn` config 会静默退化为纯 `mafgs`
  （MCGN guidance 项为零）—— 所以空路径不会崩，只是拿不到 side model 的收益。
- **Difficulty ladder** 在 `maze_variants_v2/`（校准过的 v2），不是旧的 `maze_variants/`。实验要指向 v2。
- **坐标约定（coordinate convention）**（如果你要扩展 geometry 代码）：世界坐标 `xy = (j*4 − 4, i*4 − 4)`；
  格子 `i = round((y+4)/4)`, `j = round((x+4)/4)`；grid 0=free 1=wall；giant 尺寸 (12,16)。

---

*交接时生成。所有代码已提交到 `guidance-claude` 分支（截至 `8b14797`）。*
