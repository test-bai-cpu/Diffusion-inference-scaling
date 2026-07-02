# 项目交接文档 — 面向扩散模型 Pointmaze 规划器的「地图感知推理期搜索」

**分支：** `guidance-claude` · **仓库：** `Diffusion-inference-scaling` · **最新提交：** `8b14797`
**基线论文：** arXiv 2505.23614v2 · **基准环境：** OGBench `pointmaze-giant-navigate-v0`

---

## 0. 先看这一条 —— 最重要的一句话

所有**几何、指标、辅助模型**相关的部分，都是在沙盒里（CPU）用**真实迷宫数据**跑出来的，可以信任。
唯一**没有跑**的是需要扩散规划器真正 rollout 的部分 —— 也就是端到端的成功率／碰撞率随难度退化曲线，
因为沙盒里**没有 GPU、也没有 `maze` conda 环境**。这部分是以「已测试、可直接运行的代码」形式交付的。
唯一一张展示规划器对比数字的图（`degradation_demo_curves.png`）是**合成的、仅作示意**，并已明确打上标注。

所以你接手要做的事，本质上就是：**跑一遍代码已经准备好的 GPU 实验，把那张示意图替换成真实结果图。**

---

## 1. 一段话讲清目标

拿一个**冻结的、已预训练好的**扩散轨迹规划器，只在**推理期**（外加一个小的学习式辅助模型）把它变得更好，
专门针对基线规划器在 giant 迷宫里表现出的两种失败模式：

- **死胡同陷入（dead-end trapping）** —— 规划器开进一个死角后再也出不来。
- **对角穿墙（diagonal corner-cutting）** —— 路径从两个仅在角点相接的墙块之间斜穿过去；
  论文用的**验证器**以为那个缝是通的，但**仿真器**当它是堵死的，于是这一幕会「悄无声息」地失败。

三个目标：
- **(a)** 在规划器已知的地图上，提高成功率、降低碰撞率。
- **(b)** 让规划器具备**地图感知**能力，从而能泛化到**全新的、没见过的**地图。
- **(c)** 造一个**地图变体生成器**，让它的难度「等级」带来**单调**的成功率下降 ——
  也就是让「第 3 级比第 1 级难」这句话在实测中真的成立。

---

## 2. 当前进度 —— 计划的 8 步全部完成（以代码形式）

| # | 步骤 | 提交 | 沙盒内有真实结果？ |
|---|------|------|--------------------|
| 1 | 结果指标插桩（把 成功／碰撞／对角穿墙／死胡同／停滞 分开统计） | `c7bd8c1` | ✅ 在真实 giant 迷宫上验证 |
| 2 | 校准后的难度指标 | `1809fff`, `0539e69` | ✅ 60 次真实 DFS 运行上 Spearman −0.53 |
| 3 | 带单调难度阶梯的变体生成器 | `ade4159` | ✅ 阶梯已生成并校准 |
| 4 | 角点安全的间隙验证器 | `517079c` | ✅ 修复 4 个真实夹缝的对角缝 bug |
| 5 | 修正的到目标距离场 + 死胡同检测器 | `67bdc16`, `bf43e64` | ✅ 场 RMS 0.28→0.03，越墙违规 0% |
| 6 | **MAFGS** —— 地图感知可行性引导搜索 | `b5fc881` | ✅ 逻辑在真实几何上测试（GPU 依赖已 mock） |
| 7 | **MCGN** —— 学习式地图条件引导辅助模型 | `addc043` | ✅ 在 CPU 上可训练、可迁移 |
| 8 | 评估协议 + 优雅退化报告 | `dd2c986` | ⚠️ 评估框架已测试；**规划器实验未跑** |

跨 6 个测试套件共 **67 项 CPU 检查全部通过**。

---

## 3. 每个模块是什么、在哪里

以下路径均相对仓库根目录。

### 指标插桩 —— `pointmaze/search/episode_metrics.py`
`score_episode()` 把一次 rollout 变成一组**分离的指标**，而不是单个成功位：
`success`（total_reward>0）、`collision_rate`（精确 box-SDF 接触，ball_radius=0.5，contact_margin=0.15）、
`cornercut_rate`（到对角夹缝的接近度，半径 0.9）、`deadend_frac`、`stalled_prog`、`final_gap`、`steps`。
已接入 `base_pipeline.sample()`（用 `'pointmaze' in dataset` 保护）；`run.py` 写出 `results_detailed.csv`；
`plot_metrics.py` 画出分解图。**这正是让你看清「为什么失败」而不只是「失败了」的东西。**

### 难度指标 —— `maze_update/difficulty_metric.py`
6 个特征的组合（`spectral, sp_ratio, net_shift, deadend_delta, novelty_x_elong, n_blocked`）。
对 60 次实测规划器成功率做校准：**秩相关 −0.53**，而朴素的「相对难度指数(RDI)」只有 −0.30
（它把 20% 的等级对搞反了）。`calibrate_difficulty.py` + `difficulty_calibration.png` 可复现。

### 变体生成器 —— `maze_update/generate_variants.py` → `maze_update/maze_variants_v2/giant_task{1..5}.json`
只做「加墙块」编辑（绝不打开墙、绝不生成非法迷宫），分成 **4 级 × 3 变体**，难度沿等级非递减。
即插即用的 OGBench 数据格式。`variant_ladder.png` 是总览图。

### 角点安全验证器 —— `pointmaze/search/maze_verifier_clearance.py`
`CornerSafeMazeVerifier`：精确 box 符号距离 + 间隙余量 + 线段超采样 + 显式的**对角夹缝惩罚**。
这就是对角穿墙的修复：旧验证器给标准对角缝打 0.00（以为通），现在打 3.69（惩罚）。
4 个真实 giant 夹缝全部修复。由 `test_verifier_geometry.py` 证明。

### 距离场 v2 —— `pointmaze/search/distance_field_v2.py` + `distance_field_verifier_v2.py`
4-连通 BFS 到目标 + 尊重墙体的平滑（不会**穿墙**模糊）+ 严格上坡的墙体填充，
外加**死胡同口袋检测器**（叶子剪枝）。相对真实测地线，场的保真度 RMS 从 0.28 → 0.03，
且**没有任何越墙的假捷径**（原来 8%）。这是引导搜索、并检测陷阱的「哪边真的朝向目标」信号。

### MAFGS —— `pointmaze/search/methods/mafgs.py`（`--method mafgs`）
目标 (a) 的主贡献。在扩散采样过程中它会：
- **引导（guide）** 每一步去噪：用「间隙损失 + 目标场下降（+ 可选的 MCGN 项）」的加权和。
- **门控（gate）** 可行性（纯解析）：如果某个候选子轨迹出现碰撞、对角穿墙、停滞、或死胡同占用超过容差，
  就**回溯**并重新采样。
- 关键旋钮默认值（在 `configs.py`）：`clear_weight=1.0, field_weight=0.5, margin=0.7,
  corner_radius=0.9, collision_tol=0.05, stall_tol=0.10, deadend_tol=0.15`。
  搜索预算：`recur_depth=12, budget=20, threshold=6, start_step=12`。
- **一个值得知道的设计选择：** 学习式辅助模型只进入*引导*，绝不进入*门控*。
  神经网络的坏预测可以「轻推」，但永远无法覆盖解析安全检查。

### MCGN —— `pointmaze/sidemodel/`（目标 (b)，地图感知的来源）
小型 CNN（`model.py`，patch=15 的占据裁剪 + 目标 → 2D 方向），训练目标是预测任一格子上的
**测地线下降方向**。采用**多地图 + D4 数据增强**训练（`train.py`, `dataset.py`）——
D4 增强正是它能泛化的关键。`sidemodel_verifier.py` 把它封装成 MAFGS 的 MCGN 引导项。
- **真实迁移结果（CPU）：** 在 task {1,2,4,5} 的变体布局上训练，留出 task 3
  （布局*和*目标都没见过）：留出集方向余弦 **0.46 对比 0.33**（目标方向基线）；
  在「必须绕路」的格子上（朴素指向目标是错的），从 **−0.22 → +0.22**。
  这就是地图感知在从未见过的地图上起作用。

### 评估框架 —— `pointmaze/run_eval.py`, `plot_degradation.py`, `EVALUATION.md`
- `run_eval.py`：扫描 **4 配置 × 5 任务 × 4 等级 × N 种子**，写出整洁、可续跑的 `eval_results.csv`。
  4 个配置共享同一冻结骨干：`dfs`（论文基线）、`field`（dfs + 距离场）、`mafgs`、`mafgs+mcgn`。
- `plot_degradation.py`：产出 `degradation_curves.png`（成功／碰撞／死胡同 随等级变化，带 95% 置信区间
  + 一个「相对 dfs 基线的领先幅度」面板）和 `degradation_report.md`
  （每个方法的单调性 Spearman、汇总表、最难等级的领先幅度、迁移段落）。
- `EVALUATION.md`：标准协议 —— 精确命令、「优雅退化」应长什么样、以及完整测试清单。

---

## 4. 真实 vs 合成 —— 诚实的账本

**真实（沙盒内用真实数据算出，可信）：**
- 对角穿墙修复：4 个真实 giant 夹缝上，对角缝惩罚 0.00 → 3.69。
- 距离场：相对真实测地线 RMS 0.28 → 0.03；越墙捷径 0%（原 8%）。
- 难度校准：60 次真实 DFS 上 Spearman −0.53（组合）对比 −0.30（RDI）。
- MCGN 迁移：留出集余弦 0.46 对比 0.33；必须绕路格子 −0.22 → +0.22。
- 指标行为：干净路径 → 全零失败 + 成功；贴墙 → 碰撞=1.0；穿夹缝 → 对角穿墙>0；
  死胡同乱走 → 死胡同 0.21 / 停滞 0.49。

**合成（示意用占位，切勿当作结果引用）：**
- `degradation_demo_curves.png` —— 规划器 成功／碰撞／死胡同 曲线。
  **已打上红色「ILLUSTRATIVE — synthetic numbers」标注。** 仅展示好结果*应有的形状*。
- `degradation_report_example.md` + `eval_results_demo.csv` —— 同一合成来源。

这些合成文件存在的唯一目的，是让你能提前看到 `plot_degradation.py` 会输出什么样子。
一旦你跑完真实实验，它们就会被同样形状的真实文件替换。

---

## 5. 如何接手 —— 环境 + 精确命令

在 `pointmaze/` 目录下，用你的 GPU `maze` conda 环境（基线规划器已经在用的那个）运行。

**第 A 步 —— 训练 MCGN 辅助模型**（GPU 上几分钟；仅 `mafgs+mcgn` 配置需要）：
```
python sidemodel/train.py \
  --maze_json ../maze_update/maze_variants_v2/giant_task1.json \
              ../maze_update/maze_variants_v2/giant_task2.json \
              ../maze_update/maze_variants_v2/giant_task4.json \
              ../maze_update/maze_variants_v2/giant_task5.json \
  --holdout task3 --epochs 40 --patch 15 --out sidemodel/mcgn_giant.pt
```
它会打印 留出集余弦 + 45° 以内准确率 + 目标方向基线，方便你在花时间跑实验前先确认迁移数字。

**第 B 步 —— 跑评估实验**（这是真正的实验；它会驱动扩散规划器）：
```
python run_eval.py \
  --methods dfs field mafgs mafgs+mcgn \
  --tasks 1 2 3 4 5 --levels 0 1 2 3 --seeds 0 1 2 \
  --maze_json_dir ../maze_update/maze_variants_v2 \
  --mcgn_ckpt sidemodel/mcgn_giant.pt \
  --device cuda:0 --out eval_results.csv
```
`eval_results.csv` 是**增量写入、可续跑**的 —— 可以放心 Ctrl-C 后重启，或一次只跑一个方法。

**第 C 步 —— 生成真实图与报告：**
```
python plot_degradation.py eval_results.csv degradation
# -> degradation_curves.png  +  degradation_report.md  （替换掉那些 demo 文件）
```

**大规模扫描前的冒烟测试：** 先用一个方法、一个种子起步
（`--methods dfs mafgs --tasks 1 --seeds 0`），确认规划器能加载、CSV 有行写出，再放大规模。

---

## 6. 该跑哪些实验（按优先级）

1. **主对比** —— 完整的第 B 步扫描。它回答目标 (a)：在已知地图上，MAFGS 相对 `dfs` 论文基线
   是否提高成功率、降低碰撞／对角穿墙？看汇总表和「相对 dfs 的领先幅度」面板。
2. **优雅退化／单调性** —— 用同一个 CSV，`plot_degradation.py` 报告每个方法的 Spearman(等级, 成功率)。
   这验证目标 (c)：对每个方法，成功率应随等级升高单调下降。
3. **地图迁移／地图感知** —— 对比 `mafgs` 与 `mafgs+mcgn`，重点看*更难*的等级和 task 3（MCGN 的留出任务）。
   如果 MCGN 恰好在「朴素指向目标会失败」的地方帮助最大，那就端到端确认了目标 (b)。
4. **消融（可选，有了框架很便宜）：** `field` 对 `dfs` 隔离距离场的贡献；
   `mafgs` 对 `mafgs+mcgn` 隔离学习式辅助模型；调 `mafgs_corner_weight` / `mafgs_deadend_tol`
   隔离各失败模式门控。

---

## 7. 开始前如何确认没有东西坏掉

全部可在 CPU 上跑，从 `pointmaze/` 目录：
```
python search/test_episode_metrics.py     # 4/4   结果指标
python search/test_verifier_geometry.py   # all   角点安全验证器（对角缝修复）
python search/test_distance_field.py      # 5/5   场保真度 + 死胡同检测器
python search/test_mafgs.py               # 8/8   MAFGS 引导+门控逻辑（GPU 依赖已 mock）
python sidemodel/test_mcgn.py             # 31/31 MCGN 模型/数据集/迁移/验证器
python test_run_eval.py                   # 23/23 扫描记账 + 绘图器 + 报告
```
最近一次全跑：**全绿（67 项检查）。**

---

## 8. 注意事项与需要留心的坑

- **唯一没跑的就是那个实验扫描。** 如果你 `maze` 环境里的规划器 API 与 `run_eval.py` 的假设不同
  （它通过 `search.script_utils` 调 `get_pipe(args).experiment()`），那这就是第一个要检查的对接点。
  框架是针对*被 mock 的* pipe 测试的，所以这个契约有文档、但没跟真实规划器对跑过。
- **MCGN 检查点路径** 必须在第 A 步 `--out` 和第 B 步 `--mcgn_ckpt` 之间一致
  （默认 `sidemodel/mcgn_giant.pt`）。若 `mcgn_ckpt` 为空，`mafgs+mcgn` 配置会静默退化为纯 `mafgs`
  （MCGN 引导项为零）—— 所以空路径不会崩，只是拿不到辅助模型的收益。
- **难度阶梯** 在 `maze_variants_v2/`（校准过的 v2），不是旧的 `maze_variants/`。实验要指向 v2。
- **坐标约定**（如果你要扩展几何代码）：世界坐标 `xy = (j*4 − 4, i*4 − 4)`；
  格子 `i = round((y+4)/4)`, `j = round((x+4)/4)`；网格 0=空 1=墙；giant 尺寸 (12,16)。

---

*交接时生成。所有代码已提交到 `guidance-claude` 分支（截至 `8b14797`）。*
