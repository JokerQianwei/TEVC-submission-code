# CLAUDE.md
始终用中文回复

## gstack

### Web Browsing
- Always use the `/browse` skill from gstack for all web browsing tasks.
- **Never** use `mcp__claude-in-chrome__*` tools.

### Available Skills
- `/office-hours` — YC-style brainstorming and idea validation
- `/plan-ceo-review` — CEO/founder-mode plan review (scope, ambition, strategy)
- `/plan-eng-review` — Engineering manager plan review (architecture, edge cases, tests)
- `/plan-design-review` — Designer's eye plan review (UI/UX dimensions)
- `/design-consultation` — Create a design system and DESIGN.md
- `/review` — Pre-landing PR code review
- `/ship` — Ship workflow (tests, review, version bump, PR)
- `/browse` — Headless browser for QA, screenshots, and site interaction
- `/qa` — QA test a web app and fix bugs found
- `/qa-only` — QA test and report only (no fixes)
- `/design-review` — Visual design QA and polish
- `/setup-browser-cookies` — Import cookies from your real browser for authenticated testing
- `/retro` — Weekly engineering retrospective
- `/investigate` — Systematic debugging with root cause analysis
- `/document-release` — Post-ship documentation update
- `/codex` — OpenAI Codex second opinion (review, challenge, consult)
- `/careful` — Safety warnings before destructive commands
- `/freeze` — Restrict edits to a specific directory
- `/guard` — Full safety mode (careful + freeze)
- `/unfreeze` — Remove freeze edit restrictions
- `/gstack-upgrade` — Upgrade gstack to latest version

## 默认执行命令:
### ProGA
```bash
python batch_runner.py  --recircle false
```

### ProGA-WS
每次实验，一个 batch 是 20 次 run， 不清空上个 run 最后一代的前缀，保留到下一代。下一代的gen1 不再是从头生成，而是复用上一代的最后一代前缀，条件生成
```bash
python batch_runner.py --recircle true --total_runs 20
```

---

## 0) 易错认知 (新会话必读)

**以下是历史会话中踩过的坑，新会话务必先读：**

### recircle 的真实机制（Inter-run Prefix Carryover）

**⚠️ 这是最容易理解错的地方。**

`recircle` 控制的不仅是 Gen1 是否有前缀，而是**跨 run 的 prefix state 传递**。关键实现细节：

1. **Sampler 在同一个 GPU worker 的多个 run 之间被复用**（`batch_runner.py:153-157`）。模型对象不会在 run 之间重新创建。
2. **`recircle=True` 时，Gen1 不清除 prefix state**（`generation.py:340-342`）。上一个 run 最后一代设置的 `self.model.config.sampling.prefix` 会被保留下来。
3. **实际行为**：
   - **Run 1 Gen1**：sampler 新建，prefix=None → **无条件采样**（初始化，和 ProGA 一样）
   - **Run 1 Gen2-30**：sigmoid schedule，prefix 长度递增
   - **Run 2 Gen1**：prefix = Run 1 Gen30 的 parent prefix（**不被清除**）→ **条件生成**
   - **Run 2 Gen2-30**：sigmoid schedule，prefix 长度递增
   - ...以此类推到 Run 20
4. **`recircle=False`**（ProGA）：每个 run 的 Gen1 都显式清除 prefix（`prefix = None`），强制从无条件分布出发。

**错误理解**：~~"20 个 run 之间没有信息传递"~~ → 实际上 prefix state 通过 sampler 复用在 run 之间隐式传递。
**错误理解**：~~"热启动只是 Gen1 用初始种群做前缀"~~ → 实际上 Run 2+ 的 Gen1 用的是**上一个 run 末代的 parent prefix**，不是初始种群。

### ProGA-WS 的资源消耗
- **ProGA-WS 和 ProGA 的单 run 计算量完全相同。** 不是 20 倍开销。
- ProGA 和 ProGA-WS 均执行 20 个独立 run，每个 run 内部的 docking 调用次数一模一样（可能会因为被阈值拒绝数量不同，导致的细微差别）。

### 分子产出定义
- 一次完整优化 (30 代) 中取**最后一代**的 docking score 最低（最好）的一条分子作为该 run 的产出。
- 此定义与 FragEvo 一致（FragEvo 为 10 代取最好一条）。
- 代码位置：`batch_runner.py:246` → `df.sort_values('docking_score').iloc[0]`

### BD-MLM (SoftBD) 模型
- 代码在 `model/` 目录下，架构是 Block-Diffusion Transformer (DDiT backbone)。
- config 中写 `small-50M`，实际参数量是 **55M**（50M 是配置简称）。
- 来源于团队自己的 SoftMol 论文，使用 SoftMol 提供的 ZINC-Curated 数据集重新训练。
- **选择 55M 而非 SoftMol 主实验的 89M 的原因**：推理速度快 2 倍，生成质量不变。
- 训练时重新过滤了序列长度 >70 的分子（原数据集 max_length=72，BOS/EOS 拼接后会截断极少量序列引入噪音），10 epochs，选最低 validation loss 的 checkpoint。

### Docking 工具不统一
- **5-target benchmark** (parp1/fa7/5ht1b/braf/jak2)：使用 **QuickVina 2** (`qvina02`)，对齐 GEAM/f-RAG 基线。
- **10-target benchmark** (Beyond-Affinity)：使用 **AutoDock Vina** (`vina` Python API)，对齐 Beyond-Affinity 基线。
- exhaustiveness=1（如实报道，不刻意强化）。
- 两套工具都在 `utils/docking_runner.py` 中实现，通过 `config.yaml:docking.tool` 切换。

### 初始种群
- `utils/initial_population/initial_population.smi`，100 个分子。
- 来自 ZINC，采用与 FragEvo/AutoGrow 4.0 相同的分子量分层抽样策略（<100, 100-150, 150-200, 200-250 Da, 比例 2:3:4:1）。
- 直接使用了 FragEvo 提供的初始种群。

## 1) 项目目标（必须先对齐）

SoftGA 是一个 **Target-specific molecular design** 框架：
- 在 **约束** `QED >= 阈值` 且 `SA <= 阈值` 下，尽量优化（最小化）`docking score`。
- 优化方式是：`SoftBD 生成注入 + GA(交叉/突变) + Selection(ffhs/nsgaii)` 的迭代流程。

当前默认约束（`config.yaml`）：
- `qed_min = 0.5`
- `sa_max = 5.0`

## 2) 一眼看懂执行主链

单次运行主入口：`main.py`
- 调用 `execute.py::SoftGAWorkflowExecutor`
- 每个 receptor 跑一个完整进化流程

核心迭代在 `execute.py`：
1. `run_initial_generation()`：初始种群去重 + docking + QED/SA
2. 每代 `run_generation_step(g)`：
   - 从父代 docked 文件提取 SMILES
   - `run_softbd_process()` 生成
   - `run_ga_operations()` 交叉 + 突变 + 过滤
   - `run_offspring_evaluation()` 子代 docking
   - `run_selection()` 按 `selection_mode`（`ffhs`/`nsgaii`）选下一代
   - `run_selected_population_evaluation()` 只做分析，不影响选择

## 3) 真实算法行为

### 3.1 SoftBD 生成（`generation.py`）

#### Generation 1（recircle 控制）
- 当 `recircle=False`（ProGA）：显式清除 prefix → 无条件采样 `initial_samples`（默认 1000）。
- 当 `recircle=True`（ProGA-WS）：不清除 prefix state。
  - **Run 1**：prefix 默认为 None → 仍然是无条件采样
  - **Run 2+**：prefix 保留上一个 run Gen30 的 parent prefix → 条件生成
- 过滤规则：
  - RDKit 可解析
  - 单组分（不含 `.`）
  - canonical 去重
- 选择策略：`MaxMinPicker`，把初始种群 fingerprint 作为固定锚点，
  从有效样本里选 `gen1_n_select`（默认 100）个，倾向于：
  - 与初始种群距离更大
  - 彼此也更分散

#### Generation >= 2（动态前缀补全）
- 每个父代构造 prefix，保留 token 比例由 `calculate_keep_ratio()` 决定。
- 支持策略：`linear/aggressive/super_aggressive/sigmoid/piecewise/cosine/step/step_20_40_60_80`。
- 每父代生成 `samples_per_parent` 个候选（默认 10）。
- 候选筛选：
  - 必须有效、单组分、且不与父代完全相同
  - 计算 `dist = 1 - tanimoto(parent, child)`
  - 若有 `dist >= tanimoto_threshold` 的候选：随机选一个
  - 否则兜底选 `dist` 最大者
- 最终每个父代最多贡献 1 个子代（可为 0 个）。

### 3.2 交叉（`crossover.py`）
- 基于 GD_GA 风格的图操作（ring/non-ring crossover）
- 父本配对先按 tanimoto 范围筛选（默认 `0.2 ~ 0.8`）。
- 目标产量：`crossover.number_of_crossovers`（默认 100）。
- 输出为纯 SMILES；后续统一过 `utils/filter.py`。

### 3.3 突变（`mutation.py`）
- 当前为 **Graph-edit 突变**（GB-GA 风格）。
- 目标产量：`number_of_mutants`（默认 100）；最多尝试由 `max_attempts` 或 `max_attempts_multiplier` 控制。
- 每轮从父代池随机选 parent，随机采样编辑操作（如 `insert_atom`、`change_bond_order`、`delete_cyclic_bond`、`add_ring`、`delete_atom`、`change_atom`、`append_atom`）并生成候选。
- 候选需通过：sanitize、环结构约束、尺寸约束、canonical、单组分与去重（排除输入池与已产出重复）。

### 3.4 过滤（`utils/filter.py`）
- 当前做：sanitize + canonical + 单组分约束 + 去重。
- 单组分约束：显式去掉含 `.` 的多组分分子，并校验 RDKit fragment 数为 1。
- 不做 Lipinski/PAINS 等硬过滤。

### 3.5 选择（`selection.py`）
- 支持两种模式：`ffhs`（默认）与 `nsgaii`。
- `ffhs`:
1. docking 前：对子代执行约束过滤（`QED >= qed_min` 且 `SA <= sa_max`）。
2. 选择时先取可行分子并按 `docking_score` 升序。
3. 数量不足时在不可行集上用 NSGA-II 补齐（支持 crowding distance）。
- `nsgaii`:
1. docking 前：**不做** QED/SA 约束过滤。
2. 选择时 parent + offspring 合并后，对全部候选直接做 NSGA-II。
3. 该模式下不使用 `constraints` 参与选择。

共性行为：
- parent + offspring 合并时，重复 SMILES 取更优 docking（更小）。
- 默认多目标：`docking_score` minimize、`qed_score` maximize、`sa_score` minimize。
- 输出默认 `with_scores`：`SMILES docking qed sa`。

## 4) 关键文件速查

- `main.py`：CLI 入口，参数覆盖，单 receptor 运行
- `execute.py`：工作流编排（最关键）
- `generation.py`：SoftBD 采样与动态前缀策略
- `selection.py`：FFHS / NSGA-II 双模式选择
- `crossover.py`：图交叉
- `mutation.py`：Graph-edit 突变
- `utils/docking_runner.py`：obabel + qvina02 docking（两套工具都在此）
- `utils/chem_metrics.py`：QED/SA 缓存
- `config.yaml`：所有默认参数
- `batch_runner.py`：多 GPU / 多 run 批处理
- `model/`：BD-MLM (SoftBD) 模型代码，Block-Diffusion Transformer
- `weights/50M-epoch9-1145000.ckpt`：55M 模型权重
- `results/ablation/`：消融实验数据（8 个 CSV，每个 variant 取前 120 条）

## 5) 论文写作上下文

### 论文定位
- 投稿目标：**IEEE Transactions on Evolutionary Computation (TEVC)**，Regular Paper，13 页
- 论文标题：*A Dynamic Prompt-Guided Graph Genetic Algorithm for Constrained Expensive Molecular Optimization*
- 核心定位：**带约束的昂贵单目标优化**（不是多目标优化）。Docking 是唯一优化目标，QED/SA 是约束门槛。

### 论文核心故事线

ProGA 的核心机制是 **Dynamic Prefix Prompting**，在两个时间尺度上运作：

1. **Intra-run Progressive Prefixing**（代际层）：单个 run 内，BD-MLM 的 prefix 长度通过 sigmoid schedule 从 ρ_min 递增到 ρ_max，实现 exploration → exploitation 的平滑过渡。
2. **Inter-run Prefix Carryover**（跨 run 层）：启用 recircle 后，上一个 run 末代的 prefix state 传递到下一个 run 的 Gen1，跳过冷启动探索阶段。对应变体 **ProGA-WS**（Warm Start）。

配合 **FFHS**（Feasibility-First Hierarchical Selection）：利用 docking 与 QED/SA 之间的 cost asymmetry，先用廉价约束过滤不可行分子，再对可行分子做单目标 docking 排序，集中 oracle budget。

### 论文结构
- 详见 `TEVC-LaTeX/paper_structure.md`（完整大纲，含每个章节的内容说明和变动清单）

### 论文目录
- `TEVC-LaTeX/proga.tex`：主论文文件
- `TEVC-LaTeX/paper_structure.md`：**论文结构大纲**（章节安排、内容规划、术语一致性速查）
- `TEVC-LaTeX/CLAUDE.md`：**TEVC 写作专用系统提示词**（语态、结构、自检清单），写论文时必须加载此文件

### 参考论文目录 (`TEVC-LaTeX/paper/`)
- `FragEvo.md`：**最重要的参考**。语言模型指导 GA 优化分子的同类工作。参考其实验架构和写作风格。**注意：此论文未正式发表，不能引用。**
- `softmol.md`：BD-MLM (SoftBD) 模型的来源论文。ProGA 使用了其模型架构和 ZINC-Curated 数据集。
- `Beyond_Affinity.md`：第二个 benchmark (10-target) 的来源论文。基线数据直接取自此论文。
- `Graph-GA.md`: 交叉和突变操作的来源论文

### 两个 Benchmark 的来源
1. **5-target Novel Hit Benchmark** (parp1/fa7/5ht1b/braf/jak2)：来源于 f-RAG 和 GEAM 的评测框架。Hit 定义：QED≥0.5, SA≤5.0, docking<-10.0, novel to initial population。
2. **10-target Beyond-Affinity Benchmark**：来源于 Beyond-Affinity 论文。无 hit 概念，只报 top-k 指标。

### 术语映射（代码 → 论文）
- SoftGA → **ProGA**
- SoftGA-WS / recircle=True → **ProGA-WS**（Warm Start variant, inter-run prefix carryover enabled）
- SoftBD → **BD-MLM** (Block-Diffusion Molecular Language Model)
- recircle → **inter-run prefix carryover**
- 动态前缀/keep ratio schedule → **intra-run progressive prefixing**
- prefix → **generation prompt**（在论文 II-B 中定义：SMILES prefix as generation prompt）

## 10) 最小阅读集（新会话建议）

当任务是：
- 调参/算法问题：读 `config.yaml`, `execute.py`, `generation.py`, `selection.py`
- GA 操作问题：加读 `crossover.py`, `mutation.py`, `utils/filter.py`
- 对接问题：加读 `utils/docking_runner.py`
- 批跑问题：加读 `batch_runner.py`
- **论文写作**：读 `TEVC-LaTeX/CLAUDE.md`（写作规范）+ `TEVC-LaTeX/paper_structure.md`（结构大纲）+ `TEVC-LaTeX/proga.tex`（当前论文）
