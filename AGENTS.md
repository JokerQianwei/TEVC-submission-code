# SoftGA AgentS

Hello agent. You are one of the most talented programmers of your generation.
You are looking forward to putting those talents to use to improve SoftGA.

## 0) Philosophy
- Push your reasoning to 100% of your capacity.
- Code style: Concise, where one line outweighs a thousand; reject redundant defensive and compatibility code.
- Code style: Concise, where one line outweighs a thousand; reject redundant defensive and compatibility code.
- 始终用简单且富有逻辑的语言风格回复

## 0.5) gstack
- 所有网页浏览一律使用 gstack 的 `/browse` 技能。
- 严禁使用任何 `mcp__claude-in-chrome__*` 工具。
- 项目内共享技能位于 `.agents/skills/`。
- 若团队成员本机尚未注册 gstack，先运行 `mkdir -p ~/.codex/skills && cp -R .agents/skills/gstack ~/.codex/skills/gstack && cd ~/.codex/skills/gstack && ./setup --host codex`。
- 当前仓库对 Codex 实际可用的 gstack 技能为：`/office-hours`, `/plan-ceo-review`, `/plan-eng-review`, `/plan-design-review`, `/design-consultation`, `/review`, `/ship`, `/browse`, `/qa`, `/qa-only`, `/design-review`, `/setup-browser-cookies`, `/retro`, `/investigate`, `/document-release`, `/careful`, `/freeze`, `/guard`, `/unfreeze`, `/gstack-upgrade`。
- 注意：上游 README 会列出 `/codex`，但 Codex host 当前不会生成 `gstack-codex`，因此本仓库不要假定 `/codex` 可用。

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

#### Generation 1（无条件生成）
- 从模型采样 `initial_samples`（默认 1000）。
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
- 关键参数：`number_of_mutants`、`max_attempts`、`max_attempts_multiplier`、`mutation_rate`、`average_size`、`size_stdev`、`per_parent_trials`、`operator_probs`。
- 可输出 lineage（jsonl），关键字段为 `mutation_rule=graph_edit` 与 `edit_op`。

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
- `utils/docking_runner.py`：obabel + qvina02 docking
- `utils/chem_metrics.py`：QED/SA 缓存
- `config.yaml`：所有默认参数
- `batch_runner.py`：多 GPU / 多 run 批处理

## 5) I/O 与目录约定

每次 run 输出到：`<output_dir>/<receptor>/`
- `generation_0/initial_population_docked.smi`
- `generation_k/offspring_docked.smi`
- `generation_{k+1}/initial_population_docked.smi`（下一代父代）
- `execution_config_snapshot.json`
- `chem_metric_cache.json`
- `pop.csv`（当前 active 种群）
- `removed_ind_act_history.csv`

文件格式约定：
- 纯分子池：每行第一列是 SMILES
- docked 文件：至少两列 `SMILES docking_score`
- 选择输出（with_scores）：`SMILES docking qed sa`

## 7) 运行命令模板

单任务：
```bash
python main.py  --receptor parp1
```

批量：
```bash
python batch_runner.py \
  --config config.yaml \
  --receptor parp1 \
  --output_dir output/exp1 \
  --gpu_ids 0 \
  --tasks_per_gpu 2 \
  --total_runs 40 \
  --seed 42
```

## 8) 改代码时的硬约束

- 先保接口兼容：`execute.py` 是中枢，不要随意改 I/O 文件名与列格式。
- `selection.py` 修改后必须保持：
  - 能读 parent + offspring docked 文件
  - 输出 `with_scores` 兼容下游
  - `ffhs` 与 `nsgaii` 两种模式都可用（且语义不混淆）
- `generation.py` 修改时注意：
  - Gen1 和 Gen>=2 路径分离，不要混淆
  - `tanimoto_threshold` 当前是作用在“距离 dist=1-sim”，不是相似度
- `utils/filter.py` 当前故意轻过滤，不要默认加重过滤导致搜索空间骤缩。
- docking 依赖本地 `utils/docking/qvina02` 和 `obabel`；排障优先查环境与可执行权限。

## 9) 常见排障优先级

1. 先看 `<run_dir>/logs/worker_*.log` 与 `run.log`
2. 再看某代目录下：
   - `softbd_logs/`（生成失败细节）
   - `docking_results/final_scored.smi`
   - `*.lineage.jsonl`（若启用 lineage）
3. 检查 `chem_metric_cache.json` 是否损坏或路径异常
4. 检查 `config.yaml` 的覆盖项是否被 CLI 覆盖

## 10) 最小阅读集（新会话建议）

当任务是：
- 调参/算法问题：读 `config.yaml`, `execute.py`, `generation.py`, `selection.py`
- GA 操作问题：加读 `crossover.py`, `mutation.py`, `utils/filter.py`
- 对接问题：加读 `utils/docking_runner.py`
- 批跑问题：加读 `batch_runner.py`
