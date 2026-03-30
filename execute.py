"""
SoftGA 混合工作流执行脚本
==========================
1. 种群初始化和评估
2. 基于 SoftBD 生成新的分子
3. 对父代和生成分子进行遗传算法操作(交叉、突变)
4. 对新生成的子代进行评估
5. 通过 FFHS 选择策略筛选出下一代种群
6. 对精英种群进行评分分析并导出结果
7. 继续迭代
"""
import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import threading
import queue
import csv
import hashlib
import random
import secrets
from utils.config_snapshot import save_config_snapshot #保存参数（快照）
from utils.config_loader import load_config, resolve_config_path
import multiprocessing  
import shutil  
from rdkit import Chem
from utils.chem_metrics import ChemMetricCache

# 移除全局日志配置，避免多进程日志冲突
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# 确保logger有基本的handler，但不会与其他进程冲突
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_CONFIG = "./config.yaml"
SOFTBD_RANDOM_SEED_BITS = 31
ORACLE_SUCCESS_FILENAME = "oracle_calls_success.csv"
ORACLE_ALL_FILENAME = "oracle_calls_all.csv"
ORACLE_FAILURE_DOCKING_SCORE = 99.9
ORACLE_SUCCESS_HEADERS = [
    "call_idx",
    "smiles",
    "docking_score",
    "oracle_score",
    "qed",
    "sa",
    "generation",
    "phase",
]
ORACLE_ALL_HEADERS = [
    "submit_idx",
    "success_call_idx",
    "smiles",
    "docking_score",
    "oracle_score",
    "qed",
    "sa",
    "generation",
    "phase",
    "status",
]


class SoftGAWorkflowExecutor:    #工作流；主函数/入口文件就是在调用这个类
    def __init__(
        self,
        config_path: str,
        receptor_name: str,
        output_dir_override: Optional[str] = None,
        num_processors_override: Optional[int] = None,
        initial_population_file_override: Optional[str] = None,
        strategy_mode_override: Optional[str] = None,
        max_generations_override: Optional[int] = None,
        seed_override: Optional[int] = None,
        selection_mode_override: Optional[str] = None,
        samples_per_parent_override: Optional[int] = None,
        tanimoto_threshold_override: Optional[float] = None,
        min_keep_ratio_override: Optional[float] = None,
        max_keep_ratio_override: Optional[float] = None,
        recircle_override: Optional[bool] = None,
        softbd_enable_override: Optional[bool] = None,
        softbd_seed_mode_override: Optional[str] = None,
        gpu_override: Optional[str] = None,
        block_size_override: Optional[int] = None,
        length_override: Optional[int] = None,
        temperature_override: Optional[float] = None,
        nucleus_p_override: Optional[float] = None,
        gpu_max_batch_size_override: Optional[int] = None,
        steps_override: Optional[int] = None,
        initial_samples_override: Optional[int] = None,
        gen1_n_select_override: Optional[int] = None,
        gen1_selection_mode_override: Optional[str] = None,
        number_of_crossovers_override: Optional[int] = None,
        pair_min_tanimoto_override: Optional[float] = None,
        pair_max_tanimoto_override: Optional[float] = None,
        number_of_mutants_override: Optional[int] = None,
        qed_min_override: Optional[float] = None,
        sa_max_override: Optional[float] = None,
        enable_crowding_distance_override: Optional[bool] = None,
        cleanup_intermediate_files_override: Optional[bool] = None,
        docking_tool_override: Optional[str] = None,
        docking_exhaustiveness_override: Optional[int] = None,
        max_oracle_calls_override: Optional[int] = None,
        softbd_sampler=None,
        plot_top1: bool = False,
    ):
        """
        初始化SoftGA工作流执行器。        
        Args:
            config_path (str): 配置文件路径。
            receptor_name (str): 目标受体名称（必填）。
            output_dir_override (Optional[str]): 覆盖配置文件中的输出目录。
            num_processors_override (Optional[int]): 覆盖配置文件中的处理器数量。
            initial_population_file_override (Optional[str]): 覆盖 workflow.initial_population_file。
            strategy_mode_override (Optional[str]): 覆盖 softbd 策略模式。
            max_generations_override (Optional[int]): 覆盖最大代数。
            seed_override (Optional[int]): 覆盖随机种子。
            selection_mode_override (Optional[str]): 覆盖选择模式。
            qed_min_override (Optional[float]): 覆盖 FFHS 约束的 QED 下限。
            sa_max_override (Optional[float]): 覆盖 FFHS 约束的 SA 上限。
            samples_per_parent_override (Optional[int]): 覆盖每父代样本数。
            tanimoto_threshold_override (Optional[float]): 覆盖tanimoto阈值。
            min_keep_ratio_override (Optional[float]): 覆盖最小保留比例。
            max_keep_ratio_override (Optional[float]): 覆盖最大保留比例。
            block_size_override (Optional[int]): 覆盖block_size。
            length_override (Optional[int]): 覆盖length。
            temperature_override (Optional[float]): 覆盖temperature。
            nucleus_p_override (Optional[float]): 覆盖nucleus_p。
            gpu_max_batch_size_override (Optional[int]): 覆盖gpu_max_batch_size。
            initial_samples_override (Optional[int]): 覆盖initial_samples。
            gen1_selection_mode_override (Optional[str]): 覆盖 Gen1 选择模式 (maxmin | random)。
            number_of_crossovers_override (Optional[int]): 覆盖 crossover.number_of_crossovers。
            pair_min_tanimoto_override (Optional[float]): 覆盖 crossover.pair_min_tanimoto。
            pair_max_tanimoto_override (Optional[float]): 覆盖 crossover.pair_max_tanimoto。
            number_of_mutants_override (Optional[int]): 覆盖 mutation.number_of_mutants。
            enable_crowding_distance_override (Optional[bool]): 覆盖 selection.*_settings.enable_crowding_distance。
        """
        cli_overrides = {
            "output_dir": output_dir_override,
            "num_processors": num_processors_override,
            "initial_population_file": initial_population_file_override,
            "strategy_mode": strategy_mode_override,
            "max_generations": max_generations_override,
            "seed": seed_override,
            "selection_mode": selection_mode_override,
            "qed_min": qed_min_override,
            "sa_max": sa_max_override,
            "samples_per_parent": samples_per_parent_override,
            "tanimoto_threshold": tanimoto_threshold_override,
            "min_keep_ratio": min_keep_ratio_override,
            "max_keep_ratio": max_keep_ratio_override,
            "recircle": recircle_override,
            "softbd_enable": softbd_enable_override,
            "softbd_seed_mode": softbd_seed_mode_override,
            "gpu": gpu_override,
            "block_size": block_size_override,
            "length": length_override,
            "temperature": temperature_override,
            "nucleus_p": nucleus_p_override,
            "gpu_max_batch_size": gpu_max_batch_size_override,
            "steps": steps_override,
            "initial_samples": initial_samples_override,
            "gen1_n_select": gen1_n_select_override,
            "gen1_selection_mode": gen1_selection_mode_override,
            "number_of_crossovers": number_of_crossovers_override,
            "pair_min_tanimoto": pair_min_tanimoto_override,
            "pair_max_tanimoto": pair_max_tanimoto_override,
            "number_of_mutants": number_of_mutants_override,
            "enable_crowding_distance": enable_crowding_distance_override,
            "cleanup_intermediate_files": cleanup_intermediate_files_override,
            "docking_tool": docking_tool_override,
            "docking_exhaustiveness": docking_exhaustiveness_override,
            "max_oracle_calls": max_oracle_calls_override,
        }
        # 仅记录用户显式传入的覆盖参数，方便快照复现实验
        self._cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None and v != ""}

        self.config_path = config_path
        self.softbd_sampler = softbd_sampler
        self.config = self._load_config()        
        if cleanup_intermediate_files_override is not None:
            if 'performance' not in self.config:
                self.config['performance'] = {}
            self.config['performance']['cleanup_intermediate_files'] = bool(cleanup_intermediate_files_override)
            logger.info(
                "运行时覆盖 cleanup_intermediate_files 为: %s",
                bool(cleanup_intermediate_files_override),
            )
        # 应用处理器数量覆盖
        if num_processors_override is not None:
            if 'performance' not in self.config:
                self.config['performance'] = {}
            self.config['performance']['number_of_processors'] = num_processors_override
            logger.info(f"运行时覆盖处理器数量为: {num_processors_override}")

        if initial_population_file_override is not None:
            if 'workflow' not in self.config:
                self.config['workflow'] = {}
            self.config['workflow']['initial_population_file'] = str(initial_population_file_override)
            logger.info(f"运行时覆盖 initial_population_file 为: {initial_population_file_override}")
            
        # 应用 SoftBD 策略模式覆盖
        if strategy_mode_override:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'dynamic_strategy' not in self.config['softbd']:
                self.config['softbd']['dynamic_strategy'] = {}
            self.config['softbd']['dynamic_strategy']['strategy_mode'] = strategy_mode_override
            logger.info(f"运行时覆盖 SoftBD 策略模式为: {strategy_mode_override}")

        # 应用每父代样本数覆盖
        if samples_per_parent_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'generation_params' not in self.config['softbd']:
                self.config['softbd']['generation_params'] = {}
            self.config['softbd']['generation_params']['samples_per_parent'] = samples_per_parent_override
            logger.info(f"运行时覆盖 samples_per_parent 为: {samples_per_parent_override}")

        # 应用 tanimoto 阈值覆盖
        if tanimoto_threshold_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'generation_params' not in self.config['softbd']:
                self.config['softbd']['generation_params'] = {}
            self.config['softbd']['generation_params']['tanimoto_threshold'] = tanimoto_threshold_override
            logger.info(f"运行时覆盖 tanimoto_threshold 为: {tanimoto_threshold_override}")

        # 应用最小保留比例覆盖
        if min_keep_ratio_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'dynamic_strategy' not in self.config['softbd']:
                self.config['softbd']['dynamic_strategy'] = {}
            self.config['softbd']['dynamic_strategy']['min_keep_ratio'] = min_keep_ratio_override
            logger.info(f"运行时覆盖 min_keep_ratio 为: {min_keep_ratio_override}")

        # 应用最大保留比例覆盖
        if max_keep_ratio_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'dynamic_strategy' not in self.config['softbd']:
                self.config['softbd']['dynamic_strategy'] = {}
            self.config['softbd']['dynamic_strategy']['max_keep_ratio'] = max_keep_ratio_override
            logger.info(f"运行时覆盖 max_keep_ratio 为: {max_keep_ratio_override}")

        if recircle_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            self.config['softbd']['recircle'] = bool(recircle_override)
            logger.info(f"运行时覆盖 recircle 为: {bool(recircle_override)}")

        if softbd_enable_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            self.config['softbd']['enable'] = bool(softbd_enable_override)
            logger.info(f"运行时覆盖 softbd.enable 为: {bool(softbd_enable_override)}")

        if softbd_seed_mode_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            self.config['softbd']['seed_mode'] = str(softbd_seed_mode_override)
            logger.info(f"运行时覆盖 softbd.seed_mode 为: {softbd_seed_mode_override}")
        
        # 应用 SoftBD GPU 覆盖
        if gpu_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            self.config['softbd']['gpu'] = str(gpu_override)
            logger.info(f"运行时覆盖 SoftBD gpu 为: {gpu_override}")
        
        # 应用 block_size 覆盖
        if block_size_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            self.config['softbd']['block_size'] = block_size_override
            logger.info(f"运行时覆盖 block_size 为: {block_size_override}")

        # 应用 length 覆盖
        if length_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            self.config['softbd']['length'] = length_override
            logger.info(f"运行时覆盖 length 为: {length_override}")

        # 应用 temperature 覆盖
        if temperature_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            self.config['softbd']['temperature'] = temperature_override
            logger.info(f"运行时覆盖 temperature 为: {temperature_override}")
        
        # 应用 nucleus_p 覆盖
        if nucleus_p_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'generation_params' not in self.config['softbd']:
                self.config['softbd']['generation_params'] = {}
            self.config['softbd']['generation_params']['nucleus_p'] = nucleus_p_override
            logger.info(f"运行时覆盖 nucleus_p 为: {nucleus_p_override}")

        # 应用 gpu_max_batch_size 覆盖
        if gpu_max_batch_size_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'generation_params' not in self.config['softbd']:
                self.config['softbd']['generation_params'] = {}
            self.config['softbd']['generation_params']['gpu_max_batch_size'] = gpu_max_batch_size_override
            logger.info(f"运行时覆盖 gpu_max_batch_size 为: {gpu_max_batch_size_override}")

        # 应用 steps 覆盖
        if steps_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'generation_params' not in self.config['softbd']:
                self.config['softbd']['generation_params'] = {}
            self.config['softbd']['generation_params']['steps'] = int(steps_override)
            logger.info(f"运行时覆盖 steps 为: {steps_override}")
        
        # 应用 initial_samples 覆盖
        if initial_samples_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'generation_params' not in self.config['softbd']:
                self.config['softbd']['generation_params'] = {}
            self.config['softbd']['generation_params']['initial_samples'] = initial_samples_override
            logger.info(f"运行时覆盖 initial_samples 为: {initial_samples_override}")

        # 应用 gen1_n_select 覆盖
        if gen1_n_select_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'generation_params' not in self.config['softbd']:
                self.config['softbd']['generation_params'] = {}
            self.config['softbd']['generation_params']['gen1_n_select'] = int(gen1_n_select_override)
            logger.info(f"运行时覆盖 gen1_n_select 为: {gen1_n_select_override}")

        if gen1_selection_mode_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'generation_params' not in self.config['softbd']:
                self.config['softbd']['generation_params'] = {}
            self.config['softbd']['generation_params']['gen1_selection_mode'] = str(gen1_selection_mode_override)
            logger.info(f"运行时覆盖 gen1_selection_mode 为: {gen1_selection_mode_override}")

        if number_of_crossovers_override is not None:
            if 'crossover' not in self.config:
                self.config['crossover'] = {}
            self.config['crossover']['number_of_crossovers'] = int(number_of_crossovers_override)
            logger.info(f"运行时覆盖 number_of_crossovers 为: {number_of_crossovers_override}")

        if pair_min_tanimoto_override is not None:
            if 'crossover' not in self.config:
                self.config['crossover'] = {}
            self.config['crossover']['pair_min_tanimoto'] = float(pair_min_tanimoto_override)
            logger.info(f"运行时覆盖 pair_min_tanimoto 为: {pair_min_tanimoto_override}")

        if pair_max_tanimoto_override is not None:
            if 'crossover' not in self.config:
                self.config['crossover'] = {}
            self.config['crossover']['pair_max_tanimoto'] = float(pair_max_tanimoto_override)
            logger.info(f"运行时覆盖 pair_max_tanimoto 为: {pair_max_tanimoto_override}")

        if number_of_mutants_override is not None:
            if 'mutation' not in self.config:
                self.config['mutation'] = {}
            self.config['mutation']['number_of_mutants'] = int(number_of_mutants_override)
            logger.info(f"运行时覆盖 number_of_mutants 为: {number_of_mutants_override}")



        # 应用最大代数覆盖
        if max_generations_override is not None:
            if 'workflow' not in self.config:
                self.config['workflow'] = {}
            self.config['workflow']['max_generations'] = max_generations_override
            logger.info(f"运行时覆盖最大代数为: {max_generations_override}")

        # 应用随机种子覆盖
        if seed_override is not None:
            if 'workflow' not in self.config:
                self.config['workflow'] = {}
            self.config['workflow']['seed'] = seed_override
            
            # 同时更新 softbd 和 docking 中的 seed 以保持一致性，如果它们存在
            if 'softbd' in self.config:
                # softbd config usually doesn't have direct seed at root but we can set it if needed or rely on workflow seed
                # softbd_generate.py reads from workflow seed or passed arg.
                pass 
            if 'docking' in self.config:
                self.config['docking']['seed'] = seed_override
            
            logger.info(f"运行时覆盖随机种子为: {seed_override}")

        # 应用选择模式覆盖
        if selection_mode_override:
            if 'selection' not in self.config:
                self.config['selection'] = {}
            self.config['selection']['selection_mode'] = selection_mode_override
            logger.info(f"运行时覆盖选择模式为: {selection_mode_override}")

        if enable_crowding_distance_override is not None:
            selection_cfg = self.config.setdefault('selection', {})
            enable_crowding = bool(enable_crowding_distance_override)
            selection_cfg.setdefault('ffhs_settings', {})['enable_crowding_distance'] = enable_crowding
            selection_cfg.setdefault('nsgaii_settings', {})['enable_crowding_distance'] = enable_crowding
            logger.info(f"运行时覆盖 enable_crowding_distance 为: {enable_crowding}")

        if qed_min_override is not None or sa_max_override is not None:
            selection_cfg = self.config.setdefault('selection', {})
            ffhs_cfg = selection_cfg.setdefault('ffhs_settings', {})
            constraints = ffhs_cfg.setdefault('constraints', {})
            if qed_min_override is not None:
                constraints['qed_min'] = float(qed_min_override)
                logger.info(f"运行时覆盖 constraints.qed_min 为: {qed_min_override}")
            if sa_max_override is not None:
                constraints['sa_max'] = float(sa_max_override)
                logger.info(f"运行时覆盖 constraints.sa_max 为: {sa_max_override}")

        if docking_tool_override is not None:
            if 'docking' not in self.config:
                self.config['docking'] = {}
            self.config['docking']['tool'] = str(docking_tool_override)
            logger.info(f"运行时覆盖 docking.tool 为: {docking_tool_override}")
        if docking_exhaustiveness_override is not None:
            if 'docking' not in self.config:
                self.config['docking'] = {}
            self.config['docking']['exhaustiveness'] = int(docking_exhaustiveness_override)
            logger.info(f"运行时覆盖 docking.exhaustiveness 为: {docking_exhaustiveness_override}")

        self.run_params = {}
        self._setup_parameters_and_paths(receptor_name, output_dir_override)
        self.max_oracle_calls: Optional[int] = (
            int(max_oracle_calls_override) if max_oracle_calls_override is not None else None
        )
        if self.max_oracle_calls is not None and self.max_oracle_calls <= 0:
            raise ValueError("max_oracle_calls 必须是正整数")
        self.oracle_tracking_enabled = self.max_oracle_calls is not None
        self.oracle_success_calls = 0
        self.oracle_submit_calls = 0
        self.oracle_success_file_path: Optional[Path] = (
            self.output_dir / ORACLE_SUCCESS_FILENAME if self.oracle_tracking_enabled else None
        )
        self.oracle_all_file_path: Optional[Path] = (
            self.output_dir / ORACLE_ALL_FILENAME if self.oracle_tracking_enabled else None
        )
        if self.oracle_tracking_enabled:
            self._initialize_oracle_logs()
            logger.info(
                "Oracle 预算模式已启用: max_oracle_calls=%s, success_csv=%s, all_csv=%s",
                self.max_oracle_calls,
                self.oracle_success_file_path,
                self.oracle_all_file_path,
            )
        self.cleanup_intermediate_files = bool(
            self.config.get("performance", {}).get("cleanup_intermediate_files", False)
        )
        self.run_params["cleanup_intermediate_files"] = self.cleanup_intermediate_files
        self.plot_top1 = bool(plot_top1)
        self.run_params["plot_top1"] = self.plot_top1
        self.run_params["oracle_tracking_enabled"] = self.oracle_tracking_enabled
        self.run_params["max_oracle_calls"] = self.max_oracle_calls
        self.run_params["cli_overrides"] = dict(self._cli_overrides)
        cache_path = None if self.cleanup_intermediate_files else (self.output_dir / "chem_metric_cache.json")
        self.metric_cache = ChemMetricCache(cache_path)
        self._save_run_parameters()
        self.lineage_tracker_path: Optional[Path] = (
            self.output_dir / "lineage_tracker.json" if self.enable_lineage_tracking else None
        )
        self.lineage_tracker = self._load_lineage_tracker()
        self.history_paths: Dict[str, str] = {}
        self.smiles_to_history: Dict[str, str] = {}
        self.history_records: Dict[str, Dict] = {}
        self.removed_history_records: Dict[str, Dict] = {}
        self.current_active_histories: Set[str] = set()
        self.history_root_counter = 0
        self.placeholder_roots: Dict[int, str] = {}
        self.last_offspring_histories: Set[str] = set()
        self.last_offspring_smiles: Set[str] = set()
        logger.info(f"SoftGA工作流初始化完成, 输出目录: {self.output_dir}")
        logger.info(f"最大迭代代数: {self.max_generations}")

    def _load_config(self) -> dict:#加载配置文件
        cfg_path = resolve_config_path(self.config_path, PROJECT_ROOT)
        self.config_path = str(cfg_path)
        return load_config(str(cfg_path), PROJECT_ROOT)

    def _append_docking_engine_args(self, docking_args: List[str]) -> None:
        docking_cfg = self.config.get('docking', {}) or {}
        docking_tool = docking_cfg.get('tool')
        docking_exhaustiveness = docking_cfg.get('exhaustiveness')
        if docking_tool:
            docking_args.extend(['--docking_tool', str(docking_tool)])
        if docking_exhaustiveness is not None:
            docking_args.extend(['--exhaustiveness', str(int(docking_exhaustiveness))])

    def _initialize_oracle_logs(self) -> None:
        if not self.oracle_tracking_enabled:
            return
        for path, headers in (
            (self.oracle_success_file_path, ORACLE_SUCCESS_HEADERS),
            (self.oracle_all_file_path, ORACLE_ALL_HEADERS),
        ):
            assert path is not None
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()

    @staticmethod
    def _append_csv_rows(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
        if not rows:
            return
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(rows)

    def _remaining_oracle_budget(self) -> Optional[int]:
        if not self.oracle_tracking_enabled:
            return None
        return max(0, int(self.max_oracle_calls - self.oracle_success_calls))

    def _oracle_budget_reached(self) -> bool:
        if not self.oracle_tracking_enabled:
            return False
        return self.oracle_success_calls >= int(self.max_oracle_calls)

    def _get_softbd_strategy_progress(self, generation: int) -> Optional[float]:
        if not self.oracle_tracking_enabled or generation <= 1:
            return None
        max_calls = int(self.max_oracle_calls or 0)
        if max_calls <= 0:
            return None
        progress = max(0.0, min(1.0, float(self.oracle_success_calls) / float(max_calls)))
        logger.info(
            "第 %s 代: SoftBD 前缀策略改用 Oracle 进度 %.3f (%s/%s)",
            generation,
            progress,
            self.oracle_success_calls,
            self.max_oracle_calls,
        )
        return progress

    def _prepare_docking_input_for_oracle(
        self,
        smiles_file: Path,
        generation: int,
        phase: str,
    ) -> Tuple[Path, List[str]]:
        input_path = Path(smiles_file)
        submitted_smiles = self._read_smiles_from_file(input_path)
        if not self.oracle_tracking_enabled:
            return input_path, submitted_smiles

        remaining = self._remaining_oracle_budget()
        if remaining is None:
            return input_path, submitted_smiles
        if remaining <= 0:
            logger.info("第 %s 代(%s): Oracle 预算已耗尽，跳过 docking。", generation, phase)
            return input_path, []
        if len(submitted_smiles) <= remaining:
            return input_path, submitted_smiles

        budget_file = input_path.with_name(f"{input_path.stem}_oracle_budget.smi")
        kept = 0
        with open(input_path, "r", encoding="utf-8") as fin, open(
            budget_file, "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                if not line.strip():
                    continue
                if kept >= remaining:
                    break
                fout.write(line if line.endswith("\n") else f"{line}\n")
                kept += 1
        logger.info(
            "第 %s 代(%s): Oracle 预算截断 %s -> %s (remaining=%s)",
            generation,
            phase,
            len(submitted_smiles),
            kept,
            remaining,
        )
        return budget_file, submitted_smiles[:kept]

    def _record_oracle_calls(
        self,
        submitted_smiles: List[str],
        docked_file: Path,
        generation: int,
        phase: str,
    ) -> None:
        if not self.oracle_tracking_enabled or not submitted_smiles:
            return

        score_by_smiles: Dict[str, float] = {}
        docked_path = Path(docked_file)
        if docked_path.exists():
            with open(docked_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    smi = parts[0]
                    try:
                        score = float(parts[1])
                    except ValueError:
                        continue
                    if smi not in score_by_smiles:
                        score_by_smiles[smi] = score

        success_rows: List[Dict] = []
        all_rows: List[Dict] = []
        for smi in submitted_smiles:
            docking_score = float(score_by_smiles.get(smi, ORACLE_FAILURE_DOCKING_SCORE))
            status = "success" if smi in score_by_smiles else "fail"
            oracle_score = -docking_score
            qed, sa = self.metric_cache.get_or_compute(smi)
            self.oracle_submit_calls += 1
            success_call_idx: Optional[int] = None
            if status == "success":
                self.oracle_success_calls += 1
                success_call_idx = self.oracle_success_calls
                success_rows.append(
                    {
                        "call_idx": success_call_idx,
                        "smiles": smi,
                        "docking_score": docking_score,
                        "oracle_score": oracle_score,
                        "qed": qed,
                        "sa": sa,
                        "generation": generation,
                        "phase": phase,
                    }
                )
            all_rows.append(
                {
                    "submit_idx": self.oracle_submit_calls,
                    "success_call_idx": "" if success_call_idx is None else success_call_idx,
                    "smiles": smi,
                    "docking_score": docking_score,
                    "oracle_score": oracle_score,
                    "qed": qed,
                    "sa": sa,
                    "generation": generation,
                    "phase": phase,
                    "status": status,
                }
            )

        assert self.oracle_success_file_path is not None
        assert self.oracle_all_file_path is not None
        self._append_csv_rows(
            self.oracle_success_file_path,
            ORACLE_SUCCESS_HEADERS,
            success_rows,
        )
        self._append_csv_rows(
            self.oracle_all_file_path,
            ORACLE_ALL_HEADERS,
            all_rows,
        )
        self.metric_cache.flush()

    def _run_docking_and_record_oracle(
        self,
        smiles_file: Path,
        output_file: Path,
        generation_dir: Path,
        generation: int,
        phase: str,
    ) -> Tuple[bool, List[str]]:
        docking_input_file, submitted_smiles = self._prepare_docking_input_for_oracle(
            smiles_file,
            generation=generation,
            phase=phase,
        )
        if self.oracle_tracking_enabled and not submitted_smiles:
            return True, submitted_smiles

        num_processors = self.config.get('performance', {}).get('number_of_processors')
        docking_args = [
            '--smiles_file', str(docking_input_file),
            '--output_file', str(output_file),
            '--generation_dir', str(generation_dir),
            '--config_file', self.config_path,
            '--seed', str(getattr(self, "seed", 42)),
        ]
        if self.receptor_name:
            docking_args.extend(['--receptor', self.receptor_name])
        if num_processors is not None:
            docking_args.extend(['--number_of_processors', str(num_processors)])
        self._append_docking_engine_args(docking_args)

        docking_succeeded = self._run_script('utils/docking_runner.py', docking_args)
        self._record_oracle_calls(submitted_smiles, output_file, generation=generation, phase=phase)
        return docking_succeeded, submitted_smiles

    def _setup_parameters_and_paths(self, receptor_name: str, output_dir_override: Optional[str]):        
        project_root_cfg = Path(self.config.get('paths', {}).get('project_root', '.'))
        if not project_root_cfg.is_absolute():
            project_root_cfg = (PROJECT_ROOT / project_root_cfg).resolve()
        self.project_root = project_root_cfg
        workflow_config = self.config.get('workflow', {})
        seed_value = workflow_config.get("seed", 42)
        try:
            self.seed = int(seed_value)
        except (TypeError, ValueError):
            self.seed = 42
        random.seed(self.seed)
        self.run_params["seed"] = self.seed
        self._resolve_softbd_seed()
        self.enable_lineage_tracking = bool(workflow_config.get("enable_lineage_tracking", False))
        self.run_params["enable_lineage_tracking"] = self.enable_lineage_tracking
        # 记录配置和根目录
        self.run_params['config_file_path'] = self.config_path
        self.run_params['project_root'] = str(self.project_root)
        # 确定输出目录
        if output_dir_override:
            output_dir_name = output_dir_override
        else:
            output_dir_name = workflow_config.get('output_directory', 'SoftGA_output')
        base_output_dir = self.project_root / output_dir_name
        self.run_params['base_output_dir'] = str(base_output_dir)
        self.receptor_name = str(receptor_name).strip()
        if not self.receptor_name:
            raise ValueError("receptor_name 不能为空，请显式传入 --receptor")
        self.output_dir = base_output_dir / self.receptor_name
        self.run_params['receptor_name'] = self.receptor_name
        self.run_params['run_specific_output_dir'] = str(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 加载工作流核心参数
        self.max_generations = workflow_config.get('max_generations', 10)
        self.initial_population_file = workflow_config.get('initial_population_file')
        self.run_params['max_generations'] = self.max_generations
        if self.initial_population_file:
            p = Path(self.initial_population_file)
            if not p.is_absolute():
                p = (self.project_root / p).resolve()
            self.initial_population_file = str(p)
        self.run_params['initial_population_file'] = self.initial_population_file
        # 记录选择模式
        self.run_params['selection_mode'] = self._get_selection_mode()

    @staticmethod
    def _normalize_selection_mode(mode: Optional[str]) -> str:
        return "nsgaii" if str(mode or "").strip().lower() == "nsgaii" else "ffhs"

    def _get_selection_mode(self) -> str:
        selection_config = self.config.get("selection", {})
        raw_mode = str(selection_config.get("selection_mode", "ffhs")).strip().lower()
        mode = self._normalize_selection_mode(raw_mode)
        if raw_mode not in {"ffhs", "nsgaii"}:
            logger.warning(f"selection.selection_mode 非法: {raw_mode}，回退到 ffhs")
        return mode

    def _get_selection_n_select(self, mode: str) -> int:
        selection_config = self.config.get("selection", {})
        settings_key = "nsgaii_settings" if mode == "nsgaii" else "ffhs_settings"
        settings = selection_config.get(settings_key, {})
        return int(settings.get("n_select", 100))

    def _get_ffhs_constraints(self) -> Tuple[float, float]:
        selection_config = self.config.get("selection", {})
        ffhs_config = selection_config.get("ffhs_settings", {}) or {}
        constraints = ffhs_config.get("constraints", {}) or {}
        qed_min = float(constraints.get("qed_min", 0.5))
        sa_max = float(constraints.get("sa_max", 5.0))
        return qed_min, sa_max

    def _resolve_softbd_seed(self) -> None:
        softbd_cfg = self.config.setdefault('softbd', {})
        softbd_cfg.pop('random_seed_bits', None)
        seed_mode_raw = str(softbd_cfg.get('seed_mode', 'workflow')).strip().lower()
        if seed_mode_raw not in {'workflow', 'random_per_run'}:
            logger.warning(f"softbd.seed_mode 非法: {seed_mode_raw}，回退到 workflow")
            seed_mode_raw = 'workflow'
        softbd_cfg['seed_mode'] = seed_mode_raw

        if seed_mode_raw == 'random_per_run':
            softbd_seed = secrets.randbits(SOFTBD_RANDOM_SEED_BITS)
        else:
            softbd_seed = int(self.seed)

        self.softbd_seed_mode = seed_mode_raw
        self.softbd_random_seed_bits = SOFTBD_RANDOM_SEED_BITS
        self.softbd_seed = int(softbd_seed)
        self.run_params['softbd_seed_mode'] = self.softbd_seed_mode
        self.run_params['softbd_random_seed_bits'] = self.softbd_random_seed_bits
        self.run_params['softbd_seed_effective'] = self.softbd_seed
        logger.info(
            f"SoftBD seed 已确定: mode={self.softbd_seed_mode}, "
            f"random_seed_bits={self.softbd_random_seed_bits}, seed={self.softbd_seed}"
        )

    def _save_run_parameters(self):
        """保存本次运行的完整参数快照。"""
        snapshot_file_path = self.output_dir / "execution_config_snapshot.json"
        success = save_config_snapshot(
            original_config=self.config,
            execution_context=self.run_params,
            output_file_path=str(snapshot_file_path)
        )
        if success:
            logger.info(f"完整的执行配置快照已保存到: {snapshot_file_path}")
        else:
            logger.error("保存执行配置快照失败")
    def _load_lineage_tracker(self) -> Dict[str, List[Dict]]:
        """从磁盘加载既有的血统记录。"""
        if not getattr(self, "enable_lineage_tracking", False):
            return {}
        if self.output_dir and hasattr(self, "output_dir"):
            path = getattr(self, "lineage_tracker_path", None)
        else:
            path = None
        if path and path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
                logger.warning("血统跟踪文件格式异常，已忽略原有记录。")
            except Exception as exc:
                logger.warning(f"无法加载血统跟踪文件 {path}: {exc}")
        return {}
    def _save_lineage_tracker(self) -> None:
        """将血统跟踪记录持久化到磁盘。"""
        if not getattr(self, "enable_lineage_tracking", False):
            return
        if not self.lineage_tracker_path:
            return
        try:
            with open(self.lineage_tracker_path, 'w', encoding='utf-8') as f:
                json.dump(self.lineage_tracker, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error(f"保存血统跟踪文件失败: {exc}")
    def _write_jsonl(self, output_path: Path, entries: List[Dict]) -> None:
        """将记录写入 JSONL 文件。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in entries:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    def _read_jsonl(self, input_path: Path) -> List[Dict]:
        """读取 JSONL 文件并返回字典列表。"""
        if not input_path or not input_path.exists():
            return []
        entries: List[Dict] = []
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"无法解析血统记录: {line}")
        except Exception as exc:
            logger.warning(f"读取血统文件 {input_path} 失败: {exc}")
        return entries
    def _read_smiles_from_file(self, file_path: Path, first_column_only: bool = True) -> List[str]:
        """读取SMILES文件，默认只返回第一列。"""
        smiles: List[str] = []
        if not file_path or not file_path.exists():
            return smiles
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    smiles.append(parts[0] if first_column_only else parts)
        except Exception as exc:
            logger.warning(f"读取文件 {file_path} 时发生错误: {exc}")
        return smiles

    def _update_lineage_tracker(self, lineage_entries: List[Dict]) -> None:
        """更新内存中的血统跟踪数据并同步到磁盘。"""
        if not self.enable_lineage_tracking:
            return
        if not lineage_entries:
            return
        for entry in lineage_entries:
            child = entry.get("child")
            if not child:
                continue
            history = self.lineage_tracker.setdefault(child, [])
            history.append({
                "generation": entry.get("generation"),
                "sources": entry.get("sources", [])
            })
        self._save_lineage_tracker()

    def _record_initial_population(self, formatted_file: Path) -> None:
        """记录初代种群的血统来源。"""
        if not self.enable_lineage_tracking:
            return
        smiles_list = self._read_smiles_from_file(formatted_file)
        if not smiles_list:
            return
        entries: List[Dict] = []
        for smi in smiles_list:
            self._ensure_history(smi, generation=0)
            history = self.lineage_tracker.get(smi)
            if history:
                continue
            sources = [{
                "operation": "initial_population",
                "parents": []
            }]
            self.lineage_tracker.setdefault(smi, []).append({
                "generation": 0,
                "sources": sources
            })
            entries.append({
                "generation": 0,
                "child": smi,
                "sources": sources
            })
        if entries:
            lineage_file = formatted_file.parent / "initial_population_lineage.jsonl"
            self._write_jsonl(lineage_file, entries)
            self._save_lineage_tracker()
            logger.info(f"初代种群血统记录已保存到: {lineage_file}")
    def _short_hash(self, value: str) -> str:
        return hashlib.md5(value.encode('utf-8')).hexdigest()[:6]
    def _register_history(self, smiles: str, history: str) -> None:
        self.history_paths[smiles] = history
        self.smiles_to_history[smiles] = history
    def _create_root_history(self, smiles: str) -> str:
        if smiles in self.smiles_to_history:
            return self.smiles_to_history[smiles]
        token = f"ROOT-{self.history_root_counter}"
        self.history_root_counter += 1
        history = token
        self._register_history(smiles, history)
        return history
    def _ensure_history(self, smiles: str, generation: Optional[int] = None) -> str:
        history = self.smiles_to_history.get(smiles)
        if history:
            return history
        return self._create_root_history(smiles)
    def _create_generation_placeholder_root(self, generation: int) -> str:
        placeholder = self.placeholder_roots.get(generation)
        if placeholder is None:
            placeholder = f"GEN{generation}-ROOT"
            self.placeholder_roots[generation] = placeholder
        return placeholder
    def _build_operation_token(self, operation: str, parents: List[str], generation: int) -> str:
        op = (operation or "GEN").upper()
        parent_ids = []
        for parent in parents:
            parent_history = self.smiles_to_history.get(parent)
            if not parent_history:
                parent_history = self._ensure_history(parent, generation=generation)
            parent_ids.append(self._short_hash(parent_history))
        if not parent_ids:
            parent_ids.append(f"G{generation}")
        return f"{op}-{'_'.join(parent_ids)}"
    def _derive_history(self, smiles: str, generation: int, sources: List[Dict]) -> str:
        existing = self.smiles_to_history.get(smiles)
        if existing:
            return existing
        parent_history = None
        op_tokens: List[str] = []
        if sources:
            for source in sources:
                operation = source.get("operation", "GEN")
                parents = source.get("parents") or []
                if parent_history is None and parents:
                    for parent in parents:
                        parent_history = self.smiles_to_history.get(parent)
                        if parent_history:
                            break
                op_tokens.append(self._build_operation_token(operation, parents, generation))
            if parent_history is None and sources[0].get("parents"):
                first_parent = sources[0]["parents"][0]
                parent_history = self._ensure_history(first_parent, generation=generation)
        if parent_history is None:
            parent_history = self._create_generation_placeholder_root(generation)
        if not op_tokens:
            op_tokens.append(self._build_operation_token("GEN", [], generation))
        history = f"{parent_history}|{'_'.join(op_tokens)}"
        self._register_history(smiles, history)
        return history
    def _assign_histories_to_offspring(self, generation: int, lineage_entries: List[Dict]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for entry in lineage_entries:
            child = entry.get("child")
            if not child:
                continue
            history = self._derive_history(child, generation, entry.get("sources", []))
            entry["history_data"] = history
            mapping[child] = history
        return mapping
    def _compute_metrics(self, smiles: str, docking_score: Optional[float]) -> Dict[str, Optional[float]]:
        metrics: Dict[str, Optional[float]] = {
            "docking_score": docking_score,
            "total": docking_score
        }
        qed, sa = self.metric_cache.get_or_compute(smiles)
        metrics["qed"] = qed
        metrics["sa"] = sa
        return metrics
    def _upsert_history_record(self, history: str, smiles: str, generation: int, docking_score: Optional[float], mark_active: bool) -> None:
        record = self.history_records.get(history, {
            "smiles": smiles,
            "history_data": history,
            "generation_created": generation,
            "status": "inactive"
        })
        metrics = self._compute_metrics(smiles, docking_score)
        record["smiles"] = smiles
        record["history_data"] = history
        record.setdefault("generation_created", generation)
        record["last_generation"] = generation
        record["metrics"] = metrics
        record["docking_score"] = docking_score
        if mark_active:
            record["status"] = "active"
        elif record.get("status") not in ("removed", "active"):
            record["status"] = "inactive"
        self.history_records[history] = record
    def _ingest_population_metrics(self, docked_file: Path, generation: int, mark_active: bool = False) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        path_obj = Path(docked_file)
        if not path_obj.exists():
            return mapping
        with open(path_obj, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                smiles = parts[0]
                docking_score: Optional[float] = None
                if len(parts) >= 2:
                    try:
                        docking_score = float(parts[1])
                    except ValueError:
                        docking_score = None
                history = self._ensure_history(smiles, generation=generation)
                mapping[smiles] = history
                self._upsert_history_record(history, smiles, generation, docking_score, mark_active)
        self.metric_cache.flush()
        return mapping

    def _append_qed_sa_to_docked_file(self, docked_file: Path) -> bool:
        """为 docked 文件补全 QED/SA 列（原地覆写）。"""
        path_obj = Path(docked_file)
        if not path_obj.exists() or self._count_molecules(str(path_obj)) == 0:
            return True
        args = ["--input_file", str(path_obj), "--output_file", str(path_obj)]
        cache_path = getattr(self.metric_cache, "cache_path", None)
        if cache_path:
            args += ["--cache_file", str(cache_path)]
        return self._run_script("utils/evo_qedsa_eachmol.py", args)

    @staticmethod
    def _write_ranked_by_docking(input_file: Path, output_file: Path) -> bool:
        """
        将 docked 文件按 docking_score(第2列) 升序排序后写入新文件。
        - 保留整行（包含 QED/SA 等列），只改变行顺序
        - 无法解析 docking_score 的行会排到末尾
        """
        input_file = Path(input_file)
        output_file = Path(output_file)
        if not input_file.exists():
            return False

        rows = []
        with open(input_file, "r", encoding="utf-8") as f:
            for order, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                docking = None
                if len(parts) >= 2:
                    try:
                        docking = float(parts[1])
                    except Exception:
                        docking = None
                # 稳定排序：docking -> 原始顺序
                key = (0, docking, order) if docking is not None else (1, 0.0, order)
                rows.append((key, line))

        rows.sort(key=lambda x: x[0])
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for _, line in rows:
                f.write(line + "\n")
        return True
    def _mark_histories_active(self, histories: Set[str], generation: int) -> None:
        for history in histories:
            record = self.history_records.get(history)
            if not record:
                continue
            record["status"] = "active"
            record["last_generation"] = generation
            self.history_records[history] = record
        self.current_active_histories = set(histories)
    def _mark_histories_removed(self, histories: Set[str], generation: int) -> None:
        for history in histories:
            record = self.history_records.get(history)
            if not record:
                continue
            if record.get("status") == "removed":
                continue
            record["status"] = "removed"
            record["removed_generation"] = generation
            self.history_records[history] = record
            self.removed_history_records[history] = record
    def _format_float(self, value: Optional[float]) -> str:
        if value is None:
            return ""
        try:
            return f"{float(value):.6f}"
        except Exception:
            return ""
    def _export_evomo_files(self) -> None:
        pop_file = self.output_dir / "pop.csv"
        removed_file = self.output_dir / "removed_ind_act_history.csv"
        pop_headers = ["smiles", "total", "qed", "sa", "docking_score", "history_data"]
        with open(pop_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(pop_headers)
            for history, record in sorted(self.history_records.items()):
                if record.get("status") != "active":
                    continue
                metrics = record.get("metrics", {})
                writer.writerow([
                    record.get("smiles", ""),
                    self._format_float(metrics.get("total")),
                    self._format_float(metrics.get("qed")),
                    self._format_float(metrics.get("sa")),
                    self._format_float(metrics.get("docking_score")),
                    record.get("history_data", history)
                ])
        with open(removed_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["history_data", "total", "qed", "sa", "docking_score", "smiles"])
            for history, record in sorted(self.removed_history_records.items()):
                metrics = record.get("metrics", {})
                writer.writerow([
                    record.get("history_data", history),
                    self._format_float(metrics.get("total")),
                    self._format_float(metrics.get("qed")),
                    self._format_float(metrics.get("sa")),
                    self._format_float(metrics.get("docking_score")),
                    record.get("smiles", "")
                ])
    

    def _run_script(self, script_path: str, args: List[str]) -> bool:
        """
        统一的脚本执行函数，通过流式处理输出防止死锁，并增加超时保护。
        
        Args:
            script_path (str): 相对于项目根目录的脚本路径。
            args (List[str]): 脚本的命令行参数列表。
            
        Returns:
            bool: 脚本是否执行成功。
        """
        full_script_path = self.project_root / script_path
        cmd = ['python', str(full_script_path)] + args
        logger.debug(f"Executing command: {' '.join(cmd)}")

        env = os.environ.copy()
        seed_value = str(getattr(self, "seed", 42))
        env["PYTHONHASHSEED"] = seed_value
        try:
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                cwd=str(self.project_root),
                env=env,
                close_fds=True
            ) as process:
                
                # 创建队列来从线程中接收输出
                q_stdout = queue.Queue()
                q_stderr = queue.Queue()

                # 创建并启动线程来实时读取输出
                thread_stdout = threading.Thread(target=self._read_stream, args=(process.stdout, q_stdout))
                thread_stderr = threading.Thread(target=self._read_stream, args=(process.stderr, q_stderr))
                thread_stdout.start()
                thread_stderr.start()

                # 等待进程结束，设置超时
                try:
                    process.wait(timeout=3600)  # 1小时超时
                except subprocess.TimeoutExpired:
                    logger.error(f"Script {script_path} timed out (1 hour). Terminating...")
                    process.kill()  # 强制杀死进程
                    # 再等待一小段时间确保线程能读取完最后的信息
                    thread_stdout.join(timeout=5)
                    thread_stderr.join(timeout=5)
                    # 记录日志并返回失败
                    self._log_subprocess_output(script_path, q_stdout, q_stderr, "after timeout")
                    return False
                
                # 进程正常结束后，等待读取线程完成
                thread_stdout.join()
                thread_stderr.join()

                # 收集并记录输出
                stdout_str, stderr_str = self._log_subprocess_output(script_path, q_stdout, q_stderr, "final")

                if process.returncode == 0:
                    logger.info(f"Script {script_path} executed successfully.")
                    return True
                else:
                    logger.error(f"Script {script_path} failed with return code {process.returncode}.")
                    # 在失败时，即使没有stderr，也记录stdout，可能包含线索
                    if stderr_str:
                        logger.error(f"Error output (stderr):\n{stderr_str}")
                    if stdout_str:
                        logger.error(f"Standard output (stdout):\n{stdout_str}")
                    return False
                    
        except Exception as e:
            logger.error(f"An exception occurred while trying to run script {script_path}: {e}", exc_info=True)
            return False

    def _read_stream(self, stream, q: queue.Queue):
        """实时读取流（stdout/stderr）并放入队列"""
        try:
            for line in iter(stream.readline, ''):
                q.put(line)
        finally:
            stream.close()

    def _log_subprocess_output(self, script_path: str, q_stdout: queue.Queue, q_stderr: queue.Queue, context: str) -> Tuple[str, str]:
        """从队列中收集并记录子进程的输出"""
        stdout_lines = []
        while not q_stdout.empty():
            stdout_lines.append(q_stdout.get_nowait())
        stdout_str = "".join(stdout_lines)

        stderr_lines = []
        while not q_stderr.empty():
            stderr_lines.append(q_stderr.get_nowait())
        stderr_str = "".join(stderr_lines)

        if stdout_str:
            logger.debug(f"--- stdout for {script_path} ({context}) ---\n{stdout_str}")
            logger.debug(f"--- end stdout for {script_path} ---")
        
        return stdout_str, stderr_str

    def _count_molecules(self, file_path: str) -> int:
        """统计SMILES文件中的分子数量"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                count = sum(1 for line in f if line.strip())
            return count
        except FileNotFoundError:
            return 0
    def _remove_duplicates_from_smiles_file(self, input_file: str, output_file: str) -> int:
        """
        去除SMILES文件中的重复分子,并为每个分子添加唯一ID。
        输出格式: SMILES  ligand_id_X
        增加文件锁防护，避免并发访问冲突。
        """
        import time
        import random
        
        # 添加随机延迟，避免多进程同时访问文件
        time.sleep(random.uniform(0.1, 0.5))
        
        try:
            unique_smiles = set()
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        smiles = parts[0]
                        if smiles:
                            unique_smiles.add(smiles)            
            unique_smiles_list = sorted(list(unique_smiles))            
            
            # 使用临时文件写入，然后原子性重命名，避免写入冲突
            temp_output_file = output_file + f".tmp_{os.getpid()}_{int(time.time())}"
            with open(temp_output_file, 'w', encoding='utf-8') as f:
                for i, smiles in enumerate(unique_smiles_list):
                    f.write(f"{smiles}\tligand_id_{i}\n")
            
            # 原子性重命名
            import shutil
            shutil.move(temp_output_file, output_file)
            
            logger.info(f"去重完成: {len(unique_smiles_list)} 个独特分子保存到 {output_file}")
            return len(unique_smiles_list)
        except Exception as e:
            logger.error(f"去重过程中发生错误: {e}")
            # 清理可能的临时文件
            temp_file = output_file + f".tmp_{os.getpid()}_{int(time.time())}"
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            return 0

    def _extract_smiles_from_docked_file(self, docked_file: str, output_smiles_file: str) -> bool:
        """从带对接分数的文件中提取纯SMILES,用于遗传操作或分解"""
        try:
            with open(docked_file, 'r') as infile, open(output_smiles_file, 'w') as outfile:
                for line in infile:
                    line = line.strip()
                    if line:
                        smiles = line.split()[0]
                        outfile.write(f"{smiles}\n")
            return True
        except Exception as e:
            logger.error(f"从 {docked_file} 提取SMILES时出错: {e}")
            return False

    def _execute_ga_stage(
        self,
        ga_op_name: str,
        ga_script: str,
        input_pool_file: str,
        raw_output_file: Path,
        filtered_output_file: Path,
        raw_lineage_file: Path,
        filtered_lineage_file: Path
    ) -> Tuple[bool, Optional[Path]]:
        """辅助函数，用于运行一个GA阶段（如交叉）及其后续的过滤，并返回(是否成功, 过滤后的血统文件路径)。"""
        logger.info(f"开始执行 {ga_op_name}...")
        
        # 运行GA操作
        ga_args = [
            '--smiles_file', input_pool_file,
            '--output_file', str(raw_output_file),
            '--config_file', self.config_path,
            '--seed', str(getattr(self, "seed", 42)),
        ]
        if ga_script == 'crossover.py':
            n_cross = self._cli_overrides.get("number_of_crossovers")
            if n_cross is not None:
                ga_args.extend(['--number_of_crossovers', str(int(n_cross))])
            pair_min = self._cli_overrides.get("pair_min_tanimoto")
            if pair_min is not None:
                ga_args.extend(['--pair_min_tanimoto', str(float(pair_min))])
            pair_max = self._cli_overrides.get("pair_max_tanimoto")
            if pair_max is not None:
                ga_args.extend(['--pair_max_tanimoto', str(float(pair_max))])
        elif ga_script == 'mutation.py':
            n_mut = self._cli_overrides.get("number_of_mutants")
            if n_mut is not None:
                ga_args.extend(['--number_of_mutants', str(int(n_mut))])
        if self.enable_lineage_tracking:
            ga_args.extend(['--lineage_file', str(raw_lineage_file)])

        ga_succeeded = self._run_script(ga_script, ga_args)
        if not ga_succeeded:
            logger.error(f"'{ga_op_name}' 脚本执行失败。")
            return False, None

        # 运行过滤器
        filter_succeeded = self._run_script('utils/filter.py', [
            '--smiles_file', str(raw_output_file),
            '--output_file', str(filtered_output_file),
            '--config_file', self.config_path,
        ])
        if not filter_succeeded:
            logger.error(f"'{ga_op_name}' 过滤失败。")
            return False, None

        if not self.enable_lineage_tracking:
            logger.info(f"'{ga_op_name}' 操作完成, 生成 {self._count_molecules(str(filtered_output_file))} 个过滤后的分子。")
            return True, None

        filtered_entries = self._filter_lineage_entries(raw_lineage_file, filtered_output_file)
        self._write_jsonl(filtered_lineage_file, filtered_entries)
        logger.info(f"'{ga_op_name}' 操作完成, 生成 {self._count_molecules(str(filtered_output_file))} 个过滤后的分子。")
        return True, filtered_lineage_file

    def _filter_lineage_entries(self, raw_lineage_file: Path, filtered_output_file: Path) -> List[Dict]:
        """根据过滤后的SMILES保留有效的血统记录。"""
        raw_entries = self._read_jsonl(raw_lineage_file)
        if not raw_entries:
            return []
        filtered_smiles = set(self._read_smiles_from_file(filtered_output_file))
        if not filtered_smiles:
            return []
        kept_entries: List[Dict] = []
        for entry in raw_entries:
            child = entry.get("child")
            if not child or child not in filtered_smiles:
                continue
            filtered_entry = dict(entry)
            filtered_entry["child"] = child
            filtered_entry["parents"] = list(filtered_entry.get("parents", []))
            kept_entries.append(filtered_entry)
        return kept_entries

    def _combine_files(self, file_list: List[str], output_file: str) -> bool:
        """合并多个SMILES文件到一个文件"""
        try:
            with open(output_file, 'w') as outf:
                for file_path in file_list:
                    if not file_path or not Path(file_path).exists():
                        continue
                    with open(file_path, 'r') as inf:
                        for line in inf:
                            line = line.strip()
                            if line:
                                outf.write(line + '\n')
            return True
        except Exception as e:
            logger.error(f"合并文件时发生错误: {e}")
            return False
    def _append_qed_sa_to_file(self, file_path: Path) -> bool:
        """为只有 SMILES 的文件添加 metrics 列 (qed, sa)，不包含 docking_score。"""
        path_obj = Path(file_path)
        if not path_obj.exists() or self._count_molecules(str(path_obj)) == 0:
            return True
            
        temp_file = path_obj.with_suffix('.tmp')
        
        try:
            processed_lines = []
            with open(path_obj, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    smiles = parts[0]
                    # 计算 metrics (docking_score 传入 None，我们不使用它)
                    metrics = self._compute_metrics(smiles, None)
                    
                    # 格式化输出: SMILES qed sa
                    qed_str = f"{metrics['qed']:.4f}" if metrics['qed'] is not None else "0.0000"
                    sa_str = f"{metrics['sa']:.4f}" if metrics['sa'] is not None else "10.0000"
                    
                    new_line = f"{smiles}\t{qed_str}\t{sa_str}"
                    processed_lines.append(new_line)
            
            # 写入临时文件
            with open(temp_file, 'w', encoding='utf-8') as f:
                for line in processed_lines:
                    f.write(line + '\n')
            
            # 替换原文件
            shutil.move(str(temp_file), str(path_obj))
            return True
            
        except Exception as e:
            logger.error(f"处理文件 {file_path} 添加指标时出错: {e}")
            if temp_file.exists():
                os.remove(temp_file)
            return False

    def _filter_smiles_file_by_constraints(self, file_path: Path, qed_min: float, sa_max: float) -> Tuple[bool, int, int]:
        """按 QED/SA 约束过滤分子文件（保留原始行格式）。"""
        path_obj = Path(file_path)
        if not path_obj.exists() or self._count_molecules(str(path_obj)) == 0:
            return True, 0, 0

        temp_file = path_obj.with_suffix('.constraint_filtered.tmp')
        total_count = 0
        kept_count = 0
        kept_lines: List[str] = []

        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    total_count += 1
                    smiles = parts[0]
                    metrics = self._compute_metrics(smiles, None)
                    qed = metrics.get("qed")
                    sa = metrics.get("sa")
                    if qed is None or sa is None:
                        continue
                    if qed >= qed_min and sa <= sa_max:
                        kept_lines.append(line)
                        kept_count += 1

            with open(temp_file, 'w', encoding='utf-8') as f:
                for line in kept_lines:
                    f.write(line + '\n')

            shutil.move(str(temp_file), str(path_obj))
            return True, total_count, kept_count

        except Exception as e:
            logger.error(f"约束过滤文件 {file_path} 时出错: {e}")
            if temp_file.exists():
                os.remove(temp_file)
            return False, total_count, kept_count

    def _count_smiles_passing_constraints(self, file_path: Path, qed_min: float, sa_max: float) -> Tuple[int, int]:
        """仅统计文件中通过 QED/SA 约束的分子数量，不修改文件内容。"""
        path_obj = Path(file_path)
        if not path_obj.exists() or self._count_molecules(str(path_obj)) == 0:
            return 0, 0

        total_count = 0
        kept_count = 0
        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if not parts:
                        continue
                    total_count += 1
                    smiles = parts[0]
                    metrics = self._compute_metrics(smiles, None)
                    qed = metrics.get("qed")
                    sa = metrics.get("sa")
                    if qed is None or sa is None:
                        continue
                    if qed >= qed_min and sa <= sa_max:
                        kept_count += 1
        except Exception as e:
            logger.warning(f"统计约束通过数量时出错 ({file_path}): {e}")
        return total_count, kept_count

    def run_ga_operations(self, parent_smiles_file: str, softbd_generated_file: Optional[str], generation: int) -> Optional[Tuple[str, str, Optional[str], Optional[str]]]:
        """
        串行执行遗传算法操作（交叉和突变）以避免死锁。
        
        Args:
            parent_smiles_file (str): 父代SMILES文件路径。
            softbd_generated_file (Optional[str]): SoftBD 生成的SMILES文件路径。
            generation (int): 当前代数。
            
        Returns:
            Optional[Tuple[str, str, Optional[str], Optional[str]]]: 
                成功则返回 (交叉后代文件, 突变后代文件, 交叉血统文件, 突变血统文件)。
        """
        logger.info(f"第 {generation} 代: 开始串行执行遗传算法操作...")
        gen_dir = self.output_dir / f"generation_{generation}"

        # 1. 合并父代和 SoftBD 产出，作为 GA 操作输入
        ga_input_pool_file = gen_dir / "ga_input_pool.smi"
        files_to_combine = [parent_smiles_file]
        if softbd_generated_file:
            files_to_combine.append(softbd_generated_file)
        
        if not self._combine_files(files_to_combine, str(ga_input_pool_file)):
            logger.error(f"第 {generation} 代: 合并父代和 SoftBD 产出失败。")
            return None
        logger.info(f"第 {generation} 代: GA操作输入池已创建,共 {self._count_molecules(str(ga_input_pool_file))} 个分子。")

        # 2. 串行执行交叉和突变以避免死锁
        crossover_raw_file = gen_dir / "crossover_raw.smi"
        crossover_filtered_file = gen_dir / "crossover_filtered.smi"
        crossover_raw_lineage = gen_dir / "crossover_raw_lineage.jsonl"
        crossover_filtered_lineage = gen_dir / "crossover_filtered_lineage.jsonl"
        mutation_raw_file = gen_dir / "mutation_raw.smi"
        mutation_filtered_file = gen_dir / "mutation_filtered.smi"
        mutation_raw_lineage = gen_dir / "mutation_raw_lineage.jsonl"
        mutation_filtered_lineage = gen_dir / "mutation_filtered_lineage.jsonl"

        # 执行交叉操作
        logger.info(f"第 {generation} 代: 开始交叉操作...")
        crossover_ok, crossover_lineage_path = self._execute_ga_stage(
            "交叉", 'crossover.py',
            str(ga_input_pool_file), crossover_raw_file, crossover_filtered_file,
            crossover_raw_lineage, crossover_filtered_lineage
        )
        
        if not crossover_ok:
            logger.error(f"第 {generation} 代: 交叉操作失败。")
            return None
            
        # debug 模式才为中间文件补全 QED/SA，cleanup 模式下跳过以减少写盘
        if not self.cleanup_intermediate_files:
            self._append_qed_sa_to_file(crossover_raw_file)
            self._append_qed_sa_to_file(crossover_filtered_file)

        # 执行变异操作
        logger.info(f"第 {generation} 代: 开始变异操作...")
        mutation_ok, mutation_lineage_path = self._execute_ga_stage(
            "突变", 'mutation.py',
            str(ga_input_pool_file), mutation_raw_file, mutation_filtered_file,
            mutation_raw_lineage, mutation_filtered_lineage
        )
        
        if not mutation_ok:
            logger.error(f"第 {generation} 代: 变异操作失败。")
            return None
            
        # debug 模式才为中间文件补全 QED/SA，cleanup 模式下跳过以减少写盘
        if not self.cleanup_intermediate_files:
            self._append_qed_sa_to_file(mutation_raw_file)
            self._append_qed_sa_to_file(mutation_filtered_file)

        logger.info(f"第 {generation} 代: 交叉和变异操作串行完成。")
        return (
            str(crossover_filtered_file),
            str(mutation_filtered_file),
            str(crossover_lineage_path) if crossover_lineage_path else None,
            str(mutation_lineage_path) if mutation_lineage_path else None
        )

    def run_offspring_evaluation(
        self,
        crossover_file: str,
        mutation_file: str,
        generation: int,
        crossover_lineage_file: Optional[str] = None,
        mutation_lineage_file: Optional[str] = None
    ) -> Optional[Tuple[str, Optional[str]]]:
        """
        执行子代种群的评估（对接），并生成血统记录。
        
        Args:
            crossover_file (str): 交叉后代文件路径。
            mutation_file (str): 突变后代文件路径。
            generation (int): 当前代数。
            crossover_lineage_file (Optional[str]): 交叉产出对应的血统文件。
            mutation_lineage_file (Optional[str]): 突变产出对应的血统文件。
            
        Returns:
            Optional[Tuple[str, Optional[str]]]: (子代对接结果文件路径, 血统记录文件路径)。
        """
        logger.info(f"第 {generation} 代: 开始子代评估...")
        gen_dir = self.output_dir / f"generation_{generation}"

        # 1. 合并交叉和突变结果
        offspring_raw_file = gen_dir / "offspring_combined_raw.smi"
        if not self._combine_files([crossover_file, mutation_file], str(offspring_raw_file)):
            logger.error(f"第 {generation} 代: 子代合并失败。")
            return None
        
        # 2. 对子代进行去重和格式化（为对接做准备）
        offspring_formatted_file = gen_dir / "offspring_formatted_for_docking.smi"
        offspring_count = self._remove_duplicates_from_smiles_file(
            str(offspring_raw_file), 
            str(offspring_formatted_file)
        )
        selection_mode = self._get_selection_mode()
        if selection_mode == "ffhs":
            qed_min, sa_max = self._get_ffhs_constraints()

            crossover_total, crossover_kept = self._count_smiles_passing_constraints(
                Path(crossover_file), qed_min, sa_max
            )
            mutation_total, mutation_kept = self._count_smiles_passing_constraints(
                Path(mutation_file), qed_min, sa_max
            )
            logger.info(
                f"第 {generation} 代: 交叉产物 docking前约束过滤统计，保留 {crossover_kept}/{crossover_total} 个分子 "
                f"(QED>={qed_min}, SA<={sa_max})。"
            )
            logger.info(
                f"第 {generation} 代: 突变产物 docking前约束过滤统计，保留 {mutation_kept}/{mutation_total} 个分子 "
                f"(QED>={qed_min}, SA<={sa_max})。"
            )

            constraint_filter_ok, before_filter_count, offspring_count = self._filter_smiles_file_by_constraints(
                offspring_formatted_file,
                qed_min,
                sa_max
            )
            if not constraint_filter_ok:
                logger.error(f"第 {generation} 代: 子代约束过滤失败。")
                return None
            logger.info(
                f"第 {generation} 代: docking前约束过滤完成，保留 {offspring_count}/{before_filter_count} 个分子 "
                f"(QED>={qed_min}, SA<={sa_max})。"
            )
        else:
            logger.info(f"第 {generation} 代: selection_mode=nsgaii，跳过 docking前 QED/SA 约束过滤。")
        offspring_lineage_file = gen_dir / "offspring_lineage.jsonl" if self.enable_lineage_tracking else None
        if offspring_count == 0:
            if selection_mode == "ffhs":
                logger.warning(f"第 {generation} 代: 经过去重与约束过滤后，无有效子代分子。")
            else:
                logger.warning(f"第 {generation} 代: 经过去重后，无有效子代分子。")
            # 创建一个空的对接文件和血统文件，避免后续步骤报错
            offspring_docked_file = gen_dir / "offspring_docked.smi"
            open(offspring_docked_file, 'a').close()
            self.last_offspring_histories = set()
            self.last_offspring_smiles = set()
            if offspring_lineage_file:
                self._write_jsonl(offspring_lineage_file, [])
                return str(offspring_docked_file), str(offspring_lineage_file)
            return str(offspring_docked_file), None

        if self.enable_lineage_tracking and offspring_lineage_file:
            unique_smiles = self._read_smiles_from_file(offspring_formatted_file)
            crossover_entries = self._read_jsonl(Path(crossover_lineage_file)) if crossover_lineage_file else []
            mutation_entries = self._read_jsonl(Path(mutation_lineage_file)) if mutation_lineage_file else []
            offspring_lineage_entries = self._combine_lineage_records(
                generation,
                unique_smiles,
                crossover_entries,
                mutation_entries
            )
            self._assign_histories_to_offspring(generation, offspring_lineage_entries)
            self._write_jsonl(offspring_lineage_file, offspring_lineage_entries)
            self._update_lineage_tracker(offspring_lineage_entries)

        logger.info(f"子代格式化完成: 共 {offspring_count} 个独特分子准备对接。")

        offspring_docked_file = gen_dir / "offspring_docked.smi"
        docking_succeeded, submitted_smiles = self._run_docking_and_record_oracle(
            offspring_formatted_file,
            offspring_docked_file,
            gen_dir,
            generation=generation,
            phase="offspring",
        )
        if self.oracle_tracking_enabled and not submitted_smiles:
            logger.warning(f"第 {generation} 代: Oracle 预算剩余额度为 0，本代跳过子代 docking。")
            open(offspring_docked_file, 'a').close()
            self.last_offspring_histories = set()
            self.last_offspring_smiles = set()
            return str(offspring_docked_file), str(offspring_lineage_file) if offspring_lineage_file else None

        if not docking_succeeded:
            logger.error(f"第 {generation} 代: 子代对接失败。")
            return None

        docked_count = self._count_molecules(str(offspring_docked_file))
        logger.info(f"第 {generation} 代: 子代评估完成，{docked_count} 个分子已对接。")

        offspring_mapping = self._ingest_population_metrics(offspring_docked_file, generation, mark_active=False)
        self.last_offspring_histories = set(offspring_mapping.values())
        self.last_offspring_smiles = set(offspring_mapping.keys())
        if not self.cleanup_intermediate_files:
            if not self._append_qed_sa_to_docked_file(offspring_docked_file):
                logger.error(f"第 {generation} 代: 子代QED/SA补全失败。")
                return None

        return str(offspring_docked_file), str(offspring_lineage_file) if offspring_lineage_file else None
    def _combine_lineage_records(
        self,
        generation: int,
        unique_smiles: List[str],
        crossover_entries: List[Dict],
        mutation_entries: List[Dict]
    ) -> List[Dict]:
        """合并交叉与突变的血统记录，并对齐到去重后的子代集合。"""
        lineage_map: Dict[str, List[Dict]] = {}
        for entry in crossover_entries + mutation_entries:
            child = entry.get("child")
            if not child:
                continue
            source_info = {
                "operation": entry.get("operation"),
                "parents": entry.get("parents", [])
            }
            if entry.get("operation") == "mutation":
                if "mutation_rule" in entry:
                    source_info["mutation_rule"] = entry["mutation_rule"]
                if "mutation_reaction_id" in entry:
                    source_info["mutation_reaction_id"] = entry["mutation_reaction_id"]
                if "complementary_molecules" in entry:
                    source_info["complementary_molecules"] = entry["complementary_molecules"]
            sources = lineage_map.setdefault(child, [])
            if source_info not in sources:
                sources.append(source_info)

        lineage_entries: List[Dict] = []
        for smi in unique_smiles:
            sources = lineage_map.get(smi)
            if not sources:
                continue
            lineage_entries.append({
                "generation": generation,
                "child": smi,
                "sources": sources
            })
        return lineage_entries
    def _save_next_generation_lineage(self, generation: int, next_parents_file: str, offspring_lineage_file: Optional[str]) -> None:
        """保存下一代父代的血统信息，便于追踪分子来源。"""
        if not self.enable_lineage_tracking:
            return
        if not next_parents_file:
            return
        gen_dir = self.output_dir / f"generation_{generation}"
        output_path = gen_dir / "next_generation_parents_lineage.jsonl"
        parents_smiles = self._read_smiles_from_file(Path(next_parents_file))
        offspring_entries = self._read_jsonl(Path(offspring_lineage_file)) if offspring_lineage_file else []
        offspring_map = {entry.get("child"): entry for entry in offspring_entries}

        records: List[Dict] = []
        for smi in parents_smiles:
            history = self.lineage_tracker.get(smi, [])
            latest_sources: List[Dict] = []
            origin = "unknown"
            if history:
                latest_event = history[-1]
                latest_sources = latest_event.get("sources", [])
                origin = "offspring" if latest_event.get("generation") == generation else "carryover"
            elif smi in offspring_map:
                latest_sources = offspring_map[smi].get("sources", [])
                origin = "offspring"
            records.append({
                "generation": generation,
                "child": smi,
                "origin": origin,
                "sources": latest_sources
            })

        self._write_jsonl(output_path, records)

    def run_selection(self, parent_docked_file: str, offspring_docked_file: str, generation: int) -> Optional[str]:
        """
        执行选择操作，从父代和子代中选出下一代。
        
        Args:
            parent_docked_file (str): 父代对接结果文件路径。
            offspring_docked_file (str): 子代对接结果文件路径。
            generation (int): 当前代数。
            
        Returns:
            Optional[str]: 成功则返回下一代父代文件路径,失败则返回None。
        """
        logger.info(f"第 {generation} 代: 开始选择操作...")
        next_parents_file = self.output_dir / f"generation_{generation+1}" / "initial_population_docked.smi"
        next_parents_file.parent.mkdir(exist_ok=True)

        selection_mode = self._get_selection_mode()
        n_select = self._get_selection_n_select(selection_mode)
        selection_args = [
            '--docked_file', offspring_docked_file,
            '--parent_file', parent_docked_file,
            '--output_file', str(next_parents_file),
            '--n_select', str(n_select),
            '--selection_mode', selection_mode,
            '--config_file', self.config_path,
            '--output_format', 'with_scores',
        ]
        if selection_mode == "ffhs":
            qed_min, sa_max = self._get_ffhs_constraints()
            selection_args.extend(['--qed_min', str(qed_min), '--sa_max', str(sa_max)])
        cache_path = getattr(self.metric_cache, "cache_path", None)
        if cache_path:
            selection_args.extend(['--cache_file', str(cache_path)])
        if self.cleanup_intermediate_files:
            selection_args.extend(['--disable_front_report', '--disable_cache'])
        selection_succeeded = self._run_script('selection.py', selection_args)

        if not selection_succeeded or self._count_molecules(str(next_parents_file)) == 0:
            logger.error(f"第 {generation} 代: 选择操作失败或未选出任何分子。")
            return None
        
        selected_count = self._count_molecules(str(next_parents_file))
        logger.info(f"选择操作完成 ({selection_mode.upper()}): 选出 {selected_count} 个分子作为下一代父代。")
        
        return str(next_parents_file)

    def run_selected_population_evaluation(self, selected_parents_file: str, generation: int) -> bool:
        """
        对选择后的精英种群（下一代父代）进行评分分析        
        Args:
            selected_parents_file (str): 选择后的下一代父代文件路径
            generation (int): 当前代数            
        Returns:
            bool: 评分分析是否成功
        """
        if self.cleanup_intermediate_files:
            return True

        logger.info(f"第 {generation} 代: 开始对选择后的精英种群进行评分分析")
        
        gen_dir = self.output_dir / f"generation_{generation}"
        scoring_report_file = gen_dir / f"generation_{generation}_evaluation.txt"

        scoring_args = [
            '--current_population_docked_file', str(selected_parents_file),
            '--initial_population_file', self.initial_population_file,
            '--output_file', str(scoring_report_file),
        ]
        if self._get_selection_mode() == "ffhs":
            qed_min, sa_max = self._get_ffhs_constraints()
            scoring_args.extend(['--qed_min', str(qed_min), '--sa_max', str(sa_max)])

        scoring_succeeded = self._run_script('utils/scoring.py', scoring_args)
        if scoring_succeeded:
            logger.info(f"第 {generation} 代: 精英种群评分分析完成，报告保存到 {scoring_report_file}")
        else:
            logger.warning(f"第 {generation} 代: 精英种群评分分析失败，但不影响主流程")            
        return scoring_succeeded

    def run_complete_workflow(self):
        """
        执行完整的SoftGA工作流。
        """
        import time
        from datetime import timedelta
        
        start_time = time.time()
        logger.info(f"开始执行完整的SoftGA工作流程 (输出目录: {self.output_dir})")
        
        # 第0步：初代种群处理
        current_parents_docked_file = self.run_initial_generation()
        if not current_parents_docked_file:
            logger.error("初代种群处理失败，工作流终止。")
            return False
        
        logger.info(f"初代种群处理成功，结果文件: {current_parents_docked_file}")

        # 开始迭代
        for generation in range(1, self.max_generations + 1):
            if self._oracle_budget_reached():
                logger.info(
                    "Oracle 预算已达标 (%s/%s)，停止后续代。",
                    self.oracle_success_calls,
                    self.max_oracle_calls,
                )
                break
            next_parents_file = self.run_generation_step(generation, current_parents_docked_file)
            if not next_parents_file:
                logger.error(f"第 {generation} 代处理失败，工作流终止。")
                return False
            current_parents_docked_file = next_parents_file

            if self._oracle_budget_reached():
                logger.info(
                    "第 %s 代完成后 Oracle 预算达标 (%s/%s)，提前收敛。",
                    generation,
                    self.oracle_success_calls,
                    self.max_oracle_calls,
                )
                break
        
        end_time = time.time()
        total_duration = str(timedelta(seconds=int(end_time - start_time)))
        
        budget_unmet = self.oracle_tracking_enabled and not self._oracle_budget_reached()
        if budget_unmet:
            logger.error(
                "Oracle 预算未达标: success_calls=%s < max_oracle_calls=%s",
                self.oracle_success_calls,
                self.max_oracle_calls,
            )
        logger.info("=" * 60)
        logger.info("SoftGA工作流程全部完成!")
        logger.info(f"最终优化种群保存在: {current_parents_docked_file}")
        logger.info(f"本次优化总耗时: {total_duration}")
        if self.oracle_tracking_enabled:
            logger.info(
                "Oracle 计数: success_calls=%s, submit_calls=%s, max_oracle_calls=%s",
                self.oracle_success_calls,
                self.oracle_submit_calls,
                self.max_oracle_calls,
            )
        logger.info("=" * 60)
        self._export_evomo_files()
        if self.plot_top1 and not self.cleanup_intermediate_files:
            ok = self._run_script(
                "utils/plot_top1_docking_curve.py",
                ["--run_dir", str(self.output_dir)],
            )
            if not ok:
                logger.warning("Top1 曲线绘图失败，但不影响主流程")
        if self.cleanup_intermediate_files:
            self._prune_output_to_pop_only()

        return not budget_unmet

    def run_initial_generation(self) -> Optional[str]:
        """
        执行初代种群的处理，包括去重、格式化和对接。
        
        Returns:
            Optional[str]: 成功则返回初代种群对接结果文件的路径，失败则返回None。
        """
        logger.info("开始处理初代种群 (Generation 0)...")
        
        gen_dir = self.output_dir / "generation_0"
        gen_dir.mkdir(exist_ok=True)
        
        # 1. 检查初始种群文件是否存在
        if not Path(self.initial_population_file).exists():
            logger.error(f"初始种群文件未找到: {self.initial_population_file}")
            return None
        
        # 2. 去重并格式化初始种群
        initial_formatted_file = gen_dir / "initial_population_formatted.smi"
        unique_count = self._remove_duplicates_from_smiles_file(
            self.initial_population_file, 
            str(initial_formatted_file)
        )
        if unique_count == 0:
            logger.error("初始种群文件为空或处理失败。")
            return None

        # 记录初代种群血统信息
        self._record_initial_population(initial_formatted_file)
        
        # 3. 对初代种群进行对接
        initial_docked_file = gen_dir / "initial_population_docked.smi"
        docking_succeeded, submitted_smiles = self._run_docking_and_record_oracle(
            initial_formatted_file,
            initial_docked_file,
            gen_dir,
            generation=0,
            phase="initial",
        )
        if self.oracle_tracking_enabled and not submitted_smiles:
            logger.error("初代种群在 Oracle 预算限制下没有可对接分子。")
            return None
        docked_count = self._count_molecules(str(initial_docked_file))
        if not docking_succeeded or docked_count == 0:
            logger.error("初代种群对接失败或未生成任何有效对接结果。")
            return None
        mapping = self._ingest_population_metrics(initial_docked_file, generation=0, mark_active=True)
        self._mark_histories_active(set(mapping.values()), generation=0)
        if not self.cleanup_intermediate_files:
            if not self._append_qed_sa_to_docked_file(initial_docked_file):
                logger.error("初代种群QED/SA补全失败，工作流终止。")
                return None
            ranked_file = initial_docked_file.with_name("initial_population_ranked.smi")
            if not self._write_ranked_by_docking(initial_docked_file, ranked_file):
                logger.warning(f"初代种群: initial_population_ranked.smi 写入失败: {ranked_file}")
        
        logger.info(f"初代种群对接完成: {docked_count} 个分子已评分。")
        return str(initial_docked_file)

    def run_softbd_process(self, parent_smiles_file: str, generation: int) -> Optional[str]:
        """
        运行 SoftBD 生成流程。
        
        Args:
            parent_smiles_file (str): 父代 SMILES 文件路径。
            generation (int): 当前代数。
            
        Returns:
            Optional[str]: 生成分子结果文件路径。
        """
        logger.info(f"第 {generation} 代: 开始 SoftBD 生成流程...")
        gen_dir = self.output_dir / f"generation_{generation}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        softbd_cfg = self.config.get('softbd') or {}

        if self.softbd_sampler is None:
            from generation import SoftBDSampler
            self.softbd_sampler = SoftBDSampler(softbd_cfg, seed=int(self.softbd_seed), log_dir=gen_dir / 'softbd_logs')

        gen_params = softbd_cfg.get('generation_params', {})
        strategy_progress = self._get_softbd_strategy_progress(generation)
        output_path = self.softbd_sampler.generate(
            parent_file=parent_smiles_file,
            generation=int(generation),
            output_dir=str(gen_dir),
            seed=int(self.softbd_seed),
            initial_samples=int(gen_params.get('initial_samples', 100)),
            max_generations=int(self.max_generations),
            initial_population_file=str(self.initial_population_file) if self.initial_population_file else None,
            strategy_progress=strategy_progress,
        )
        if not output_path:
            return None

        output_file = Path(output_path)
        if self._count_molecules(str(output_file)) == 0:
            return None

        generated_count = self._count_molecules(str(output_file))
        logger.info(f"第 {generation} 代: SoftBD 生成完成，产出 {generated_count} 个新分子。")
        
        # 注册历史记录 (兼容原有血统追踪)
        softbd_smiles = self._read_smiles_from_file(output_file)
        placeholder_root = self._create_generation_placeholder_root(generation)
        for smi in softbd_smiles:
            if smi in self.smiles_to_history:
                continue
            # 使用 SOFTBD 前缀区分
            op_token = f"SOFTBD-{generation}-{self._short_hash(smi)}"
            history = f"{placeholder_root}|{op_token}"
            self._register_history(smi, history)
            self._upsert_history_record(history, smi, generation, None, mark_active=False)
            
        return str(output_file)

    def run_generation_step(self, generation: int, current_parents_docked_file: str):
        """
        执行单代SoftGA的完整流程。
        """
        logger.info(f"========== 开始第 {generation} 代进化 ==========")
        gen_dir = self.output_dir / f"generation_{generation}"
        gen_dir.mkdir(exist_ok=True)
        # 1. 从父代对接文件中提取纯SMILES
        parent_smiles_file = gen_dir / "current_parent_smiles.smi"
        if not self._extract_smiles_from_docked_file(current_parents_docked_file, str(parent_smiles_file)):
            logger.error(f"第{generation}代: 无法从父代文件提取SMILES,工作流终止")
            return None

        # 2. SoftBD 生成
        softbd_generated_file = self.run_softbd_process(str(parent_smiles_file), generation)

        # 3. 遗传操作
        ga_children_files = self.run_ga_operations(str(parent_smiles_file), softbd_generated_file, generation)
        if not ga_children_files:
            logger.error(f"第{generation}代: 遗传操作失败，工作流终止。")
            return None
        
        crossover_file, mutation_file, crossover_lineage, mutation_lineage = ga_children_files

        # 4. 子代评估（对接，同时生成血统记录）
        offspring_result = self.run_offspring_evaluation(
            crossover_file,
            mutation_file,
            generation,
            crossover_lineage,
            mutation_lineage
        )
        if not offspring_result:
            logger.error(f"第{generation}代: 子代评估失败，工作流终止。")
            return None
        offspring_docked_file, offspring_lineage_file = offspring_result

        # 5. 选择
        next_parents_docked_file = self.run_selection(
            current_parents_docked_file, 
            offspring_docked_file, 
            generation
        )
        if not next_parents_docked_file:
            logger.error(f"第{generation}代: 选择操作失败，工作流终止。")
            return None

        selected_mapping = self._ingest_population_metrics(next_parents_docked_file, generation, mark_active=True)
        if not self.cleanup_intermediate_files:
            if not self._append_qed_sa_to_docked_file(Path(next_parents_docked_file)):
                logger.error(f"第 {generation} 代: 父代QED/SA补全失败，工作流终止。")
                return None
            ranked_file = Path(next_parents_docked_file).with_name("initial_population_ranked.smi")
            if not self._write_ranked_by_docking(Path(next_parents_docked_file), ranked_file):
                logger.warning(f"第 {generation} 代: initial_population_ranked.smi 写入失败: {ranked_file}")
        new_active_histories = set(selected_mapping.values())
        candidate_histories = set(self.current_active_histories)
        candidate_histories.update(self.last_offspring_histories)
        removed_histories = candidate_histories - new_active_histories
        if removed_histories:
            self._mark_histories_removed(removed_histories, generation)
        self._mark_histories_active(new_active_histories, generation)
        self.last_offspring_histories = set()
        self.last_offspring_smiles = set()

        
        # 保存下一代父代的血统信息，便于追踪
        if self.enable_lineage_tracking:
            self._save_next_generation_lineage(
                generation,
                next_parents_docked_file,
                offspring_lineage_file
            )

        # 6. 对选择后的精英种群进行评分分析（cleanup 模式跳过）
        if not self.cleanup_intermediate_files:
            self.run_selected_population_evaluation(next_parents_docked_file, generation)

        # 7. 清理临时文件（如果启用）
        self._cleanup_generation_files(generation)

        logger.info(f"========== 第 {generation} 代进化完成 ==========")
        return next_parents_docked_file

    def _cleanup_generation_files(self, generation_num: int):
        """
        cleanup 模式下按代清理：删除整代目录，降低峰值 IO/存储占用。
        
        Args:
            generation_num (int): 要清理的代数
        """
        if not self.cleanup_intermediate_files:
            return

        to_delete = [self.output_dir / f"generation_{generation_num}"]
        if generation_num == 1:
            to_delete.append(self.output_dir / "generation_0")

        for gen_dir in to_delete:
            try:
                if gen_dir.exists():
                    shutil.rmtree(gen_dir)
                    logger.info(f"已清理目录: {gen_dir}")
            except Exception as e:
                logger.warning(f"清理目录失败 {gen_dir}: {e}")

    def _prune_output_to_pop_only(self) -> None:
        """cleanup 模式最终裁剪：保留 pop.csv 与 oracle_calls*.csv。"""
        if not self.cleanup_intermediate_files:
            return

        pop_file = self.output_dir / "pop.csv"
        if not pop_file.exists():
            logger.warning("cleanup 模式最终裁剪跳过：未找到 pop.csv")
            return

        try:
            keep_files = {pop_file.name}
            if self.oracle_tracking_enabled:
                keep_files.update({ORACLE_SUCCESS_FILENAME, ORACLE_ALL_FILENAME})
            for item in self.output_dir.iterdir():
                if item.is_file() and item.name in keep_files:
                    continue
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            logger.info("cleanup 模式最终裁剪完成，保留: %s", ", ".join(sorted(keep_files)))
        except Exception as e:
            logger.warning(f"cleanup 模式最终裁剪失败: {e}")

# --- 主函数入口 ---
def main():
    """主函数，用于解析命令行参数和启动工作流"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SoftGA工作流执行器')
    parser.add_argument('--config', type=str, 
                       default=DEFAULT_CONFIG,
                       help='配置文件路径')
    parser.add_argument('--receptor', type=str, required=True,
                       help='要运行的目标受体名称（如 parp1 或 6GL8）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='(可选) 指定输出目录，覆盖配置文件中的设置')
    parser.add_argument('--initial_population_file', type=str, default=None,
                       help='(可选) 覆盖 workflow.initial_population_file')
    parser.add_argument('--pair_min_tanimoto', type=float, default=None,
                       help='(可选) 覆盖 crossover.pair_min_tanimoto')
    parser.add_argument('--pair_max_tanimoto', type=float, default=None,
                       help='(可选) 覆盖 crossover.pair_max_tanimoto')
    parser.add_argument('--qed_min', type=float, default=None,
                       help='(可选) 覆盖 selection.ffhs_settings.constraints.qed_min')
    parser.add_argument('--sa_max', type=float, default=None,
                       help='(可选) 覆盖 selection.ffhs_settings.constraints.sa_max')
    parser.add_argument(
        '--softbd_enable',
        nargs='?',
        const='true',
        default=None,
        help='(可选) 覆盖 softbd.enable (true/false)',
    )
    parser.add_argument('--docking_exhaustiveness', type=int, default=None,
                       help='(可选) 覆盖 docking.exhaustiveness')
    parser.add_argument('--max_oracle_calls', type=int, default=None,
                       help='(可选) Oracle 预算（仅成功 docking 计数）')
    parser.add_argument(
        '--enable_crowding_distance',
        nargs='?',
        const='true',
        default=None,
        help='(可选) 覆盖 selection.*_settings.enable_crowding_distance (true/false)',
    )
    
    args = parser.parse_args()
    
    try:
        executor = SoftGAWorkflowExecutor(
            args.config,
            args.receptor,
            args.output_dir,
            initial_population_file_override=args.initial_population_file,
            pair_min_tanimoto_override=args.pair_min_tanimoto,
            pair_max_tanimoto_override=args.pair_max_tanimoto,
            qed_min_override=args.qed_min,
            sa_max_override=args.sa_max,
            softbd_enable_override=(
                str(args.softbd_enable).strip().lower() in ("1", "true", "yes", "y", "on")
                if args.softbd_enable is not None
                else None
            ),
            docking_exhaustiveness_override=args.docking_exhaustiveness,
            enable_crowding_distance_override=(
                str(args.enable_crowding_distance).strip().lower() in ("1", "true", "yes", "y", "on")
                if args.enable_crowding_distance is not None
                else None
            ),
            max_oracle_calls_override=args.max_oracle_calls,
        )
        success = executor.run_complete_workflow()
        if not success:
            logger.error("SoftGA工作流执行失败。")
            return 1
        
        logger.info("SoftGA工作流成功完成!")

    except Exception as e:
        logger.critical(f"工作流执行过程中发生严重错误: {e}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
