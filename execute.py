""" SoftGA hybrid workflow execution script
==========================
1. Population initialization and evaluation
2. Generate new molecules based on SoftBD
3. Perform genetic algorithm operations (crossover, mutation) on the parent and generated molecules
4. Evaluate the newly generated offspring
5. Screen out the next generation population through FFHS selection strategy
6. Perform score analysis on the elite population and export the results
7. Continue iterating """
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
from utils.config_snapshot import save_config_snapshot #Save parameters (snapshot)
from utils.config_loader import load_config, resolve_config_path
import multiprocessing  
import shutil  
from rdkit import Chem
from utils.chem_metrics import ChemMetricCache

# Remove global log configuration to avoid multi-process log conflicts
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Make sure the logger has basic handlers but does not conflict with other processes
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


class SoftGAWorkflowExecutor:    #Workflow; the main function/entry file is calling this class
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
        """         Initialize the SoftGA workflow executor.        
        Args:
            config_path (str): Configuration file path.
            receptor_name (str): Target receptor name (required).
            output_dir_override (Optional[str]): Override the output directory in the configuration file.
            num_processors_override (Optional[int]): Override the number of processors in the configuration file.
            initial_population_file_override (Optional[str]): Override workflow.initial_population_file.
            strategy_mode_override (Optional[str]): Override softbd strategy mode.
            max_generations_override (Optional[int]): Maximum number of generations to override.
            seed_override (Optional[int]): Override the random seed.
            selection_mode_override (Optional[str]): Override selection mode.
            qed_min_override (Optional[float]): Override the QED lower bound for FFHS constraints.
            sa_max_override (Optional[float]): Override the SA upper limit of FFHS constraints.
            samples_per_parent_override (Optional[int]): Number of samples per parent to override.
            tanimoto_threshold_override (Optional[float]): Override the tanimoto threshold.
            min_keep_ratio_override (Optional[float]): Override the minimum keep ratio.
            max_keep_ratio_override (Optional[float]): Override the maximum retention ratio.
            block_size_override (Optional[int]): Override block_size.
            length_override (Optional[int]): Override length.
            temperature_override (Optional[float]): Override temperature.
            nucleus_p_override (Optional[float]): Override nucleus_p.
            gpu_max_batch_size_override (Optional[int]): Override gpu_max_batch_size.
            initial_samples_override (Optional[int]): Override initial_samples.
            gen1_selection_mode_override (Optional[str]): Override Gen1 selection mode (maxmin | random).
            number_of_crossovers_override (Optional[int]): Override crossover.number_of_crossovers.
            pair_min_tanimoto_override (Optional[float]): Override crossover.pair_min_tanimoto.
            pair_max_tanimoto_override (Optional[float]): Override crossover.pair_max_tanimoto.
            number_of_mutants_override (Optional[int]): Override mutation.number_of_mutants.
            enable_crowding_distance_override (Optional[bool]): Overrides selection.*_settings.enable_crowding_distance.         """
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
        # Only the coverage parameters explicitly passed in by the user are recorded to facilitate snapshot replication experiments.
        self._cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None and v != ""}

        self.config_path = config_path
        self.softbd_sampler = softbd_sampler
        self.config = self._load_config()        
        if cleanup_intermediate_files_override is not None:
            if 'performance' not in self.config:
                self.config['performance'] = {}
            self.config['performance']['cleanup_intermediate_files'] = bool(cleanup_intermediate_files_override)
            logger.info(
                "Runtime override cleanup_intermediate_files is: %s",
                bool(cleanup_intermediate_files_override),
            )
        # Application processor number coverage
        if num_processors_override is not None:
            if 'performance' not in self.config:
                self.config['performance'] = {}
            self.config['performance']['number_of_processors'] = num_processors_override
            logger.info(f"The number of processors covered during runtime is: {num_processors_override}")

        if initial_population_file_override is not None:
            if 'workflow' not in self.config:
                self.config['workflow'] = {}
            self.config['workflow']['initial_population_file'] = str(initial_population_file_override)
            logger.info(f"Override initial_population_file at runtime to: {initial_population_file_override}")
            
        # Apply SoftBD Policy Mode Override
        if strategy_mode_override:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'dynamic_strategy' not in self.config['softbd']:
                self.config['softbd']['dynamic_strategy'] = {}
            self.config['softbd']['dynamic_strategy']['strategy_mode'] = strategy_mode_override
            logger.info(f"The runtime override SoftBD policy mode is: {strategy_mode_override}")

        # Apply samples per parent coverage
        if samples_per_parent_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'generation_params' not in self.config['softbd']:
                self.config['softbd']['generation_params'] = {}
            self.config['softbd']['generation_params']['samples_per_parent'] = samples_per_parent_override
            logger.info(f"Runtime override samples_per_parent to: {samples_per_parent_override}")

        # Apply tanimoto threshold override
        if tanimoto_threshold_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'generation_params' not in self.config['softbd']:
                self.config['softbd']['generation_params'] = {}
            self.config['softbd']['generation_params']['tanimoto_threshold'] = tanimoto_threshold_override
            logger.info(f"Runtime override tanimoto_threshold is: {tanimoto_threshold_override}")

        # Apply minimum retention ratio coverage
        if min_keep_ratio_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'dynamic_strategy' not in self.config['softbd']:
                self.config['softbd']['dynamic_strategy'] = {}
            self.config['softbd']['dynamic_strategy']['min_keep_ratio'] = min_keep_ratio_override
            logger.info(f"Runtime override min_keep_ratio is: {min_keep_ratio_override}")

        # Apply maximum retention ratio coverage
        if max_keep_ratio_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'dynamic_strategy' not in self.config['softbd']:
                self.config['softbd']['dynamic_strategy'] = {}
            self.config['softbd']['dynamic_strategy']['max_keep_ratio'] = max_keep_ratio_override
            logger.info(f"Runtime override max_keep_ratio is: {max_keep_ratio_override}")

        if recircle_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            self.config['softbd']['recircle'] = bool(recircle_override)
            logger.info(f"Runtime override recircle is: {bool(recircle_override)}")

        if softbd_enable_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            self.config['softbd']['enable'] = bool(softbd_enable_override)
            logger.info(f"Runtime override softbd.enable is: {bool(softbd_enable_override)}")

        if softbd_seed_mode_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            self.config['softbd']['seed_mode'] = str(softbd_seed_mode_override)
            logger.info(f"Runtime override softbd.seed_mode to: {softbd_seed_mode_override}")
        
        # Apply SoftBD GPU overlay
        if gpu_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            self.config['softbd']['gpu'] = str(gpu_override)
            logger.info(f"Runtime override for SoftBD gpu is: {gpu_override}")
        
        # Apply block_size override
        if block_size_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            self.config['softbd']['block_size'] = block_size_override
            logger.info(f"Runtime override block_size is: {block_size_override}")

        # Apply length override
        if length_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            self.config['softbd']['length'] = length_override
            logger.info(f"Runtime override length is: {length_override}")

        # Apply temperature override
        if temperature_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            self.config['softbd']['temperature'] = temperature_override
            logger.info(f"The runtime override temperature is: {temperature_override}")
        
        # Apply nucleus_p override
        if nucleus_p_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'generation_params' not in self.config['softbd']:
                self.config['softbd']['generation_params'] = {}
            self.config['softbd']['generation_params']['nucleus_p'] = nucleus_p_override
            logger.info(f"The runtime override of nucleus_p is: {nucleus_p_override}")

        # Apply gpu_max_batch_size override
        if gpu_max_batch_size_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'generation_params' not in self.config['softbd']:
                self.config['softbd']['generation_params'] = {}
            self.config['softbd']['generation_params']['gpu_max_batch_size'] = gpu_max_batch_size_override
            logger.info(f"Runtime override gpu_max_batch_size is: {gpu_max_batch_size_override}")

        # Apply steps override
        if steps_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'generation_params' not in self.config['softbd']:
                self.config['softbd']['generation_params'] = {}
            self.config['softbd']['generation_params']['steps'] = int(steps_override)
            logger.info(f"The runtime override steps is: {steps_override}")
        
        # Apply initial_samples override
        if initial_samples_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'generation_params' not in self.config['softbd']:
                self.config['softbd']['generation_params'] = {}
            self.config['softbd']['generation_params']['initial_samples'] = initial_samples_override
            logger.info(f"Runtime override initial_samples is: {initial_samples_override}")

        # Apply gen1_n_select override
        if gen1_n_select_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'generation_params' not in self.config['softbd']:
                self.config['softbd']['generation_params'] = {}
            self.config['softbd']['generation_params']['gen1_n_select'] = int(gen1_n_select_override)
            logger.info(f"Runtime override gen1_n_select is: {gen1_n_select_override}")

        if gen1_selection_mode_override is not None:
            if 'softbd' not in self.config:
                self.config['softbd'] = {}
            if 'generation_params' not in self.config['softbd']:
                self.config['softbd']['generation_params'] = {}
            self.config['softbd']['generation_params']['gen1_selection_mode'] = str(gen1_selection_mode_override)
            logger.info(f"Runtime override gen1_selection_mode to: {gen1_selection_mode_override}")

        if number_of_crossovers_override is not None:
            if 'crossover' not in self.config:
                self.config['crossover'] = {}
            self.config['crossover']['number_of_crossovers'] = int(number_of_crossovers_override)
            logger.info(f"Runtime override number_of_crossovers is: {number_of_crossovers_override}")

        if pair_min_tanimoto_override is not None:
            if 'crossover' not in self.config:
                self.config['crossover'] = {}
            self.config['crossover']['pair_min_tanimoto'] = float(pair_min_tanimoto_override)
            logger.info(f"Runtime override pair_min_tanimoto is: {pair_min_tanimoto_override}")

        if pair_max_tanimoto_override is not None:
            if 'crossover' not in self.config:
                self.config['crossover'] = {}
            self.config['crossover']['pair_max_tanimoto'] = float(pair_max_tanimoto_override)
            logger.info(f"Runtime override pair_max_tanimoto is: {pair_max_tanimoto_override}")

        if number_of_mutants_override is not None:
            if 'mutation' not in self.config:
                self.config['mutation'] = {}
            self.config['mutation']['number_of_mutants'] = int(number_of_mutants_override)
            logger.info(f"Runtime override number_of_mutants is: {number_of_mutants_override}")



        # Apply maximum algebraic coverage
        if max_generations_override is not None:
            if 'workflow' not in self.config:
                self.config['workflow'] = {}
            self.config['workflow']['max_generations'] = max_generations_override
            logger.info(f"The maximum number of generations covered during runtime is: {max_generations_override}")

        # Apply random seed override
        if seed_override is not None:
            if 'workflow' not in self.config:
                self.config['workflow'] = {}
            self.config['workflow']['seed'] = seed_override
            
            # Also update seeds in softbd and docking for consistency, if they exist
            if 'softbd' in self.config:
                # softbd config usually doesn't have direct seed at root but we can set it if needed or rely on workflow seed
                # softbd_generate.py reads from workflow seed or passed arg.
                pass 
            if 'docking' in self.config:
                self.config['docking']['seed'] = seed_override
            
            logger.info(f"The runtime coverage random seed is: {seed_override}")

        # Apply selection mode override
        if selection_mode_override:
            if 'selection' not in self.config:
                self.config['selection'] = {}
            self.config['selection']['selection_mode'] = selection_mode_override
            logger.info(f"The runtime coverage selection mode is: {selection_mode_override}")

        if enable_crowding_distance_override is not None:
            selection_cfg = self.config.setdefault('selection', {})
            enable_crowding = bool(enable_crowding_distance_override)
            selection_cfg.setdefault('ffhs_settings', {})['enable_crowding_distance'] = enable_crowding
            selection_cfg.setdefault('nsgaii_settings', {})['enable_crowding_distance'] = enable_crowding
            logger.info(f"Runtime override enable_crowding_distance is: {enable_crowding}")

        if qed_min_override is not None or sa_max_override is not None:
            selection_cfg = self.config.setdefault('selection', {})
            ffhs_cfg = selection_cfg.setdefault('ffhs_settings', {})
            constraints = ffhs_cfg.setdefault('constraints', {})
            if qed_min_override is not None:
                constraints['qed_min'] = float(qed_min_override)
                logger.info(f"Runtime override constraints.qed_min is: {qed_min_override}")
            if sa_max_override is not None:
                constraints['sa_max'] = float(sa_max_override)
                logger.info(f"Runtime override constraints.sa_max is: {sa_max_override}")

        if docking_tool_override is not None:
            if 'docking' not in self.config:
                self.config['docking'] = {}
            self.config['docking']['tool'] = str(docking_tool_override)
            logger.info(f"Runtime override docking.tool is: {docking_tool_override}")
        if docking_exhaustiveness_override is not None:
            if 'docking' not in self.config:
                self.config['docking'] = {}
            self.config['docking']['exhaustiveness'] = int(docking_exhaustiveness_override)
            logger.info(f"Runtime override docking.exhaustiveness is: {docking_exhaustiveness_override}")

        self.run_params = {}
        self._setup_parameters_and_paths(receptor_name, output_dir_override)
        self.max_oracle_calls: Optional[int] = (
            int(max_oracle_calls_override) if max_oracle_calls_override is not None else None
        )
        if self.max_oracle_calls is not None and self.max_oracle_calls <= 0:
            raise ValueError("max_oracle_calls must be a positive integer")
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
                "Oracle budget mode is enabled: max_oracle_calls=%s, success_csv=%s, all_csv=%s",
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
        logger.info(f"SoftGAWorkflow initialization is completed, output directory: {self.output_dir}")
        logger.info(f"Maximum number of iterations: {self.max_generations}")

    def _load_config(self) -> dict:#Load configuration file
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
            "Generation %s: SoftBD prefix strategy changed to use Oracle progress %.3f (%s/%s)",
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
            logger.info("Generation %s (%s): Oracle budget exhausted, skipping docking.", generation, phase)
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
            "Generation %s (%s): Oracle budget truncation %s -> %s (remaining=%s)",
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
        # Record configuration and root directory
        self.run_params['config_file_path'] = self.config_path
        self.run_params['project_root'] = str(self.project_root)
        # Determine the output directory
        if output_dir_override:
            output_dir_name = output_dir_override
        else:
            output_dir_name = workflow_config.get('output_directory', 'SoftGA_output')
        base_output_dir = self.project_root / output_dir_name
        self.run_params['base_output_dir'] = str(base_output_dir)
        self.receptor_name = str(receptor_name).strip()
        if not self.receptor_name:
            raise ValueError("receptor_name cannot be empty, please explicitly pass in --receptor")
        self.output_dir = base_output_dir / self.receptor_name
        self.run_params['receptor_name'] = self.receptor_name
        self.run_params['run_specific_output_dir'] = str(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Load workflow core parameters
        self.max_generations = workflow_config.get('max_generations', 10)
        self.initial_population_file = workflow_config.get('initial_population_file')
        self.run_params['max_generations'] = self.max_generations
        if self.initial_population_file:
            p = Path(self.initial_population_file)
            if not p.is_absolute():
                p = (self.project_root / p).resolve()
            self.initial_population_file = str(p)
        self.run_params['initial_population_file'] = self.initial_population_file
        # Record selection mode
        self.run_params['selection_mode'] = self._get_selection_mode()

    @staticmethod
    def _normalize_selection_mode(mode: Optional[str]) -> str:
        return "nsgaii" if str(mode or "").strip().lower() == "nsgaii" else "ffhs"

    def _get_selection_mode(self) -> str:
        selection_config = self.config.get("selection", {})
        raw_mode = str(selection_config.get("selection_mode", "ffhs")).strip().lower()
        mode = self._normalize_selection_mode(raw_mode)
        if raw_mode not in {"ffhs", "nsgaii"}:
            logger.warning(f"selection.selection_mode Illegal: {raw_mode}, fallback to ffhs")
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
            logger.warning(f"softbd.seed_mode Illegal: {seed_mode_raw}, fall back to workflow")
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
            f"SoftBD seed Determined: mode={self.softbd_seed_mode},"
            f"random_seed_bits={self.softbd_random_seed_bits}, seed={self.softbd_seed}"
        )

    def _save_run_parameters(self):
        """Save a complete parameter snapshot of this run."""
        snapshot_file_path = self.output_dir / "execution_config_snapshot.json"
        success = save_config_snapshot(
            original_config=self.config,
            execution_context=self.run_params,
            output_file_path=str(snapshot_file_path)
        )
        if success:
            logger.info(f"The complete execution configuration snapshot has been saved to: {snapshot_file_path}")
        else:
            logger.error("Failed to save execution configuration snapshot")
    def _load_lineage_tracker(self) -> Dict[str, List[Dict]]:
        """Load existing lineage records from disk."""
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
                logger.warning("The format of the lineage tracking file is abnormal and the original records have been ignored.")
            except Exception as exc:
                logger.warning(f"Unable to load lineage tracking file {path}: {exc}")
        return {}
    def _save_lineage_tracker(self) -> None:
        """Persist lineage tracking records to disk."""
        if not getattr(self, "enable_lineage_tracking", False):
            return
        if not self.lineage_tracker_path:
            return
        try:
            with open(self.lineage_tracker_path, 'w', encoding='utf-8') as f:
                json.dump(self.lineage_tracker, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.error(f"Failed to save lineage tracking file: {exc}")
    def _write_jsonl(self, output_path: Path, entries: List[Dict]) -> None:
        """Write records to a JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in entries:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    def _read_jsonl(self, input_path: Path) -> List[Dict]:
        """Reads a JSONL file and returns a list of dictionaries."""
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
                        logger.warning(f"Unable to parse ancestry record: {line}")
        except Exception as exc:
            logger.warning(f"Failed to read lineage file {input_path}: {exc}")
        return entries
    def _read_smiles_from_file(self, file_path: Path, first_column_only: bool = True) -> List[str]:
        """Read SMILES files, only the first column is returned by default."""
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
            logger.warning(f"An error occurred while reading file {file_path}: {exc}")
        return smiles

    def _update_lineage_tracker(self, lineage_entries: List[Dict]) -> None:
        """Update lineage tracking data in memory and sync to disk."""
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
        """Record the ancestry of the first generation population."""
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
            logger.info(f"The first generation population pedigree record has been saved to: {lineage_file}")
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
        """Complete QED/SA columns for docked files (overwrite in place)."""
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
        """         Sort the docked files in ascending order by docking_score (column 2) and write to the new file.
        - Keep the entire row (including columns such as QED/SA) and only change the row order
        - Lines that cannot parse docking_score will be queued to the end         """
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
                # Stable sorting: docking -> original order
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
        """         Unified script execution functions, preventing deadlocks by streaming output, and adding timeout protection.
        
        Args:
            script_path (str): The script path relative to the project root directory.
            args (List[str]): List of command line arguments for the script.
            
        Returns:
            bool: Whether the script was executed successfully.         """
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
                
                # Create a queue to receive output from the thread
                q_stdout = queue.Queue()
                q_stderr = queue.Queue()

                # Create and start threads to read output in real time
                thread_stdout = threading.Thread(target=self._read_stream, args=(process.stdout, q_stdout))
                thread_stderr = threading.Thread(target=self._read_stream, args=(process.stderr, q_stderr))
                thread_stdout.start()
                thread_stderr.start()

                # Wait for the process to end and set a timeout
                try:
                    process.wait(timeout=3600)  # 1 hour timeout
                except subprocess.TimeoutExpired:
                    logger.error(f"Script {script_path} timed out (1 hour). Terminating...")
                    process.kill()  # Force kill process
                    # Wait a short period of time to ensure that the thread can read the last information
                    thread_stdout.join(timeout=5)
                    thread_stderr.join(timeout=5)
                    # Log and return failure
                    self._log_subprocess_output(script_path, q_stdout, q_stderr, "after timeout")
                    return False
                
                # After the process ends normally, wait for the reading thread to complete
                thread_stdout.join()
                thread_stderr.join()

                # Collect and log output
                stdout_str, stderr_str = self._log_subprocess_output(script_path, q_stdout, q_stderr, "final")

                if process.returncode == 0:
                    logger.info(f"Script {script_path} executed successfully.")
                    return True
                else:
                    logger.error(f"Script {script_path} failed with return code {process.returncode}.")
                    # On failure, log stdout even if there is no stderr, may contain clues
                    if stderr_str:
                        logger.error(f"Error output (stderr):\n{stderr_str}")
                    if stdout_str:
                        logger.error(f"Standard output (stdout):\n{stdout_str}")
                    return False
                    
        except Exception as e:
            logger.error(f"An exception occurred while trying to run script {script_path}: {e}", exc_info=True)
            return False

    def _read_stream(self, stream, q: queue.Queue):
        """Read the stream (stdout/stderr) in real time and put it into the queue"""
        try:
            for line in iter(stream.readline, ''):
                q.put(line)
        finally:
            stream.close()

    def _log_subprocess_output(self, script_path: str, q_stdout: queue.Queue, q_stderr: queue.Queue, context: str) -> Tuple[str, str]:
        """Collect and log the output of child processes from the queue"""
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
        """Count the number of molecules in a SMILES file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                count = sum(1 for line in f if line.strip())
            return count
        except FileNotFoundError:
            return 0
    def _remove_duplicates_from_smiles_file(self, input_file: str, output_file: str) -> int:
        """         Remove duplicate molecules in SMILES files and add unique IDs to each molecule.
        Output format: SMILES ligand_id_X
        Add file lock protection to avoid concurrent access conflicts.         """
        import time
        import random
        
        # Add random delays to avoid multiple processes accessing files at the same time
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
            
            # Use temporary files to write and then rename atomically to avoid write conflicts
            temp_output_file = output_file + f".tmp_{os.getpid()}_{int(time.time())}"
            with open(temp_output_file, 'w', encoding='utf-8') as f:
                for i, smiles in enumerate(unique_smiles_list):
                    f.write(f"{smiles}\tligand_id_{i}\n")
            
            # Atomic renaming
            import shutil
            shutil.move(temp_output_file, output_file)
            
            logger.info(f"Deduplication completed: {len(unique_smiles_list)} unique molecules saved to {output_file}")
            return len(unique_smiles_list)
        except Exception as e:
            logger.error(f"An error occurred during deduplication: {e}")
            # Clean possible temporary files
            temp_file = output_file + f".tmp_{os.getpid()}_{int(time.time())}"
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            return 0

    def _extract_smiles_from_docked_file(self, docked_file: str, output_smiles_file: str) -> bool:
        """Extract pure SMILES from files with docking scores for genetic manipulation or decomposition"""
        try:
            with open(docked_file, 'r') as infile, open(output_smiles_file, 'w') as outfile:
                for line in infile:
                    line = line.strip()
                    if line:
                        smiles = line.split()[0]
                        outfile.write(f"{smiles}\n")
            return True
        except Exception as e:
            logger.error(f"Error extracting SMILES from {docked_file}: {e}")
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
        """Auxiliary function used to run a GA stage (such as crossover) and its subsequent filtering, and return (whether successful, filtered lineage file path)."""
        logger.info(f"Start executing {ga_op_name}...")
        
        # Run GA operations
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
            logger.error(f"'{ga_op_name}' Script execution failed.")
            return False, None

        # run filter
        filter_succeeded = self._run_script('utils/filter.py', [
            '--smiles_file', str(raw_output_file),
            '--output_file', str(filtered_output_file),
            '--config_file', self.config_path,
        ])
        if not filter_succeeded:
            logger.error(f"'{ga_op_name}' Filtering failed.")
            return False, None

        if not self.enable_lineage_tracking:
            logger.info(f"'{ga_op_name}' The operation is complete, generating {self._count_molecules(str(filtered_output_file))} filtered molecules.")
            return True, None

        filtered_entries = self._filter_lineage_entries(raw_lineage_file, filtered_output_file)
        self._write_jsonl(filtered_lineage_file, filtered_entries)
        logger.info(f"'{ga_op_name}' The operation is complete, generating {self._count_molecules(str(filtered_output_file))} filtered molecules.")
        return True, filtered_lineage_file

    def _filter_lineage_entries(self, raw_lineage_file: Path, filtered_output_file: Path) -> List[Dict]:
        """Keep valid ancestry records based on filtered SMILES."""
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
        """Merge multiple SMILES files into one file"""
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
            logger.error(f"An error occurred while merging files: {e}")
            return False
    def _append_qed_sa_to_file(self, file_path: Path) -> bool:
        """Add metrics column (qed, sa) for files with only SMILES, excluding docking_score."""
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
                    # Compute metrics (docking_score is passed in None, we don't use it)
                    metrics = self._compute_metrics(smiles, None)
                    
                    # Formatted output: SMILES qed sa
                    qed_str = f"{metrics['qed']:.4f}" if metrics['qed'] is not None else "0.0000"
                    sa_str = f"{metrics['sa']:.4f}" if metrics['sa'] is not None else "10.0000"
                    
                    new_line = f"{smiles}\t{qed_str}\t{sa_str}"
                    processed_lines.append(new_line)
            
            # Write to temporary file
            with open(temp_file, 'w', encoding='utf-8') as f:
                for line in processed_lines:
                    f.write(line + '\n')
            
            # Replace original file
            shutil.move(str(temp_file), str(path_obj))
            return True
            
        except Exception as e:
            logger.error(f"Error adding indicator while processing file {file_path}: {e}")
            if temp_file.exists():
                os.remove(temp_file)
            return False

    def _filter_smiles_file_by_constraints(self, file_path: Path, qed_min: float, sa_max: float) -> Tuple[bool, int, int]:
        """Filter molecule files by QED/SA constraints (preserving original row formatting)."""
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
            logger.error(f"Error while constraining filter file {file_path}: {e}")
            if temp_file.exists():
                os.remove(temp_file)
            return False, total_count, kept_count

    def _count_smiles_passing_constraints(self, file_path: Path, qed_min: float, sa_max: float) -> Tuple[int, int]:
        """Only the number of molecules constrained by QED/SA in the file is counted and the file content is not modified."""
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
            logger.warning(f"Error when statistical constraint passes quantity ({file_path}): {e}")
        return total_count, kept_count

    def run_ga_operations(self, parent_smiles_file: str, softbd_generated_file: Optional[str], generation: int) -> Optional[Tuple[str, str, Optional[str], Optional[str]]]:
        """         Perform genetic algorithm operations (crossover and mutation) serially to avoid deadlocks.
        
        Args:
            parent_smiles_file (str): Parent SMILES file path.
            softbd_generated_file (Optional[str]): SMILES file path generated by SoftBD.
            generation (int): Current generation.
            
        Returns:
            Optional[Tuple[str, str, Optional[str], Optional[str]]]:
                Returns (cross descendant file, mutation descendant file, cross lineage file, mutation lineage file) on success.         """
        logger.info(f"Generation {generation}: Start serial execution of genetic algorithm operations...")
        gen_dir = self.output_dir / f"generation_{generation}"

        # 1. Merge parent and SoftBD outputs as GA operation input
        ga_input_pool_file = gen_dir / "ga_input_pool.smi"
        files_to_combine = [parent_smiles_file]
        if softbd_generated_file:
            files_to_combine.append(softbd_generated_file)
        
        if not self._combine_files(files_to_combine, str(ga_input_pool_file)):
            logger.error(f"Generation {generation}: Merging parent and SoftBD output failed.")
            return None
        logger.info(f"Generation {generation}: The GA operation input pool has been created, with a total of {self._count_molecules(str(ga_input_pool_file))} molecules.")

        # 2. Execute crossovers and mutations serially to avoid deadlocks
        crossover_raw_file = gen_dir / "crossover_raw.smi"
        crossover_filtered_file = gen_dir / "crossover_filtered.smi"
        crossover_raw_lineage = gen_dir / "crossover_raw_lineage.jsonl"
        crossover_filtered_lineage = gen_dir / "crossover_filtered_lineage.jsonl"
        mutation_raw_file = gen_dir / "mutation_raw.smi"
        mutation_filtered_file = gen_dir / "mutation_filtered.smi"
        mutation_raw_lineage = gen_dir / "mutation_raw_lineage.jsonl"
        mutation_filtered_lineage = gen_dir / "mutation_filtered_lineage.jsonl"

        # Perform crossover operation
        logger.info(f"Generation {generation}: Starting crossover operation...")
        crossover_ok, crossover_lineage_path = self._execute_ga_stage(
            "cross", 'crossover.py',
            str(ga_input_pool_file), crossover_raw_file, crossover_filtered_file,
            crossover_raw_lineage, crossover_filtered_lineage
        )
        
        if not crossover_ok:
            logger.error(f"Generation {generation}: Crossover operation failed.")
            return None
            
        # Only use debug mode to complete QED/SA for intermediate files, and skip it in cleanup mode to reduce disk writing.
        if not self.cleanup_intermediate_files:
            self._append_qed_sa_to_file(crossover_raw_file)
            self._append_qed_sa_to_file(crossover_filtered_file)

        # Perform mutation operations
        logger.info(f"Generation {generation}: Start mutation operation...")
        mutation_ok, mutation_lineage_path = self._execute_ga_stage(
            "mutation", 'mutation.py',
            str(ga_input_pool_file), mutation_raw_file, mutation_filtered_file,
            mutation_raw_lineage, mutation_filtered_lineage
        )
        
        if not mutation_ok:
            logger.error(f"Generation {generation}: Mutation operation failed.")
            return None
            
        # Only use debug mode to complete QED/SA for intermediate files, and skip it in cleanup mode to reduce disk writing.
        if not self.cleanup_intermediate_files:
            self._append_qed_sa_to_file(mutation_raw_file)
            self._append_qed_sa_to_file(mutation_filtered_file)

        logger.info(f"Generation {generation}: Crossover and mutation operations are completed serially.")
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
        """         Perform evaluation (docking) of progeny populations and generate lineage records.
        
        Args:
            crossover_file (str): Crossover descendant file path.
            mutation_file (str): Path to mutation offspring file.
            generation (int): Current generation.
            crossover_lineage_file (Optional[str]): The lineage file corresponding to the crossover output.
            mutation_lineage_file (Optional[str]): The lineage file corresponding to the mutation output.
            
        Returns:
            Optional[Tuple[str, Optional[str]]]: (Progeny docking result file path, pedigree record file path).         """
        logger.info(f"Generation {generation}: Start descendant evaluation...")
        gen_dir = self.output_dir / f"generation_{generation}"

        # 1. Merge crossover and mutation results
        offspring_raw_file = gen_dir / "offspring_combined_raw.smi"
        if not self._combine_files([crossover_file, mutation_file], str(offspring_raw_file)):
            logger.error(f"Generation {generation}: Failed to merge offspring.")
            return None
        
        # 2. Deduplicate and format the children (preparing for docking)
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
                f"Generation {generation}: constrained filtering statistics of cross products before docking, retaining {crossover_kept}/{crossover_total} molecules"
                f"(QED>={qed_min}, SA<={sa_max})."
            )
            logger.info(
                f"Generation {generation}: Constraint filtering statistics of mutation products before docking, retaining {mutation_kept}/{mutation_total} molecules"
                f"(QED>={qed_min}, SA<={sa_max})."
            )

            constraint_filter_ok, before_filter_count, offspring_count = self._filter_smiles_file_by_constraints(
                offspring_formatted_file,
                qed_min,
                sa_max
            )
            if not constraint_filter_ok:
                logger.error(f"Generation {generation}: Descendant constraint filtering failed.")
                return None
            logger.info(
                f"Generation {generation}: Constraint filtering before docking is completed, retaining {offspring_count}/{before_filter_count} molecules"
                f"(QED>={qed_min}, SA<={sa_max})."
            )
        else:
            logger.info(f"Generation {generation}: selection_mode=nsgaii, skip QED/SA constraint filtering before docking.")
        offspring_lineage_file = gen_dir / "offspring_lineage.jsonl" if self.enable_lineage_tracking else None
        if offspring_count == 0:
            if selection_mode == "ffhs":
                logger.warning(f"Generation {generation}: After deduplication and constraint filtering, there are no valid descendant molecules.")
            else:
                logger.warning(f"Generation {generation}: After deduplication, there are no valid descendant molecules.")
            # Create an empty docking file and lineage file to avoid errors in subsequent steps.
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

        logger.info(f"Progeny formatting complete: A total of {offspring_count} unique molecules ready for docking.")

        offspring_docked_file = gen_dir / "offspring_docked.smi"
        docking_succeeded, submitted_smiles = self._run_docking_and_record_oracle(
            offspring_formatted_file,
            offspring_docked_file,
            gen_dir,
            generation=generation,
            phase="offspring",
        )
        if self.oracle_tracking_enabled and not submitted_smiles:
            logger.warning(f"Generation {generation}: The remaining Oracle budget is 0, and this generation skips child generation docking.")
            open(offspring_docked_file, 'a').close()
            self.last_offspring_histories = set()
            self.last_offspring_smiles = set()
            return str(offspring_docked_file), str(offspring_lineage_file) if offspring_lineage_file else None

        if not docking_succeeded:
            logger.error(f"Generation {generation}: Failed to connect offspring.")
            return None

        docked_count = self._count_molecules(str(offspring_docked_file))
        logger.info(f"Passage {generation}: Progeny evaluation completed, {docked_count} molecules have been docked.")

        offspring_mapping = self._ingest_population_metrics(offspring_docked_file, generation, mark_active=False)
        self.last_offspring_histories = set(offspring_mapping.values())
        self.last_offspring_smiles = set(offspring_mapping.keys())
        if not self.cleanup_intermediate_files:
            if not self._append_qed_sa_to_docked_file(offspring_docked_file):
                logger.error(f"Generation {generation}: QED/SA completion of the descendant failed.")
                return None

        return str(offspring_docked_file), str(offspring_lineage_file) if offspring_lineage_file else None
    def _combine_lineage_records(
        self,
        generation: int,
        unique_smiles: List[str],
        crossover_entries: List[Dict],
        mutation_entries: List[Dict]
    ) -> List[Dict]:
        """Merge the cross and mutated lineage records and align them to the deduplicated progeny set."""
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
        """Preserve the ancestry information of the next generation’s parents to facilitate tracing the source of molecules."""
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
        """         Perform a selection operation to select the next generation from its parents and children.
        
        Args:
            parent_docked_file (str): Parent docking result file path.
            offspring_docked_file (str): Offspring docking result file path.
            generation (int): Current generation.
            
        Returns:
            Optional[str]: Returns the next generation parent file path if successful, and None if failed.         """
        logger.info(f"Generation {generation}: Starting selection operation...")
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
            logger.error(f"Generation {generation}: The selection operation failed or no molecules were selected.")
            return None
        
        selected_count = self._count_molecules(str(next_parents_file))
        logger.info(f"Selection operation completed ({selection_mode.upper()}): {selected_count} molecules are selected as the next generation parents.")
        
        return str(next_parents_file)

    def run_selected_population_evaluation(self, selected_parents_file: str, generation: int) -> bool:
        """         Perform score analysis on the selected elite population (parents of the next generation)
        Args:
            selected_parents_file (str): Selected next generation parent file path
            generation (int): current generation
        Returns:
            bool: whether the score analysis was successful         """
        if self.cleanup_intermediate_files:
            return True

        logger.info(f"Generation {generation}: Start scoring analysis of the selected elite population")
        
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
            logger.info(f"Generation {generation}: Elite population score analysis is completed and the report is saved to {scoring_report_file}")
        else:
            logger.warning(f"Generation {generation}: Elite population score analysis failed, but it does not affect the main process")            
        return scoring_succeeded

    def run_complete_workflow(self):
        """         Execute the complete SoftGA workflow.         """
        import time
        from datetime import timedelta
        
        start_time = time.time()
        logger.info(f"Start executing the complete SoftGA workflow (output directory: {self.output_dir})")
        
        # Step 0: Primary population processing
        current_parents_docked_file = self.run_initial_generation()
        if not current_parents_docked_file:
            logger.error("The initial generation population processing failed and the workflow was terminated.")
            return False
        
        logger.info(f"The first generation population processing was successful, result file: {current_parents_docked_file}")

        # Start iterating
        for generation in range(1, self.max_generations + 1):
            if self._oracle_budget_reached():
                logger.info(
                    "Oracle budget has been reached (%s/%s) and subsequent generations are stopped.",
                    self.oracle_success_calls,
                    self.max_oracle_calls,
                )
                break
            next_parents_file = self.run_generation_step(generation, current_parents_docked_file)
            if not next_parents_file:
                logger.error(f"Generation {generation} failed and the workflow terminated.")
                return False
            current_parents_docked_file = next_parents_file

            if self._oracle_budget_reached():
                logger.info(
                    "After the completion of the %s generation, the Oracle budget reaches the target (%s/%s) and converges early.",
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
                "Oracle budget not met: success_calls=%s < max_oracle_calls=%s",
                self.oracle_success_calls,
                self.max_oracle_calls,
            )
        logger.info("=" * 60)
        logger.info("The SoftGA workflow is all completed!")
        logger.info(f"The final optimized population is saved in: {current_parents_docked_file}")
        logger.info(f"Total time spent on this optimization: {total_duration}")
        if self.oracle_tracking_enabled:
            logger.info(
                "Oracle count: success_calls=%s, submit_calls=%s, max_oracle_calls=%s",
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
                logger.warning("Top1 curve drawing fails, but does not affect the main process")
        if self.cleanup_intermediate_files:
            self._prune_output_to_pop_only()

        return not budget_unmet

    def run_initial_generation(self) -> Optional[str]:
        """         Perform processing of primary populations, including deduplication, formatting, and docking.
        
        Returns:
            Optional[str]: Returns the path to the first-generation population docking result file if successful, and None if failed.         """
        logger.info("Start processing the first generation population (Generation 0)...")
        
        gen_dir = self.output_dir / "generation_0"
        gen_dir.mkdir(exist_ok=True)
        
        # 1. Check whether the initial population file exists
        if not Path(self.initial_population_file).exists():
            logger.error(f"Initial population file not found: {self.initial_population_file}")
            return None
        
        # 2. Remove duplicates and format the initial population
        initial_formatted_file = gen_dir / "initial_population_formatted.smi"
        unique_count = self._remove_duplicates_from_smiles_file(
            self.initial_population_file, 
            str(initial_formatted_file)
        )
        if unique_count == 0:
            logger.error("The initial population file is empty or processing failed.")
            return None

        # Record the ancestry information of the first generation population
        self._record_initial_population(initial_formatted_file)
        
        # 3. Docking the primary population
        initial_docked_file = gen_dir / "initial_population_docked.smi"
        docking_succeeded, submitted_smiles = self._run_docking_and_record_oracle(
            initial_formatted_file,
            initial_docked_file,
            gen_dir,
            generation=0,
            phase="initial",
        )
        if self.oracle_tracking_enabled and not submitted_smiles:
            logger.error("The primary population has no dockable molecules within the Oracle budget constraints.")
            return None
        docked_count = self._count_molecules(str(initial_docked_file))
        if not docking_succeeded or docked_count == 0:
            logger.error("The docking of the primary population failed or no valid docking results were generated.")
            return None
        mapping = self._ingest_population_metrics(initial_docked_file, generation=0, mark_active=True)
        self._mark_histories_active(set(mapping.values()), generation=0)
        if not self.cleanup_intermediate_files:
            if not self._append_qed_sa_to_docked_file(initial_docked_file):
                logger.error("The first-generation population QED/SA completion failed and the workflow was terminated.")
                return None
            ranked_file = initial_docked_file.with_name("initial_population_ranked.smi")
            if not self._write_ranked_by_docking(initial_docked_file, ranked_file):
                logger.warning(f"Initial population: initial_population_ranked.smi Writing failure: {ranked_file}")
        
        logger.info(f"The first generation population docking is completed: {docked_count} molecules have been scored.")
        return str(initial_docked_file)

    def run_softbd_process(self, parent_smiles_file: str, generation: int) -> Optional[str]:
        """         Run the SoftBD build process.
        
        Args:
            parent_smiles_file (str): Parent SMILES file path.
            generation (int): Current generation.
            
        Returns:
            Optional[str]: Generate molecular result file path.         """
        logger.info(f"Generation {generation}: Starting SoftBD generation process...")
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
        logger.info(f"Generation {generation}: SoftBD generation is completed and {generated_count} new molecules are produced.")
        
        # Registration history (compatible with original lineage tracking)
        softbd_smiles = self._read_smiles_from_file(output_file)
        placeholder_root = self._create_generation_placeholder_root(generation)
        for smi in softbd_smiles:
            if smi in self.smiles_to_history:
                continue
            # Use SOFTBD prefix to distinguish
            op_token = f"SOFTBD-{generation}-{self._short_hash(smi)}"
            history = f"{placeholder_root}|{op_token}"
            self._register_history(smi, history)
            self._upsert_history_record(history, smi, generation, None, mark_active=False)
            
        return str(output_file)

    def run_generation_step(self, generation: int, current_parents_docked_file: str):
        """         Execute the complete process of a single generation of SoftGA.         """
        logger.info(f"========== Start generation {generation} evolution ==========")
        gen_dir = self.output_dir / f"generation_{generation}"
        gen_dir.mkdir(exist_ok=True)
        # 1. Extract pure SMILES from parent docking file
        parent_smiles_file = gen_dir / "current_parent_smiles.smi"
        if not self._extract_smiles_from_docked_file(current_parents_docked_file, str(parent_smiles_file)):
            logger.error(f"Generation {generation}: Unable to extract SMILES from parent file, workflow terminated")
            return None

        # 2. SoftBD generation
        softbd_generated_file = self.run_softbd_process(str(parent_smiles_file), generation)

        # 3. Genetic manipulation
        ga_children_files = self.run_ga_operations(str(parent_smiles_file), softbd_generated_file, generation)
        if not ga_children_files:
            logger.error(f"Generation {generation}: The genetic operation failed and the workflow was terminated.")
            return None
        
        crossover_file, mutation_file, crossover_lineage, mutation_lineage = ga_children_files

        # 4. Progeny evaluation (docking, generating pedigree records at the same time)
        offspring_result = self.run_offspring_evaluation(
            crossover_file,
            mutation_file,
            generation,
            crossover_lineage,
            mutation_lineage
        )
        if not offspring_result:
            logger.error(f"Generation {generation}: Child evaluation failed and workflow terminated.")
            return None
        offspring_docked_file, offspring_lineage_file = offspring_result

        # 5. Select
        next_parents_docked_file = self.run_selection(
            current_parents_docked_file, 
            offspring_docked_file, 
            generation
        )
        if not next_parents_docked_file:
            logger.error(f"Generation {generation}: The selection operation failed and the workflow terminated.")
            return None

        selected_mapping = self._ingest_population_metrics(next_parents_docked_file, generation, mark_active=True)
        if not self.cleanup_intermediate_files:
            if not self._append_qed_sa_to_docked_file(Path(next_parents_docked_file)):
                logger.error(f"Generation {generation}: The parent generation QED/SA completion failed and the workflow was terminated.")
                return None
            ranked_file = Path(next_parents_docked_file).with_name("initial_population_ranked.smi")
            if not self._write_ranked_by_docking(Path(next_parents_docked_file), ranked_file):
                logger.warning(f"Generation {generation}: initial_population_ranked.smi Write failed: {ranked_file}")
        new_active_histories = set(selected_mapping.values())
        candidate_histories = set(self.current_active_histories)
        candidate_histories.update(self.last_offspring_histories)
        removed_histories = candidate_histories - new_active_histories
        if removed_histories:
            self._mark_histories_removed(removed_histories, generation)
        self._mark_histories_active(new_active_histories, generation)
        self.last_offspring_histories = set()
        self.last_offspring_smiles = set()

        
        # Save the pedigree information of the next generation’s parents for easy tracking
        if self.enable_lineage_tracking:
            self._save_next_generation_lineage(
                generation,
                next_parents_docked_file,
                offspring_lineage_file
            )

        # 6. Perform score analysis on the selected elite population (skip in cleanup mode)
        if not self.cleanup_intermediate_files:
            self.run_selected_population_evaluation(next_parents_docked_file, generation)

        # 7. Clean temporary files (if enabled)
        self._cleanup_generation_files(generation)

        logger.info(f"========== Generation {generation} evolution completed ==========")
        return next_parents_docked_file

    def _cleanup_generation_files(self, generation_num: int):
        """         Clean up by generation in cleanup mode: delete the entire generation directory to reduce peak IO/storage usage.
        
        Args:
            generation_num (int): generation to be cleaned         """
        if not self.cleanup_intermediate_files:
            return

        to_delete = [self.output_dir / f"generation_{generation_num}"]
        if generation_num == 1:
            to_delete.append(self.output_dir / "generation_0")

        for gen_dir in to_delete:
            try:
                if gen_dir.exists():
                    shutil.rmtree(gen_dir)
                    logger.info(f"Cleaned directory: {gen_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean directory {gen_dir}: {e}")

    def _prune_output_to_pop_only(self) -> None:
        """Cleanup mode final trimming: keep pop.csv and oracle_calls*.csv."""
        if not self.cleanup_intermediate_files:
            return

        pop_file = self.output_dir / "pop.csv"
        if not pop_file.exists():
            logger.warning("cleanup mode final crop skipped: pop.csv not found")
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
            logger.info("The cleanup mode is finally trimmed and retained: %s", ", ".join(sorted(keep_files)))
        except Exception as e:
            logger.warning(f"cleanup Pattern final clipping failed: {e}")

# --- Main function entry ---
def main():
    """Main function, used to parse command line parameters and start workflow"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SoftGA workflow executor')
    parser.add_argument('--config', type=str, 
                       default=DEFAULT_CONFIG,
                       help='Configuration file path')
    parser.add_argument('--receptor', type=str, required=True,
                       help='Target receptor name to run on (e.g. parp1 or 6GL8)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='(Optional) Specify the output directory, overriding the settings in the configuration file')
    parser.add_argument('--initial_population_file', type=str, default=None,
                       help='(Optional) Override workflow.initial_population_file')
    parser.add_argument('--pair_min_tanimoto', type=float, default=None,
                       help='(Optional) Override crossover.pair_min_tanimoto')
    parser.add_argument('--pair_max_tanimoto', type=float, default=None,
                       help='(Optional) Override crossover.pair_max_tanimoto')
    parser.add_argument('--qed_min', type=float, default=None,
                       help='(Optional) Override selection.ffhs_settings.constraints.qed_min')
    parser.add_argument('--sa_max', type=float, default=None,
                       help='(Optional) Override selection.ffhs_settings.constraints.sa_max')
    parser.add_argument(
        '--softbd_enable',
        nargs='?',
        const='true',
        default=None,
        help='(optional) Override softbd.enable (true/false)',
    )
    parser.add_argument('--docking_exhaustiveness', type=int, default=None,
                       help='(Optional) Override docking.exhaustiveness')
    parser.add_argument('--max_oracle_calls', type=int, default=None,
                       help='(Optional) Oracle budget (successful docking count only)')
    parser.add_argument(
        '--enable_crowding_distance',
        nargs='?',
        const='true',
        default=None,
        help='(Optional) Override selection.*_settings.enable_crowding_distance (true/false)',
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
            logger.error("SoftGA workflow execution failed.")
            return 1
        
        logger.info("SoftGA workflow completed successfully!")

    except Exception as e:
        logger.critical(f"A serious error occurred during workflow execution: {e}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
