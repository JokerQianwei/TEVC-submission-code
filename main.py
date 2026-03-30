import argparse
import logging
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_CONFIG = "./config.yaml"

from execute import SoftGAWorkflowExecutor
from utils.config_loader import load_config
from utils.cpu_utils import get_available_cpu_cores

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SOFTGA_MAIN")


def _to_bool(value, default: bool = True) -> bool:
    if value is None:
        return bool(default)
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def run_workflow_for_receptor(
    config_path: str,
    receptor_name: str,
    output_dir: Optional[str],
    num_processors: int,
    initial_population_file: Optional[str] = None,
    strategy_mode: Optional[str] = None,
    max_generations: Optional[int] = None,
    seed: Optional[int] = None,
    selection_mode: Optional[str] = None,
    samples_per_parent: Optional[int] = None,
    tanimoto_threshold: Optional[float] = None,
    min_keep_ratio: Optional[float] = None,
    max_keep_ratio: Optional[float] = None,
    recircle: Optional[bool] = None,
    softbd_enable: Optional[bool] = None,
    softbd_seed_mode: Optional[str] = None,
    gpu: Optional[str] = None,
    block_size: Optional[int] = None,
    length: Optional[int] = None,
    temperature: Optional[float] = None,
    nucleus_p: Optional[float] = None,
    gpu_max_batch_size: Optional[int] = None,
    steps: Optional[int] = None,
    initial_samples: Optional[int] = None,
    gen1_n_select: Optional[int] = None,
    gen1_selection_mode: Optional[str] = None,
    number_of_crossovers: Optional[int] = None,
    pair_min_tanimoto: Optional[float] = None,
    pair_max_tanimoto: Optional[float] = None,
    number_of_mutants: Optional[int] = None,
    qed_min: Optional[float] = None,
    sa_max: Optional[float] = None,
    enable_crowding_distance: Optional[bool] = None,
    cleanup_intermediate_files: Optional[bool] = None,
    docking_tool: Optional[str] = None,
    docking_exhaustiveness: Optional[int] = None,
    max_oracle_calls: Optional[int] = None,
    plot_top1: bool = False,
    softbd_sampler=None,
) -> Tuple[str, bool]:
    """     Run the full SoftGA workflow for a single receptor (process-safe wrapper).

    Args:
        config_path: Path to the YAML config file.
        receptor_name: Receptor name.
        output_dir: Output root directory.
        num_processors: CPU cores allocated to this worker.
        initial_population_file: Override workflow.initial_population_file.
        strategy_mode: SoftBD dynamic strategy mode.
        max_generations: Maximum number of generations.
        seed: Random seed.
        selection_mode: Selection mode (ffhs | nsgaii).
        qed_min: FFHS constraint QED lower bound.
        sa_max: FFHS constraint SA upper bound.
        samples_per_parent: Samples per parent.
        tanimoto_threshold: Tanimoto threshold.
        min_keep_ratio: Min keep ratio.
        max_keep_ratio: Max keep ratio.
        block_size: SoftBD block size.
        length: SoftBD generation length.
        temperature: SoftBD temperature.
        nucleus_p: SoftBD nucleus_p.
        gpu_max_batch_size: SoftBD gpu_max_batch_size.
        initial_samples: SoftBD initial_samples.
        gen1_selection_mode: SoftBD Gen1 selection mode (maxmin | random).
        number_of_crossovers: crossover.number_of_crossovers override value.
        pair_min_tanimoto: crossover.pair_min_tanimoto override value.
        pair_max_tanimoto: crossover.pair_max_tanimoto override value.
        number_of_mutants: mutation.number_of_mutants override value.
        enable_crowding_distance: Overrides selection.*_settings.enable_crowding_distance.
        max_oracle_calls: Oracle budget (only counts successful docking).

    Returns:
        Tuple[str, bool]: (receptor_display_name, success)     """
    receptor_display_name = receptor_name
    
    # Configure receptor-specific logger
    log_handler = None
    try:
        cfg = load_config(config_path, PROJECT_ROOT)
        
        base_out = output_dir or cfg.get('workflow', {}).get('output_directory', 'SoftGA_output')
        rec_dir = receptor_name
        
        log_path = os.path.join(base_out, rec_dir, "logs", f"worker_{os.getpid()}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        log_handler = logging.FileHandler(log_path, encoding='utf-8')
        log_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(log_handler)
    except Exception as e:
        logger.warning(f"Failed to setup receptor log: {e}")

    logger.info("=" * 80)
    logger.info(
        "Starting worker process for receptor '%s' (PID: %s)",
        receptor_display_name,
        os.getpid(),
    )
    logger.info("Allocated CPU cores: %s", num_processors)
    if strategy_mode:
        logger.info("Overriding strategy_mode: %s", strategy_mode)
    if max_generations is not None:
        logger.info("Overriding max_generations: %s", max_generations)
    if seed is not None:
        logger.info("Overriding seed: %s", seed)
    if selection_mode:
        logger.info("Overriding selection_mode: %s", selection_mode)
    if qed_min is not None:
        logger.info("Overriding qed_min: %s", qed_min)
    if sa_max is not None:
        logger.info("Overriding sa_max: %s", sa_max)
    if samples_per_parent is not None:
        logger.info("Overriding samples_per_parent: %s", samples_per_parent)
    if tanimoto_threshold is not None:
        logger.info("Overriding tanimoto_threshold: %s", tanimoto_threshold)
    if min_keep_ratio is not None:
        logger.info("Overriding min_keep_ratio: %s", min_keep_ratio)
    if max_keep_ratio is not None:
        logger.info("Overriding max_keep_ratio: %s", max_keep_ratio)
    if recircle is not None:
        logger.info("Overriding recircle: %s", recircle)
    if softbd_enable is not None:
        logger.info("Overriding softbd_enable: %s", softbd_enable)
    if softbd_seed_mode is not None:
        logger.info("Overriding softbd_seed_mode: %s", softbd_seed_mode)
    if gpu is not None:
        logger.info("Overriding gpu: %s", gpu)
    if block_size is not None:
        logger.info("Overriding block_size: %s", block_size)
    if length is not None:
        logger.info("Overriding length: %s", length)
    if temperature is not None:
        logger.info("Overriding temperature: %s", temperature)
    if nucleus_p is not None:
        logger.info("Overriding nucleus_p: %s", nucleus_p)
    if gpu_max_batch_size is not None:
        logger.info("Overriding gpu_max_batch_size: %s", gpu_max_batch_size)
    if steps is not None:
        logger.info("Overriding steps: %s", steps)
    if initial_samples is not None:
        logger.info("Overriding initial_samples: %s", initial_samples)
    if gen1_n_select is not None:
        logger.info("Overriding gen1_n_select: %s", gen1_n_select)
    if gen1_selection_mode is not None:
        logger.info("Overriding gen1_selection_mode: %s", gen1_selection_mode)
    if number_of_crossovers is not None:
        logger.info("Overriding number_of_crossovers: %s", number_of_crossovers)
    if pair_min_tanimoto is not None:
        logger.info("Overriding pair_min_tanimoto: %s", pair_min_tanimoto)
    if pair_max_tanimoto is not None:
        logger.info("Overriding pair_max_tanimoto: %s", pair_max_tanimoto)
    if number_of_mutants is not None:
        logger.info("Overriding number_of_mutants: %s", number_of_mutants)
    if enable_crowding_distance is not None:
        logger.info("Overriding enable_crowding_distance: %s", enable_crowding_distance)
    if cleanup_intermediate_files is not None:
        logger.info("Overriding cleanup_intermediate_files: %s", cleanup_intermediate_files)
    if docking_tool is not None:
        logger.info("Overriding docking_tool: %s", docking_tool)
    if docking_exhaustiveness is not None:
        logger.info("Overriding docking_exhaustiveness: %s", docking_exhaustiveness)
    if max_oracle_calls is not None:
        logger.info("Overriding max_oracle_calls: %s", max_oracle_calls)
    if initial_population_file is not None:
        logger.info("Overriding initial_population_file: %s", initial_population_file)
    logger.info("=" * 80)
    
    try:
        executor_kwargs = dict(
            config_path=config_path,
            receptor_name=receptor_name,
            output_dir_override=output_dir,
            num_processors_override=num_processors,
            strategy_mode_override=strategy_mode,
            max_generations_override=max_generations,
            seed_override=seed,
            selection_mode_override=selection_mode,
            qed_min_override=qed_min,
            sa_max_override=sa_max,
            samples_per_parent_override=samples_per_parent,
            tanimoto_threshold_override=tanimoto_threshold,
            min_keep_ratio_override=min_keep_ratio,
            max_keep_ratio_override=max_keep_ratio,
            recircle_override=recircle,
            softbd_enable_override=softbd_enable,
            softbd_seed_mode_override=softbd_seed_mode,
            gpu_override=gpu,
            block_size_override=block_size,
            length_override=length,
            temperature_override=temperature,
            nucleus_p_override=nucleus_p,
            gpu_max_batch_size_override=gpu_max_batch_size,
            steps_override=steps,
            initial_samples_override=initial_samples,
            gen1_n_select_override=gen1_n_select,
            gen1_selection_mode_override=gen1_selection_mode,
            number_of_crossovers_override=number_of_crossovers,
            pair_min_tanimoto_override=pair_min_tanimoto,
            pair_max_tanimoto_override=pair_max_tanimoto,
            number_of_mutants_override=number_of_mutants,
            enable_crowding_distance_override=enable_crowding_distance,
            softbd_sampler=softbd_sampler,
            plot_top1=plot_top1,
        )
        if cleanup_intermediate_files is not None:
            executor_kwargs["cleanup_intermediate_files_override"] = cleanup_intermediate_files
        if docking_tool is not None:
            executor_kwargs["docking_tool_override"] = docking_tool
        if docking_exhaustiveness is not None:
            executor_kwargs["docking_exhaustiveness_override"] = docking_exhaustiveness
        if max_oracle_calls is not None:
            executor_kwargs["max_oracle_calls_override"] = max_oracle_calls
        if initial_population_file is not None:
            executor_kwargs["initial_population_file_override"] = initial_population_file
        executor = SoftGAWorkflowExecutor(**executor_kwargs)
        success = executor.run_complete_workflow()
        if success:
            logger.info(
                "Worker finished successfully: receptor '%s' (PID: %s)",
                receptor_display_name,
                os.getpid(),
            )
            logger.info("Lineage tracking saved to: %s", executor.lineage_tracker_path)
            logger.info(
                "EvoMol exports: %s/pop.csv, %s/removed_ind_act_history.csv",
                executor.output_dir,
                executor.output_dir,
            )
        else:
            logger.error(
                "Worker failed: receptor '%s' (PID: %s)",
                receptor_display_name,
                os.getpid(),
            )
        return receptor_display_name, success

    except Exception as e:
        logger.critical(
            "Unhandled exception while running workflow for receptor '%s': %s",
            receptor_display_name,
            e,
            exc_info=True,
        )
        return receptor_display_name, False
    finally:
        if log_handler:
            log_handler.close()
            logging.getLogger().removeHandler(log_handler)


def run_cli(argv=None, softbd_sampler=None) -> int:
    parser = argparse.ArgumentParser(
        description="SoftGA hybrid molecule generation entry point",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=DEFAULT_CONFIG,
        help='Path to the main config YAML file',
    )
    parser.add_argument(
        '--receptor',
        type=str,
        required=True,
        help='Receptor name to run (e.g., parp1 or 6GL8)',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='(Optional) Output root directory override',
    )
    parser.add_argument(
        '--initial_population_file',
        type=str,
        default=None,
        help='(Optional) Override workflow.initial_population_file',
    )
    parser.add_argument(
        '--strategy_mode',  # Prefix retention policy
        type=str,
        default=None,
        help='Support linear, aggressive, super_aggressive, sigmoid, piecewise, cosine, step, step_20_40_60_80 modes',
    )
    parser.add_argument(
        '--max_generations', # generative algebra
        type=int,
        default=None,
        help='(Optional) Maximum number of generations',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='(Optional) Random seed',
    )
    parser.add_argument(
        '--selection_mode',
        type=str,
        default=None,
        help='ffhs | nsgaii',
    )
    parser.add_argument(
        '--qed_min',
        type=float,
        default=None,
        help='(Optional) Overrides selection.ffhs_settings.constraints.qed_min',
    )
    parser.add_argument(
        '--sa_max',
        type=float,
        default=None,
        help='(Optional) Overrides selection.ffhs_settings.constraints.sa_max',
    )
    parser.add_argument(
        '--samples_per_parent',
        type=int,
        default=None,
        help='(Optional) SoftBD samples per parent',
    )
    parser.add_argument(
        '--tanimoto_threshold',
        type=float,
        default=None,
        help='(Optional) SoftBD tanimoto threshold',
    )
    parser.add_argument(
        '--min_keep_ratio',
        type=float,
        default=None,
        help='(Optional) SoftBD min keep ratio',
    )
    parser.add_argument(
        '--max_keep_ratio',
        type=float,
        default=None,
        help='(Optional) SoftBD max keep ratio',
    )
    parser.add_argument(
        '--recircle',
        nargs='?',
        const='true',
        default=None,
        help='(Optional) SoftBD recircle switch (true/false), controls whether to reuse the previous round of prefix across runs',
    )
    parser.add_argument(
        '--softbd_enable',
        nargs='?',
        const='true',
        default=None,
        help='(Optional) SoftBD master switch (true/false)',
    )
    parser.add_argument(
        '--softbd_seed_mode',
        type=str,
        default=None,
        help='(Optional) SoftBD seed mode: workflow | random_per_run',
    )
    parser.add_argument(
        '--gpu',
        type=str,
        default=None,
        help='(Optional) SoftBD gpu id (string/int)',
    )
    parser.add_argument(
        '--block_size', 
        type=int,
        default=None,
        help='(Optional) SoftBD block size',
    )
    parser.add_argument(
        '--length',
        type=int,
        default=None,
        help='(Optional) SoftBD generation length',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='(Optional) SoftBD temperature',
    )
    parser.add_argument(
        '--nucleus_p',
        type=float,
        default=None,
        help='(Optional) SoftBD nucleus_p',
    )
    parser.add_argument(
        '--number_of_processors',
        type=int,
        default=None,
        help='Control the number of CPU cores used by a single task',
    )
    parser.add_argument(
        '--gpu_max_batch_size',
        type=int,
        default=None,
        help='Controls the maximum batch size for SoftBD generation',
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=None,
        help='(Optional) SoftBD diffusion steps (algo.T)',
    )
    parser.add_argument(
        '--initial_samples',
        type=int,
        default=None,
        help='(Optional) SoftBD initial_samples',
    )
    parser.add_argument(
        '--gen1_n_select',
        type=int,
        default=None,
        help='(Optional) SoftBD Gen1 selected molecules cap',
    )
    parser.add_argument(
        '--gen1_selection_mode',
        type=str,
        default=None,
        help='(Optional) SoftBD Gen1 selection mode: maxmin | random',
    )
    parser.add_argument(
        '--number_of_crossovers',
        type=int,
        default=None,
        help='(Optional) Override crossover.number_of_crossovers',
    )
    parser.add_argument(
        '--pair_min_tanimoto',
        type=float,
        default=None,
        help='(Optional) Override crossover.pair_min_tanimoto',
    )
    parser.add_argument(
        '--pair_max_tanimoto',
        type=float,
        default=None,
        help='(Optional) Override crossover.pair_max_tanimoto',
    )
    parser.add_argument(
        '--number_of_mutants',
        type=int,
        default=None,
        help='(Optional) Overrides mutation.number_of_mutants',
    )
    parser.add_argument(
        '--enable_crowding_distance',
        nargs='?',
        const='true',
        default=None,
        help='(Optional) Overrides selection.*_settings.enable_crowding_distance (true/false)',
    )
    parser.add_argument(
        '--plot_top1',
        nargs='?',
        const='true',
        default='true',
        help='(Optional) Draw the Top1 curve at the end of the run (true/false)',
    )
    parser.add_argument(
        '--cleanup_intermediate_files',
        nargs='?',
        const='true',
        default=None,
        help='(Optional) Override cleanup_intermediate_files in configuration (true/false)',
    )
    parser.add_argument(
        '--docking_tool',
        type=str,
        default=None,
        help='(Optional) docking engine: qvina02 | vina',
    )
    parser.add_argument(
        '--docking_exhaustiveness',
        type=int,
        default=None,
        help='(Optional) Override docking.exhaustiveness',
    )
    parser.add_argument(
        '--max_oracle_calls',
        type=int,
        default=None,
        help='(Optional) Oracle budget (successful docking count only); not set to turn off budget mode',
    )

    args = parser.parse_args(argv)
    plot_top1 = str(args.plot_top1).strip().lower() in ("1", "true", "yes", "y")
    recircle = _to_bool(args.recircle, default=True) if args.recircle is not None else None
    softbd_enable = _to_bool(args.softbd_enable, default=True) if args.softbd_enable is not None else None
    enable_crowding_distance = (
        _to_bool(args.enable_crowding_distance, default=True)
        if args.enable_crowding_distance is not None
        else None
    )
    cleanup_intermediate_files_override = (
        _to_bool(args.cleanup_intermediate_files, default=True)
        if args.cleanup_intermediate_files is not None
        else None
    )

    try:
        config = load_config(args.config, PROJECT_ROOT)
    except Exception as exc:
        logger.critical("Failed to load config YAML: %s (%s)", args.config, exc)
        return 1

    receptor_to_run = args.receptor

    cleanup_intermediate_files = (
        cleanup_intermediate_files_override
        if cleanup_intermediate_files_override is not None
        else bool(config.get("performance", {}).get("cleanup_intermediate_files", False))
    )
    file_handler = None
    if not cleanup_intermediate_files:
        log_dir = os.path.join(args.output_dir, 'logs') if args.output_dir else 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"softga_run_{os.getpid()}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)

    logger.info("Receptor to run: %s", receptor_to_run)

    performance_config = config.get('performance', {})
    # The number_of_processors passed in the command line will be used first. If not, the number_of_processors in the configuration file will be used.
    inner_processors_config = args.number_of_processors if args.number_of_processors is not None else performance_config.get('number_of_processors', -1)
    
    if inner_processors_config is None:
        inner_processors_config = -1

    try:
        if inner_processors_config == -1:
            available_cores, _cpu_usage = get_available_cpu_cores(cpu_threshold=95.0)
            cores_per_receptor = max(1, int(available_cores * 0.95))
        else:
            cores_per_receptor = inner_processors_config

        logger.info("CPU cores for receptor %s: %s", receptor_to_run, cores_per_receptor)

        workflow_kwargs = dict(
            initial_population_file=args.initial_population_file,
            strategy_mode=args.strategy_mode,
            max_generations=args.max_generations,
            seed=args.seed,
            selection_mode=args.selection_mode,
            qed_min=args.qed_min,
            sa_max=args.sa_max,
            samples_per_parent=args.samples_per_parent,
            tanimoto_threshold=args.tanimoto_threshold,
            min_keep_ratio=args.min_keep_ratio,
            max_keep_ratio=args.max_keep_ratio,
            recircle=recircle,
            softbd_enable=softbd_enable,
            softbd_seed_mode=args.softbd_seed_mode,
            gpu=args.gpu,
            block_size=args.block_size,
            length=args.length,
            temperature=args.temperature,
            nucleus_p=args.nucleus_p,
            gpu_max_batch_size=args.gpu_max_batch_size,
            steps=args.steps,
            initial_samples=args.initial_samples,
            gen1_n_select=args.gen1_n_select,
            gen1_selection_mode=args.gen1_selection_mode,
            number_of_crossovers=args.number_of_crossovers,
            pair_min_tanimoto=args.pair_min_tanimoto,
            pair_max_tanimoto=args.pair_max_tanimoto,
            number_of_mutants=args.number_of_mutants,
            enable_crowding_distance=enable_crowding_distance,
            docking_tool=args.docking_tool,
            docking_exhaustiveness=args.docking_exhaustiveness,
            softbd_sampler=softbd_sampler,
            plot_top1=plot_top1,
        )
        if cleanup_intermediate_files_override is not None:
            workflow_kwargs["cleanup_intermediate_files"] = cleanup_intermediate_files_override
        if args.max_oracle_calls is not None:
            workflow_kwargs["max_oracle_calls"] = args.max_oracle_calls
        receptor_display_name, success = run_workflow_for_receptor(
            args.config,
            receptor_to_run,
            args.output_dir,
            cores_per_receptor,
            **workflow_kwargs,
        )

        logger.info("=" * 80)
        logger.info("All SoftGA workflows finished")
        logger.info("Successful receptors: %s", [receptor_display_name] if success else [])
        logger.info("Failed receptors: %s", [] if success else [receptor_display_name])
        logger.info("=" * 80)
        return 0 if success else 1
    finally:
        if file_handler:
            file_handler.close()
            logging.getLogger().removeHandler(file_handler)


def main() -> None:
    raise SystemExit(run_cli())

if __name__ == "__main__":
    # On Windows or macOS, using 'spawn' (or 'forkserver') avoids issues with forking.
    if sys.platform.startswith('win') or sys.platform.startswith('darwin'):
        multiprocessing.set_start_method('spawn', force=True)
    
    main()
