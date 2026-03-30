import argparse
import logging
import multiprocessing as mp
import os
import secrets
import shutil
import sys
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
import pandas as pd

from utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
SOFTGA_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = "./config.yaml"


def _clean_rest(rest):
    return [x for x in rest if x and x.strip()]


def _to_bool(value, default: bool = True) -> bool:
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _make_batch_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


def _parse_overrides(rest):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--config', default=DEFAULT_CONFIG)
    p.add_argument('--length', type=int)
    p.add_argument('--block_size', type=int)
    p.add_argument('--temperature', type=float)
    p.add_argument('--nucleus_p', type=float)
    p.add_argument('--gpu_max_batch_size', type=int)
    p.add_argument('--samples_per_parent', type=int)
    p.add_argument('--tanimoto_threshold', type=float)
    p.add_argument('--steps', type=int)
    p.add_argument('--initial_samples', type=int)
    p.add_argument('--gen1_selection_mode', type=str)
    p.add_argument('--strategy_mode', type=str)
    p.add_argument('--min_keep_ratio', type=float)
    p.add_argument('--max_keep_ratio', type=float)
    p.add_argument('--recircle', nargs='?', const='true', default=None)
    p.add_argument('--softbd_enable', nargs='?', const='true', default=None)
    p.add_argument('--softbd_seed_mode', type=str)
    p.add_argument('--cleanup_intermediate_files', nargs='?', const='true', default=None)
    return p.parse_known_args(rest)[0]


def _load_softbd_cfg(config_path: str, overrides) -> dict:
    cfg = load_config(config_path, SOFTGA_ROOT)
    softbd = dict(cfg.get('softbd', {}))
    softbd.pop('random_seed_bits', None)
    softbd['gpu'] = '0'

    for k in ['length', 'block_size', 'temperature']:
        v = getattr(overrides, k)
        if v is not None:
            softbd[k] = v

    gen = dict(softbd.get('generation_params', {}))
    for k in ['nucleus_p', 'gpu_max_batch_size', 'samples_per_parent', 'tanimoto_threshold', 'steps', 'initial_samples', 'gen1_selection_mode']:
        v = getattr(overrides, k)
        if v is not None:
            gen[k] = v
    softbd['generation_params'] = gen

    dyn = dict(softbd.get('dynamic_strategy', {}))
    for k in ['strategy_mode', 'min_keep_ratio', 'max_keep_ratio']:
        v = getattr(overrides, k)
        if v is not None:
            dyn[k] = v
    softbd['dynamic_strategy'] = dyn
    if getattr(overrides, 'recircle', None) is not None:
        softbd['recircle'] = _to_bool(overrides.recircle, default=True)
    if getattr(overrides, 'softbd_enable', None) is not None:
        softbd['enable'] = _to_bool(overrides.softbd_enable, default=True)
    if getattr(overrides, 'softbd_seed_mode', None) is not None:
        softbd['seed_mode'] = str(overrides.softbd_seed_mode)

    return softbd


def _is_cleanup_enabled(config_path: str, cleanup_override=None) -> bool:
    if cleanup_override is not None:
        return bool(cleanup_override)
    try:
        cfg = load_config(config_path, SOFTGA_ROOT)
    except Exception:
        return False
    return bool(cfg.get("performance", {}).get("cleanup_intermediate_files", False))


def _prune_run_to_pop_only(run_dir: str, receptor: str) -> bool:
    run_path = Path(run_dir)
    receptor_dir = run_path / receptor
    pop_file = receptor_dir / "pop.csv"
    if not pop_file.exists():
        return False

    try:
        pop_content = pop_file.read_bytes()
        oracle_payloads = []
        for oracle_file in sorted(receptor_dir.glob("oracle_calls*.csv")):
            oracle_payloads.append((oracle_file.name, oracle_file.read_bytes()))
        for child in run_path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        receptor_dir.mkdir(parents=True, exist_ok=True)
        pop_file.write_bytes(pop_content)
        for name, payload in oracle_payloads:
            (receptor_dir / name).write_bytes(payload)
        return True
    except Exception:
        return False

def worker(q, gpu_id: int, rest, receptor: str, seed: int):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    sys.path.insert(0, os.path.dirname(__file__))
    import main as softga_main
    from generation import SoftBDSampler

    rest = _clean_rest(rest)
    overrides = _parse_overrides(rest)
    config_path = overrides.config
    cleanup_override = (
        _to_bool(overrides.cleanup_intermediate_files, default=True)
        if overrides.cleanup_intermediate_files is not None
        else None
    )
    cleanup_enabled = _is_cleanup_enabled(config_path, cleanup_override)
    sampler = None
    sampler_initialized = False

    while True:
        run_dir = q.get()
        if run_dir is None:
            q.task_done()
            return

        try:
            os.makedirs(run_dir, exist_ok=True)
            run_log = os.path.join(run_dir, 'run.log')
            if not sampler_initialized:
                softbd_cfg = _load_softbd_cfg(config_path, overrides)
                sampler = SoftBDSampler(softbd_cfg, seed=int(seed), log_dir=Path(run_dir) / '_softbd_init')
                sampler_initialized = True

            argv = list(rest) + ['--output_dir', run_dir, '--receptor', receptor, '--seed', str(seed)]
            with open(run_log, 'w', encoding='utf-8') as f:
                with redirect_stdout(f), redirect_stderr(f):
                    rc = softga_main.run_cli(argv=argv, softbd_sampler=sampler)
            if rc == 0:
                if cleanup_enabled:
                    pruned = _prune_run_to_pop_only(run_dir, receptor)
                    if not pruned:
                        logging.warning(f"cleanup enabled but prune skipped (missing pop.csv?): {run_dir}")
                logging.info(f"Done: {run_dir}")
            else:
                logging.error(f"Fail {run_dir}: exit={rc}")
        except Exception as e:
            logging.error(f"Fail {run_dir}: {e}")
        finally:
            q.task_done()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu_ids", type=int, nargs='+', required=True)
    p.add_argument("--tasks_per_gpu", type=int, default=1)
    p.add_argument("--total_runs", type=int, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--receptor", required=True)
    p.add_argument("--seed", default=42)
    p.add_argument("--recircle", nargs='?', const='true', default=None, help="SoftBD recircle 开关 (true/false)")
    p.add_argument("--softbd_enable", nargs='?', const='true', default=None, help="SoftBD 总开关 (true/false)")
    p.add_argument("--softbd_seed_mode", type=str, default=None, help="SoftBD seed mode: workflow | random_per_run")
    p.add_argument("--gen1_selection_mode", type=str, default=None, help="SoftBD Gen1 selection mode: maxmin | random")
    p.add_argument(
        "--cleanup_intermediate_files",
        nargs='?',
        const='true',
        default=None,
        help="覆盖配置中的 cleanup_intermediate_files (true/false)",
    )
    p.add_argument(
        "--batch_id",
        default=None,
        help="批次子目录名；默认自动生成 `时间戳-随机码`，避免同 seed 多次运行覆盖。",
    )
    args, rest = p.parse_known_args()
    rest = _clean_rest(rest)
    if args.recircle is not None:
        rest += ["--recircle", str(args.recircle)]
    if args.softbd_enable is not None:
        rest += ["--softbd_enable", str(args.softbd_enable)]
    if args.softbd_seed_mode is not None:
        rest += ["--softbd_seed_mode", str(args.softbd_seed_mode)]
    if args.gen1_selection_mode is not None:
        rest += ["--gen1_selection_mode", str(args.gen1_selection_mode)]
    if args.cleanup_intermediate_files is not None:
        rest += ["--cleanup_intermediate_files", str(args.cleanup_intermediate_files)]

    ctx = mp.get_context('spawn')
    q = ctx.JoinableQueue()
    seed_base = os.path.join(args.output_dir, args.receptor, f"seed{args.seed}")
    batch_id = args.batch_id or _make_batch_id()
    base = os.path.join(seed_base, batch_id)
    
    logging.info(f"Queuing {args.total_runs} tasks to {base}...")
    for i in range(args.total_runs):
        run_dir = os.path.join(base, f"run_{i}")
        q.put(run_dir)

    procs = [
        ctx.Process(target=worker, args=(q, gpu, list(rest), args.receptor, int(args.seed)))
        for gpu in args.gpu_ids
        for _ in range(args.tasks_per_gpu)
    ]
    for _ in range(len(procs)):
        q.put(None)
    for p_ in procs:
        p_.start()
    q.join()
    for p_ in procs:
        p_.join()

    logging.info("Summarizing results...")
    results = []
    for i in range(args.total_runs):
        csv_path = os.path.join(base, f"run_{i}", args.receptor, "pop.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    continue
                results.append(df.sort_values('docking_score').iloc[0])
            except Exception as e:
                logging.warning(f"Skip {csv_path}: {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(base, "summary.csv"), index=False)
        stats = (
            f"Mean Vina: {df['docking_score'].mean():.4f}\n"
            f"Median Vina: {df['docking_score'].median():.4f}\n"
            f"Top 5% Vina: {df['docking_score'].quantile(0.05):.4f}\n"
            f"Mean QED: {df['qed'].mean():.4f}\n"
            f"Median QED: {df['qed'].median():.4f}\n"
            f"Mean SA: {df['sa'].mean():.4f}\n"
            f"Median SA: {df['sa'].median():.4f}"
        )
        with open(os.path.join(base, "summary_stats.txt"), "w") as f: f.write(stats)
        logging.info(f"Stats:\n{stats}")
    else: logging.warning("No results found.")

if __name__ == "__main__":
    main()
