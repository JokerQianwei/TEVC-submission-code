import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')


def summarize_seed(seed_dir: Path) -> int:
    rows = []
    runs = sorted((p for p in seed_dir.glob('run_*') if p.is_dir()), key=lambda p: int(p.name.split('_')[-1]))
    for run_dir in runs:
        pop_files = sorted(run_dir.glob('*/pop.csv'))
        if not pop_files:
            continue
        df = pd.read_csv(pop_files[0])
        if df.empty:
            continue
        filtered = df[(df['qed'] > 0.5) & (df['sa'] < 5)]
        pool = filtered if not filtered.empty else df
        best = pool.sort_values('docking_score').iloc[0].copy()
        best['run_dir'] = run_dir.name
        rows.append(best)

    if not rows:
        return 0

    summary = pd.DataFrame(rows)
    summary.to_csv(seed_dir / 'summary.csv', index=False)
    stats = (
        f"Mean Vina: {summary['docking_score'].mean():.4f}\n"
        f"Top 5% Vina: {summary['docking_score'].quantile(0.05):.4f}\n"
        f"Median Vina: {summary['docking_score'].median():.4f}\n"
        f"Mean QED: {summary['qed'].mean():.4f}\n"
        f"Mean SA: {summary['sa'].mean():.4f}"
    )
    (seed_dir / 'summary_stats.txt').write_text(stats, encoding='utf-8')
    return len(summary)


def find_seed_dirs(root: Path):
    # 兼容两种结构：
    # 1) .../seed42/run_*
    # 2) .../seed42/<batch_id>/run_*
    def has_runs(d: Path) -> bool:
        return d.is_dir() and any(c.is_dir() and c.name.startswith('run_') for c in d.iterdir())

    if has_runs(root):
        return [root]

    result = []
    for seed_dir in sorted(p for p in root.rglob('seed*') if p.is_dir()):
        if has_runs(seed_dir):
            result.append(seed_dir)
            continue
        result.extend(sorted(child for child in seed_dir.iterdir() if has_runs(child)))

    # 去重并保持顺序
    seen = set()
    dedup = []
    for p in result:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        dedup.append(p)
    return dedup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', help='实验结果目录，如 .../batch-multi/test11-crossover-backup')
    args = parser.parse_args()

    root = Path(args.result_dir).expanduser().resolve()
    seed_dirs = find_seed_dirs(root)
    if not seed_dirs:
        logging.warning(f'No seed dirs found under {root}')
        return

    total_rows = 0
    for seed_dir in seed_dirs:
        n = summarize_seed(seed_dir)
        total_rows += n
        if n:
            logging.info(f'Summarized {seed_dir}: {n} runs')
        else:
            logging.warning(f'No valid pop.csv in {seed_dir}')
    logging.info(f'Finished. total selected runs: {total_rows}')


if __name__ == '__main__':
    main()
