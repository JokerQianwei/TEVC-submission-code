import argparse
import csv
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

def _pick_col(columns: Sequence[str], candidates: tuple[str, ...], name: str) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(f"missing {name} column, candidates={candidates}, got={list(columns)}")


def _scan_pop_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root] if root.name == "pop.csv" else []
    pop_files: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        if "pop.csv" in filenames:
            pop_files.append(Path(dirpath) / "pop.csv")
    return pop_files


def _to_valid_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _extract_top1(pop_csv: Path) -> Optional[Dict[str, Any]]:
    try:
        with pop_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return None

            score_col = _pick_col(reader.fieldnames, ("docking_score", "ds", "rv"), "docking score")
            smiles_col = _pick_col(reader.fieldnames, ("smiles", "smi", "SMILES"), "smiles")

            best_row: Optional[Dict[str, Any]] = None
            best_score = float("inf")
            for row in reader:
                score = _to_valid_float(row.get(score_col))
                if score is None:
                    continue
                if score < best_score:
                    best_score = score
                    best_row = row
    except (OSError, csv.Error):
        return None
    if best_row is None:
        return None

    row = dict(best_row)
    row[score_col] = best_score
    row["source_pop_csv"] = str(pop_csv)
    row["score_col"] = score_col
    row["smiles_col"] = smiles_col
    row["run_id"] = pop_csv.parent.parent.name if pop_csv.parent.parent.name.startswith("run_") else ""
    row["run_num"] = int(row["run_id"].split("_", 1)[1]) if row["run_id"].startswith("run_") else -1
    return row


def _collect_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    return fieldnames


def _run_single_input(input_path: Path, output_path: Optional[Path], workers: int) -> Tuple[int, int, Path]:
    root = input_path.expanduser().resolve()
    out = output_path.expanduser().resolve() if output_path else (
        (root.parent if root.is_file() else root) / "top1_per_pop.csv"
    )

    pop_files = _scan_pop_files(root)
    rows: List[Dict[str, Any]] = []
    if workers == 1:
        for pop in pop_files:
            item = _extract_top1(pop)
            if item is not None:
                rows.append(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for item in executor.map(_extract_top1, pop_files):
                if item is not None:
                    rows.append(item)

    if not rows:
        return 0, len(pop_files), out

    if any("run_num" in r for r in rows):
        rows.sort(key=lambda r: (r.get("run_num", -1), r.get("source_pop_csv", "")))
    else:
        rows.sort(key=lambda r: r.get("source_pop_csv", ""))

    fieldnames = _collect_fieldnames(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), len(pop_files), out


def _list_dirs_at_depth(root: Path, depth: int) -> List[Path]:
    if not root.is_dir():
        return []
    if depth <= 1:
        return sorted([p for p in root.iterdir() if p.is_dir()])
    out: List[Path] = []
    root_depth = len(root.parts)
    for dirpath, dirnames, _ in os.walk(root, topdown=True):
        current = Path(dirpath)
        rel_depth = len(current.parts) - root_depth
        if rel_depth == depth:
            out.append(current)
            dirnames[:] = []
            continue
        if rel_depth > depth:
            dirnames[:] = []
    return sorted(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Collect top1 (minimum docking_score) of pop.csv")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("-i", "--input", help="Root directory (pop.csv will be looked up recursively) or a single pop.csv file")
    g.add_argument("-r", "--run-dir", help="Process all subdirectories below the directory in parallel (equivalent to executing multiple times in parallel -i)")
    p.add_argument("-o", "--output", default=None, help="Output CSV path; only available in -i mode, defaults to <input>/top1_per_pop.csv")
    p.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        help="-i: Number of threads to read pop.csv; -r: Number of parallel subdirectories",
    )
    p.add_argument(
        "-d",
        "--run-depth",
        type=int,
        default=1,
        help="Directory level processed by -r mode (default 1, indicates the next subdirectory; 2 indicates the next two levels of directories)",
    )
    args = p.parse_args()
    workers = max(1, args.workers)

    if args.input:
        n_rows, n_pops, out = _run_single_input(
            input_path=Path(args.input),
            output_path=Path(args.output) if args.output else None,
            workers=workers,
        )
        if n_rows == 0:
            print(f"No valid pop.csv found under: {Path(args.input).expanduser().resolve()}")
            return
        print(f"Collected {n_rows} top1 rows from {n_pops} pop.csv files")
        print(f"Saved to: {out}")
        return

    if args.output:
        p.error("-o is not supported in -r mode; each subdirectory will write its own top1_per_pop.csv")
    if args.run_depth < 1:
        p.error("--run-depth must >= 1")

    run_root = Path(args.run_dir).expanduser().resolve()
    subdirs = _list_dirs_at_depth(run_root, args.run_depth)
    if not subdirs:
        print(f"No subdirectories found under: {run_root} at depth={args.run_depth}")
        return

    parallelism = workers if workers > 1 else min(32, len(subdirs))
    print(
        f"Running -i over {len(subdirs)} subdirectories at depth={args.run_depth} "
        f"in parallel (workers={parallelism})"
    )

    def _task(subdir: Path) -> Tuple[Path, int, int, Path]:
        n_rows, n_pops, out = _run_single_input(subdir, None, 1)
        return subdir, n_rows, n_pops, out

    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        results = list(executor.map(_task, subdirs))

    ok = 0
    for subdir, n_rows, n_pops, out in results:
        if n_rows == 0:
            print(f"[NO_DATA] {subdir} (scanned pop.csv: {n_pops})")
            continue
        ok += 1
        print(f"[OK] {subdir} -> {out} (rows={n_rows}, pop.csv={n_pops})")
    print(f"Completed: {ok}/{len(subdirs)} subdirectories produced output")


if __name__ == "__main__":
    main()
