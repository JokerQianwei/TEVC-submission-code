import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

"""
 python tools/to_collect_top1_to_results.py \
  -r output/ablation \
  -o results/ablation \
  --max-depth 2 \
  --overwrite

 python tools/to_collect_top1_to_results.py \
  -r output/sensitivity \
  -o results/sensitivity \
  --max-depth 3 \
  --overwrite

 python tools/to_collect_top1_to_results.py \
  -r output/benchmark/main/recircleTue \
  -o results/benchmark/recircleTrue \
  --max-depth 2 \
  --overwrite

 python tools/to_collect_top1_to_results.py \
  -r output/additional \
  -o results/additional \
  --max-depth 2 \
  --overwrite

"""

def _iter_top1_files(root: Path, max_depth: Optional[int]) -> Iterable[Path]:
    if root.is_file():
        if root.name == "top1_per_pop.csv":
            yield root
        return
    root_depth = len(root.parts)
    direct = root / "top1_per_pop.csv"
    if direct.exists():
        yield direct

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        current = Path(dirpath)
        rel_depth = len(current.parts) - root_depth
        if max_depth is not None and rel_depth >= max_depth:
            dirnames[:] = []
        if "top1_per_pop.csv" in filenames and current != root:
            yield current / "top1_per_pop.csv"


def _safe_name(parts: List[str]) -> str:
    cleaned = []
    for p in parts:
        s = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in p.strip())
        s = "_".join(filter(None, s.split("_")))
        if s:
            cleaned.append(s)
    return "__".join(cleaned)


def _build_name(src: Path, root: Path, keep_levels: int) -> str:
    rel_parent = src.parent.relative_to(root) if src.is_relative_to(root) else src.parent
    rel_parts = list(rel_parent.parts[-keep_levels:]) if keep_levels > 0 else list(rel_parent.parts)
    base = _safe_name([root.name] + rel_parts)
    return f"{base}.csv"


def _dedup_target(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    idx = 2
    while True:
        candidate = path.with_name(f"{stem}__{idx}{suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def _copy_or_move(src: Path, dst: Path, move: bool) -> None:
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def _count_csv_data_rows(csv_path: Path) -> int:
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def _collect(
    roots: List[Path],
    out_dir: Path,
    move: bool,
    keep_levels: int,
    max_depth: Optional[int],
    overwrite: bool,
) -> List[Tuple[str, str, int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mappings: List[Tuple[str, str, int]] = []
    for root in roots:
        for src in _iter_top1_files(root, max_depth=max_depth):
            name = _build_name(src=src, root=root, keep_levels=keep_levels)
            dst = out_dir / name
            if not overwrite:
                dst = _dedup_target(dst)
            _copy_or_move(src, dst, move=move)
            data_rows = _count_csv_data_rows(dst)
            mappings.append((str(src.resolve()), str(dst.resolve()), data_rows))
            print(f"[OK] {src} -> {dst} (rows={data_rows})")
    return mappings


def _write_manifest(rows: List[Tuple[str, str, int]], out_dir: Path) -> Path:
    manifest = out_dir / "top1_manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_top1_csv", "saved_as", "data_rows"])
        writer.writerows(rows)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize top1_per_pop.csv to the results directory and rename them uniformly")
    p.add_argument(
        "-r",
        "--roots",
        nargs="+",
        required=True,
        help="Enter the root directory (can be multiple) or a single top1_per_pop.csv file",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        default="results/top1_collected",
        help="Output directory, default results/top1_collected",
    )
    p.add_argument(
        "--move",
        action="store_true",
        help="Move files (default copy)",
    )
    p.add_argument(
        "--keep-levels",
        type=int,
        default=3,
        help="Number of relative levels to keep in filenames (default 3)",
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum level of recursive search (default 3, pass -1 to indicate no limit)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="If the target file already exists, overwrite it (default automatically appends __2/__3 to prevent conflicts)",
    )
    args = p.parse_args()

    roots = [Path(x).expanduser().resolve() for x in args.roots]
    out_dir = Path(args.out_dir).expanduser().resolve()

    mappings = _collect(
        roots=roots,
        out_dir=out_dir,
        move=args.move,
        keep_levels=max(0, args.keep_levels),
        max_depth=None if args.max_depth < 0 else max(0, args.max_depth),
        overwrite=args.overwrite,
    )
    if not mappings:
        print("No top1_per_pop.csv found.")
        return

    manifest = _write_manifest(mappings, out_dir)
    print(f"Completed: {len(mappings)} files -> {out_dir}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
