import argparse
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

TARGET_NAMES = ("parp1", "fa7", "5ht1b", "braf", "jak2")
DEFAULT_JOBS = 20
TOP5_PATTERN = re.compile(r"docking_score:\s*top5%:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")


def infer_target(csv_path: Path) -> str | None:
    name = csv_path.stem.lower()
    hits = [target for target in TARGET_NAMES if target in name]
    if len(hits) == 1:
        return hits[0]
    return None


def iter_csv_files(input_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    glob_func = input_dir.rglob if recursive else input_dir.glob
    files = sorted(path for path in glob_func(pattern) if path.is_file())
    return [path for path in files if not path.stem.endswith("_top5")]


def default_output_path(input_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return input_dir.parent / f"batch_eval_{input_dir.name}_{stamp}.txt"


def default_target_stats_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_target_top5_stats.txt")


def format_skip_block(csv_path: Path) -> str:
    return "\n".join(
        [
            "=" * 50,
            f"[SKIP] {csv_path} (cannot infer target)",
            "",
        ]
    )


def extract_top5_value(stdout: str) -> float | None:
    match = TOP5_PATTERN.search(stdout)
    if match is None:
        return None
    return float(match.group(1))


def run_eval(
    index: int, csv_path: str, target: str, eval_script: str
) -> tuple[int, str, str, str, float | None]:
    lines = [
        "=" * 50,
        f"[RUN ] {csv_path} (target={target})",
    ]
    try:
        proc = subprocess.run(
            [sys.executable, eval_script, "-i", csv_path, "-t", target],
            text=True,
            capture_output=True,
        )
    except Exception as exc:
        lines.append(f"[FAIL] {csv_path} (exception={exc})")
        lines.append("")
        return index, "fail", "\n".join(lines), target, None

    top5_value = None
    if proc.stdout:
        lines.append(proc.stdout.rstrip())
        top5_value = extract_top5_value(proc.stdout)
    if proc.returncode == 0:
        lines.append(f"[ OK ] {csv_path}")
        status = "success"
    else:
        if proc.stderr:
            lines.append(proc.stderr.rstrip())
        lines.append(f"[FAIL] {csv_path} (exit={proc.returncode})")
        status = "fail"
    lines.append("")
    return index, status, "\n".join(lines), target, top5_value


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--input-dir", type=Path, required=True, help="Directory containing CSV files")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Batch result txt path")
    parser.add_argument("-j", "--jobs", type=int, default=DEFAULT_JOBS, help="Max parallel eval workers")
    parser.add_argument("--pattern", type=str, default="*.csv", help="Glob pattern for CSV files")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories recursively")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    if args.jobs < 1:
        raise ValueError(f"--jobs must be >= 1, got {args.jobs}")

    output_path = args.output.resolve() if args.output else default_output_path(input_dir)
    target_stats_output_path = default_target_stats_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now().astimezone()

    csv_files = iter_csv_files(input_dir, args.pattern, args.recursive)
    eval_script = str(Path(__file__).with_name("eval_csv.py").resolve())

    total = len(csv_files)
    success = 0
    skip = 0
    fail = 0
    blocks = [""] * total
    tasks: list[tuple[int, str, str]] = []
    target_top5_values: dict[str, list[float]] = defaultdict(list)

    for index, csv_path in enumerate(csv_files):
        target = infer_target(csv_path)
        if target is None:
            skip += 1
            blocks[index] = format_skip_block(csv_path)
            continue
        tasks.append((index, str(csv_path), target))

    if tasks:
        workers = min(args.jobs, len(tasks))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_eval, index, csv_path, target, eval_script): index
                for index, csv_path, target in tasks
            }
            for future in as_completed(futures):
                index, status, block, target, top5_value = future.result()
                blocks[index] = block
                if status == "success":
                    success += 1
                    if top5_value is not None:
                        target_top5_values[target].append(top5_value)
                else:
                    fail += 1
    else:
        workers = 0

    with output_path.open("w", encoding="utf-8") as out:
        print(f"Batch eval start: {start_time.strftime('%a %b %d %H:%M:%S %Z %Y')}", file=out)
        print(f"Input dir: {input_dir}", file=out)
        print(f"Output file: {output_path}", file=out)
        print(f"Workers: {workers}", file=out)
        print(file=out)
        for block in blocks:
            if block:
                print(block, file=out)

        print("=" * 50, file=out)
        print(f"Summary: total={total}, success={success}, skip={skip}, fail={fail}", file=out)
        print(
            f"Batch eval end: {datetime.now().astimezone().strftime('%a %b %d %H:%M:%S %Z %Y')}",
            file=out,
        )

    with target_stats_output_path.open("w", encoding="utf-8") as out:
        print(f"Target top5% stats from: {output_path}", file=out)
        print(
            f"Generated at: {datetime.now().astimezone().strftime('%a %b %d %H:%M:%S %Z %Y')}",
            file=out,
        )
        print(file=out)
        for target in TARGET_NAMES:
            scores = target_top5_values.get(target, [])
            if not scores:
                print(f"{target}\tn=0\tmean_top5=NA\tstd_top5=NA", file=out)
                continue
            mean_val = statistics.mean(scores)
            std_val = statistics.stdev(scores) if len(scores) >= 2 else 0.0
            values = ",".join(f"{score:.3f}" for score in sorted(scores))
            print(
                f"{target}\tn={len(scores)}\tmean_top5={mean_val:.3f}\tstd_top5={std_val:.3f}\tvalues={values}",
                file=out,
            )

    print(output_path)
    print(target_stats_output_path)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
