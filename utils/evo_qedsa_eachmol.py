#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Line-by-line completion of QED/SA two columns for docked files.

The input is usually: SMILES <tab|space> docking_score [extra...]
The output is: SMILES docking_score QED SA [extra...]
If the original file already contains >= 4 columns, the 3/4th column will be regarded as the old QED/SA and overwritten, and the 5th and subsequent additional columns will be retained. """

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.chem_metrics import ChemMetricCache


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except Exception:
        return ""


def _parse_line(line: str) -> Optional[List[str]]:
    s = line.strip()
    if not s:
        return None
    return s.replace("\t", " ").split()


def _annotate_lines(lines: List[str], cache: ChemMetricCache) -> Tuple[List[str], Dict[str, int]]:
    parsed: List[List[str]] = []
    smiles_set = set()
    for line in lines:
        parts = _parse_line(line)
        if not parts:
            continue
        parsed.append(parts)
        smiles_set.add(parts[0])

    for smi in smiles_set:
        cache.get_or_compute(smi)
    cache.flush()

    out_lines: List[str] = []
    for parts in parsed:
        smi = parts[0]
        ds = parts[1] if len(parts) >= 2 else ""
        qed, sa = cache.get(smi)

        extra: List[str] = []
        if len(parts) >= 4:
            extra = parts[4:]
        elif len(parts) == 3:
            extra = parts[2:]

        new_parts = [smi, ds, _fmt(qed), _fmt(sa)] + extra
        out_lines.append("\t".join(new_parts) + "\n")

    stats = {"lines_in": len(parsed), "unique_smiles": len(smiles_set), "lines_out": len(out_lines)}
    return out_lines, stats


def main():
    p = argparse.ArgumentParser(description="Append QED/SA to each molecule line")
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--cache_file", default=None, help="(optional) ChemMetricCache json path")
    args = p.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    cache_path = Path(args.cache_file) if args.cache_file else None
    cache = ChemMetricCache(cache_path)

    lines = input_path.read_text(encoding="utf-8").splitlines(True)
    out_lines, stats = _annotate_lines(lines, cache)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(f"{output_path.suffix}.tmp_{os.getpid()}")
    tmp.write_text("".join(out_lines), encoding="utf-8")
    tmp.replace(output_path)

    print(f"OK lines={stats['lines_out']} unique={stats['unique_smiles']}")


if __name__ == "__main__":
    main()
