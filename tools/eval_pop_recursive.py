#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem
from tdc import Evaluator, Oracle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INITIAL_POP = PROJECT_ROOT / "utils/initial_population/initial_population.smi"

QED_ORACLE = Oracle("qed")
SA_ORACLE = Oracle("sa")
DIVERSITY_EVAL = Evaluator(name="Diversity")
NOVELTY_EVAL = Evaluator(name="Novelty")

METRIC_COLUMNS = ["top_1", "top_10", "top_100", "novelty", "diversity", "qed", "sa"]


def _pick_column(df: pd.DataFrame, candidates: Iterable[str], name: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing column {name}, candidate column = {tuple(candidates)}, current column = {list(df.columns)}")


def _is_valid_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _load_initial_smiles(path: Path) -> list[str]:
    seen: set[str] = set()
    smiles_list: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            smiles = parts[0].strip()
            if not smiles or smiles in seen:
                continue
            if _is_valid_smiles(smiles):
                seen.add(smiles)
                smiles_list.append(smiles)
    return smiles_list


def _load_and_rank_pop(path: Path) -> tuple[list[str], list[float]]:
    df = pd.read_csv(path)
    smiles_col = _pick_column(df, ("smiles", "smi", "SMILES"), "SMILES")
    docking_col = _pick_column(df, ("docking_score", "total", "score"), "docking_score")

    df = df[[smiles_col, docking_col]].copy()
    df = df.dropna(subset=[smiles_col, docking_col])
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df[df[smiles_col] != ""]
    df[docking_col] = pd.to_numeric(df[docking_col], errors="coerce")
    df = df.dropna(subset=[docking_col])

    if df.empty:
        return [], []

    best_idx = df.groupby(smiles_col, sort=False)[docking_col].idxmin()
    df = df.loc[best_idx]
    df = df.sort_values(by=docking_col, ascending=True).reset_index(drop=True)

    return df[smiles_col].tolist(), df[docking_col].astype(float).tolist()


def _compute_metrics(smiles_sorted: list[str], docking_sorted: list[float], initial_smiles: list[str]) -> dict[str, float]:
    n_total = len(docking_sorted)
    n10 = min(10, n_total)
    n100 = min(100, n_total)

    top_1 = docking_sorted[0] if n_total else float("nan")
    top_10 = _safe_mean(docking_sorted[:n10])
    top_100 = _safe_mean(docking_sorted[:n100])

    top100_smiles = smiles_sorted[:n100]
    valid_top100 = [s for s in top100_smiles if _is_valid_smiles(s)]

    if valid_top100:
        novelty = float(NOVELTY_EVAL(valid_top100, initial_smiles)) if initial_smiles else float("nan")
        diversity = float(DIVERSITY_EVAL(valid_top100)) if len(valid_top100) > 1 else 0.0
        qed = _safe_mean([float(x) for x in QED_ORACLE(valid_top100)])
        sa = _safe_mean([float(x) for x in SA_ORACLE(valid_top100)])
    else:
        novelty = float("nan")
        diversity = float("nan")
        qed = float("nan")
        sa = float("nan")

    return {
        "top_1": top_1,
        "top_10": top_10,
        "top_100": top_100,
        "novelty": novelty,
        "diversity": diversity,
        "qed": qed,
        "sa": sa,
        "n_total": float(n_total),
        "n_top100_valid": float(len(valid_top100)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recursively evaluate all pop.csv in the directory and summarize the average")
    parser.add_argument("-i", "--input_dir", required=True, help="Enter the directory and search recursively for pop.csv")
    parser.add_argument(
        "--initial_population_file",
        default=str(DEFAULT_INITIAL_POP),
        help="Initial population SMILES file used for Novelty calculations",
    )
    parser.add_argument(
        "--output_prefix",
        default="eval_pop",
        help="Output file prefix (*_per_file.csv and *_mean.csv will be output)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    initial_pop_file = Path(args.initial_population_file).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not initial_pop_file.exists():
        raise FileNotFoundError(f"Initial population file does not exist: {initial_pop_file}")

    pop_files = sorted(input_dir.rglob("pop.csv"))
    if not pop_files:
        print(f"pop.csv not found in directory: {input_dir}")
        return

    initial_smiles = _load_initial_smiles(initial_pop_file)
    if not initial_smiles:
        raise ValueError(f"No SMILES available in initial population file: {initial_pop_file}")

    rows: list[dict[str, float | str]] = []
    for pop_file in pop_files:
        smiles_sorted, docking_sorted = _load_and_rank_pop(pop_file)
        metrics = _compute_metrics(smiles_sorted, docking_sorted, initial_smiles)
        row: dict[str, float | str] = {
            "pop_csv": str(pop_file),
            "n_total": metrics["n_total"],
            "n_top100_valid": metrics["n_top100_valid"],
        }
        for col in METRIC_COLUMNS:
            row[col] = metrics[col]
        rows.append(row)

    per_file_df = pd.DataFrame(rows)
    mean_row = {"pop_csv": "MEAN", "n_total": per_file_df["n_total"].mean(), "n_top100_valid": per_file_df["n_top100_valid"].mean()}
    for col in METRIC_COLUMNS:
        mean_row[col] = per_file_df[col].mean()
    mean_df = pd.DataFrame([mean_row])

    out_per_file = input_dir / f"{args.output_prefix}_per_file.csv"
    out_mean = input_dir / f"{args.output_prefix}_mean.csv"
    per_file_df.to_csv(out_per_file, index=False)
    mean_df.to_csv(out_mean, index=False)

    print(f"Input directory: {input_dir}")
    print(f"Initial population: {initial_pop_file}")
    print(f"Total number of pop.csv found: {len(pop_files)}")
    print("\\n[Indicators per pop.csv]")
    print(per_file_df.to_string(index=False, justify="left"))
    print("\\n[Mean of all pop.csv]")
    print(mean_df.to_string(index=False, justify="left"))
    print(f"\nSaved: {out_per_file}")
    print(f"Saved: {out_mean}")


if __name__ == "__main__":
    main()
