#!/usr/bin/env python3
"""Evaluate CSV with TopK docking/QED/SA statistics and save sorted outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SUMMARY_COLUMNS = [
    "input_csv",
    "n_raw",
    "n_clean",
    "n_dedup",
    "n_final",
    "k_top100",
    "k_top10",
    "docking_mean",
    "docking_top100_mean",
    "docking_top10_mean",
    "docking_top1",
    "qed_top100_mean",
    "sa_top100_mean",
    "qed_top10_mean",
    "sa_top10_mean",
]


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...], name: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing {name} column, candidates={candidates}, got={list(df.columns)}")


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.mean())


def _resolve_output(path: str | None, default: Path) -> Path:
    return Path(path).expanduser().resolve() if path else default


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", type=str, required=True, help="Input CSV file")
    parser.add_argument(
        "-o",
        "--output_sorted",
        type=str,
        default=None,
        help="Output path for sorted CSV (default: <input_stem>_sorted.csv)",
    )
    parser.add_argument(
        "-s",
        "--output_summary",
        type=str,
        default=None,
        help="Output path for summary CSV (default: <input_stem>_summary.csv)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_sorted = _resolve_output(args.output_sorted, input_path.with_name(f"{input_path.stem}_sorted.csv"))
    output_summary = _resolve_output(args.output_summary, input_path.with_name(f"{input_path.stem}_summary.csv"))

    df_raw = pd.read_csv(input_path)
    n_raw = int(len(df_raw))

    smiles_col = _pick_column(df_raw, ("smiles", "smi", "SMILES"), "SMILES")
    docking_col = _pick_column(df_raw, ("docking_score", "rv", "total", "score", "ds"), "docking score")
    if "qed" not in df_raw.columns or "sa" not in df_raw.columns:
        raise ValueError(f"Input CSV must contain qed/sa columns, got={list(df_raw.columns)}")

    df = df_raw.dropna(subset=[smiles_col, docking_col, "qed", "sa"]).copy()
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df[df[smiles_col] != ""]
    df[docking_col] = pd.to_numeric(df[docking_col], errors="coerce")
    df["qed"] = pd.to_numeric(df["qed"], errors="coerce")
    df["sa"] = pd.to_numeric(df["sa"], errors="coerce")
    df = df.dropna(subset=[docking_col, "qed", "sa"])
    n_clean = int(len(df))

    if not df.empty:
        best_idx = df.groupby(smiles_col, sort=False)[docking_col].idxmin()
        df = df.loc[best_idx].copy()
    n_dedup = int(len(df))

    df = df.sort_values(by=docking_col, ascending=True).reset_index(drop=True)
    n_final = int(len(df))

    k_top100 = min(100, n_final)
    k_top10 = min(10, n_final)
    top100 = df.head(k_top100)
    top10 = df.head(k_top10)

    docking_mean = _safe_mean(df[docking_col])
    docking_top100_mean = _safe_mean(top100[docking_col])
    docking_top10_mean = _safe_mean(top10[docking_col])
    docking_top1 = float(df[docking_col].iloc[0]) if n_final > 0 else float("nan")
    qed_top100_mean = _safe_mean(top100["qed"])
    sa_top100_mean = _safe_mean(top100["sa"])
    qed_top10_mean = _safe_mean(top10["qed"])
    sa_top10_mean = _safe_mean(top10["sa"])

    summary_row = {
        "input_csv": str(input_path),
        "n_raw": n_raw,
        "n_clean": n_clean,
        "n_dedup": n_dedup,
        "n_final": n_final,
        "k_top100": k_top100,
        "k_top10": k_top10,
        "docking_mean": docking_mean,
        "docking_top100_mean": docking_top100_mean,
        "docking_top10_mean": docking_top10_mean,
        "docking_top1": docking_top1,
        "qed_top100_mean": qed_top100_mean,
        "sa_top100_mean": sa_top100_mean,
        "qed_top10_mean": qed_top10_mean,
        "sa_top10_mean": sa_top10_mean,
    }

    output_sorted.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_sorted, index=False)
    pd.DataFrame([summary_row], columns=SUMMARY_COLUMNS).to_csv(output_summary, index=False)

    print(f"Input CSV:\t\t\t{input_path}")
    print(f"Detected columns:\t\tSMILES={smiles_col}, docking={docking_col}, qed=qed, sa=sa")
    print(f"Rows raw/clean/dedup/final:\t{n_raw}/{n_clean}/{n_dedup}/{n_final}")
    print("=" * 60)
    print(
        "docking_score: "
        f"mean={docking_mean:.6f}; top100_mean={docking_top100_mean:.6f}; "
        f"top10_mean={docking_top10_mean:.6f}; top1={docking_top1:.6f}"
    )
    print(
        f"Top100 (k={k_top100}) QED/SA:\t{qed_top100_mean:.4f}/{sa_top100_mean:.4f}\n"
        f"Top10  (k={k_top10}) QED/SA:\t{qed_top10_mean:.4f}/{sa_top10_mean:.4f}"
    )
    print(f"Saved sorted CSV:\t\t{output_sorted}")
    print(f"Saved summary CSV:\t\t{output_summary}")


if __name__ == "__main__":
    main()
