#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

TARGETS: tuple[str, ...] = ("parp1", "fa7", "5ht1b", "braf", "jak2")
SEEDS: tuple[int, ...] = (0, 1, 2)
QED_THR = 0.5
SA_RAW_THR = 5.0
HIT_DOCKING_THR_BY_TARGET: dict[str, float] = {
    "parp1": -10.0,
    "fa7": -8.5,
    "5ht1b": -8.7845,
    "braf": -10.3,
    "jak2": -9.1,
}

METHOD_SPECS = (
    {
        "key": "f-rag",
        "latex_label": "f-rag",
        "build_path": lambda root, seed, target: root / f"seed{seed}" / f"{target}_{seed}.csv",
    },
    {
        "key": "genmol",
        "latex_label": "GenMol",
        "build_path": lambda root, seed, target: root / f"seed{seed}" / f"genmol_{target}_merged.csv",
    },
)


def _canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...], name: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing {name} column, candidates={candidates}, got={list(df.columns)}")


def _load_baseline_csv(path: Path, *, max_rows: int) -> pd.DataFrame:
    df = pd.read_csv(path).iloc[:max_rows].copy()
    smi_col = _pick_column(df, ("SMILES", "smiles", "smi"), "SMILES")
    dock_col = _pick_column(df, ("DS", "docking_score", "rv"), "docking score")
    qed_col = _pick_column(df, ("QED", "qed"), "QED")
    sa_col = _pick_column(df, ("SA", "sa"), "SA")

    df = df.dropna(subset=[smi_col, dock_col, qed_col, sa_col]).copy()
    df[smi_col] = df[smi_col].astype(str).str.strip()
    df = df[df[smi_col] != ""].copy()
    df[dock_col] = pd.to_numeric(df[dock_col], errors="coerce")
    df[qed_col] = pd.to_numeric(df[qed_col], errors="coerce")
    df[sa_col] = pd.to_numeric(df[sa_col], errors="coerce")
    df = df.dropna(subset=[dock_col, qed_col, sa_col]).copy()

    df["canonical_smiles"] = df[smi_col].map(_canonicalize_smiles)
    df = df.dropna(subset=["canonical_smiles"]).copy()
    df = df.drop_duplicates(subset=["canonical_smiles"]).reset_index(drop=True)

    out = pd.DataFrame(
        {
            "smiles": df["canonical_smiles"],
            "qed": df[qed_col],
            "sa_raw": 10.0 - 9.0 * df[sa_col],
            "docking_score": -df[dock_col],
        }
    )
    return out


def _evaluate_file(path: Path, *, target: str, max_rows: int) -> dict[str, float | int | str]:
    df = _load_baseline_csv(path, max_rows=max_rows)
    n_total_clean = len(df)
    if n_total_clean == 0:
        return {
            "n_total_clean": 0,
            "n_hit": 0,
            "hit_ratio": 0.0,
            "sa_raw_min": float("nan"),
            "sa_raw_max": float("nan"),
            "docking_score_min": float("nan"),
            "docking_score_max": float("nan"),
        }

    hit_thr = HIT_DOCKING_THR_BY_TARGET[target]
    hit_mask = (
        (df["qed"] > QED_THR)
        & (df["sa_raw"] < SA_RAW_THR)
        & (df["docking_score"] < hit_thr)
    )
    n_hit = int(hit_mask.sum())
    return {
        "n_total_clean": n_total_clean,
        "n_hit": n_hit,
        "hit_ratio": 100.0 * n_hit / n_total_clean,
        "sa_raw_min": float(df["sa_raw"].min()),
        "sa_raw_max": float(df["sa_raw"].max()),
        "docking_score_min": float(df["docking_score"].min()),
        "docking_score_max": float(df["docking_score"].max()),
    }


def _format_pm(mean: float, std: float) -> str:
    return f"{mean:.3f} $\\\\pm$ {std:.3f}"


def _build_method_summary(
    *,
    method_key: str,
    latex_label: str,
    root: Path,
    build_path,
    max_rows: int,
) -> tuple[list[dict[str, object]], str]:
    rows: list[dict[str, object]] = []
    latex_cells: list[str] = []

    for target in TARGETS:
        per_seed: list[dict[str, float | int | str]] = []
        for seed in SEEDS:
            path = build_path(root, seed, target)
            if not path.exists():
                raise FileNotFoundError(path)
            metrics = _evaluate_file(path, target=target, max_rows=max_rows)
            metrics.update(
                {
                    "method": method_key,
                    "target": target,
                    "seed": seed,
                    "path": str(path),
                }
            )
            per_seed.append(metrics)

        ratios = pd.Series([float(item["hit_ratio"]) for item in per_seed], dtype=float)
        std = float(ratios.std(ddof=1)) if len(ratios) > 1 else 0.0
        row: dict[str, object] = {
            "method": method_key,
            "target": target,
            "mean": float(ratios.mean()),
            "std": std,
        }
        for item in per_seed:
            seed = int(item["seed"])
            row[f"seed{seed}_ratio"] = float(item["hit_ratio"])
            row[f"seed{seed}_n_hit"] = int(item["n_hit"])
            row[f"seed{seed}_n_total_clean"] = int(item["n_total_clean"])
            row[f"seed{seed}_sa_raw_min"] = float(item["sa_raw_min"])
            row[f"seed{seed}_sa_raw_max"] = float(item["sa_raw_max"])
            row[f"seed{seed}_docking_score_min"] = float(item["docking_score_min"])
            row[f"seed{seed}_docking_score_max"] = float(item["docking_score_max"])
            row[f"seed{seed}_path"] = str(item["path"])
        rows.append(row)
        latex_cells.append(_format_pm(float(row["mean"]), float(row["std"])))

    latex_row = f"{latex_label} & " + " & ".join(latex_cells) + r" \\"
    return rows, latex_row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute hit ratio for f-rag and GenMol baselines on the SoftMol benchmark."
    )
    parser.add_argument(
        "--f-rag-root",
        type=Path,
        default=Path("results/softmol/f-rag/original"),
    )
    parser.add_argument(
        "--genmol-root",
        type=Path,
        default=Path("results/softmol/genmol/original"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/softmol/hit_ratio_summary.csv"),
    )
    parser.add_argument("--max-rows", type=int, default=3000)
    args = parser.parse_args()

    roots = {
        "f-rag": args.f_rag_root.resolve(),
        "genmol": args.genmol_root.resolve(),
    }

    all_rows: list[dict[str, object]] = []
    latex_rows: list[str] = []
    for spec in METHOD_SPECS:
        rows, latex_row = _build_method_summary(
            method_key=spec["key"],
            latex_label=spec["latex_label"],
            root=roots[spec["key"]],
            build_path=spec["build_path"],
            max_rows=args.max_rows,
        )
        all_rows.extend(rows)
        latex_rows.append(latex_row)

    summary = pd.DataFrame(all_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)

    print(f"Saved hit ratio summary to: {args.output_csv}")
    print()
    print("LaTeX rows:")
    for row in latex_rows:
        print(row)


if __name__ == "__main__":
    main()
