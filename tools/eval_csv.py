"""Evaluate SoftGA summary CSV.
Pipeline: clean -> deduplicate -> QED/SA filter -> novelty filter -> docking hit filter -> top5% stats.
Expected columns: smiles, qed, sa, docking_score
"""

import argparse
import math
from pathlib import Path

import pandas as pd
from rdkit import Chem
from tdc import Evaluator

MAX_ROWS = 3000
QED_THR = 0.5
SA_THR = 5.0
TOP_FRAC = 0.05
DIVERSITY_EVALUATOR = Evaluator("diversity")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INITIAL_POP = PROJECT_ROOT / "utils/initial_population/initial_population.smi"

# Hard-coded docking thresholds (lower is better, i.e. more negative is better)
HIT_DOCKING_THR_BY_TARGET: dict[str, float] = {
    "parp1": -10.0,
    "fa7": -8.5,
    "5ht1b": -8.7845,
    "braf": -10.3,
    "jak2": -9.1,
}


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...], name: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing {name} column, candidates={candidates}, got={list(df.columns)}")


def _canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def _load_initial_population(path: Path) -> set[str]:
    initial_smiles: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            canonical = _canonicalize_smiles(parts[0].strip())
            if canonical:
                initial_smiles.add(canonical)
    return initial_smiles


def _compute_diversity(smiles_list: list[str]) -> float:
    valid = [s for s in smiles_list if Chem.MolFromSmiles(s)]
    uniq = list(set(valid))
    return float(DIVERSITY_EVALUATOR(uniq)) if len(uniq) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", type=str, default="output/summary.csv")
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="parp1",
        choices=["parp1", "fa7", "5ht1b", "braf", "jak2"],
    )
    parser.add_argument(
        "--initial-population-file",
        type=Path,
        default=DEFAULT_INITIAL_POP,
        help="Novelty reference SMILES file",
    )
    args = parser.parse_args()

    hit_thr = HIT_DOCKING_THR_BY_TARGET[args.target]
    path = Path(args.input)
    initial_pop_path = args.initial_population_file.resolve()
    if not initial_pop_path.exists():
        raise FileNotFoundError(f"Initial population file not found: {initial_pop_path}")
    initial_population = _load_initial_population(initial_pop_path)
    if not initial_population:
        raise ValueError(f"No valid SMILES found in initial population file: {initial_pop_path}")

    df = pd.read_csv(path).iloc[:MAX_ROWS].copy()
    print(f"Raw samples (first {MAX_ROWS} rows):\t{len(df)}")
    print(f"Novelty reference molecules:\t{len(initial_population)}")

    smi_col = _pick_column(df, ("smiles", "smi"), "SMILES")
    dock_col = _pick_column(df, ("docking_score", "rv"), "docking score")
    if "qed" not in df.columns or "sa" not in df.columns:
        raise ValueError(f"Input CSV must contain qed/sa columns, got={list(df.columns)}")

    df = df.dropna(subset=[smi_col, dock_col, "qed", "sa"])
    df[smi_col] = df[smi_col].astype(str).str.strip()
    df = df[df[smi_col] != ""]
    df = df.drop_duplicates(subset=[smi_col])
    df[dock_col] = pd.to_numeric(df[dock_col], errors="coerce")
    df["qed"] = pd.to_numeric(df["qed"], errors="coerce")
    df["sa"] = pd.to_numeric(df["sa"], errors="coerce")
    df = df.dropna(subset=[dock_col, "qed", "sa"])
    df["canonical_smiles"] = df[smi_col].map(_canonicalize_smiles)
    df = df.dropna(subset=["canonical_smiles"])
    df["is_novel"] = ~df["canonical_smiles"].isin(initial_population)

    n_total = len(df)
    print(f"Deduplicated samples:\t\t{n_total}")
    if n_total == 0:
        return
    print(f"Diversity (deduplicated):\t{_compute_diversity(df[smi_col].tolist()):.4f}")

    df_qs = df[(df["qed"] > QED_THR) & (df["sa"] < SA_THR)]
    print(f"After QED/SA filter:\t\t{len(df_qs)}/{n_total}\t({len(df_qs) / n_total:.2%})")
    if df_qs.empty:
        return

    df_qsn = df_qs[df_qs["is_novel"]]
    print(
        f"After Novelty filter:\t\t{len(df_qsn)}/{len(df_qs)}\t"
        f"({len(df_qsn) / len(df_qs):.2%})"
    )
    if df_qsn.empty:
        return

    df_hit = df_qsn[df_qsn[dock_col] < hit_thr]
    n_hit = len(df_hit)
    print(
        f"Hit (Novel, QED>{QED_THR}, SA<{SA_THR}, docking<{hit_thr:.4g} @ {args.target}):\t"
        f"{n_hit}/{n_total}\t({n_hit / n_total:.2%})"
    )
    if df_hit.empty:
        return
    print(f"Diversity (hit set):\t\t{_compute_diversity(df_hit[smi_col].tolist()):.4f}")

    df_hit = df_hit.sort_values(by=dock_col, ascending=True)
    docking_vals = df_hit[dock_col].to_numpy()
    docking_mean = float(docking_vals.mean())
    docking_median = float(df_hit[dock_col].median())
    docking_top1 = float(docking_vals[0])
    k_top5 = max(1, int(math.ceil(len(docking_vals) * TOP_FRAC)))
    docking_top5_mean = float(docking_vals[:k_top5].mean())
    qed_median = float(df_hit["qed"].median())
    sa_median = float(df_hit["sa"].median())

    print("=" * 50)
    print(
        f"docking_score: top5%: {docking_top5_mean:.6f}; "
        f"mean: {docking_mean:.6f}; median: {docking_median:.6f}; top1: {docking_top1:.6f}"
    )
    print(f"Hit median QED/SA:\t\t{qed_median:.4f}/{sa_median:.4f}")

    top5 = df_hit.head(k_top5).drop(columns=["canonical_smiles", "is_novel"], errors="ignore")
    print(f"top5%  QED:\t\t{float(top5['qed'].mean()):.4f}")
    print(f"top5%  SA:\t\t{float(top5['sa'].mean()):.4f}")
    print(f"top5%  Diversity:\t\t{_compute_diversity(top5[smi_col].tolist()):.4f}")

    top5_path = path.with_name(path.stem + "_top5.csv")
    top5.to_csv(top5_path, index=False)
    print(f"Saved top5% samples to:\t\t{top5_path}")


if __name__ == "__main__":
    main()
