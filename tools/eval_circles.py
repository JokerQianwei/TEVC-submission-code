#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Calculate the "#Circles of generated hit molecules" for SoftGA CSV.

Current input CSV adaptation:
- Listing support: smiles/smi/SMILES, docking_score/rv, qed/QED, sa/SA
- Only the first 3000 entries are evaluated by default

hit definition (default):
- SMILES deduplication + RDKit parsable
- QED > 0.5
- SA < 5.0
- docking_score < target threshold (smaller is better) """

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

TARGETS: tuple[str, ...] = ("parp1", "fa7", "5ht1b", "braf", "jak2")

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


def _load_softga_csv(path: Path, *, max_rows: int) -> pd.DataFrame:
    df = pd.read_csv(path).iloc[:max_rows].copy()
    smi_col = _pick_column(df, ("smiles", "smi", "SMILES"), "SMILES")
    dock_col = _pick_column(df, ("docking_score", "rv"), "docking score")
    qed_col = _pick_column(df, ("qed", "QED"), "QED")
    sa_col = _pick_column(df, ("sa", "SA"), "SA")

    df = df.dropna(subset=[smi_col, dock_col, qed_col, sa_col]).copy()
    df[smi_col] = df[smi_col].astype(str).str.strip()
    df = df[df[smi_col] != ""]
    df = df.drop_duplicates(subset=[smi_col]).reset_index(drop=True)

    df[dock_col] = pd.to_numeric(df[dock_col], errors="coerce")
    df[qed_col] = pd.to_numeric(df[qed_col], errors="coerce")
    df[sa_col] = pd.to_numeric(df[sa_col], errors="coerce")
    df = df.dropna(subset=[dock_col, qed_col, sa_col]).copy()

    df = df.rename(columns={smi_col: "smi", dock_col: "dock", qed_col: "qed", sa_col: "sa"}).copy()
    df["mol"] = df["smi"].apply(Chem.MolFromSmiles)
    df = df.dropna(subset=["mol"]).drop(columns=["mol"]).reset_index(drop=True)
    return df[["smi", "dock", "qed", "sa"]]


def _morgan_fps(smiles: Sequence[str], *, radius: int = 2, nbits: int = 1024) -> List[DataStructs.ExplicitBitVect]:
    fps: List[DataStructs.ExplicitBitVect] = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits))
    return fps


def _divide_evenly(n: int, seq: Sequence) -> List[List]:
    n = max(1, int(n))
    length = len(seq)
    q, r = divmod(length, n)
    out: List[List] = []
    start = 0
    for i in range(n):
        size = q + (1 if i < r else 0)
        out.append(list(seq[start : start + size]))
        start += size
    return out


class NCircles:
    def __init__(self, *, threshold: float = 0.75) -> None:
        self.t = float(threshold)

    def _get_circles(self, fps: Iterable[DataStructs.ExplicitBitVect]) -> List[DataStructs.ExplicitBitVect]:
        circles: List[DataStructs.ExplicitBitVect] = []
        for fp in fps:
            if circles:
                sims = DataStructs.BulkTanimotoSimilarity(fp, circles)
                if (1.0 - max(sims)) <= self.t:
                    continue
            circles.append(fp)
        return circles

    def measure(self, fps: List[DataStructs.ExplicitBitVect], *, n_chunk: int = 64) -> int:
        vecs: List[DataStructs.ExplicitBitVect] = list(fps)
        for i in range(3):
            n = max(1, n_chunk // (2**i))
            chunks = _divide_evenly(n, vecs)
            circles_list = [self._get_circles(chunk) for chunk in chunks]
            vecs = [c for ls in circles_list for c in ls]
            random.shuffle(vecs)
        vecs = self._get_circles(vecs)
        return len(vecs)


def main() -> None:
    parser = argparse.ArgumentParser(description="SoftGA CSV: Calculate #Circles (hit set)")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Enter CSV")
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        required=True,
        choices=list(TARGETS),
        help="Target receptor for hitting docking threshold",
    )
    parser.add_argument("--max-rows", type=int, default=3000)
    parser.add_argument("--qed-thr", type=float, default=0.5)
    parser.add_argument("--sa-thr", type=float, default=5.0)
    parser.add_argument("--circle-dist-thr", type=float, default=0.75)
    parser.add_argument("--circle-n-chunk", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    hit_thr = HIT_DOCKING_THR_BY_TARGET[args.target]
    nc = NCircles(threshold=args.circle_dist_thr)

    df_raw = pd.read_csv(args.input).iloc[: args.max_rows].copy()
    print(f"Raw samples (first {args.max_rows} rows):\t{len(df_raw)}")

    df = _load_softga_csv(args.input, max_rows=args.max_rows)
    n_dedup = len(df)
    print(f"Deduplicated valid samples:\t\t{n_dedup}")
    if n_dedup == 0:
        print("#Circles (hit set):\t\t0")
        return

    df_hit = df[(df["qed"] > args.qed_thr) & (df["sa"] < args.sa_thr) & (df["dock"] < hit_thr)].copy()
    n_hit = len(df_hit)
    print(
        f"Hit (QED>{args.qed_thr}, SA<{args.sa_thr}, docking<{hit_thr:.4g} @ {args.target}):\t"
        f"{n_hit}/{n_dedup}\t({n_hit / n_dedup:.2%})"
    )
    if df_hit.empty:
        print("#Circles (hit set):\t\t0")
        return

    fps = _morgan_fps(df_hit["smi"].tolist())
    random.seed(args.seed)
    n_circle = nc.measure(fps, n_chunk=args.circle_n_chunk) if fps else 0
    print(f"#Circles (hit set):\t\t{n_circle}")
    print(
        f"Params: circle_dist_thr={args.circle_dist_thr}, "
        f"circle_n_chunk={args.circle_n_chunk}, seed={args.seed}"
    )


if __name__ == "__main__":
    main()
