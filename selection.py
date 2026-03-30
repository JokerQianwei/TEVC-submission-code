#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" SoftGA selection strategy
1. ffhs: First select feasible molecules (QED/SA constraints), and if insufficient, use NSGA-II to complete the infeasible set
2. nsgaii: All candidates are directly selected using NSGA-II. """

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from utils.chem_metrics import ChemMetricCache
from utils.config_loader import load_config, resolve_config_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _normalize_selection_mode(mode: Optional[str]) -> str:
    return "nsgaii" if str(mode or "").strip().lower() == "nsgaii" else "ffhs"


def _default_objectives() -> List[Dict]:
    return [
        {"name": "docking_score", "direction": "minimize"},
        {"name": "qed_score", "direction": "maximize"},
        {"name": "sa_score", "direction": "minimize"},
    ]


def _load_selection_cfg(config_file: Optional[str]) -> Dict:
    if not config_file:
        return {}
    path = resolve_config_path(config_file, Path(__file__).resolve().parent)
    cfg = load_config(str(path), Path(__file__).resolve().parent)
    return cfg.get("selection", {}) or {}


def _load_docked(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            smi = parts[0]
            try:
                score = float(parts[1])
            except ValueError:
                continue
            rows.append({"smiles": smi, "docking_score": score})
    return rows


def _merge_parent_child(parent_file: Optional[str], child_file: str) -> List[Dict]:
    merged: Dict[str, Dict] = {}
    for path in [parent_file, child_file]:
        if not path:
            continue
        for row in _load_docked(path):
            smi = row["smiles"]
            if smi not in merged or row["docking_score"] < merged[smi]["docking_score"]:
                merged[smi] = row
    return list(merged.values())


def _add_qed_sa(rows: List[Dict], cache: ChemMetricCache) -> List[Dict]:
    for row in rows:
        qed, sa = cache.get_or_compute(row["smiles"])
        row["qed_score"] = 0.0 if qed is None else float(qed)
        row["sa_score"] = 10.0 if sa is None else float(sa)
    cache.flush()
    return rows


def _objective_matrix(rows: List[Dict], objectives_cfg: List[Dict]) -> np.ndarray:
    if not objectives_cfg:
        objectives_cfg = _default_objectives()

    mat: List[List[float]] = []
    for row in rows:
        vals = []
        for obj in objectives_cfg:
            name = obj.get("name", "").strip()
            direction = obj.get("direction", "minimize").strip().lower()
            value = float(row.get(name, 0.0))
            if direction == "maximize":
                value = -value
            vals.append(value)
        mat.append(vals)
    return np.array(mat, dtype=float)


def _non_dominated_sort(objs: np.ndarray) -> List[List[int]]:
    n = len(objs)
    if n == 0:
        return []

    dominates = [[] for _ in range(n)]
    dominated_count = np.zeros(n, dtype=int)
    fronts: List[List[int]] = []

    for i in range(n):
        for j in range(i + 1, n):
            i_dom_j = np.all(objs[i] <= objs[j]) and np.any(objs[i] < objs[j])
            j_dom_i = np.all(objs[j] <= objs[i]) and np.any(objs[j] < objs[i])
            if i_dom_j:
                dominates[i].append(j)
                dominated_count[j] += 1
            elif j_dom_i:
                dominates[j].append(i)
                dominated_count[i] += 1

    cur = np.where(dominated_count == 0)[0].tolist()
    while cur:
        fronts.append(cur)
        nxt = []
        for i in cur:
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    nxt.append(j)
        cur = nxt
    return fronts


def _crowding_distance(objs: np.ndarray, front: List[int]) -> np.ndarray:
    n = len(front)
    if n == 0:
        return np.array([])
    if n <= 2:
        return np.array([np.inf] * n)

    sub = objs[front, :]
    dist = np.zeros(n)
    m = sub.shape[1]
    for k in range(m):
        idx = np.argsort(sub[:, k])
        dist[idx[0]] = np.inf
        dist[idx[-1]] = np.inf
        lo, hi = sub[idx[0], k], sub[idx[-1], k]
        if hi == lo:
            continue
        for i in range(1, n - 1):
            dist[idx[i]] += (sub[idx[i + 1], k] - sub[idx[i - 1], k]) / (hi - lo)
    return dist


def _select_nsga2(rows: List[Dict], n_select: int, objectives_cfg: List[Dict], enable_crowding: bool) -> List[Dict]:
    if not rows or n_select <= 0:
        return []

    objs = _objective_matrix(rows, objectives_cfg)
    fronts = _non_dominated_sort(objs)
    selected: List[Dict] = []

    for rank, front in enumerate(fronts, start=1):
        for idx in front:
            rows[idx]["front_rank"] = rank

        if len(selected) + len(front) <= n_select:
            if enable_crowding:
                dist = _crowding_distance(objs, front)
                ordered = [x for x, _ in sorted(zip(front, dist), key=lambda t: (-t[1], t[0]))]
            else:
                ordered = sorted(front, key=lambda i: objs[i, 0])
            selected.extend(rows[i] for i in ordered)
            continue

        remain = n_select - len(selected)
        if enable_crowding:
            dist = _crowding_distance(objs, front)
            ordered = [x for x, _ in sorted(zip(front, dist), key=lambda t: (-t[1], t[0]))]
        else:
            ordered = sorted(front, key=lambda i: objs[i, 0])
        selected.extend(rows[i] for i in ordered[:remain])
        break

    return selected


def _select_ffhs(
    rows: List[Dict],
    n_select: int,
    objectives_cfg: List[Dict],
    enable_crowding: bool,
    qed_min: float,
    sa_max: float,
) -> List[Dict]:
    for row in rows:
        qed = float(row.get("qed_score", 0.0))
        sa = float(row.get("sa_score", 10.0))
        row["is_feasible"] = bool(qed >= qed_min and sa <= sa_max)
        row["constraint_violation"] = max(0.0, qed_min - qed) + max(0.0, sa - sa_max)

    feasible = sorted(
        (r for r in rows if r["is_feasible"]),
        key=lambda r: float(r.get("docking_score", 0.0)),
    )
    if len(feasible) >= n_select:
        for row in feasible[:n_select]:
            row["front_rank"] = 0
        return feasible[:n_select]

    selected = list(feasible)
    remain = n_select - len(selected)
    infeasible = [r for r in rows if not r["is_feasible"]]
    if remain > 0 and infeasible:
        selected.extend(_select_nsga2(infeasible, remain, objectives_cfg, enable_crowding))
    return selected


def _select_nsgaii(rows: List[Dict], n_select: int, objectives_cfg: List[Dict], enable_crowding: bool) -> List[Dict]:
    for row in rows:
        row.setdefault("is_feasible", False)
        row.setdefault("constraint_violation", 0.0)
    return _select_nsga2(rows, n_select, objectives_cfg, enable_crowding)


def select_molecules(
    rows: List[Dict],
    n_select: int,
    objectives_cfg: List[Dict],
    enable_crowding: bool,
    qed_min: float,
    sa_max: float,
    selection_mode: str = "ffhs",
) -> List[Dict]:
    if _normalize_selection_mode(selection_mode) == "nsgaii":
        return _select_nsgaii(rows, n_select, objectives_cfg, enable_crowding)
    return _select_ffhs(rows, n_select, objectives_cfg, enable_crowding, qed_min, sa_max)


def _save_smiles_only(rows: List[Dict], out: str) -> None:
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(f"{row['smiles']}\n")


def _save_with_scores(rows: List[Dict], out: str) -> None:
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                f"{row['smiles']}\t{float(row['docking_score']):.6f}\t{float(row['qed_score']):.4f}\t"
                f"{float(row['sa_score']):.4f}\n"
            )


def _save_front_report(rows: List[Dict], out: str) -> None:
    with open(out, "w", encoding="utf-8") as f:
        f.write("smiles\tfront_rank\tis_feasible\tconstraint_violation\tdocking_score\tqed_score\tsa_score\n")
        for row in rows:
            f.write(
                f"{row['smiles']}\t{row.get('front_rank', '')}\t{row.get('is_feasible', False)}\t"
                f"{float(row.get('constraint_violation', 0.0)):.6f}\t{float(row['docking_score']):.6f}\t"
                f"{float(row['qed_score']):.4f}\t{float(row['sa_score']):.4f}\n"
            )


def _print_stats(selected: List[Dict], selection_mode: str) -> None:
    if not selected:
        logger.warning("No molecules selected")
        return
    scores = [float(r["docking_score"]) for r in selected]
    logger.info(
        "Selection completed (%s): %d molecules, docking[min=%.4f, max=%.4f, mean=%.4f]",
        selection_mode,
        len(selected),
        min(scores),
        max(scores),
        sum(scores) / len(scores),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SoftGA molecule selection")
    parser.add_argument("--docked_file", required=True)
    parser.add_argument("--parent_file", default=None)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--n_select", type=int, default=None)
    parser.add_argument("--selection_mode", type=str, default=None)
    parser.add_argument("--output_format", choices=["smiles_only", "with_scores"], default="with_scores")
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--qed_min", type=float, default=None)
    parser.add_argument("--sa_max", type=float, default=None)
    parser.add_argument("--cache_file", type=str, default=None)
    parser.add_argument("--front_report_file", type=str, default=None)
    parser.add_argument("--disable_front_report", action="store_true", default=False)
    parser.add_argument("--disable_cache", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    selection_cfg = _load_selection_cfg(args.config_file)
    selection_mode = _normalize_selection_mode(args.selection_mode or selection_cfg.get("selection_mode", "ffhs"))
    mode_cfg_key = "nsgaii_settings" if selection_mode == "nsgaii" else "ffhs_settings"
    mode_cfg = selection_cfg.get(mode_cfg_key) or {}

    if args.verbose or bool(selection_cfg.get("verbose", False)) or bool(mode_cfg.get("verbose", False)):
        logger.setLevel(logging.DEBUG)

    n_select = args.n_select if args.n_select is not None else int(mode_cfg.get("n_select", 100))
    objectives_cfg = mode_cfg.get("objectives") or _default_objectives()
    enable_crowding = bool(mode_cfg.get("enable_crowding_distance", True))

    constraints_cfg = (selection_cfg.get("ffhs_settings") or {}).get("constraints", {})
    qed_min = float(args.qed_min if args.qed_min is not None else constraints_cfg.get("qed_min", 0.5))
    sa_max = float(args.sa_max if args.sa_max is not None else constraints_cfg.get("sa_max", 5.0))

    cache_path: Optional[str] = None
    if not args.disable_cache:
        cache_path = args.cache_file
        if not cache_path:
            run_root = Path(args.output_file).resolve().parent.parent
            cache_path = str(run_root / "chem_metric_cache.json")
    cache = ChemMetricCache(Path(cache_path) if cache_path else None)

    rows = _merge_parent_child(args.parent_file, args.docked_file)
    if not rows:
        logger.error("No candidate molecules available")
        raise SystemExit(1)

    rows = _add_qed_sa(rows, cache)
    selected = select_molecules(
        rows,
        n_select,
        objectives_cfg,
        enable_crowding,
        qed_min,
        sa_max,
        selection_mode=selection_mode,
    )
    if not selected:
        logger.error("Selection failed, result is empty")
        raise SystemExit(1)

    if args.output_format == "smiles_only":
        _save_smiles_only(selected, args.output_file)
    else:
        _save_with_scores(selected, args.output_file)

    if not args.disable_front_report:
        front_report = args.front_report_file or str(Path(args.output_file).with_suffix(".fronts.tsv"))
        _save_front_report(selected, front_report)
    _print_stats(selected, selection_mode)


if __name__ == "__main__":
    main()
