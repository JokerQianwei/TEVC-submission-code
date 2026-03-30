# Code adapted from https://github.com/jensengroup/GB_GA/blob/master/mutate.py
"""Graph-edit mutation for SoftGA."""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_CONFIG = str(PROJECT_ROOT / "config.yaml")

from utils.config_loader import load_config, resolve_config_path

rdBase.DisableLog("rdApp.error")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ATOM_CHOICES = ["#6", "#7", "#8", "#9", "#16", "#17", "#35"]
ATOM_PROBS = [0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14]
OPS = ("insert_atom", "change_bond_order", "delete_cyclic_bond", "add_ring", "delete_atom", "change_atom", "append_atom")
OP_PROBS = [0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.15]


def _to_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _canon_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        return None
    if any(b.GetBondType() == Chem.BondType.UNSPECIFIED for b in mol.GetBonds()):
        return None
    can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return None if "." in can else can


def _load_unique_pool(smiles_file: str) -> List[str]:
    seen, pool = set(), []
    with open(smiles_file, "r", encoding="utf-8") as f:
        for line in f:
            smi = line.split()[0].strip() if line.strip() else ""
            can = _canon_smiles(smi) if smi else None
            if can and can not in seen:
                seen.add(can)
                pool.append(can)
    return pool


def _ring_ok(mol) -> bool:
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]")):
        return True
    ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts("[R]=[R]=[R]"))
    rings = mol.GetRingInfo().AtomRings()
    macro_cycle = (max((len(r) for r in rings), default=0) > 6)
    small_ring_db = mol.HasSubstructMatch(Chem.MolFromSmarts("[r3,r4]=[r3,r4]"))
    return (not ring_allene) and (not macro_cycle) and (not small_ring_db)


def _mol_ok(mol, avg_size: float, size_stdev: float) -> bool:
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False
    target_size = size_stdev * np.random.randn() + avg_size
    n_atoms = mol.GetNumAtoms()
    return 5 < n_atoms < target_size


def _delete_atom_smarts() -> str:
    choices = [
        "[*:1]~[D1]>>[*:1]",
        "[*:1]~[D2]~[*:2]>>[*:1]-[*:2]",
        "[*:1]~[D3](~[*;!H0:2])~[*:3]>>[*:1]-[*:2]-[*:3]",
        "[*:1]~[D4](~[*;!H0:2])(~[*;!H0:3])~[*:4]>>[*:1]-[*:2]-[*:3]-[*:4]",
        "[*:1]~[D4](~[*;!H0;!H1:2])(~[*:3])~[*:4]>>[*:1]-[*:2](-[*:3])-[*:4]",
    ]
    probs = [0.25, 0.25, 0.25, 0.1875, 0.0625]
    return str(np.random.choice(choices, p=probs))


def _append_atom_smarts() -> str:
    choices = [
        ("single", ["C", "N", "O", "F", "S", "Cl", "Br"], [1 / 7.0] * 7),
        ("double", ["C", "N", "O"], [1 / 3.0] * 3),
        ("triple", ["C", "N"], [1 / 2.0] * 2),
    ]
    bo = choices[int(np.random.choice([0, 1, 2], p=[0.60, 0.35, 0.05]))]
    bond_order, atoms, probs = bo
    atom = str(np.random.choice(atoms, p=probs))
    if bond_order == "single":
        return "[*;!H0:1]>>[*:1]-" + atom
    if bond_order == "double":
        return "[*;!H0;!H1:1]>>[*:1]=" + atom
    return "[*;H3:1]>>[*:1]#" + atom


def _insert_atom_smarts() -> str:
    choices = [
        ("single", ["C", "N", "O", "S"], [1 / 4.0] * 4),
        ("double", ["C", "N"], [1 / 2.0] * 2),
        ("triple", ["C"], [1.0]),
    ]
    bo = choices[int(np.random.choice([0, 1, 2], p=[0.60, 0.35, 0.05]))]
    bond_order, atoms, probs = bo
    atom = str(np.random.choice(atoms, p=probs))
    if bond_order == "single":
        return "[*:1]~[*:2]>>[*:1]" + atom + "[*:2]"
    if bond_order == "double":
        return "[*;!H0:1]~[*:2]>>[*:1]=" + atom + "-[*:2]"
    return "[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#" + atom + "-[*:2]"


def _change_bond_order_smarts() -> str:
    choices = [
        "[*:1]!-[*:2]>>[*:1]-[*:2]",
        "[*;!H0:1]-[*;!H0:2]>>[*:1]=[*:2]",
        "[*:1]#[*:2]>>[*:1]=[*:2]",
        "[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#[*:2]",
    ]
    return str(np.random.choice(choices, p=[0.45, 0.45, 0.05, 0.05]))


def _delete_cyclic_bond_smarts() -> str:
    return "[*:1]@[*:2]>>([*:1].[*:2])"


def _add_ring_smarts() -> str:
    choices = [
        "[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1",
        "[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1",
    ]
    return str(np.random.choice(choices, p=[0.05, 0.05, 0.45, 0.45]))


def _change_atom_smarts(mol) -> Optional[str]:
    present = [x for x in ATOM_CHOICES if mol.HasSubstructMatch(Chem.MolFromSmarts(f"[{x}]"))]
    if not present:
        return None
    x = str(np.random.choice(present))
    y_candidates = [a for a in ATOM_CHOICES if a != x]
    y = str(np.random.choice(y_candidates, p=np.array([ATOM_PROBS[ATOM_CHOICES.index(a)] for a in y_candidates]) / sum(ATOM_PROBS[ATOM_CHOICES.index(a)] for a in y_candidates)))
    return "[X:1]>>[Y:1]".replace("X", x).replace("Y", y)


def _mutate_graph_once(parent_smi: str, mutation_rate: float, avg_size: float, size_stdev: float, per_parent_trials: int, op_probs: List[float]) -> Optional[Tuple[str, str]]:
    if random.random() > mutation_rate:
        return None
    parent_mol = Chem.MolFromSmiles(parent_smi)
    if not parent_mol:
        return None
    parent_can = Chem.MolToSmiles(parent_mol, canonical=True, isomericSmiles=True)
    for _ in range(per_parent_trials):
        mol = Chem.Mol(parent_mol)
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass
        op = str(np.random.choice(OPS, p=op_probs))
        if op == "insert_atom":
            smarts = _insert_atom_smarts()
        elif op == "change_bond_order":
            smarts = _change_bond_order_smarts()
        elif op == "delete_cyclic_bond":
            smarts = _delete_cyclic_bond_smarts()
        elif op == "add_ring":
            smarts = _add_ring_smarts()
        elif op == "delete_atom":
            smarts = _delete_atom_smarts()
        elif op == "change_atom":
            smarts = _change_atom_smarts(mol)
        else:
            smarts = _append_atom_smarts()
        if not smarts:
            continue
        rxn = AllChem.ReactionFromSmarts(smarts)
        trials = rxn.RunReactants((mol,))
        if not trials:
            continue
        valid = []
        for item in trials:
            child = item[0]
            if _mol_ok(child, avg_size, size_stdev) and _ring_ok(child):
                child_smi = _canon_smiles(Chem.MolToSmiles(child, canonical=True, isomericSmiles=True))
                if child_smi and child_smi != parent_can:
                    valid.append(child_smi)
        if valid:
            return random.choice(valid), op
    return None


def _normalized_probs(raw: Optional[List[float]], default: List[float]) -> List[float]:
    vals = default if not raw or len(raw) != len(default) else [float(x) for x in raw]
    s = sum(vals)
    return default if s <= 0 else [x / s for x in vals]


def main():
    parser = argparse.ArgumentParser(description="mutation (graph-edit)")
    parser.add_argument("--smiles_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--config_file", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--lineage_file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--number_of_mutants", type=int, default=None)
    args = parser.parse_args()

    config_path = resolve_config_path(args.config_file, PROJECT_ROOT)
    config = load_config(str(config_path), PROJECT_ROOT)
    cfg = config.get("mutation") or config.get("mutation_rnx") or {}

    seed = int(args.seed if args.seed is not None else config.get("workflow", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    pool = _load_unique_pool(args.smiles_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lineage_path = Path(args.lineage_file) if args.lineage_file else None
    if lineage_path:
        lineage_path.parent.mkdir(parents=True, exist_ok=True)

    if not pool:
        output_path.write_text("", encoding="utf-8")
        if lineage_path:
            lineage_path.write_text("", encoding="utf-8")
        return

    n = _to_int(args.number_of_mutants if args.number_of_mutants is not None else cfg.get("number_of_mutants", 100), 100)
    max_attempts = _to_int(cfg.get("max_attempts"), n * _to_int(cfg.get("max_attempts_multiplier", 200), 200))
    mutation_rate = _to_float(cfg.get("mutation_rate", 1.0), 1.0)
    avg_size = _to_float(cfg.get("average_size", 39.15), 39.15)
    size_stdev = _to_float(cfg.get("size_stdev", 3.50), 3.50)
    per_parent_trials = _to_int(cfg.get("per_parent_trials", 10), 10)
    op_probs = _normalized_probs(cfg.get("operator_probs"), OP_PROBS)

    input_set, out, out_set, lineage = set(pool), [], set(), []
    attempts = 0
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        parent = random.choice(pool)
        result = _mutate_graph_once(parent, mutation_rate, avg_size, size_stdev, per_parent_trials, op_probs)
        if not result:
            continue
        child, op = result
        if child in input_set or child in out_set:
            continue
        out.append(child)
        out_set.add(child)
        if lineage_path:
            lineage.append(
                {
                    "child": child,
                    "operation": "mutation",
                    "parents": [parent],
                    "mutation_rule": "graph_edit",
                    "edit_op": op,
                }
            )

    output_path.write_text("".join(f"{s}\n" for s in out), encoding="utf-8")
    logger.info(f"mutation_graph: {len(out)}/{n} (attempts={attempts}, seed={seed})")
    if lineage_path:
        lineage_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lineage), encoding="utf-8")


if __name__ == "__main__":
    main()
