#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import argparse
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
DEFAULT_CONFIG = str(Path(PROJECT_ROOT) / "config.yaml")
from typing import Dict
from utils.autogrow_compat import mol_object_handling as MOH


def _sanitize_and_normalize_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol = MOH.check_sanitization(mol)
    mol = MOH.try_deprotanation(mol)
    if mol is None:
        return None, None
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), mol


def _is_single_component_smiles(smiles: str, mol: Chem.Mol) -> bool:
    if not smiles or mol is None or "." in smiles:
        return False
    try:
        return len(Chem.GetMolFrags(mol, asMols=False)) == 1
    except Exception:
        return False


def run_pre_docking_filter_operation(input_file: str, output_file: str) -> str:
    """Pre-docking filtering: only ensures legal structure and deduplication, without hard filtering such as Lipinski/PAINS."""
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    smiles = [line.split()[0].strip() for line in open(input_file, 'r', encoding='utf-8') if line.strip()]
    seen, kept = set(), []
    for smi in smiles:
        can, mol = _sanitize_and_normalize_smiles(smi)
        if can and _is_single_component_smiles(can, mol) and can not in seen:
            seen.add(can)
            kept.append(can)
    open(output_file, 'w', encoding='utf-8').write(''.join(s + "\n" for s in kept))
    return output_file

# The current workflow only uses pre-docking deduplication filtering, where the interface is retained but external filter implementation is no longer bound.
FILTER_CLASS_MAP = {}


def init_filters_from_config(config: Dict):
    cfg = config.get('filter', {})
    return [cls() for k, cls in FILTER_CLASS_MAP.items() if cfg.get(f'enable_{k}', False)]


def run_filter_operation(config: Dict, input_file: str, output_file: str) -> str:
    """     Perform molecular filtering operations
    Args:
        config: configuration parameter dictionary
        input_file: input file path
        output_file: output file path
    Returns:
        str: filtered result file path     """    
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    population = [line.split()[0].strip() for line in open(input_file, 'r') if line.strip()]
    filters = init_filters_from_config(config)
    kept, seen = [], set()
    for smi in population:
        can, mol = _sanitize_and_normalize_smiles(smi)
        if mol is None:
            continue
        if not _is_single_component_smiles(can, mol):
            continue
        if all(f.run_filter(mol) for f in filters):
            if can not in seen:
                seen.add(can)
                kept.append(can)
    open(output_file, 'w').write(''.join(s + '\n' for s in kept))
    return output_file

def main():
    p = argparse.ArgumentParser(description='filter')
    p.add_argument('--smiles_file', type=str, required=True)
    p.add_argument('--output_file', type=str, required=True)
    p.add_argument('--config_file', type=str, default=DEFAULT_CONFIG)
    # p.add_argument('--mode', type=str, default='full', choices=['full', 'pre_docking'])
    a = p.parse_args()

    # config = {} if a.mode == 'pre_docking' else json.load(open(a.config_file, 'r', encoding='utf-8'))
    # out = run_pre_docking_filter_operation(a.smiles_file, a.output_file) if a.mode == 'pre_docking' else run_filter_operation(config, a.smiles_file, a.output_file)
    out = run_pre_docking_filter_operation(a.smiles_file, a.output_file)
    print(out)

if __name__ == "__main__":
    main()
