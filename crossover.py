# Code adapted from https://github.com/jensengroup/GB_GA/blob/master/crossover.py
import os
import sys
import json
import argparse
import random
import logging
from pathlib import Path
import numpy as np

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.MolStandardize import rdMolStandardize

rdBase.DisableLog('rdApp.error')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "config.yaml")
from utils.autogrow_compat import mol_object_handling as MOH
from utils.config_loader import load_config, resolve_config_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global defaults, can be overridden by config
AVERAGE_SIZE = 39.15
SIZE_STDEV = 3.50


def _sanitize(smiles: str):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol = MOH.check_sanitization(mol)
    mol = MOH.try_deprotanation(mol)
    return None if mol is None else rdMolStandardize.Uncharger().uncharge(mol)


def _fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _tanimoto(fp1, fp2) -> float:
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


# -------------------------------------------------------------------------
# GD_GA Crossover Logic
# -------------------------------------------------------------------------

def cut(mol):
    if not mol.HasSubstructMatch(Chem.MolFromSmarts('[*]-;!@[*]')):
        return None

    bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[*]-;!@[*]')))  # single bond not in ring
    bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]

    fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])

    try:
        return Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
    except ValueError:
        return None


def cut_ring(mol):
    for i in range(10):
        if random.random() < 0.5:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]@[R]@[R]@[R]')):
                return None
            bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[R]@[R]@[R]@[R]')))
            bis = ((bis[0], bis[1]), (bis[2], bis[3]),)
        else:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]@[R;!D2]@[R]')):
                return None
            bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[R]@[R;!D2]@[R]')))
            bis = ((bis[0], bis[1]), (bis[1], bis[2]),)

        bs = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bis]

        fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1), (1, 1)])

        try:
            fragments = Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
            if len(fragments) == 2:
                return fragments
        except ValueError:
            return None

    return None


def ring_OK(mol):
    if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]')):
        return True

    ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts('[R]=[R]=[R]'))

    cycle_list = mol.GetRingInfo().AtomRings()
    max_cycle_length = max([len(j) for j in cycle_list]) if cycle_list else 0
    macro_cycle = max_cycle_length > 6

    double_bond_in_small_ring = mol.HasSubstructMatch(Chem.MolFromSmarts('[r3,r4]=[r3,r4]'))

    return not ring_allene and not macro_cycle and not double_bond_in_small_ring


def mol_ok(mol):
    try:
        Chem.SanitizeMol(mol)
        target_size = SIZE_STDEV * np.random.randn() + AVERAGE_SIZE
        if mol.GetNumAtoms() > 5 and mol.GetNumAtoms() < target_size:
            return True
        else:
            return False
    except ValueError:
        return False


def crossover_ring(parent_A, parent_B):
    ring_smarts = Chem.MolFromSmarts('[R]')
    if not parent_A.HasSubstructMatch(ring_smarts) and not parent_B.HasSubstructMatch(ring_smarts):
        return None

    rxn_smarts1 = ['[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]', '[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]']
    rxn_smarts2 = ['([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]', '([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]']

    for i in range(10):
        fragments_A = cut_ring(parent_A)
        fragments_B = cut_ring(parent_B)

        if fragments_A is None or fragments_B is None:
            return None

        new_mol_trial = []
        for rs in rxn_smarts1:
            rxn1 = AllChem.ReactionFromSmarts(rs)
            # new_mol_trial = [] # Redundant reset loop in original GD_GA? kept consistent with logic
            for fa in fragments_A:
                for fb in fragments_B:
                    new_mol_trial.append(rxn1.RunReactants((fa, fb))[0])

        new_mols = []
        for rs in rxn_smarts2:
            rxn2 = AllChem.ReactionFromSmarts(rs)
            for m in new_mol_trial:
                m = m[0]
                if mol_ok(m):
                    try:
                        new_mols += list(rxn2.RunReactants((m,)))
                    except Exception:
                        pass # Reactants mismatch

        new_mols2 = []
        for m in new_mols:
            m = m[0]
            if mol_ok(m) and ring_OK(m):
                new_mols2.append(m)

        if len(new_mols2) > 0:
            return random.choice(new_mols2)

    return None


def crossover_non_ring(parent_A, parent_B):
    for i in range(10):
        fragments_A = cut(parent_A)
        fragments_B = cut(parent_B)
        if fragments_A is None or fragments_B is None:
            return None
        rxn = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
        new_mol_trial = []
        for fa in fragments_A:
            for fb in fragments_B:
                new_mol_trial.append(rxn.RunReactants((fa, fb))[0])

        new_mols = []
        for mol in new_mol_trial:
            mol = mol[0]
            if mol_ok(mol):
                new_mols.append(mol)

        if len(new_mols) > 0:
            return random.choice(new_mols)

    return None


def crossover(parent_A, parent_B):
    parent_smiles = [Chem.MolToSmiles(parent_A), Chem.MolToSmiles(parent_B)]
    try:
        Chem.Kekulize(parent_A, clearAromaticFlags=True)
        Chem.Kekulize(parent_B, clearAromaticFlags=True)
    except ValueError:
        pass

    for i in range(10):
        if random.random() <= 0.5:
            # print 'non-ring crossover'
            new_mol = crossover_non_ring(parent_A, parent_B)
            if new_mol is not None:
                new_smiles = Chem.MolToSmiles(new_mol)
                if new_smiles is not None and new_smiles not in parent_smiles:
                    return new_mol
        else:
            # print 'ring crossover'
            new_mol = crossover_ring(parent_A, parent_B)
            if new_mol is not None:
                new_smiles = Chem.MolToSmiles(new_mol)
                if new_smiles is not None and new_smiles not in parent_smiles:
                    return new_mol
    return None

# -------------------------------------------------------------------------

def main():
    global AVERAGE_SIZE, SIZE_STDEV
    
    parser = argparse.ArgumentParser(description='crossover')
    parser.add_argument('--smiles_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--config_file', type=str, default=DEFAULT_CONFIG)
    parser.add_argument('--lineage_file', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--number_of_crossovers', type=int, default=None)
    parser.add_argument('--pair_min_tanimoto', type=float, default=None)
    parser.add_argument('--pair_max_tanimoto', type=float, default=None)
    args = parser.parse_args()

    cfg_path = resolve_config_path(args.config_file, Path(PROJECT_ROOT))
    config = load_config(str(cfg_path), Path(PROJECT_ROOT))
    cfg = config.get('crossover', {})
    
    # Update globals from config
    AVERAGE_SIZE = float(cfg.get('average_size', 39.15))
    SIZE_STDEV = float(cfg.get('size_stdev', 3.50))
    
    seed = int(args.seed if args.seed is not None else config.get('workflow', {}).get('seed', 42))
    
    # Setup global random seeds as GD_GA logic uses global random state
    random.seed(seed)
    np.random.seed(seed)
    
    # We still use a local rng for shuffling indices to match the outer loop logic structure 
    # but the crossover internals use global random.
    rng = random.Random(seed)

    pool = []  # (smi, mol, fp)
    seen = set()
    for line in open(args.smiles_file, 'r'):
        smi = line.split()[0].strip()
        mol = _sanitize(smi)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if can in seen:
            continue
        seen.add(can)
        pool.append((can, mol, _fp(mol)))

    if not pool:
        open(args.output_file, 'w').close()
        return

    n = int(args.number_of_crossovers if args.number_of_crossovers is not None else cfg.get('number_of_crossovers', 100))
    pair_min = float(args.pair_min_tanimoto if args.pair_min_tanimoto is not None else cfg.get('pair_min_tanimoto', 0.2))
    pair_max = float(args.pair_max_tanimoto if args.pair_max_tanimoto is not None else cfg.get('pair_max_tanimoto', 0.8))
    # bond_trials removed as it was specific to BRICS
    max_attempts = n * int(cfg.get('max_attempts_multiplier', 50))

    input_set = {s for s, _, _ in pool}
    ids = list(range(len(pool)))
    rng.shuffle(ids)

    children, lineage = [], []
    attempts = 0
    generated_smiles_set = set() # Track children to avoid duplicates
    
    while len(children) < n and attempts < max_attempts:
        attempts += 1
        if not ids:
            ids = list(range(len(pool)))
            rng.shuffle(ids)
        i1 = ids.pop()
        s1, m1, fp1 = pool[i1]

        i2 = None
        cand = list(range(len(pool)))
        rng.shuffle(cand)
        for i in cand:
            if i == i1:
                continue
            sim = _tanimoto(fp1, pool[i][2])
            if pair_min <= sim <= pair_max:
                i2 = i
                break
        if i2 is None:
            continue

        s2, m2, _fp2 = pool[i2]
        
        # New Crossover Call
        # Note: m1 and m2 might be modified in crossover (Kekulize), 
        # but since we generate them fresh or they are objects, logic should hold.
        # Ideally we should pass copies if we want to preserve pool integrity, 
        # but crossover modifies in place for Kekulize only. 
        # RDKit Mols are mutable. But Kekulize is usually fine.
        # To be safe, we can copy, but for performance we might skip.
        # The original GD_GA logic operates on Mol objects directly.
        
        child_mol = crossover(m1, m2)
        
        if child_mol is None:
            continue
            
        try:
            child_smi = Chem.MolToSmiles(child_mol, canonical=True, isomericSmiles=True)
        except:
            continue
            
        if not child_smi or child_smi in input_set or child_smi in generated_smiles_set:
            continue
            
        children.append(child_smi)
        generated_smiles_set.add(child_smi)
        
        if config.get('workflow', {}).get('enable_lineage_tracking'):
            lineage.append({'child': child_smi, 'operation': 'crossover', 'parents': [s1, s2]})

    open(args.output_file, 'w').write(''.join(f'{s}\n' for s in children))
    logger.info(f"crossover: {len(children)}/{n} (attempts={attempts})")

    if args.lineage_file and lineage:
        open(args.lineage_file, 'w', encoding='utf-8').write(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in lineage)
        )

if __name__ == "__main__":
    main()
