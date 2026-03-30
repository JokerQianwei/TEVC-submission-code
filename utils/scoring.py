# Indicator calculation (comparison/analysis)
import argparse
import os
import numpy as np
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import QED
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from tdc import Oracle, Evaluator

# Use TDC for all metric evaluations
qed_evaluator = Oracle('qed')
sa_evaluator = Oracle('sa')
diversity_evaluator = Evaluator(name='Diversity')
# NOTE: TDC's Novelty evaluator requires an initial SMILES list for initialization
# We will create it dynamically in the main function based on the parameters

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def calculate_sa_scores(smiles_list: list) -> list:
    """Use TDC Oracle to calculate SA scores in batches."""
    if not smiles_list:
        return []
    print(f"Use TDC to batch calculate the SA fraction of {len(smiles_list)} molecules...")
    return sa_evaluator(smiles_list)

def calculate_qed_scores(smiles_list: list) -> list:
    """Use TDC Oracle to calculate QED scores in batches."""
    if not smiles_list:
        return []
    print(f"Batch calculation of QED scores for {len(smiles_list)} molecules using TDC...")
    return qed_evaluator(smiles_list)

def load_smiles_from_file(filepath):   #Load smile
    smiles_list = []    
    with open(filepath, 'r') as f:
        for line in f:
            smiles = line.strip().split()[0] 
            if smiles:
                smiles_list.append(smiles)    
    return smiles_list

def load_smiles_and_scores_from_file(filepath):   #Load smile and score: output file (with fraction) after docking
    molecules = []
    scores = []
    smiles_list = []    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                smiles = parts[0]
                try:
                    score = float(parts[1])
                    molecules.append(smiles)
                    scores.append(score)
                    smiles_list.append(smiles)
                except ValueError:
                    print(f"Warning: Could not parse score for SMILES: {smiles}")
            elif len(parts) == 1: # If only SMILES is present, no score
                smiles_list.append(parts[0])    
    return smiles_list, molecules, scores

def get_rdkit_mols(smiles_list): #smiles-----mol
    mols = []
    valid_smiles = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            mols.append(mol)
            valid_smiles.append(s)
        else:
            print(f"Warning: Could not parse SMILES: {s}")
    return mols, valid_smiles

def calculate_docking_stats(scores):
    """Calculates Top-1, Top-10 mean, Top-100 mean docking scores."""    
    sorted_scores = sorted(scores) # Docking scores, lower is better
    top1_score = sorted_scores[0] if len(sorted_scores) >= 1 else np.nan    #top1
    top10_scores = sorted_scores[:10]
    top10_mean = np.mean(top10_scores) if top10_scores else np.nan           #top10
    top100_scores = sorted_scores[:100]
    top100_mean = np.mean(top100_scores) if top100_scores else np.nan        #top100
    return top1_score, top10_mean, top100_mean

def calculate_novelty(current_smiles: list, initial_smiles_list: list) -> float:
    """Novelty is calculated using TDC Evaluator."""
    if not current_smiles:
        return 0.0
    # Correct usage: Pass parameters directly by position
    novelty_evaluator = Evaluator(name='Novelty')
    return novelty_evaluator(current_smiles, initial_smiles_list)

def calculate_top100_diversity(smiles_list: list) -> float:
    """Use TDC Evaluator to calculate the top-100 diversity."""
    top_smiles = smiles_list[:min(100, len(smiles_list))]
    if not top_smiles:
        return 0.0
    return diversity_evaluator(top_smiles)

def print_calculation_results(results):    
    print("Calculation Results:")
    print(results)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a generation of molecules.")
    parser.add_argument("--current_population_docked_file", type=str, required=True,
                        help="Path to the SMILES file of the current population with docking scores (SMILES score per line).")
    parser.add_argument("--initial_population_file", type=str, required=True,
                        help="Path to the SMILES file of the initial population (for novelty calculation).")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output file to save calculated metrics (e.g., results.txt or results.csv).")
    # Top-1 feasibility statistic threshold (aligned with FFHS selection/final Top-1 rules)
    parser.add_argument("--qed_min", type=float, default=0.5, help="Feasible rule: QED >= qed_min (default: 0.5)")
    parser.add_argument("--sa_max", type=float, default=5.0, help="Feasible rule: SA <= sa_max (default: 5.0)")
    
    args = parser.parse_args()
    print(f"Processing population file: {args.current_population_docked_file}")
    print(f"Using initial population for novelty: {args.initial_population_file}")
    print(f"Saving results to: {args.output_file}")

    # Load current SMILES and docking scores
    current_smiles_list, scored_molecules_smiles, docking_scores = load_smiles_and_scores_from_file(args.current_population_docked_file)
    
    # Docking score sorting
    if scored_molecules_smiles and docking_scores:
        molecules_with_scores = sorted(zip(scored_molecules_smiles, docking_scores), key=lambda x: x[1])
        sorted_smiles = [s for s, _ in molecules_with_scores]
    else:
        sorted_smiles = current_smiles_list # If there are no scores, the original order is used
        
    # 1. Docking Score Metrics
    top1_score, top10_mean_score, top100_mean_score = calculate_docking_stats(docking_scores)
    
    # Define the elite group (Top 100) used to calculate all attribute metrics
    smiles_for_scoring = sorted_smiles[:min(100, len(sorted_smiles))]
    score_description = f"Top {len(smiles_for_scoring)}"

    # 2. Novelty (based on Top 100 elite populations)
    initial_smiles = load_smiles_from_file(args.initial_population_file)
    novelty = calculate_novelty(smiles_for_scoring, initial_smiles)
    
    # 3. Diversity (Top 100 molecules)
    diversity = calculate_top100_diversity(smiles_for_scoring)
    
    # 4. QED & SA Scores (for Top 100)
    qed_scores = calculate_qed_scores(smiles_for_scoring)
    sa_scores = calculate_sa_scores(smiles_for_scoring)
    
    mean_qed = np.mean(qed_scores) if qed_scores else np.nan
    mean_sa = np.mean(sa_scores) if sa_scores else np.nan

    # 4.1 QED/SA corresponding to Top-1 (by docking), and "feasible Top-1" (optimal docking that meets the QED/SA threshold)
    top1_qed = np.nan
    top1_sa = np.nan
    feasible_top1_docking = np.nan
    feasible_top1_qed = np.nan
    feasible_top1_sa = np.nan
    feasible_ratio = np.nan
    if scored_molecules_smiles and docking_scores and qed_scores and sa_scores:
        n = len(smiles_for_scoring)
        top_pairs = molecules_with_scores[:n]
        if n > 0:
            top1_qed = float(qed_scores[0])
            top1_sa = float(sa_scores[0])
        feas_mask = [(float(qed_scores[i]) >= args.qed_min) and (float(sa_scores[i]) <= args.sa_max) for i in range(n)]
        feasible_ratio = float(sum(feas_mask)) / float(n) if n > 0 else np.nan
        feasible_candidates = [(top_pairs[i][1], float(qed_scores[i]), float(sa_scores[i])) for i in range(n) if feas_mask[i]]
        if feasible_candidates:
            feasible_candidates.sort(key=lambda x: x[0])  # The smaller the docking, the better
            feasible_top1_docking, feasible_top1_qed, feasible_top1_sa = feasible_candidates[0]

    # Safely handle filenames that may contain special characters
    population_filename = os.path.basename(args.current_population_docked_file)
    initial_population_filename = os.path.basename(args.initial_population_file)    
    # To avoid f-string formatting issues, use traditional string formatting
    results = "Metrics for Population: {}\n".format(population_filename)
    results += "--------------------------------------------------\n"
    results += "Total molecules processed: {}\n".format(len(current_smiles_list))
    results += "Valid RDKit molecules for properties: {}\n".format(len(sorted_smiles))
    results += "Molecules with docking scores: {}\n".format(len(docking_scores))
    results += "--------------------------------------------------\n"    
    # Handle floating point number formatting, pay attention to handling NaN situations
    if np.isnan(top1_score): #top1
        results += "Docking Score - Top 1: N/A\n"
    else:
        results += "Docking Score - Top 1: {:.4f}\n".format(top1_score)
        if not np.isnan(top1_qed) and not np.isnan(top1_sa):
            results += "Top 1 (by Docking) - QED: {:.4f}, SA: {:.4f}\n".format(top1_qed, top1_sa)
        
    if np.isnan(top10_mean_score): #top10
        results += "Docking Score - Top 10 Mean: N/A\n"
    else:
        results += "Docking Score - Top 10 Mean: {:.4f}\n".format(top10_mean_score)    

    if np.isnan(top100_mean_score): #top100
        results += "Docking Score - Top 100 Mean: N/A\n"
    else:
        results += "Docking Score - Top 100 Mean: {:.4f}\n".format(top100_mean_score)    
    results += "--------------------------------------------------\n"
    results += "Novelty (vs {}): {:.4f}\n".format(initial_population_filename, novelty)
    results += "Diversity (Top 100): {:.4f}\n".format(diversity)
    results += "--------------------------------------------------\n"    
    if np.isnan(mean_qed):
        results += "QED - {} Mean: N/A\n".format(score_description)
    else:
        results += "QED - {} Mean: {:.4f}\n".format(score_description, mean_qed)        
    if np.isnan(mean_sa):
        results += "SA Score - {} Mean: N/A\n".format(score_description)
    else:
        results += "SA Score - {} Mean: {:.4f}\n".format(score_description, mean_sa)    
    results += "--------------------------------------------------\n"
    # Feasible Top-1 statistics (used to explain the phenomenon of "Top-1 QED drops very much": extreme docking may correspond to very low QED)
    if np.isnan(feasible_ratio):
        results += "Feasible Ratio (QED>=%.2f, SA<=%.2f): N/A\n" % (args.qed_min, args.sa_max)
    else:
        results += "Feasible Ratio (QED>=%.2f, SA<=%.2f): %.4f\n" % (args.qed_min, args.sa_max, feasible_ratio)
    if np.isnan(feasible_top1_docking):
        results += "Feasible Top 1 (by Docking): N/A\n"
    else:
        results += "Feasible Top 1 - Docking: {:.4f}, QED: {:.4f}, SA: {:.4f}\n".format(
            feasible_top1_docking, feasible_top1_qed, feasible_top1_sa
        )
    results += "--------------------------------------------------\n"    
    print_calculation_results(results)
    
    # Save results to output file
    try:
        with open(args.output_file, 'w') as f:
            f.write(results)
        print(f"Results successfully saved to {args.output_file}")
    except IOError:
        print(f"Error: Could not write results to {args.output_file}")
if __name__ == "__main__":
    main()
