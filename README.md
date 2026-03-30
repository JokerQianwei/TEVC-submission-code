# ProGA: Dynamic Prompt-Guided Graph Genetic Algorithm for Constrained Molecular Optimization

This repository contains the official implementation for the paper **"A Dynamic Prompt-Guided Graph Genetic Algorithm for Constrained Molecular Optimization"**.

## Overview

ProGA is a hybrid molecular optimization framework that combines:
- **BD-MLM** (Block-Diffusion Molecular Language Model) for guided generation
- **Dynamic Prefix Prompting** with progressive prefixing strategy
- **Graph-based Genetic Operations** (crossover and mutation)
- **FFHS** (Feasibility-First Hierarchical Selection) for constrained optimization

### Key Features

- **Intra-run Progressive Prefixing**: Sigmoid-scheduled prefix length increase within each run (exploration → exploitation)
- **Inter-run Prefix Carryover** (ProGA-WS variant): Warm-start mechanism that carries prefix state across runs
- **FFHS Selection**: Cost-asymmetry-aware selection that filters with cheap constraints before expensive docking evaluation
- **Two Benchmark Support**: 5-target Novel Hit Discovery and 10-target Cross-Target Generalization

## Installation

### Environment Setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate proga
```

### Docking Tools Setup

Grant executable permissions to docking binaries:

```bash
chmod +x utils/docking/qvina02
chmod +x utils/docking/vina
```

### Verify Installation

```bash
python -c "import rdkit, meeko, vina; print('Installation successful')"
```

## Pre-trained Weights

> **Note**: Due to double-blind review requirements, model weights are temporarily unavailable. They will be released upon paper acceptance.

Once obtained, place the checkpoint file in the `./weights` directory. The experiments reported in the paper use a 55M-parameter BD-MLM model.

## Usage

### 1. Unconditional Molecule Generation

Generate molecules using BD-MLM without optimization:

```bash
python sample.py
```

### 2. ProGA Optimization (Standard)

Run ProGA with default settings (recircle disabled):

```bash
python batch_runner.py \
    --receptor parp1 \
    --output_dir output/proga/ \
    --strategy_mode sigmoid \
    --tanimoto_threshold 0.7 \
    --samples_per_parent 10 \
    --initial_samples 1000 \
    --length 128 \
    --max_generations 30 \
    --gpu_ids 0 \
    --tasks_per_gpu 1 \
    --total_runs 20 \
    --seed 42 \
    --gpu_max_batch_size 1000 \
    --number_of_processors 60 \
    --min_keep_ratio 0.1 \
    --max_keep_ratio 1.0 \
    --recircle false
```

### 3. ProGA-WS Optimization (Warm Start)

Run ProGA-WS with inter-run prefix carryover enabled. With `--total_runs 20` and `--recircle true`, the prefix state from the final generation of each run is carried over to the first generation of the next run, creating a chain of 20 warm-started runs:

```bash
python batch_runner.py \
    --receptor parp1 \
    --output_dir output/proga_ws/ \
    --strategy_mode sigmoid \
    --tanimoto_threshold 0.7 \
    --samples_per_parent 10 \
    --initial_samples 1000 \
    --length 128 \
    --max_generations 30 \
    --gpu_ids 0 \
    --tasks_per_gpu 1 \
    --total_runs 20 \
    --seed 42 \
    --gpu_max_batch_size 1000 \
    --number_of_processors 60 \
    --min_keep_ratio 0.1 \
    --max_keep_ratio 1.0 \
    --recircle true
```

## Supported Targets

### Novel Hit Discovery Benchmark (5 targets)
- `parp1` - PARP1 (Poly ADP-ribose polymerase 1)
- `jak2` - JAK2 (Janus kinase 2)
- `fa7` - Factor VIIa
- `5ht1b` - 5-HT1B receptor
- `braf` - BRAF kinase

### Cross-Target Generalization Benchmark (10 targets)
PDB IDs: `1KKQ`, `1UWH`, `5WFD`, `6AZV`, `6GL8`, `7D42`, `7OTE`, `7S1S`, `7WC7`, `8JJL`

Example using PDB ID:
```bash
python batch_runner.py --receptor 6GL8 --output_dir output/6GL8/ --total_runs 20 --seed 42
```

## Results & Reproducibility

Experimental results are organized in the `results/` directory:

```
results/
├── Novel_Hit_Discovery_Benchmark/
│   ├── ProGA/              # Standard ProGA (recircle=false)
│   │   ├── recicleFalse_parp1_seed42.csv
│   │   ├── recicleFalse_parp1_seed43.csv
│   │   └── ...
│   └── ProGA-WS/           # Warm-start variant (recircle=true)
│       ├── recicle20_parp1_seed42.csv
│       ├── recicle20_parp1_seed43.csv
│       └── ...
└── Cross-Target_Generalization_Benchmark/
    ├── ProGA/
    │   ├── 1KKQ.csv
    │   ├── 1UWH.csv
    │   └── ...
    └── ProGA-WS/
        ├── 1KKQ.csv
        ├── 1UWH.csv
        └── ...
```

Each CSV file contains generated molecules with their docking scores, QED, and SA values across multiple runs.

## Evaluation

### Collect Top-1 Molecules

Extract the best molecule from each run:

```bash
python tools/collect_top1.py --input_dir output/proga/ --output_file top1_results.csv
```

### Evaluate Results

Compute metrics (hit rate, diversity, novelty, etc.):

```bash
python tools/eval_csv.py --input_file results/Novel_Hit_Discovery_Benchmark/ProGA/recicleFalse_parp1_seed42.csv
```

### Batch Evaluation

Evaluate multiple result files:

```bash
python tools/batch_eval_csv.py --input_dir results/Novel_Hit_Discovery_Benchmark/ProGA/
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--receptor` | Target protein name or PDB ID | `parp1` |
| `--recircle` | Enable inter-run prefix carryover (ProGA-WS) | `false` |
| `--strategy_mode` | Prefix schedule strategy | `sigmoid` |
| `--max_generations` | Number of generations per run | `30` |
| `--total_runs` | Number of independent runs | `20` |
| `--initial_samples` | Samples for generation 1 | `1000` |
| `--samples_per_parent` | Samples per parent in generation 2+ | `10` |
| `--tanimoto_threshold` | Diversity threshold for offspring selection | `0.7` |
| `--min_keep_ratio` | Minimum prefix retention ratio | `0.1` |
| `--max_keep_ratio` | Maximum prefix retention ratio | `1.0` |
| `--number_of_processors` | CPU cores for docking | `60` |

## Project Structure

```
.
├── main.py                  # Single-run entry point
├── batch_runner.py          # Multi-run batch execution
├── execute.py               # Workflow orchestration
├── generation.py            # BD-MLM generation with dynamic prefixing
├── crossover.py             # Graph-based crossover operations
├── mutation.py              # Graph-edit mutation operations
├── selection.py             # FFHS and NSGA-II selection
├── sample.py                # Unconditional sampling
├── config.yaml              # Default configuration
├── environment.yml          # Conda environment specification
├── model/                   # BD-MLM model implementation
│   ├── diffusion.py
│   ├── tokenizer.py
│   ├── models/
│   └── configs/
├── utils/                   # Utilities
│   ├── docking_runner.py    # Docking execution (qvina02/vina)
│   ├── chem_metrics.py      # QED, SA calculation
│   ├── filter.py            # Molecular filtering
│   ├── scoring.py           # Scoring functions
│   └── docking/             # Docking binaries and receptor files
├── tools/                   # Evaluation scripts
│   ├── collect_top1.py
│   ├── eval_csv.py
│   └── batch_eval_csv.py
└── results/                 # Experimental results
    ├── Novel_Hit_Discovery_Benchmark/
    └── Cross-Target_Generalization_Benchmark/
```
