#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quickly calculate the docking score of a single SMILES.
python tools/quick_eval_DS.py -i "CCO" -t parp1 -v qvina02
python tools/quick_eval_DS.py -i "Cc1ccc2cc(C(=O)N=C3C[C@@H](c4ccc5c(c4)=CC(=C4NNC6=C4CC=CC6)N=5)CC=N3)ccc2c1" -t 1KKQ -v vina  """

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = CURRENT_SCRIPT_DIR.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

from utils.config_loader import load_config, resolve_config_path
from utils.docking_runner import DEFAULT_CONFIG, DEFAULT_RECEPTOR_BOXES, DockingVinaSoftGA

SUPPORTED_RECEPTORS = tuple(DEFAULT_RECEPTOR_BOXES.keys())
RECEPTOR_NAME_LOOKUP = {name.lower(): name for name in SUPPORTED_RECEPTORS}
DOCKING_TOOL_ALIASES = {
    "qvina": "qvina02",
    "qvina02": "qvina02",
    "vina": "vina",
}
SUPPORTED_DOCKING_TOOLS = ("qvina02", "vina")


def _parse_receptor(value: str) -> str:
    receptor = str(value).strip()
    canonical = RECEPTOR_NAME_LOOKUP.get(receptor.lower())
    if canonical is None:
        supported = ", ".join(SUPPORTED_RECEPTORS)
        raise argparse.ArgumentTypeError(f"Unsupported receptor: {receptor}. Optional: {supported}")
    return canonical


def _parse_docking_tool(value: str) -> str:
    tool = str(value).strip().lower()
    canonical = DOCKING_TOOL_ALIASES.get(tool)
    if canonical is None:
        supported = ", ".join(SUPPORTED_DOCKING_TOOLS)
        raise argparse.ArgumentTypeError(f"Unsupported docking mode: {value}. Optional: {supported}")
    return canonical


def _resolve_num_processors(cfg: Dict, override: Optional[int]) -> int:
    if override is not None:
        return max(1, int(override))
    num_processors = cfg.get("performance", {}).get("number_of_processors", 1)
    if num_processors == -1:
        num_processors = max(1, (os.cpu_count() or 1) - 2)
    return max(1, int(num_processors))


def _resolve_exhaustiveness(cfg: Dict, override: Optional[int]) -> int:
    if override is not None:
        return max(1, int(override))
    return max(1, int(cfg.get("docking", {}).get("exhaustiveness", 1)))


def _resolve_seed(cfg: Dict, override: Optional[int]) -> Optional[int]:
    if override is not None:
        return int(override)
    seed = cfg.get("workflow", {}).get("seed")
    return None if seed is None else int(seed)


def _compute_docking(
    smiles: str,
    receptor: str,
    docking_tool: str,
    num_processors: int,
    seed: Optional[int],
    exhaustiveness: int,
) -> float:
    docker = DockingVinaSoftGA(
        target=receptor,
        num_processors=num_processors,
        seed=seed,
        exhaustiveness=exhaustiveness,
        docking_tool=docking_tool,
    )
    try:
        scores = docker.predict([smiles])
        return float(scores[0]) if scores else 99.9
    finally:
        if hasattr(docker, "temp_dir") and docker.temp_dir:
            shutil.rmtree(docker.temp_dir, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    supported = ", ".join(SUPPORTED_RECEPTORS)
    parser = argparse.ArgumentParser(
        description="Quickly calculate the docking score of a single SMILES",
        epilog=f"15 receptors supported: {supported}",
    )
    parser.add_argument("-i", "--smiles", required=True, help="Enter molecules SMILES")
    parser.add_argument("-t", "--target", required=True, type=_parse_receptor, help="Receptor name, case insensitive")
    parser.add_argument("-v", "--docking-tool", required=True, type=_parse_docking_tool, help="docking mode: qvina02 or vina")
    parser.add_argument("--config_file", type=str, default=DEFAULT_CONFIG, help="Configuration file path")
    parser.add_argument("--number_of_processors", type=int, default=None, help="Number of concurrent processes, default is config")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, default is config.workflow.seed")
    parser.add_argument("-e", "--exhaustiveness", type=int, default=10, help="docking exhaustiveness, the default is config")
    return parser


def run_cli(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    cfg_path = resolve_config_path(args.config_file, PROJECT_ROOT_DIR)
    config = load_config(str(cfg_path), PROJECT_ROOT_DIR)
    num_processors = _resolve_num_processors(config, args.number_of_processors)
    exhaustiveness = _resolve_exhaustiveness(config, args.exhaustiveness)
    seed = _resolve_seed(config, args.seed)

    docking_score = _compute_docking(
        smiles=args.smiles,
        receptor=args.target,
        docking_tool=args.docking_tool,
        num_processors=num_processors,
        seed=seed,
        exhaustiveness=exhaustiveness,
    )

    result = {
        "smiles": args.smiles,
        "receptor": args.target,
        "docking_tool": args.docking_tool,
        "docking_score": docking_score,
        "success": docking_score <= 50.0,
        "number_of_processors": num_processors,
        "exhaustiveness": exhaustiveness,
        "seed": seed,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
