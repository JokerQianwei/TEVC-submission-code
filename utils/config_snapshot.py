#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Configuration parameter snapshot generator
==================
This module is responsible for generating and saving the complete configuration snapshot of the current GA run.
Main functions:
1. Filter out unused configuration branches based on actual execution mode
2. Collect all parameters actually used during operation
3. Generate a "clean" and "complete" parameter record file   """

import os
import sys
import json
import copy
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import load_config

# Configuration log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigSnapshotGenerator:
    """Configure snapshot generator class"""
    
    def __init__(self, original_config: Dict[str, Any], execution_context: Dict[str, Any]):
        """         Initialize configuration snapshot generator
        
        Args:
            original_config: original complete configuration dictionary
            execution_context: execution context information, including the actual mode and parameters used         """
        self.original_config = original_config
        self.execution_context = execution_context
        self.used_config = {}
        
    def generate_snapshot(self) -> Dict[str, Any]:
        """         Generate configuration snapshot
        
        Returns:
            Dict: Configuration dictionary containing the actual parameters used in the current run         """
        # Goal: Keep only the parameter branches "that will actually be used in this execution" (avoid writing the entire package of unused receptors).
        # Also avoid omissions caused by handwritten field whitelisting (e.g. constraints), so use deepcopy for "selected branches".
        logger.info("Start generating configuration parameter snapshots (tailored according to actual usage)...")
        cfg = self.original_config if isinstance(self.original_config, dict) else {}

        used: Dict[str, Any] = {}
        self.used_config = used
        self._add_execution_metadata()

        # 1) All top-level configurations except "Receptors/Selects" are retained by default.
        # Receptors/selections need to be pruned according to actual branches (for example, only keep one receptor, only keep multi/single corresponding branches).
        for key, value in cfg.items():
            if key in ("receptors", "selection", "execution_metadata"):
                continue
            used[key] = copy.deepcopy(value)

        # 4) Selection strategy: Only keep the actually used FFHS branch, and keep the branch field (including constraints) completely
        selection_cfg = cfg.get("selection")
        if isinstance(selection_cfg, dict) and selection_cfg:
            actual_mode = (
                self.execution_context.get("selection_mode")
                or selection_cfg.get("selection_mode")
                or "ffhs"
            )
            clean_selection: Dict[str, Any] = {"selection_mode": actual_mode}
            if actual_mode == "ffhs" and isinstance(selection_cfg.get("ffhs_settings"), dict):
                clean_selection["ffhs_settings"] = copy.deepcopy(selection_cfg["ffhs_settings"])
            else:
                # Unknown mode: degenerate to retain the entire selection to avoid losing parameters
                clean_selection = copy.deepcopy(selection_cfg)
                clean_selection["selection_mode"] = actual_mode
            used["selection"] = clean_selection

        # 5) Receptor configuration: only retain the receptor information actually used this time (avoid writing all receptors in target_list into snapshot)
        receptors_cfg = cfg.get("receptors")
        if isinstance(receptors_cfg, dict) and receptors_cfg:
            receptor_name = self.execution_context.get("receptor_name")
            used_receptor: Optional[Dict[str, Any]] = None

            target_list = receptors_cfg.get("target_list")
            if receptor_name and isinstance(target_list, dict) and receptor_name in target_list:
                used_receptor = copy.deepcopy(target_list[receptor_name])
                if isinstance(used_receptor, dict) and "name" not in used_receptor:
                    used_receptor["name"] = receptor_name
            else:
                default_receptor = receptors_cfg.get("default_receptor")
                if isinstance(default_receptor, dict):
                    if (not receptor_name) or (default_receptor.get("name") == receptor_name):
                        used_receptor = copy.deepcopy(default_receptor)

            if used_receptor:
                used["receptors"] = {"used_receptor": used_receptor}

        logger.info("Configuration parameter snapshot generation is completed")
        return used
    
    def _add_execution_metadata(self):
        """Add execution metadata"""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "config_file_path": self.execution_context.get("config_file_path"),
            "project_root": self.execution_context.get("project_root"),
            "base_output_dir": self.execution_context.get("base_output_dir"),
            "receptor_name": self.execution_context.get("receptor_name"),
            "run_specific_output_dir": self.execution_context.get("run_specific_output_dir"),
            "max_generations": self.execution_context.get("max_generations"),
            "initial_population_file": self.execution_context.get("initial_population_file")
        }
        cli_overrides = self.execution_context.get("cli_overrides")
        if isinstance(cli_overrides, dict) and cli_overrides:
            metadata["cli_overrides"] = cli_overrides
        # Completely save the execution context passed by the caller to facilitate repeated experiments (such as selection_mode / num_processors, etc.)
        metadata["execution_context"] = self.execution_context
        self.used_config["execution_metadata"] = metadata

def save_config_snapshot(original_config: Dict[str, Any], 
                        execution_context: Dict[str, Any], 
                        output_file_path: str) -> bool:
    """     Save a snapshot of configuration parameters to a file
    
    Args:
        original_config: original complete configuration dictionary
        execution_context: execution context information
        output_file_path: output file path
        
    Returns:
        bool: whether saved successfully     """
    try:
        # Generate configuration snapshot
        generator = ConfigSnapshotGenerator(original_config, execution_context)
        snapshot = generator.generate_snapshot()
        
        # Make sure the output directory exists
        output_dir = Path(output_file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # save to file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuration parameters snapshot saved to: {output_file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save configuration parameters snapshot: {e}", exc_info=True)
        return False

def main():
    """Main function, used for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Configuration parameter snapshot generator')
    parser.add_argument('--config', type=str, required=True, help='Original configuration file path')
    parser.add_argument('--output', type=str, required=True, help='Output snapshot file path')
    parser.add_argument('--selection_mode', type=str, default='ffhs', 
                       help='Actual selection mode used')
    
    args = parser.parse_args()
    
    # Load original configuration
    config = load_config(args.config, PROJECT_ROOT)
    
    # Build execution context (for testing)
    execution_context = {
        "config_file_path": args.config,
        "selection_mode": args.selection_mode,
        "receptor_name": "default_receptor",
        "max_generations": config.get("workflow", {}).get("max_generations", 5)
    }
    
    # Generate and save snapshots
    success = save_config_snapshot(config, execution_context, args.output)
    
    if success:
        print(f"Configuration snapshot generated successfully: {args.output}")
    else:
        print("Configuration snapshot generation failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 
