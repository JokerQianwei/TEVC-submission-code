#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置参数快照生成器
==================
该模块负责生成并保存当次GA运行的完整配置快照。
主要功能:
1. 根据实际执行模式过滤掉未使用的配置分支
2. 收集所有在运行中实际使用的参数
3. 生成一个"干净"且"完整"的参数记录文件


"""

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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigSnapshotGenerator:
    """配置快照生成器类"""
    
    def __init__(self, original_config: Dict[str, Any], execution_context: Dict[str, Any]):
        """
        初始化配置快照生成器
        
        Args:
            original_config: 原始完整配置字典
            execution_context: 执行上下文信息，包含实际使用的模式和参数
        """
        self.original_config = original_config
        self.execution_context = execution_context
        self.used_config = {}
        
    def generate_snapshot(self) -> Dict[str, Any]:
        """
        生成配置快照
        
        Returns:
            Dict: 包含当次运行实际使用参数的配置字典
        """
        # 目标：仅保留“本次执行真正会用到”的参数分支（避免把未使用的受体等整包写入）。
        # 同时避免手写字段白名单导致遗漏（例如 constraints），因此对“已选择的分支”使用 deepcopy。
        logger.info("开始生成配置参数快照（按实际使用裁剪）...")
        cfg = self.original_config if isinstance(self.original_config, dict) else {}

        used: Dict[str, Any] = {}
        self.used_config = used
        self._add_execution_metadata()

        # 1) 默认保留除“受体/选择”之外的所有顶层配置。
        # 受体/选择需要按实际分支裁剪（例如只保留一个受体、只保留 multi/single 对应分支）。
        for key, value in cfg.items():
            if key in ("receptors", "selection", "execution_metadata"):
                continue
            used[key] = copy.deepcopy(value)

        # 4) 选择策略：只保留实际使用的 FFHS 分支，并完整保留该分支字段（含 constraints）
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
                # 未知模式：退化为保留 selection 全量，避免丢参
                clean_selection = copy.deepcopy(selection_cfg)
                clean_selection["selection_mode"] = actual_mode
            used["selection"] = clean_selection

        # 5) 受体配置：仅保留本次实际使用的受体信息（避免把 target_list 里所有受体都写入快照）
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

        logger.info("配置参数快照生成完成")
        return used
    
    def _add_execution_metadata(self):
        """添加执行元数据"""
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
        # 完整保存调用方传入的执行上下文，便于复现实验（例如 selection_mode / num_processors 等）
        metadata["execution_context"] = self.execution_context
        self.used_config["execution_metadata"] = metadata

def save_config_snapshot(original_config: Dict[str, Any], 
                        execution_context: Dict[str, Any], 
                        output_file_path: str) -> bool:
    """
    保存配置参数快照到文件
    
    Args:
        original_config: 原始完整配置字典
        execution_context: 执行上下文信息
        output_file_path: 输出文件路径
        
    Returns:
        bool: 是否成功保存
    """
    try:
        # 生成配置快照
        generator = ConfigSnapshotGenerator(original_config, execution_context)
        snapshot = generator.generate_snapshot()
        
        # 确保输出目录存在
        output_dir = Path(output_file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存到文件
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        
        logger.info(f"配置参数快照已保存到: {output_file_path}")
        return True
        
    except Exception as e:
        logger.error(f"保存配置参数快照失败: {e}", exc_info=True)
        return False

def main():
    """主函数，用于测试"""
    import argparse
    
    parser = argparse.ArgumentParser(description='配置参数快照生成器')
    parser.add_argument('--config', type=str, required=True, help='原始配置文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出快照文件路径')
    parser.add_argument('--selection_mode', type=str, default='ffhs', 
                       help='实际使用的选择模式')
    
    args = parser.parse_args()
    
    # 加载原始配置
    config = load_config(args.config, PROJECT_ROOT)
    
    # 构建执行上下文（测试用）
    execution_context = {
        "config_file_path": args.config,
        "selection_mode": args.selection_mode,
        "receptor_name": "default_receptor",
        "max_generations": config.get("workflow", {}).get("max_generations", 5)
    }
    
    # 生成并保存快照
    success = save_config_snapshot(config, execution_context, args.output)
    
    if success:
        print(f"配置快照生成成功: {args.output}")
    else:
        print("配置快照生成失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 
