"""
SoftGA SoftBD 生成脚本
====================================
此脚本替换了 FragMLM (GPT) 组件。它处理：
1. 加载 SoftBD 模型（Diffusion）。
2. 基于代数的动态前缀截断。
3. SoftBD 的批量生成。
4. 过滤（有效性、唯一性、新颖性）。
5. 选择（MaxMinPicker 或 random）。
6. 全面的日志记录。
"""

import argparse
import os
import sys
import csv
import logging
import random
import math
import secrets
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
import omegaconf
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from utils.config_loader import load_config, resolve_config_path

# --- 设置路径以包含根目录模块 ---
SOFTGA_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = SOFTGA_ROOT / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

# 强制绑定 softga/model/utils.py，避免与 softga/utils 包名冲突
import importlib.util as _ilu
_utils_path = MODEL_ROOT / "utils.py"
if _utils_path.exists():
    _spec = _ilu.spec_from_file_location("utils", str(_utils_path))
    _mod = _ilu.module_from_spec(_spec)
    assert _spec is not None and _spec.loader is not None
    _spec.loader.exec_module(_mod)
    sys.modules["utils"] = _mod

import dataloader
import diffusion

# --- 日志设置 ---
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
SOFTBD_RANDOM_SEED_BITS = 31

# --- 辅助函数 ---

def _build_config(args_dict: Dict, logdir: Path) -> omegaconf.DictConfig:
    """构建 SoftBD 的 Hydra/OmegaConf 配置。"""
    base_cfg = omegaconf.OmegaConf.load(MODEL_ROOT / "configs/sample.yaml")
    
    # 检查模型配置路径
    model_name = str(args_dict.get('model', 'small-89M')).strip()
    if model_name.endswith('.yaml'):
        model_name = model_name[:-5]
    model_yaml = MODEL_ROOT / "configs/model" / f"{model_name}.yaml"
    if not model_yaml.exists():
        raise FileNotFoundError(f"找不到模型配置文件: {model_yaml}")

    model_cfg = omegaconf.OmegaConf.load(str(model_yaml))

    sampling_overrides = dict(args_dict.get("sampling") or {})
    sampling_cfg = {
        "logdir": str(logdir),
        "nucleus_p": float(args_dict.get("nucleus_p", 0.95)),
        "temperature": float(args_dict.get("temperature", 1.0)),
        # 这些标志默认值与 sample.yaml/历史行为保持一致，可在 config.yaml 的 softbd.sampling 覆盖
        "first_hitting": bool(sampling_overrides.get("first_hitting", True)),
        "top1": bool(sampling_overrides.get("top1", True)),
        "next_block_only": bool(sampling_overrides.get("next_block_only", False)),
        "kv_cache": bool(sampling_overrides.get("kv_cache", False)),
        "stop_on_eos_only": bool(sampling_overrides.get("stop_on_eos_only", True)),
        "entropy_stop": bool(sampling_overrides.get("entropy_stop", False)),
        "var_length": bool(sampling_overrides.get("var_length", True)),
        "prefix": None,  # 将在生成期间动态设置
    }

    # 如果路径不是绝对路径，则相对于 PROJECT_ROOT 解析路径。
    ckpt_path = Path(args_dict['ckpt'])
    if not ckpt_path.is_absolute():
        ckpt_path = SOFTGA_ROOT / ckpt_path
        
    vocab_path = Path(args_dict['vocab'])
    if not vocab_path.is_absolute():
        vocab_path = SOFTGA_ROOT / vocab_path

    overrides = omegaconf.OmegaConf.create({
        "seed": int(args_dict.get('seed', 42)),
        "block_size": int(args_dict.get('block_size', 4)),
        "algo": {"T": int(args_dict.get('steps', 64))},
        "model": {"length": int(args_dict.get('length', 512)), "attn_backend": "sdpa"},
        "loader": {"eval_batch_size": int(args_dict.get('batch_size', 64))},
        "sampling": sampling_cfg,
        "eval": {"checkpoint_path": str(ckpt_path.resolve())},
        "data": {"tokenizer_name_or_path": str(vocab_path.resolve())},
    })
    
    merged = omegaconf.OmegaConf.merge(base_cfg, {"model": model_cfg}, overrides)
    return merged

def calculate_keep_ratio(
    generation: int,
    max_generations: int,
    min_ratio: float,
    max_ratio: float,
    mode: str = "linear",
    progress_override: Optional[float] = None,
) -> float:
    """计算当前代需要保留的前缀比例 (0.0 - 1.0)。支持按代数或外部进度驱动。"""
    if generation <= 1: return 0.0

    # 固定阶梯策略（按代数硬编码）：Gen2=0.2, Gen3=0.4, Gen4=0.6, Gen5+=0.8
    # 说明：该模式将忽略 min_ratio/max_ratio，适用于希望前几代快速完成骨架构建的场景。
    if mode == "step_20_40_60_80":
        if progress_override is not None:
            p = max(0.0, min(1.0, float(progress_override)))
            if p < 0.25:
                return 0.2
            if p < 0.5:
                return 0.4
            if p < 0.75:
                return 0.6
            return 0.8
        if generation >= 5:
            return 0.8
        return {2: 0.2, 3: 0.4, 4: 0.6}.get(generation, 0.8)
    
    # 计算进度 0.0 -> 1.0
    if progress_override is None:
        denom = max(1, max_generations - 2)
        p = max(0.0, min(1.0, (generation - 2) / denom))
    else:
        p = max(0.0, min(1.0, float(progress_override)))
    
    # 计算保留比例系数
    if mode == "aggressive":
        progress_factor = p ** 0.5  # 快速上升
    elif mode == "super_aggressive":
        progress_factor = p ** 0.25 # 更加激进地快速上升
    elif mode == "sigmoid":
        progress_factor = 1 / (1 + math.exp(-10 * (p - 0.5)))
    elif mode == "piecewise":
        # 分段线性策略: 
        # 前 30% 阶段: 保持 min_ratio (progress_factor = 0)
        # 中 40% 阶段: 线性过渡
        # 后 30% 阶段: 保持 max_ratio (progress_factor = 1)
        if p < 0.3:
            progress_factor = 0.3
        elif p > 0.7:
            progress_factor = 1.0
        else:
            progress_factor = (p - 0.3) / 0.4
    elif mode == "cosine":
        # 余弦策略: 使用 1/2 个余弦周期 (0 到 PI)
        # progress_factor = (1 - cos(p * PI)) / 2
        progress_factor = (1.0 - math.cos(p * math.pi)) / 2.0
    elif mode == "step":
        # Step (早期饱和) 策略: 
        # 模拟 Fragment 拼接行为，在前 25% 的代数内快速从 min_ratio 增加到 max_ratio，
        # 之后一直保持 max_ratio (即只对末端进行微调)。
        # 适用于 max_generations 较大 (如 25)，但希望在前几代 (如 6) 就完成骨架构建的场景。
        saturation_point = 0.25
        if p >= saturation_point:
            progress_factor = 1.0
        else:
            progress_factor = p / saturation_point
    else:
        progress_factor = p  # linear
        
    return min_ratio + progress_factor * (max_ratio - min_ratio)

def get_fingerprint(smiles: str):
    """计算 Morgan 指纹。"""
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None

def clean_smiles(smiles: str) -> str:
    """移除 [BOS], [EOS], [PAD] tokens。"""
    return smiles.replace("[BOS]", "").replace("[EOS]", "").replace("[PAD]", "").strip()


def _to_bool(value, default: bool = True) -> bool:
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_softbd_seed(base_seed: int, softbd_config: Dict) -> Tuple[str, int]:
    cfg = softbd_config or {}
    cfg.pop("random_seed_bits", None)
    seed_mode = str(cfg.get("seed_mode", "workflow")).strip().lower()
    if seed_mode not in {"workflow", "random_per_run"}:
        logger.warning(f"softbd.seed_mode 非法: {seed_mode}，回退到 workflow")
        seed_mode = "workflow"

    if seed_mode == "random_per_run":
        softbd_seed = secrets.randbits(SOFTBD_RANDOM_SEED_BITS)
    else:
        softbd_seed = int(base_seed)

    logger.info(
        f"SoftBD seed 已确定: mode={seed_mode}, random_seed_bits={SOFTBD_RANDOM_SEED_BITS}, seed={int(softbd_seed)}"
    )
    return seed_mode, int(softbd_seed)


def _normalize_gen1_selection_mode(mode: Optional[str]) -> str:
    raw = str(mode or "").strip().lower()
    if raw in {"", "maxmin", "maxminpicker", "max_min"}:
        return "maxmin"
    if raw in {"random", "rand"}:
        return "random"
    logger.warning(f"gen1_selection_mode 非法: {mode}，回退到 maxmin")
    return "maxmin"


def is_single_component_smiles(smiles: str, mol: Chem.Mol) -> bool:
    """判定是否为单一组分（不含 '.' 且 RDKit 解析后只有 1 个 fragment）。"""
    if not smiles or "." in smiles:
        return False
    try:
        return len(Chem.GetMolFrags(mol, asMols=False)) == 1
    except Exception:
        return False


class SoftBDSampler:
    def __init__(self, softbd_config: Dict, seed: int = 42, log_dir: Optional[Path] = None):
        self.softbd_config = dict(softbd_config or {})
        self.log_dir = Path(log_dir) if log_dir is not None else Path.cwd()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._init_model(seed)

    def _seed_all(self, seed: int) -> None:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    def _init_model(self, seed: int) -> None:
        softbd_config = self.softbd_config
        ckpt_path = softbd_config.get('ckpt')
        vocab_path = softbd_config.get('vocab')
        if not ckpt_path or not vocab_path:
            raise ValueError('softbd.ckpt / softbd.vocab 缺失')

        gen_params = softbd_config.get('generation_params', {})
        sampling_cfg = dict(softbd_config.get("sampling") or {})
        args_dict = {
            'model': softbd_config.get('model', 'small-89M'),
            'ckpt': ckpt_path,
            'vocab': vocab_path,
            'length': softbd_config.get('length', 512),
            'block_size': softbd_config.get('block_size', 4),
            'steps': gen_params.get('steps', 64),
            'batch_size': gen_params.get('gpu_max_batch_size', 256),
            'nucleus_p': gen_params.get('nucleus_p', 0.95),
            'seed': int(seed),
            'temperature': softbd_config.get('temperature', 1.0),
            'sampling': sampling_cfg,
        }

        gpu_id = int(softbd_config.get('gpu', 0) or 0)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            n = torch.cuda.device_count()
            if gpu_id < 0 or gpu_id >= n:
                gpu_id = 0
            torch.cuda.set_device(gpu_id)
            self.device = torch.device('cuda', gpu_id)
        else:
            self.device = torch.device('cpu')

        self.cfg = _build_config(args_dict, self.log_dir)
        self.tokenizer = dataloader.get_tokenizer(self.cfg)
        self._seed_all(seed)

        logger.info(f"正在从 {ckpt_path} 加载 SoftBD 模型...")
        self.model = diffusion.Diffusion.load_from_checkpoint(
            self.cfg.eval.checkpoint_path,
            tokenizer=self.tokenizer,
            config=self.cfg,
            strict=False,
            weights_only=False,
        ).to(self.device)
        if self.cfg.eval.disable_ema:
            self.model.ema = None
        self.model.eval()

    def generate(
        self,
        parent_file: str,
        generation: int,
        output_dir: str,
        seed: int = 42,
        initial_samples: int = 100,
        max_generations: int = 10,
        initial_population_file: Optional[str] = None,
        strategy_progress: Optional[float] = None,
    ) -> Optional[str]:
        self._seed_all(seed)
        cfg = self.softbd_config
        recircle = _to_bool(cfg.get("recircle", True), default=True)
        gen_params = cfg.get('generation_params', {})
        sampling_cfg = dict(cfg.get("sampling") or {})
        dyn = cfg.get('dynamic_strategy', {})
        steps = int(gen_params.get('steps', getattr(self.cfg.algo, 'T', 64)))
        gpu_max_batch_size = int(gen_params.get('gpu_max_batch_size', 5000))
        samples_per_parent = int(gen_params.get('samples_per_parent', 50))

        output_path = Path(output_dir)
        log_dir = output_path / 'softbd_logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.sampling.logdir = str(log_dir)
        self.model.config.sampling.logdir = str(log_dir)
        self.model.config.sampling.nucleus_p = float(gen_params.get('nucleus_p', 0.95))
        self.model.config.sampling.temperature = float(cfg.get('temperature', 1.0))
        # 同步其他采样标志，避免“只改了 config 但运行时没生效”
        for k in ["first_hitting", "top1", "next_block_only", "kv_cache", "stop_on_eos_only", "entropy_stop", "var_length"]:
            if k in sampling_cfg:
                setattr(self.model.config.sampling, k, bool(sampling_cfg[k]))
        self.model.config.loader.eval_batch_size = gpu_max_batch_size

        final_output_file = output_path / 'gpt_generated_molecules.smi'

        if generation == 1:
            if not recircle:
                self.cfg.sampling.prefix = None
                self.model.config.sampling.prefix = None
            total_samples = int(initial_samples)
            batch_size = int(gpu_max_batch_size)
            num_batches = int(np.ceil(total_samples / batch_size))

            raw_log_file = log_dir / '3_raw_generated.smi'
            generated: List[str] = []
            with open(raw_log_file, 'w') as f_raw:
                for _ in range(num_batches):
                    cleaned = [clean_smiles(s) for s in self.model.restore_model_and_sample(num_steps=steps)]
                    generated.extend(cleaned)
                    for s in cleaned:
                        f_raw.write(s + '\n')

            valid_smiles: List[str] = []
            valid_fps = []
            seen = set()
            for smi in generated:
                # 多组分（含 '.'）直接判定为无效，避免进入后续对接/GA
                if "." in smi:
                    continue
                mol = Chem.MolFromSmiles(smi)
                if not mol:
                    continue
                can = Chem.MolToSmiles(mol, canonical=True)
                if not is_single_component_smiles(can, mol):
                    continue
                if can in seen:
                    continue
                seen.add(can)
                valid_smiles.append(can)
                valid_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

            if not valid_smiles:
                return None

            with open(log_dir / '4_valid_unique.smi', 'w') as f:
                for s in valid_smiles:
                    f.write(s + '\n')

            init_fps = []
            if initial_population_file:
                with open(initial_population_file, 'r') as f:
                    for line in f:
                        fp = get_fingerprint(line.strip().split()[0])
                        if fp:
                            init_fps.append(fp)

            gen1_n_select = int(gen_params.get("gen1_n_select", 100))
            gen1_selection_mode = _normalize_gen1_selection_mode(gen_params.get("gen1_selection_mode", "maxmin"))
            n_select = min(len(valid_smiles), max(1, gen1_n_select))
            if gen1_selection_mode == "random":
                selected = random.sample(valid_smiles, n_select)
            else:
                picker = MaxMinPicker()
                all_fps = init_fps + valid_fps
                len_init = len(init_fps)
                pick_size = min(len(all_fps), len_init + n_select)

                def dist_func(i, j):
                    return 1.0 - DataStructs.TanimotoSimilarity(all_fps[i], all_fps[j])

                picked = picker.LazyPick(dist_func, len(all_fps), pick_size, list(range(len_init)))
                selected = [valid_smiles[i - len_init] for i in picked if i >= len_init][:n_select]

            with open(log_dir / '5_selected.smi', 'w') as f:
                for s in selected:
                    f.write(s + '\n')
            with open(final_output_file, 'w') as f:
                for s in selected:
                    f.write(s + '\n')

            logger.info(
                f"SoftBD 统计: raw={len(generated)} valid_unique={len(valid_smiles)} selected={len(selected)} "
                f"gen1_selection_mode={gen1_selection_mode} NONE_VALID=0"
            )
            return str(final_output_file)

        parents: List[str] = []
        with open(parent_file, 'r') as f:
            for line in f:
                smi = line.strip().split()[0]
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    parents.append(Chem.MolToSmiles(mol, canonical=True))

        with open(log_dir / '1_input_parents.smi', 'w') as f:
            for p in parents:
                f.write(p + '\n')

        strategy_mode = str(dyn.get('strategy_mode', 'linear'))
        min_ratio = float(dyn.get('min_keep_ratio', 0.1))
        max_ratio = float(dyn.get('max_keep_ratio', 0.9))
        current_ratio = calculate_keep_ratio(
            generation,
            int(max_generations),
            min_ratio,
            max_ratio,
            mode=strategy_mode,
            progress_override=strategy_progress,
        )
        if strategy_progress is None:
            logger.info(f"动态掩码 ({strategy_mode})：目标保留比例 {current_ratio:.2f}")
        else:
            logger.info(
                "动态掩码 (%s)：目标保留比例 %.2f (oracle_progress=%.3f)",
                strategy_mode,
                current_ratio,
                float(strategy_progress),
            )

        tanimoto_threshold = float(gen_params.get('tanimoto_threshold', 0.0))
        all_tasks = []
        with open(log_dir / '2_prefixes.smi', 'w') as f_pref:
            for parent_smi in parents:
                mol = Chem.MolFromSmiles(parent_smi)
                parent_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                parent_ids = self.tokenizer.encode(parent_smi, add_special_tokens=True)
                keep = max(1, min(len(parent_ids) - 1, int(len(parent_ids) * current_ratio)))
                prefix_str = self.tokenizer.decode(parent_ids[:keep], skip_special_tokens=True)
                f_pref.write(f"{parent_smi}\t{prefix_str}\n")
                task = {'parent_smi': parent_smi, 'prefix_str': prefix_str, 'keep_tokens': keep, 'parent_fp': parent_fp}
                all_tasks.extend([task] * samples_per_parent)

        raw_generated = 0
        parent_candidates = {}
        with open(log_dir / '3_raw_generated.smi', 'w') as f_raw:
            for i in range(0, len(all_tasks), gpu_max_batch_size):
                batch = all_tasks[i:i + gpu_max_batch_size]
                prefixes = [t['prefix_str'] for t in batch]
                self.model.config.sampling.prefix = prefixes
                self.model.config.loader.eval_batch_size = len(prefixes)
                cleaned = [clean_smiles(s) for s in self.model.restore_model_and_sample(num_steps=steps)]
                for s in cleaned:
                    f_raw.write(s + '\n')
                raw_generated += len(cleaned)

                for task, gen_smi in zip(batch, cleaned):
                    # 多组分（含 '.'）直接判定为无效，避免进入候选池
                    if "." in gen_smi:
                        continue
                    cand_mol = Chem.MolFromSmiles(gen_smi)
                    if not cand_mol:
                        continue
                    cand_can = Chem.MolToSmiles(cand_mol, canonical=True)
                    if not is_single_component_smiles(cand_can, cand_mol):
                        continue
                    if cand_can == task['parent_smi']:
                        continue
                    cand_fp = AllChem.GetMorganFingerprintAsBitVect(cand_mol, 2, nBits=2048)
                    dist = 1.0 - DataStructs.TanimotoSimilarity(task['parent_fp'], cand_fp)
                    parent_candidates.setdefault(task['parent_smi'], {'info': task, 'candidates': []})['candidates'].append((cand_can, dist))

        final_selected: List[str] = []
        none_generated = 0
        none_valid = 0

        with open(log_dir / 'generation_details.csv', 'w', newline='') as csvfile:
            w = csv.writer(csvfile)
            w.writerow(['Parent_SMILES', 'Keep_Tokens', 'Prefix_SMILES', 'Generated_SMILES', 'Valid', 'Unique', 'Tanimoto_to_Parent', 'Selected', 'Selection_Note'])

            for p_smi in parents:
                group = parent_candidates.get(p_smi)
                if not group:
                    none_generated += 1
                    w.writerow([p_smi, 'N/A', 'N/A', 'NONE_GENERATED', False, False, 0.0, False, 'No Candidates'])
                    continue
                info = group['info']
                candidates = group['candidates']
                if not candidates:
                    none_valid += 1
                    w.writerow([p_smi, info['keep_tokens'], info['prefix_str'], 'NONE_VALID', False, False, 0.0, False, 'No Valid Candidates'])
                    continue

                qualified = [(s, d) for s, d in candidates if d >= tanimoto_threshold]
                if qualified:
                    child, dist = random.choice(qualified)
                    note = 'threshold_pass'
                else:
                    child, dist = max(candidates, key=lambda x: x[1])
                    note = 'fallback_max_div'

                final_selected.append(child)
                w.writerow([p_smi, info['keep_tokens'], info['prefix_str'], child, True, True, f"{dist:.4f}", True, note])

        with open(log_dir / '5_selected.smi', 'w') as f:
            for s in final_selected:
                f.write(s + '\n')
        with open(final_output_file, 'w') as f:
            for s in final_selected:
                f.write(s + '\n')

        logger.info(
            f"SoftBD 统计: parents={len(parents)} tasks={len(all_tasks)} raw={raw_generated} valid_selected={len(final_selected)} "
            f"NONE_VALID={none_valid} NONE_GENERATED={none_generated}"
        )
        return str(final_output_file)

# --- 主逻辑 ---

def run_softbd_generation(
    parent_file: str,
    generation: int,
    config_path: str,
    output_dir: str,
    softbd_config: Dict,
    seed: int = 42,
    initial_samples: int = 100,
    max_generations_override: Optional[int] = None,
    strategy_progress_override: Optional[float] = None,
) -> Optional[str]:
    logger.info(f"第 {generation} 代开始 SoftBD 生成")
    cfg_path = resolve_config_path(config_path, SOFTGA_ROOT)
    full_config = load_config(str(cfg_path), SOFTGA_ROOT)

    workflow = full_config.get('workflow', {})
    max_generations = int(max_generations_override) if max_generations_override is not None else int(workflow.get('max_generations', 10))
    initial_population_file = workflow.get('initial_population_file')
    seed_mode, softbd_seed = _resolve_softbd_seed(int(seed), softbd_config)
    softbd_config['seed_mode'] = seed_mode

    sampler = SoftBDSampler(softbd_config, seed=int(softbd_seed), log_dir=Path(output_dir) / 'softbd_logs')
    return sampler.generate(
        parent_file=parent_file,
        generation=int(generation),
        output_dir=output_dir,
        seed=int(softbd_seed),
        initial_samples=int(initial_samples),
        max_generations=max_generations,
        initial_population_file=initial_population_file,
        strategy_progress=strategy_progress_override,
    )

# --- CLI 入口点 ---

def main():
    parser = argparse.ArgumentParser(description="SoftBD 生成模块")
    parser.add_argument("--parent_file", required=True, help="输入父代 SMILES 文件")
    parser.add_argument("--generation", type=int, required=True, help="当前代数索引")
    parser.add_argument("--config_path", required=True, help="主 config.yaml 的路径")
    parser.add_argument("--output_dir", required=True, help="保存输出的目录")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial_samples", type=int, default=100, help="Initial samples count for Generation 1")
    parser.add_argument("--max_generations", type=int, default=None, help="(Optional) Override max_generations for keep-ratio schedule")
    parser.add_argument("--strategy_progress", type=float, default=None, help="(Optional) Override keep-ratio progress [0,1]")

    # 允许通过 CLI 覆盖 SoftBD 关键超参（保证上层工作流覆盖真实生效）
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--nucleus_p", type=float, default=None)
    parser.add_argument("--gpu_max_batch_size", type=int, default=None)
    parser.add_argument("--samples_per_parent", type=int, default=None)
    parser.add_argument("--tanimoto_threshold", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--gen1_selection_mode", type=str, default=None, help="Gen1 selection mode: maxmin | random")
    parser.add_argument("--strategy_mode", type=str, default=None)
    parser.add_argument("--min_keep_ratio", type=float, default=None)
    parser.add_argument("--max_keep_ratio", type=float, default=None)
    parser.add_argument("--recircle", nargs="?", const="true", default=None)
    parser.add_argument("--softbd_seed_mode", type=str, default=None)
    
    args = parser.parse_args()
    
    # 从主配置加载 SoftBD 配置部分
    cfg_path = resolve_config_path(args.config_path, SOFTGA_ROOT)
    full_config = load_config(str(cfg_path), SOFTGA_ROOT)
        
    softbd_config = full_config.get('softbd', {})
    if not softbd_config.get('enable', False):
        logger.warning("配置中禁用了 SoftBD，但脚本被调用了。")
        return

    # --- 应用 CLI 覆盖 ---
    if args.gpu is not None:
        softbd_config['gpu'] = str(args.gpu)
    if args.length is not None:
        softbd_config['length'] = int(args.length)
    if args.block_size is not None:
        softbd_config['block_size'] = int(args.block_size)
    if args.temperature is not None:
        softbd_config['temperature'] = float(args.temperature)

    gen_params = softbd_config.get('generation_params')
    if not isinstance(gen_params, dict):
        gen_params = {}
        softbd_config['generation_params'] = gen_params
    if args.nucleus_p is not None:
        gen_params['nucleus_p'] = float(args.nucleus_p)
    if args.gpu_max_batch_size is not None:
        gen_params['gpu_max_batch_size'] = int(args.gpu_max_batch_size)
    if args.samples_per_parent is not None:
        gen_params['samples_per_parent'] = int(args.samples_per_parent)
    if args.tanimoto_threshold is not None:
        gen_params['tanimoto_threshold'] = float(args.tanimoto_threshold)
    if args.steps is not None:
        gen_params['steps'] = int(args.steps)
    if args.gen1_selection_mode is not None:
        gen_params['gen1_selection_mode'] = str(args.gen1_selection_mode)

    dyn = softbd_config.get('dynamic_strategy')
    if not isinstance(dyn, dict):
        dyn = {}
        softbd_config['dynamic_strategy'] = dyn
    if args.strategy_mode is not None:
        dyn['strategy_mode'] = str(args.strategy_mode)
    if args.min_keep_ratio is not None:
        dyn['min_keep_ratio'] = float(args.min_keep_ratio)
    if args.max_keep_ratio is not None:
        dyn['max_keep_ratio'] = float(args.max_keep_ratio)
    if args.recircle is not None:
        softbd_config['recircle'] = _to_bool(args.recircle, default=True)
    if args.softbd_seed_mode is not None:
        softbd_config['seed_mode'] = str(args.softbd_seed_mode)

    result = run_softbd_generation(
        args.parent_file,
        args.generation,
        str(cfg_path),
        args.output_dir,
        softbd_config,
        args.seed,
        initial_samples=args.initial_samples,
        max_generations_override=args.max_generations,
        strategy_progress_override=args.strategy_progress,
    )
    
    if result:
        print(f"SUCCESS:{result}")
    else:
        print("FAILURE")
        sys.exit(1)

if __name__ == "__main__":
    main()
