#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SoftGA 分子对接模块
==================
qvina02: obabel + qvina02 CLI
vina: RDKit + Meeko + AutoDock Vina Python API (benchmark 对齐)
"""
import os
import sys
import shutil
import logging
import argparse
import subprocess
import tempfile
from multiprocessing import Manager, Process, Queue
from shutil import rmtree
from typing import Dict, Optional, List, Tuple
from pathlib import Path

CURRENT_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DIR = CURRENT_SCRIPT_DIR.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

from utils.config_loader import load_config, resolve_config_path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DOCKING_ASSET_PATH = os.path.join(PROJECT_ROOT_DIR, "utils/docking")
DEFAULT_CONFIG = str(PROJECT_ROOT_DIR / "config.yaml")
SUPPORTED_DOCKING_TOOLS = ("qvina02", "vina")
DEFAULT_DOCKING_TOOL = "qvina02"
DEFAULT_RECEPTOR_BOXES: Dict[str, Dict[str, Tuple[float, float, float]]] = {
    "fa7": {
        "center": (10.131, 41.879, 32.097),
        "size": (20.673, 20.198, 21.362),
    },
    "parp1": {
        "center": (26.413, 11.282, 27.238),
        "size": (18.521, 17.479, 19.995),
    },
    "5ht1b": {
        "center": (-26.602, 5.277, 17.898),
        "size": (22.5, 22.5, 22.5),
    },
    "jak2": {
        "center": (114.758, 65.496, 11.345),
        "size": (19.033, 17.929, 20.283),
    },
    "braf": {
        "center": (84.194, 6.949, -7.081),
        "size": (22.032, 19.211, 14.106),
    },
    "6GL8": {
        "center": (16.9, 2.8, 15.7),
        "size": (24.2, 21.6, 24.4),
    },
    "1UWH": {
        "center": (74.2, 43.8, 65.7),
        "size": (20.6, 25.3, 22.0),
    },
    "7OTE": {
        "center": (1.2, 8.9, -1.4),
        "size": (30.3, 18.3, 22.9),
    },
    "1KKQ": {
        "center": (74.3, 26.4, 23.8),
        "size": (24.8, 23.0, 22.0),
    },
    "5WFD": {
        "center": (24.5, -1.0, -22.7),
        "size": (19.6, 23.7, 26.2),
    },
    "7WC7": {
        "center": (-26.7, -11.3, 143.2),
        "size": (19.7, 20.0, 19.7),
    },
    "8JJL": {
        "center": (123.9, 117.4, 91.6),
        "size": (17.2, 24.1, 18.4),
    },
    "7D42": {
        "center": (18.7, -13.4, -39.8),
        "size": (22.3, 22.2, 22.7),
    },
    "7S1S": {
        "center": (-28.7, -47.2, -9.0),
        "size": (20.3, 22.7, 17.0),
    },
    "6AZV": {
        "center": (29.9, 23.3, -17.0),
        "size": (22.8, 17.9, 20.9),
    },
}

RECEPTOR_NAME_LOOKUP: Dict[str, str] = {
    key.lower(): key for key in DEFAULT_RECEPTOR_BOXES.keys()
}

def _resolve_receptor_box(target: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    if target in DEFAULT_RECEPTOR_BOXES:
        default_box = DEFAULT_RECEPTOR_BOXES[target]
        return default_box["center"], default_box["size"]
    supported = ", ".join(sorted(DEFAULT_RECEPTOR_BOXES.keys()))
    raise ValueError(f"不支持的受体: {target}。可用受体: {supported}")


def _canonicalize_receptor_name(target: str) -> str:
    receptor = str(target).strip()
    if not receptor:
        raise ValueError("受体名称不能为空")
    canonical = RECEPTOR_NAME_LOOKUP.get(receptor.lower())
    if canonical is not None:
        return canonical
    supported = ", ".join(sorted(DEFAULT_RECEPTOR_BOXES.keys()))
    raise ValueError(f"不支持的受体: {receptor}。可用受体: {supported}")


def _normalize_docking_tool(tool: Optional[str]) -> str:
    normalized = str(tool if tool is not None else DEFAULT_DOCKING_TOOL).strip().lower()
    if normalized not in SUPPORTED_DOCKING_TOOLS:
        supported = ", ".join(SUPPORTED_DOCKING_TOOLS)
        raise ValueError(f"不支持的 docking tool: {tool}。可选值: {supported}")
    return normalized


def _resolve_configured_executable(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT_DIR / path
    return path.resolve()


def _is_executable_file(path: Path) -> bool:
    return path.is_file() and os.access(str(path), os.X_OK)


def _resolve_qvina_executable(docking_executable: Optional[str]) -> str:
    checked_candidates: List[str] = []

    configured_path = None
    if docking_executable is not None:
        text = str(docking_executable).strip()
        if text:
            configured_path = _resolve_configured_executable(text)
            checked_candidates.append(str(configured_path))
            if not configured_path.exists():
                raise FileNotFoundError(
                    f"指定的 docking 可执行文件不存在: {configured_path}. "
                    "请检查 config.yaml 的 docking.executable。"
                )
            if not _is_executable_file(configured_path):
                raise PermissionError(
                    f"指定的 docking 可执行文件不可执行: {configured_path}. "
                    "请执行 chmod +x 或替换为可执行文件。"
                )
            return str(configured_path)

    candidate = Path(DOCKING_ASSET_PATH) / "qvina02"
    checked_candidates.append(str(candidate))
    if _is_executable_file(candidate):
        return str(candidate)

    checked_text = "\n".join(f"- {item}" for item in checked_candidates)
    raise FileNotFoundError(
        "找不到可用的 docking 可执行文件 (tool=qvina02)。\n"
        f"已检查:\n{checked_text}\n"
        "可通过 config.yaml:docking.executable 显式指定。"
    )

class DockingVinaSoftGA(object):
    """
    SoftGA Docking 类
    1. 资源路径指向 softga/utils/docking
    2. 使用 obabel CLI 进行格式转换，避免 Python 绑定兼容问题
    """
    def __init__(
        self,
        target,
        num_processors=1,
        seed=None,
        exhaustiveness=8,
        docking_tool: Optional[str] = None,
        docking_executable: Optional[str] = None,
        vina_energy_range: float = 4.0,
        vina_n_poses: int = 20,
    ):
        super().__init__()
        target = _canonicalize_receptor_name(target)
        self.base_dir = DOCKING_ASSET_PATH
        self.box_center, self.box_size = _resolve_receptor_box(target)

        self.protein = target
        self.docking_tool = _normalize_docking_tool(docking_tool)
        self.vina_program: Optional[str] = None
        if self.docking_tool == "qvina02":
            self.vina_program = _resolve_qvina_executable(
                docking_executable=docking_executable,
            )
        else:
            self._ensure_vina_python_stack()
        self.receptor_file = os.path.join(self.base_dir, f'{target}.pdbqt')
        
        # 检查必要文件是否存在
        if not os.path.exists(self.receptor_file):
            raise FileNotFoundError(f"找不到受体文件: {self.receptor_file}")
        if self.docking_tool == "qvina02":
            logger.info(f"Docking tool: {self.docking_tool}, executable: {self.vina_program}")
        else:
            logger.info("Docking tool: vina, backend: python-api")

        # 对接参数
        self.exhaustiveness = exhaustiveness
        self.seed = seed
        self.total_cpu = max(1, int(num_processors) if num_processors else 1)
        self.num_cpu_dock = 1
        self.num_sub_proc = self.total_cpu
        self.num_modes = 10
        self.vina_energy_range = float(vina_energy_range)
        self.vina_n_poses = int(vina_n_poses)
        self.timeout_gen3d = 30
        self.timeout_dock = 100

        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp(prefix=f"trio_dock_{target}_")
        logger.info(f"Docking temp dir created: {self.temp_dir}")
        
        # 仅 qvina02 需要 OpenBabel CLI
        if self.docking_tool == "qvina02":
            self._setup_openbabel_env()

    def _setup_openbabel_env(self):
        """配置 OpenBabel 环境变量，支持自动探测"""
        # 1. 尝试从 CONDA_PREFIX 获取
        prefix = os.environ.get('CONDA_PREFIX')
        
        # 2. 如果没有 CONDA_PREFIX，尝试从 obabel 可执行文件路径推断
        if not prefix:
            try:
                obabel_path = shutil.which('obabel')
                if obabel_path:
                    # obabel 通常在 .../bin/obabel，所以向上两级是 prefix
                    prefix = str(Path(obabel_path).resolve().parent.parent)
                    logger.info(f"Inferring OpenBabel prefix from executable: {prefix}")
            except Exception:
                pass

        if not prefix:
            logger.warning("未检测到 OpenBabel 安装路径 (CONDA_PREFIX 或 obabel 命令)，跳过环境变量自动配置。")
            return

        # 定义探测路径
        paths = {
            "BABEL_DATADIR": os.path.join(prefix, "share", "openbabel"),
            "BABEL_LIBDIR": os.path.join(prefix, "lib", "openbabel")
        }

        for env_var, base_path in paths.items():
            if not os.path.exists(base_path):
                # logger.debug(f"路径不存在: {base_path}") # 降低日志级别
                continue
                
            # 查找版本号子目录 (如 3.1.0)
            try:
                subdirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
                final_path = os.path.join(base_path, subdirs[-1]) if subdirs else base_path
                
                os.environ[env_var] = final_path
                logger.info(f"Set {env_var}: {final_path}")
            except Exception as e:
                logger.warning(f"Error setting {env_var}: {e}")

    @staticmethod
    def _ensure_vina_python_stack():
        try:
            from rdkit import Chem as _chem  # noqa: F401
            from rdkit.Chem import AllChem as _all_chem  # noqa: F401
            import meeko as _meeko  # noqa: F401
            import vina as _vina  # noqa: F401
        except Exception as e:
            raise ImportError(
                "tool=vina 需要安装并可导入 rdkit, meeko, vina（建议使用 2025-sbdd-benchmark/vina_environment.yaml）。"
            ) from e

    @staticmethod
    def _has_nonzero_coords_mol(mol_file: str, tol: float = 1e-6) -> bool:
        """检查 .mol 坐标是否存在且非全0（用于拦截坏构象）"""
        try:
            with open(mol_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception:
            return False

        if len(lines) < 4:
            return False

        counts_line = lines[3]
        atom_count = 0
        try:
            atom_count = int(counts_line[:3].strip() or "0")
        except Exception:
            parts = counts_line.split()
            if parts:
                try:
                    atom_count = int(parts[0])
                except Exception:
                    atom_count = 0

        if atom_count <= 0:
            return False

        atom_lines = lines[4:4 + atom_count]
        for line in atom_lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            except Exception:
                continue
            if abs(x) > tol or abs(y) > tol or abs(z) > tol:
                return True
        return False

    def gen_3d(self, smi, ligand_mol_file):
        """
        使用 obabel 生成 3D 构象 (.mol)。
        若直生 3D 失败或产物坐标全 0，则自动回退：
        SMILES -> raw.sdf -> --gen3D -> .mol
        返回: 是否触发了回退流程
        """
        direct_error = None
        direct_cmd = ['obabel', '-:' + smi, '--gen3D', '-O', ligand_mol_file]
        try:
            subprocess.check_output(
                direct_cmd,
                stderr=subprocess.STDOUT,
                timeout=self.timeout_gen3d,
                universal_newlines=True
            )
            if self._has_nonzero_coords_mol(ligand_mol_file):
                return False
            logger.warning("检测到全0坐标，触发3D重建回退流程。")
        except Exception as e:
            direct_error = e
            logger.warning(f"直接3D生成失败，触发回退流程: {e}")

        raw_sdf_file = os.path.splitext(ligand_mol_file)[0] + "_raw.sdf"
        fallback_sdf_file = os.path.splitext(ligand_mol_file)[0] + "_fallback3d.sdf"
        try:
            subprocess.check_output(
                ['obabel', '-:' + smi, '-O', raw_sdf_file],
                stderr=subprocess.STDOUT,
                timeout=self.timeout_gen3d,
                universal_newlines=True
            )
            subprocess.check_output(
                ['obabel', raw_sdf_file, '--gen3D', '-O', fallback_sdf_file],
                stderr=subprocess.STDOUT,
                timeout=self.timeout_gen3d,
                universal_newlines=True
            )
            subprocess.check_output(
                ['obabel', fallback_sdf_file, '-O', ligand_mol_file],
                stderr=subprocess.STDOUT,
                timeout=self.timeout_gen3d,
                universal_newlines=True
            )
        finally:
            for tmp_file in (raw_sdf_file, fallback_sdf_file):
                try:
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)
                except Exception:
                    pass

        if not self._has_nonzero_coords_mol(ligand_mol_file):
            if direct_error is not None:
                raise RuntimeError("3D构象生成失败：直接生成失败且回退后仍为全0坐标。") from direct_error
            raise RuntimeError("3D构象生成失败：回退后仍为全0坐标。")
        return True

    def convert_mol_to_pdbqt(self, mol_file, pdbqt_file):
        """使用 obabel CLI 将 .mol 转换为 .pdbqt (替代 pybel)"""
        # obabel -imol input.mol -opdbqt -O output.pdbqt
        cmd = ['obabel', '-imol', mol_file, '-opdbqt', '-O', pdbqt_file]
        subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            timeout=self.timeout_gen3d, # 复用 gen3d 的超时
            universal_newlines=True
        )

    def docking(self, receptor_file, ligand_mol_file, ligand_pdbqt_file, docking_pdbqt_file):
        """执行 Docking 流程: 转换格式 -> 调用 docking 程序(qvina02/vina)"""
        # 1. 转换 .mol -> .pdbqt
        self.convert_mol_to_pdbqt(ligand_mol_file, ligand_pdbqt_file)

        # 2. 调用 qvina02
        cmd = [
            self.vina_program,
            '--receptor', receptor_file,
            '--ligand', ligand_pdbqt_file,
            '--out', docking_pdbqt_file,
            '--center_x', str(self.box_center[0]),
            '--center_y', str(self.box_center[1]),
            '--center_z', str(self.box_center[2]),
            '--size_x', str(self.box_size[0]),
            '--size_y', str(self.box_size[1]),
            '--size_z', str(self.box_size[2]),
            '--cpu', str(self.num_cpu_dock),
            '--num_modes', str(self.num_modes),
            '--exhaustiveness', str(self.exhaustiveness)
        ]
        
        if self.seed is not None:
            cmd.extend(['--seed', str(self.seed)])
        
        result = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            timeout=self.timeout_dock,
            universal_newlines=True
        )
        
        # 3. 解析结果
        result_lines = result.split('\n')
        check_result = False
        affinity_list = list()
        
        for result_line in result_lines:
            if result_line.startswith('-----+'):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith('Writing output'):
                break
            if result_line.startswith('Refine time'):
                break
            
            lis = result_line.strip().split()
            if not lis or not lis[0].isdigit():
                break
            
            try:
                affinity = float(lis[1])
                affinity_list.append(affinity)
            except ValueError:
                continue
                
        return affinity_list

    def docking_vina_python_api(self, smi: str) -> Optional[float]:
        """Benchmark 对齐流程：RDKit + Meeko + Vina Python API。"""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        import meeko
        import vina

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        embed_status = AllChem.EmbedMolecule(mol)
        if embed_status != 0:
            return None

        meeko_prep = meeko.MoleculePreparation()
        meeko_prep.prepare(mol)
        lig_pdbqt = meeko_prep.write_pdbqt_string()

        v = vina.Vina(sf_name='vina', verbosity=0)
        v.set_receptor(self.receptor_file)
        v.set_ligand_from_string(lig_pdbqt)
        v.compute_vina_maps(center=list(self.box_center), box_size=list(self.box_size))
        v.dock(exhaustiveness=int(self.exhaustiveness), n_poses=int(self.vina_n_poses))
        energies = v.energies(n_poses=1, energy_range=float(self.vina_energy_range))
        if len(energies) == 0:
            return None
        return float(energies[0][0])

    def creator(self, q, data, num_sub_proc):
        """生产者：将任务放入队列"""
        for d in data:
            q.put(d) # (idx, smi)
        
        # 发送结束信号
        for _ in range(num_sub_proc):
            q.put('DONE')

    def docking_subprocess(self, q, return_dict, sub_id):
        """消费者子进程：执行对接任务"""
        while True:
            item = q.get()
            if item == 'DONE':
                break
            
            idx, smi = item

            if self.docking_tool == "vina":
                try:
                    score = self.docking_vina_python_api(smi)
                    return_dict[idx] = 99.9 if score is None else score
                except Exception:
                    return_dict[idx] = 99.9
                continue

            # qvina02 路径：保留原有 obabel + CLI 流程
            ligand_mol_file = os.path.join(self.temp_dir, f'ligand_{sub_id}_{idx}.mol')
            ligand_pdbqt_file = os.path.join(self.temp_dir, f'ligand_{sub_id}_{idx}.pdbqt')
            docking_pdbqt_file = os.path.join(self.temp_dir, f'dock_{sub_id}_{idx}.pdbqt')

            try:
                fallback_used = self.gen_3d(smi, ligand_mol_file)
                if fallback_used:
                    logger.info(f"子进程{sub_id}分子{idx}已使用3D重建回退流程。")
                affinity_list = self.docking(
                    self.receptor_file,
                    ligand_mol_file,
                    ligand_pdbqt_file,
                    docking_pdbqt_file
                )

                if not affinity_list:
                    return_dict[idx] = 99.9
                else:
                    return_dict[idx] = min(affinity_list)

            except Exception:
                return_dict[idx] = 99.9
                continue

    def predict(self, smiles_list):
        """
        并行预测 SMILES 列表的亲和力
        返回: 对应顺序的 affinity list
        """
        if not smiles_list:
            return []

        num_sub_proc = min(self.num_sub_proc, len(smiles_list))
        data = list(enumerate(smiles_list))
        q = Queue()
        manager = Manager()
        return_dict = manager.dict()
        
        # 启动生产者
        proc_master = Process(target=self.creator, args=(q, data, num_sub_proc))
        proc_master.start()

        # 启动消费者进程
        procs = []
        for sub_id in range(num_sub_proc):
            proc = Process(target=self.docking_subprocess, args=(q, return_dict, sub_id))
            procs.append(proc)
            proc.start()

        # 等待所有进程结束
        proc_master.join()
        for proc in procs:
            proc.join()
        
        # 整理结果
        affinity_list = []
        # keys 必须排序，以保证与输入的 smiles_list 顺序一致
        for i in range(len(smiles_list)):
            affinity_list.append(return_dict.get(i, 99.9))
            
        return affinity_list

    def __del__(self):
        """析构时清理临时目录"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp dir: {self.temp_dir}")
            except Exception as e:
                logger.error(f"Failed to clean temp dir: {e}")

def run_molecular_docking(
    config: Dict, 
    ligands_file: str, 
    generation_dir: str, 
    receptor_name: Optional[str] = None,
    override_num_processors: Optional[int] = None,
    seed: Optional[int] = None,
    override_docking_tool: Optional[str] = None,
    override_exhaustiveness: Optional[int] = None,
) -> Optional[str]:
    """
    执行 SoftGA 对接流程
    """
    logger.info("启动 SoftGA 对接流程...")
    
    # 1. 确定受体
    # 优先使用命令行传入的，否则使用配置中的默认受体
    if not receptor_name:
        receptor_name = config.get('receptors', {}).get('default_receptor', {}).get('name')
    
    if not receptor_name:
        logger.error("未指定受体名称且无法从配置中获取默认受体。")
        return None
        
    # 2. 确定并发数
    num_processors = override_num_processors
    if num_processors is None:
        num_processors = config.get('performance', {}).get('number_of_processors', 1)
        if num_processors == -1: # 自动
            num_processors = max(1, os.cpu_count() - 2)
    
    docking_config = config.get('docking', {}) or {}
    if not isinstance(docking_config, dict):
        docking_config = {}
    # 获取 exhaustiveness 配置，优先级：命令行覆盖 > config.yaml 配置 > 默认值
    exhaustiveness = (
        override_exhaustiveness
        if override_exhaustiveness is not None
        else docking_config.get('exhaustiveness', 8)
    )
    docking_tool = override_docking_tool if override_docking_tool is not None else docking_config.get('tool', DEFAULT_DOCKING_TOOL)
    docking_tool = _normalize_docking_tool(docking_tool)
    vina_energy_range = float(docking_config.get('energy_range', 4.0))
    vina_n_poses = int(docking_config.get('n_poses', 20))
    # 仅保留配置层兼容入口，不再提供 CLI 覆盖 executable
    docking_executable = docking_config.get('executable')
    if docking_tool == "qvina02" and docking_executable is not None:
        docking_executable = str(docking_executable).strip() or None
    else:
        docking_executable = None
            
    logger.info(f"Target Receptor: {receptor_name}")
    logger.info(f"Parallel Processors: {num_processors}")
    logger.info(f"Docking Exhaustiveness: {exhaustiveness}")
    logger.info(f"Docking Tool: {docking_tool}")
    if str(docking_tool).strip().lower() == "vina":
        logger.info(f"Vina API Params: n_poses={vina_n_poses}, energy_range={vina_energy_range}")
    if docking_executable:
        logger.info(f"Docking Executable (configured): {docking_executable}")
    if seed is not None:
        logger.info(f"Random Seed: {seed}")
    
    # 3. 读取 SMILES 文件
    # 输入文件格式可能是 "SMILES Name" 或仅 "SMILES"
    try:
        with open(ligands_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        clean_smiles = []
        
        for line in lines:
            parts = line.split()
            if not parts: continue
            smi = parts[0]
            clean_smiles.append(smi)
            
        if not clean_smiles:
            logger.warning("输入文件没有有效 SMILES")
            return None
            
        logger.info(f"读取到 {len(clean_smiles)} 个分子进行对接")
        
        # 4. 执行对接
        docker = DockingVinaSoftGA(
            receptor_name, 
            num_processors=num_processors, 
            seed=seed,
            exhaustiveness=exhaustiveness,
            docking_tool=docking_tool,
            docking_executable=docking_executable,
            vina_energy_range=vina_energy_range,
            vina_n_poses=vina_n_poses,
        )
        scores = docker.predict(clean_smiles)
        
        # 5. 写入结果
        # 格式：SMILES\tScore (按 Score 升序排序)
        output_dir = os.path.join(generation_dir, "docking_results")
        os.makedirs(output_dir, exist_ok=True)
        final_scores_file = os.path.join(output_dir, "final_scored.smi")
        
        results = []
        for i, score in enumerate(scores):
            # 过滤失败结果 (失败分值通常为 99.9)
            if score > 50:
                continue
            results.append((clean_smiles[i], score))
            
        # 排序：分数越低越好
        results.sort(key=lambda x: x[1])
        
        with open(final_scores_file, 'w', encoding='utf-8') as f:
            for smi, score in results:
                f.write(f"{smi}\t{score}\n")
                
        logger.info(f"对接完成，有效结果已写入: {final_scores_file}")
        return final_scores_file
        
    except Exception as e:
        logger.error(f"SoftGA 对接流程异常: {e}", exc_info=True)
        return None


def main():
    """主函数，适配 FragEvo 调用接口"""
    parser = argparse.ArgumentParser(description="SoftGA docking")
    parser.add_argument('--smiles_file', type=str, required=True, help='Input SMILES file')
    parser.add_argument('--output_file', type=str, required=True, help='Final output file path')
    parser.add_argument('--config_file', type=str, default=DEFAULT_CONFIG, help='Config file')
    parser.add_argument('--generation_dir', type=str, required=True, help='Generation directory')
    parser.add_argument('--receptor', type=str, default=None, help='Receptor name (e.g., parp1 or 6GL8)')
    parser.add_argument('--number_of_processors', type=int, default=None, help='Number of processors')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--docking_tool', type=str, default=None, help='Docking tool: qvina02 | vina')
    parser.add_argument('--exhaustiveness', type=int, default=None, help='Docking exhaustiveness override')

    args = parser.parse_args()
    
    try:
        cfg_path = resolve_config_path(args.config_file, PROJECT_ROOT_DIR)
        config = load_config(str(cfg_path), PROJECT_ROOT_DIR)
    except Exception as e:
        logging.error(f"无法加载配置文件 {args.config_file}: {e}")
        exit(1)

    final_output_file = run_molecular_docking(
        config=config,
        ligands_file=args.smiles_file,
        generation_dir=args.generation_dir,
        receptor_name=args.receptor,
        override_num_processors=args.number_of_processors,
        seed=args.seed,
        override_docking_tool=args.docking_tool,
        override_exhaustiveness=args.exhaustiveness,
    )
    
    if final_output_file and os.path.exists(final_output_file):
        # 复制到最终要求的输出路径
        try:
            shutil.copy(final_output_file, args.output_file)
            logging.info(f"最终结果已复制到: {args.output_file}")
            exit(0)
        except Exception as e:
            logger.error(f"结果复制失败: {e}")
            exit(1)
    else:
        logger.error("对接流程失败，未生成有效文件。")
        # 创建空文件防止流水线崩溃
        Path(args.output_file).touch()
        exit(1)

if __name__ == "__main__":
    main()
