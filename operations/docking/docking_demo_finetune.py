#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子对接工作流程模块
================
提供分子对接的核心功能，接受配置字典作为参数
移除了参数解析和main函数,作为模块被主流程调用
"""
import os
import sys
import glob
import subprocess
import time
import re
import shutil
from pathlib import Path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import logging
from typing import Dict, Optional, Tuple, List

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入必要的模块
from autogrow.operators.convert_files.conversion_to_3d import convert_to_3d
from autogrow.operators.convert_files.conversion_to_3d import convert_sdf_to_pdbs
from autogrow.docking.docking_class.docking_file_conversion.convert_with_mgltools import MGLToolsConversion
from autogrow.docking.docking_class.docking_file_conversion.convert_with_obabel import ObabelConversion
from autogrow.operators.convert_files.gypsum_dl.gypsum_dl.Parallelizer import Parallelizer

from autogrow.docking.docking_class.docking_class_children.vina_docking import VinaDocking
from autogrow.docking.docking_class.docking_class_children.quick_vina_2_docking import QuickVina2Docking

from autogrow.docking.execute_docking import run_docking_common
from autogrow.docking.docking_class.parent_pdbqt_converter import ParentPDBQTConverter
from autogrow.docking.docking_class.parent_dock_class import ParentDocking
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict

def vina_dock_single(ligand_file, receptor_pdbqt, results_dir, vars):
    """ 单个分子的对接函数，静默忽略失败。"""    
    out_file = os.path.join(results_dir, os.path.basename(ligand_file).replace(".pdbqt", "_out.pdbqt"))
    log_file = os.path.join(results_dir, os.path.basename(ligand_file).replace(".pdbqt", ".log"))    
    cmd = [
        vars["docking_executable"],
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_file,
        "--center_x", str(vars["center_x"]),
        "--center_y", str(vars["center_y"]),
        "--center_z", str(vars["center_z"]),
        "--size_x", str(vars["size_x"]),
        "--size_y", str(vars["size_y"]),
        "--size_z", str(vars["size_z"]),
        "--out", out_file,
        "--log", log_file
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
        score = extract_vina_score_from_pdbqt(out_file)
        if os.path.exists(log_file):#清理文件
            os.remove(log_file)
        return ligand_file, True, score
    except Exception:
        # 对接失败，静默返回NA，后续将被过滤
        return ligand_file, False, "NA"

def extract_vina_score_from_pdbqt(pdbqt_file):
    """
    从PDBQT输出文件中提取Vina对接分数    
    Args:
        pdbqt_file: PDBQT输出文件路径        
    Returns:
        str: 对接分数（负值表示更好的结合亲和力）
    """
    try:
        with open(pdbqt_file, "r") as f:
            for line in f:
                # 查找第一个VINA RESULT行（最佳分数）
                if "REMARK VINA RESULT:" in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        # REMARK VINA RESULT:      -7.6      0.000      0.000
                        # 分数是第4个元素（索引3）
                        score_str = parts[3]
                        try:
                            # 验证是否为有效的浮点数
                            float(score_str)
                            return score_str
                        except ValueError:
                            continue
    except Exception as e:
        print(f"解析PDBQT文件出错 {pdbqt_file}: {e}")    
    return "NA"

def keep_best_docking_results(results_dir):
    """    对每个分子,只保留对接分数最好的那个构象体的结果文件(_out.pdbqt 和 .log)，其余全部删除。
    """
    print(f"正在从 {results_dir} 清理非最佳构象体的对接结果...")    
    # 1. 按分子基础ID对文件进行分组
    output_files = glob.glob(os.path.join(results_dir, "*_out.pdbqt"))#
    files_by_id = defaultdict(list)    
    for file_path in output_files:
        base_name = os.path.basename(file_path)
        # 从 'naphthalene_1__1_out.pdbqt' 中提取 'naphthalene_1'
        mol_id = base_name.split('__')[0]
        files_by_id[mol_id].append(file_path)
    # 2. 遍历每个分组，找到分数最好的文件并删除其他文件
    for mol_id, file_group in files_by_id.items():
        if len(file_group) <= 1:
            continue

        best_file = None
        best_score = float('inf') 

        # 2a. 在组内找到最佳文件
        for file_path in file_group:
            score_str = extract_vina_score_from_pdbqt(file_path)
            if score_str != "NA":
                score = float(score_str)
                if score < best_score:
                    best_score = score
                    best_file = file_path
        
        if best_file is None:
            print(f"警告: 无法为ID {mol_id} 确定最佳结果，将全部保留。")
            continue
        
        # 2b. 删除组内非最佳的所有结果文件
        for file_path in file_group:
            if file_path != best_file:
                try:
                    # 删除 PDBQT 输出文件
                    os.remove(file_path)
                    
                    # 删除对应的 log 文件
                    log_file = file_path.replace("_out.pdbqt", ".log")
                    if os.path.exists(log_file):
                        os.remove(log_file)
                        
                except OSError as e:
                    print(f"删除文件失败: {file_path}, 错误: {e}")

def output_smiles_scores(smiles_file, scores_dict, output_file):
    """将成功对接的结果写入文件 不包含头 不记录NA值。"""
    # 读取SMILES与分子名映射
    smiles_map = {}
    with open(smiles_file, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    smiles_map[parts[1]] = parts[0]
    
    # 写入最终结果
    with open(output_file, "w") as out:
        for mol_name, smiles in smiles_map.items():
            score = scores_dict.get(mol_name, "NA")
            # 仅在得分有效时写入
            if score != "NA":
                out.write(f"{smiles}\t{score}\n")

class DockingWorkflow:
    """分子对接工作流程类"""    
    def __init__(self, config: Dict, generation_dir: str, ligands_file: str):
        """
        初始化工作流程        
        参数:
        :param dict config: 完整的配置参数字典
        :param str generation_dir: 当前代数专用的输出目录
        :param str ligands_file: 输入的SMILES文件路径
        """
        self.generation_dir = Path(generation_dir)
        
        # 首先从配置中初始化self.vars
        self._initialize_vars_from_config(config)
        
        # 设置输入的配体文件
        self.vars['ligands'] = ligands_file
        
        # 使用代际目录覆盖配置文件中的输出路径
        self._setup_generation_dirs()
        
        # 然后设置受体配置
        self._setup_receptor_config(config)        
        
        # 初始化运行时对象
        num_processors = self.vars.get('number_of_processors', None)
        self.vars['parallelizer'] = Parallelizer('multiprocessing', num_processors, True)
        
    def _initialize_vars_from_config(self, config: Dict):
        """从配置文件初始化vars字典"""
        # 获取各个配置部分
        docking_config = config.get('docking', {})
        paths_config = config.get('paths', {})
        
        # 初始化vars字典
        self.vars = {
            # 对接配置
            'dock_choice': docking_config.get('dock_choice', 'VinaDocking'),
            'conversion_choice': docking_config.get('conversion_choice', 'MGLToolsConversion'),
            'docking_exhaustiveness': docking_config.get('docking_exhaustiveness', 8),
            'docking_num_modes': docking_config.get('docking_num_modes', 9),
            'number_of_processors': docking_config.get('number_of_processors'),
            'max_variants_per_compound': docking_config.get('max_variants_per_compound', 3),
            'gypsum_thoroughness': docking_config.get('gypsum_thoroughness', 3),
            'gypsum_timeout_limit': docking_config.get('gypsum_timeout_limit', 15),
            'min_ph': docking_config.get('min_ph', 6.4),
            'max_ph': docking_config.get('max_ph', 8.4),
            'pka_precision': docking_config.get('pka_precision', 1.0),
            'debug_mode': docking_config.get('debug_mode', False),
            
            # 路径配置
            'output_directory': paths_config.get('temp_dir', 'GA_gpt_output/temp'),
            'ligand_dir': paths_config.get('ligand_dir', 'GA_gpt_output/ligands'),
            'sdf_dir': paths_config.get('sdf_dir', 'GA_gpt_output/ligands3D_SDFs'),
            'pdb_dir': paths_config.get('pdb_dir', 'GA_gpt_output/ligands3D_PDBs'),
            'docking_results_dir': paths_config.get('docking_results_dir', 'GA_gpt_output/docking_results'),
            
            # MGLTools路径
            'mgltools_directory': paths_config.get('mgltools_dir'),
            'mgl_python': paths_config.get('mgl_python'),
            'prepare_receptor4.py': paths_config.get('prepare_receptor4.py'),
            'prepare_ligand4.py': paths_config.get('prepare_ligand4.py'),
            
            # 对接可执行文件
            'docking_executable': paths_config.get('docking_executable'),
            'timeout_vs_gtimeout': paths_config.get('timeout_vs_gtimeout', 'timeout'),
            
            # 配体文件（运行时设置）
            # 'ligands': docking_config.get('ligands', ''), # 由构造函数直接设置
        }
        
    def _setup_generation_dirs(self):
        """创建当前代数所需的所有输出子目录"""
        self.generation_dir.mkdir(exist_ok=True, parents=True)
        
        self.vars['output_directory'] = str(self.generation_dir)
        self.vars['ligand_dir'] = str(self.generation_dir / "ligands")
        self.vars['sdf_dir'] = str(self.generation_dir / "ligands3D_SDFs")
        self.vars['pdb_dir'] = str(self.generation_dir / "ligands3D_PDBs")
        self.vars['docking_results_dir'] = str(self.generation_dir / "docking_results")
        
        # 创建这些目录
        Path(self.vars['ligand_dir']).mkdir(exist_ok=True)
        Path(self.vars['sdf_dir']).mkdir(exist_ok=True)
        Path(self.vars['pdb_dir']).mkdir(exist_ok=True)
        Path(self.vars['docking_results_dir']).mkdir(exist_ok=True)

    def _setup_receptor_config(self, config: Dict):
        """
        设置受体配置: 直接加载默认受体        
        参数:
        :param dict config: 完整的配置字典
        """       
        receptors_config = config.get('receptors', {})
        receptor_info = receptors_config.get('default_receptor')
        
        if not receptor_info or not isinstance(receptor_info, dict):
            raise ValueError("配置文件中未找到有效的 'default_receptor' 对象。")
            
        print(f"使用默认受体: {receptor_info.get('name', 'N/A')} ({receptor_info.get('description', 'No description')})")
        
        # 设置受体相关参数
        project_root = config.get('paths', {}).get('project_root', PROJECT_ROOT)
        self.vars['receptor_file'] = os.path.join(project_root, receptor_info['file'])
        self.vars['filename_of_receptor'] = self.vars['receptor_file'] # 别名，用于autogrow兼容性
        self.vars['center_x'] = receptor_info['center_x']
        self.vars['center_y'] = receptor_info['center_y'] 
        self.vars['center_z'] = receptor_info['center_z']
        self.vars['size_x'] = receptor_info['size_x']
        self.vars['size_y'] = receptor_info['size_y']
        self.vars['size_z'] = receptor_info['size_z']           
        
    
    def set_receptor(self, receptor_name: str, config: Dict):
        """
        从目标列表中动态切换到指定的受体        
        参数:
        :param str receptor_name: 目标受体名称
        :param dict config: 完整的配置字典
        """         
        receptors_config = config.get('receptors', {})
        target_list = receptors_config.get('target_list', {})
        
        if receptor_name not in target_list:
            available_receptors = list(target_list.keys())
            raise ValueError(f"目标受体 '{receptor_name}' 未在 target_list 中找到。可用目标: {available_receptors}")
            
        receptor_info = target_list[receptor_name]
        project_root = config.get('paths', {}).get('project_root', PROJECT_ROOT)
        
        # 更新受体相关参数
        self.vars['receptor_file'] = os.path.join(project_root, receptor_info['file'])
        self.vars['filename_of_receptor'] = self.vars['receptor_file']  # 兼容性
        self.vars['center_x'] = receptor_info['center_x']
        self.vars['center_y'] = receptor_info['center_y']
        self.vars['center_z'] = receptor_info['center_z']
        self.vars['size_x'] = receptor_info['size_x']
        self.vars['size_y'] = receptor_info['size_y']
        self.vars['size_z'] = receptor_info['size_z']        
        print(f"已切换到受体: {receptor_info['name']} ({receptor_info.get('description', 'No description')})")
    
    def get_target_receptors(self, config: Dict) -> List[str]:
        """
        获取所有可用于迭代的目标受体列表        
        参数:
        :param dict config: 完整的配置字典        
        返回:
        :returns: List[str]: 可用目标受体名称列表
        """                   
        receptors_config = config.get('receptors', {})
        target_list = receptors_config.get('target_list', {})
        return list(target_list.keys())
    def prepare_receptor(self) -> str:
        """
        准备受体蛋白文件。
        此函数验证预先准备好的PDBQT文件是否存在。        
        返回:returns: str: 找到的受体PDBQT文件路径。
        """
        print("正在验证受体文件...")        
        # 根据 .pdb 文件路径推断出 .pdbqt 文件路径
        receptor_pdbqt_path = self.vars["receptor_file"] + "qt"           
        print(f"成功找到受体文件: {receptor_pdbqt_path}")
        return receptor_pdbqt_path

    def prepare_ligands(self, smi_file: str) -> str:
        """准备配体分子，静默忽略转换失败的分子。"""
        print("准备配体分子...")
        # 1. SMILES转3D SDF
        ligand_dir = self.vars["ligand_dir"]
        if not os.path.exists(ligand_dir):
            os.makedirs(ligand_dir)
        convert_to_3d(self.vars, smi_file, ligand_dir)
        # 2. SDF转PDB
        sdf_dir = self.vars["sdf_dir"]
        convert_sdf_to_pdbs(self.vars, sdf_dir, sdf_dir)
        pdb_dir = sdf_dir + "_PDB"
        if not os.path.exists(pdb_dir):
            raise RuntimeError(f"PDB目录未生成: {pdb_dir}")
        # 3. PDB转PDBQT 
        pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))        
        # 实例化转换器一次，而不是在循环中
        file_conversion_class = self.pick_conversion_class(self.vars["conversion_choice"])
        file_converter = file_conversion_class(self.vars, None, test_boot=False)        
        for pdb_file in tqdm(pdb_files, desc="PDB->PDBQT转换进度"):
            pdbqt_file = pdb_file + "qt"
            # 如果文件已存在且有效，则跳过
            if os.path.exists(pdbqt_file) and os.path.getsize(pdbqt_file) > 0:
                continue            
            try:
                success, smile_name = file_converter.convert_ligand_pdb_file_to_pdbqt(pdb_file)
                # 静默处理转换失败的分子
            except Exception:
                continue            
        return pdb_dir  # 返回包含PDBQT文件的目录

    def run_docking(self, receptor_pdbqt: str, ligand_dir: str) -> str:
        print("执行分子对接...")
        #输出目录/路径
        results_dir = os.path.join(self.vars["output_directory"], "docking_results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        #配体文件查找
        ligand_files = glob.glob(os.path.join(ligand_dir, "*.pdbqt"))          
        num_workers = self.vars.get("number_of_processors")#决定使用CPU数目
        if num_workers is None:            
            num_workers = os.cpu_count() or 1 # 如果为null或未指定，则使用所有可用CPU核心
        else:            
            num_workers = int(num_workers) # 否则，使用指定数量
        print(f"将使用 {num_workers} 个CPU核心进行并行对接...")
        futures = [] #对接任务对象
        scores = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for ligand_file in ligand_files:#遍历所有配体
                futures.append(executor.submit(
                    vina_dock_single, ligand_file, receptor_pdbqt, results_dir, self.vars
                ))#添加对接任务            
            for future in tqdm(as_completed(futures), total=len(futures), desc="对接进度"):
                ligand_file, success, score_str = future.result()                   
                base_name = os.path.basename(ligand_file).replace(".pdbqt", "")  # 从完整路径中获取文件名，并去除.pdbqt后缀                  
                mol_name = base_name.split('__')[0] # 从文件名中提取基础分子名，去除构象异构体编号（如 __1, __2）               
                if success and score_str != "NA": # 只有对接成功且分数有效时才处理
                    try:
                        score_float = float(score_str)
                        current_best_score = scores.get(mol_name)                        
                        # 如果当前分子还没有记录，或者新分数更好（更低），则更新
                        if current_best_score is None or score_float < float(current_best_score):
                            scores[mol_name] = score_str                            
                    except (ValueError, TypeError):
                        if mol_name not in scores:
                            scores[mol_name] = "NA"
                else:
                    if mol_name not in scores:
                        scores[mol_name] = "NA"        
        keep_best_docking_results(results_dir)#仅保留最佳对接的mol/姿势        
        final_scores_file = os.path.join(results_dir, "final_scored.smi")
        output_smiles_scores(self.vars["ligands"], scores, final_scores_file)        
        return final_scores_file        
    @staticmethod
    def pick_conversion_class(conversion_choice: str) -> type:
        """
        选择文件转换类        
        参数:
        :param str conversion_choice: 转换方式选择        
        返回:
        :returns: type: 转换类
        """
        conversion_classes = {
            "MGLToolsConversion": MGLToolsConversion,
            "ObabelConversion": ObabelConversion
        }
        return conversion_classes.get(conversion_choice)        
    @staticmethod
    def pick_docking_class(dock_choice: str) -> type:
        """
        选择对接类        
        参数:
        :param str dock_choice: 对接程序选择
        返回:
        :returns: type: 对接类
        """
        docking_classes = {
            "VinaDocking": VinaDocking,
            "QuickVina2Docking": QuickVina2Docking
        }
        return docking_classes.get(dock_choice)

def run_molecular_docking(config: Dict, ligands_file: str, generation_dir: str, receptor_name: Optional[str] = None) -> Optional[str]:
    """
    执行完整的分子对接流程。
    
    参数:
    :param dict config: 配置字典
    :param str ligands_file: 包含SMILES和ID的文件路径
    :param str generation_dir: 当前代际的输出目录
    :param str receptor_name: (可选) 要使用的受体名称，覆盖默认受体
    
    返回:
    :return: str or None: 成功则返回包含对接分数的最终文件路径,否则返回None
    """
    logger.info("启动分子对接工作流程...")
    
    if not os.path.exists(ligands_file) or os.path.getsize(ligands_file) == 0:
        logger.error(f"输入的配体文件为空或不存在: {ligands_file}")
        return None
        
    try:
        # 实例化工作流，并传入配体文件路径
        workflow = DockingWorkflow(config, generation_dir, ligands_file)
        
        # 如果指定了受体，则切换到该受体
        if receptor_name:
            try:
                workflow.set_receptor(receptor_name, config)
            except ValueError as e:
                logger.error(f"无法设置受体 '{receptor_name}': {e}")
                return None

        # 1. 准备受体
        receptor_pdbqt = workflow.prepare_receptor()
        if not receptor_pdbqt:
            logger.error("受体准备失败")
            return None
        
        # 2. 准备配体 (SMILES -> 3D PDB -> PDBQT)
        pdbqt_dir = workflow.prepare_ligands(ligands_file)
        if not pdbqt_dir:
            logger.error("配体准备失败")
            return None
            
        # 3. 执行对接
        final_results_file = workflow.run_docking(receptor_pdbqt, pdbqt_dir)
        
        if final_results_file and os.path.exists(final_results_file):
            logger.info(f"对接成功完成。最终结果保存在: {final_results_file}")
            return final_results_file
        else:
            logger.error("对接流程失败，未生成有效的输出文件。")
            return None

    except Exception as e:
        logger.error(f"对接工作流程出现严重错误: {e}", exc_info=True)
        return None

def main():
    """主函数，用于独立运行此脚本。"""
    import argparse
    import json
    import shutil

    parser = argparse.ArgumentParser(description="分子对接流程脚本")
    parser.add_argument('--smiles_file', type=str, required=True, help='输入的SMILES文件路径')
    parser.add_argument('--output_file', type=str, required=True, help='最终输出的对接结果文件路径 (SMILES, score)')
    parser.add_argument('--config_file', type=str, default='GA_gpt/config_example.json', help='配置文件路径')
    parser.add_argument('--generation_dir', type=str, required=True, help='当前代数专用的输出目录')
    parser.add_argument('--receptor', type=str, default=None, help='要使用的受体名称，如果未提供，则使用配置文件中的默认受体')
    
    args = parser.parse_args()
    
    try:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logging.error(f"无法加载配置文件 {args.config_file}: {e}")
        exit(1)
        
    # 将smiles_file参数和receptor参数传递给对接工作流
    final_output_file = run_molecular_docking(config, args.smiles_file, args.generation_dir, args.receptor)
    
    if final_output_file:
        print(f"对接工作流成功完成: {final_output_file}")
        # 将生成的最终结果文件复制到指定的output_file
        try:
            shutil.copy(final_output_file, args.output_file)
            logging.info(f"结果已成功复制到: {args.output_file}")
            exit(0)
        except Exception as e:
            logging.error(f"无法将结果复制到 {args.output_file}: {e}")
            exit(1)
    else:
        logging.error("对接流程失败，未生成有效的输出文件。")
        # 创建一个空的输出文件，以防止下游流程因FileNotFound而崩溃
        Path(args.output_file).touch()
        exit(1)

if __name__ == "__main__":
    main()



