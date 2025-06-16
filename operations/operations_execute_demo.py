#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA操作执行脚本
==============
集成所有GA操作的完整遗传算法工作流程。
实现：初代种群 -> 迭代(交叉/突变 -> 评估 -> 选择) -> 优化分子生成
"""
import os
import sys
import json
import subprocess
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class GAWorkflowExecutor:
    """GA工作流执行器"""
    
    def __init__(self, config_path: str, receptor_name: Optional[str] = None, output_dir_override: Optional[str] = None):
        """
        初始化GA工作流执行器。
        
        Args:
            config_path (str): 配置文件路径。
            receptor_name (Optional[str]): 目标受体名称。如果为None，则使用默认受体。
            output_dir_override (Optional[str]): 覆盖配置文件中的输出目录。如果提供，将优先使用此目录。
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.receptor_name = receptor_name
        self.project_root = Path(self.config.get('paths', {}).get('project_root', PROJECT_ROOT))
        
        # 从配置中读取工作流参数
        workflow_config = self.config.get('workflow', {})
        self.max_generations = workflow_config.get('max_generations', 5)
        self.initial_population_file = workflow_config.get('initial_population_file', 'datasets/source_compounds/naphthalene_smiles.smi')
        
        # 输出目录优先级：命令行参数 > 配置文件 > 默认值
        if output_dir_override:
            output_dir_name = output_dir_override
            logger.info(f"使用命令行指定的输出目录: {output_dir_name}")
        else:
            output_dir_name = workflow_config.get('output_directory', 'GA_output')
            logger.info(f"使用配置文件中的输出目录: {output_dir_name}")

        # 创建基于受体的输出目录结构
        base_output_dir = self.project_root / output_dir_name
        
        if self.receptor_name:
            self.output_dir = base_output_dir / self.receptor_name
            logger.info(f"将为受体 '{self.receptor_name}' 在指定目录中创建输出。")
        else:
            self.output_dir = base_output_dir / "default_receptor_run"
            logger.info("未指定受体，将在默认目录中创建输出。")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        logger.info(f"GA工作流初始化完成, 输出目录: {self.output_dir}")
        logger.info(f"最大迭代代数: {self.max_generations}")
        
    def _load_config(self) -> dict:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"无法加载配置文件 {self.config_path}: {e}")
            raise
    
    def _run_script(self, script_path: str, args: List[str]) -> bool:
        """运行Python脚本，并管理输出信息"""
        full_script_path = self.project_root / script_path
        cmd = ['python', str(full_script_path)] + args
        
        logger.info(f"执行命令: {' '.join(cmd)}")
        
        try:
            # 捕获输出，但在成功时不显示stdout/stderr，以保持日志整洁
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_root), check=False)
            
            if result.returncode == 0:
                logger.info(f"脚本 {script_path} 执行成功")
                # 如果需要，可以记录一些关键的stdout信息
                # logger.debug(f"STDOUT: {result.stdout}")
                return True
            else:
                logger.error(f"脚本 {script_path} 执行失败")
                # 仅在失败时打印详细的错误输出
                logger.error(f"错误输出 (stderr):\n{result.stderr}")
                if result.stdout:
                    logger.error(f"标准输出 (stdout):\n{result.stdout}")
                return False
        except Exception as e:
            logger.error(f"执行脚本 {script_path} 时发生异常: {e}")
            return False
    
    def _count_molecules(self, file_path: str) -> int:
        """统计文件中分子数量"""
        try:
            with open(file_path, 'r') as f:
                count = sum(1 for line in f if line.strip())
            return count
        except Exception:
            return 0
    
    def _remove_duplicates_from_smiles_file(self, input_file: str, output_file: str) -> int:
        """
        去除SMILES文件中的重复分子，并为每个分子添加唯一ID。
        输出格式: SMILES  ligand_id_X
        """
        try:
            unique_smiles = set()
            with open(input_file, 'r') as f:
                for line in f:
                    smiles = line.strip().split()[0]  # 取第一列作为SMILES
                    if smiles:
                        unique_smiles.add(smiles)
            
            # 转换为列表以保证顺序
            unique_smiles_list = sorted(list(unique_smiles))
            
            with open(output_file, 'w') as f:
                for i, smiles in enumerate(unique_smiles_list):
                    ligand_id = f"ligand_id_{i}"
                    f.write(f"{smiles}\t{ligand_id}\n")
            
            logger.info(f"去重完成: {len(unique_smiles_list)} 个独特分子保存到 {output_file} (已添加ID)")
            return len(unique_smiles_list)
        except Exception as e:
            logger.error(f"去重过程中发生错误: {e}")
            return 0
    
    def _combine_files(self, file_list: List[str], output_file: str) -> bool:
        """合并多个文件"""
        try:
            with open(output_file, 'w') as outf:
                for file_path in file_list:
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as inf:
                            for line in inf:
                                line = line.strip()
                                if line:
                                    outf.write(line + '\n')
                    else:
                        logger.warning(f"文件不存在: {file_path}")
            return True
        except Exception as e:
            logger.error(f"合并文件时发生错误: {e}")
            return False
    
    def run_initial_generation(self) -> str:
        """
        执行初代种群处理
        
        Returns:
            初代种群对接结果文件路径
        """
        logger.info("=" * 60)
        logger.info("开始处理初代种群 (Generation 0)")
        logger.info("=" * 60)
        
        gen_dir = self.output_dir / "generation_0"
        gen_dir.mkdir(exist_ok=True)
        
        # 1. 去重初始种群
        initial_unique_file = gen_dir / "initial_population_unique.smi"
        unique_count = self._remove_duplicates_from_smiles_file(
            self.initial_population_file, 
            str(initial_unique_file)
        )
        
        if unique_count == 0:
            logger.error("初始种群去重失败")
            return None
        
        # 2. 对初代种群进行对接
        initial_docked_file = gen_dir / "initial_population_docked.smi"
        docking_args = [
            '--smiles_file', str(initial_unique_file),
            '--output_file', str(initial_docked_file),
            '--generation_dir', str(gen_dir),
            '--config_file', self.config_path
        ]
        if self.receptor_name:
            docking_args.extend(['--receptor', self.receptor_name])
            
        docking_succeeded = self._run_script('operations/docking/docking_demo_finetune.py', docking_args)
        
        docked_count = self._count_molecules(str(initial_docked_file))
        if not docking_succeeded or docked_count == 0:
            logger.error("初代种群对接失败或未生成任何有效对接结果。请检查对接模块配置和日志。")
            return None
        
        logger.info(f"初代种群对接完成: {docked_count} 个分子")
        
        return str(initial_docked_file)
    
    def run_genetic_operations(self, parent_file: str, generation: int) -> tuple:
        """
        执行遗传操作：交叉和突变
        
        Args:
            parent_file: 父代分子文件路径
            generation: 当前代数
            
        Returns:
            (crossover_file, mutation_file): 交叉和突变结果文件路径
        """
        logger.info(f"开始第{generation}代遗传操作")
        
        gen_dir = self.output_dir / f"generation_{generation}"
        gen_dir.mkdir(exist_ok=True)
        
        # 1. 交叉操作
        logger.info("执行交叉操作...")
        crossover_raw_file = gen_dir / "crossover_raw.smi"
        crossover_filtered_file = gen_dir / "crossover_filtered.smi"
        
        crossover_succeeded = self._run_script('operations/crossover/crossover_demo_finetune.py', [
            '--smiles_file', parent_file,
            '--output_file', str(crossover_raw_file)
        ])
        
        # 过滤交叉结果
        filter_crossover_succeeded = self._run_script('operations/filter/filter_demo.py', [
            '--smiles_file', str(crossover_raw_file),
            '--output_file', str(crossover_filtered_file)
        ])
        
        crossover_count = self._count_molecules(str(crossover_filtered_file))
        if not crossover_succeeded or not filter_crossover_succeeded:
            logger.error("交叉或过滤步骤执行失败。")
            return None, None
        
        logger.info(f"交叉操作完成: 生成 {crossover_count} 个新分子 (过滤后)")
        
        # 2. 突变操作
        logger.info("执行突变操作...")
        mutation_raw_file = gen_dir / "mutation_raw.smi"
        mutation_filtered_file = gen_dir / "mutation_filtered.smi"
        
        mutation_succeeded = self._run_script('operations/mutation/mutation_demo_finetune.py', [
            '--smiles_file', parent_file,
            '--output_file', str(mutation_raw_file)
        ])
        
        # 过滤突变结果
        filter_mutation_succeeded = self._run_script('operations/filter/filter_demo.py', [
            '--smiles_file', str(mutation_raw_file),
            '--output_file', str(mutation_filtered_file)
        ])
        
        mutation_count = self._count_molecules(str(mutation_filtered_file))
        if not mutation_succeeded or not filter_mutation_succeeded:
            logger.error("突变或过滤步骤执行失败。")
            return None, None
        
        logger.info(f"突变操作完成: 生成 {mutation_count} 个新分子 (过滤后)")
        
        return str(crossover_filtered_file), str(mutation_filtered_file)
    
    def run_offspring_evaluation(self, crossover_file: str, mutation_file: str, generation: int) -> str:
        """
        执行子代评估：合并、去重、添加ID、对接、评分
        
        Args:
            crossover_file: 交叉结果文件
            mutation_file: 突变结果文件
            generation: 当前代数
            
        Returns:
            子代对接结果文件路径
        """
        logger.info(f"开始第{generation}代子代评估")
        
        gen_dir = self.output_dir / f"generation_{generation}"
        
        # 1. 合并交叉和突变结果到一个临时文件
        offspring_raw_file = gen_dir / "offspring_combined_raw.smi"
        if not self._combine_files([crossover_file, mutation_file], str(offspring_raw_file)):
            logger.error("子代合并失败")
            return None
        
        # 2. 对合并后的文件进行去重并添加唯一ID，确保格式正确
        offspring_formatted_file = gen_dir / "offspring_formatted_for_docking.smi"
        offspring_count = self._remove_duplicates_from_smiles_file(
            str(offspring_raw_file), 
            str(offspring_formatted_file)
        )
        if offspring_count == 0:
            logger.error("子代分子在去重和格式化后为空。")
            return None
            
        logger.info(f"子代合并和格式化完成: 共 {offspring_count} 个独特的分子准备进行对接")
        
        # 3. 对格式化后的子代文件进行对接
        offspring_docked_file = gen_dir / "offspring_docked.smi"
        docking_args = [
            '--smiles_file', str(offspring_formatted_file),
            '--output_file', str(offspring_docked_file),
            '--generation_dir', str(gen_dir),
            '--config_file', self.config_path
        ]
        if self.receptor_name:
            docking_args.extend(['--receptor', self.receptor_name])
            
        docking_succeeded = self._run_script('operations/docking/docking_demo_finetune.py', docking_args)
        
        docked_count = self._count_molecules(str(offspring_docked_file))
        if not docking_succeeded or docked_count == 0:
            logger.error("子代对接失败或未生成任何有效对接结果。请检查对接模块配置和日志。")
            return None
        
        logger.info(f"子代对接完成: {docked_count} 个分子")
        
        # 4. 对子代进行评分分析
        scoring_report_file = gen_dir / f"generation_{generation}_evaluation.txt"
        if not self._run_script('operations/scoring/scoring_demo.py', [
            '--current_population_docked_file', str(offspring_docked_file),
            '--initial_population_file', self.initial_population_file,
            '--output_file', str(scoring_report_file)
        ]):
            logger.warning("子代评分分析失败，但不影响主流程")
        else:
            logger.info(f"子代评分分析完成: 报告保存到 {scoring_report_file}")
        
        return str(offspring_docked_file)
    
    def run_selection(self, parent_docked_file: str, offspring_docked_file: str, generation: int) -> str:
        """
        执行选择操作：从父代+子代中选择下一代父代
        
        Args:
            parent_docked_file: 父代对接结果文件
            offspring_docked_file: 子代对接结果文件
            generation: 当前代数
            
        Returns:
            下一代父代文件路径
        """
        logger.info(f"开始第{generation}代选择操作")
        
        gen_dir = self.output_dir / f"generation_{generation}"
        next_parents_file = gen_dir / f"next_generation_parents.smi"
        
        selection_succeeded = self._run_script('operations/selecting/molecular_selection.py', [
            '--docked_file', offspring_docked_file,
            '--parent_file', parent_docked_file,
            '--output_file', str(next_parents_file)
        ])
        
        selected_count = self._count_molecules(str(next_parents_file))
        if not selection_succeeded or selected_count == 0:
            logger.error("选择操作失败或未选出任何分子。")
            return None
        
        logger.info(f"选择操作完成: 选出 {selected_count} 个下一代父代")
        
        return str(next_parents_file)
    
    def run_complete_workflow(self):
        """执行完整的GA工作流"""
        logger.info("开始执行完整的GA工作流程")
        logger.info(f"配置文件: {self.config_path}")
        logger.info(f"输出目录: {self.output_dir}")
        
        # 第0步：初代种群处理
        current_parents_file = self.run_initial_generation()
        if not current_parents_file:
            logger.error("初代种群处理失败，工作流终止")
            return False
        
        # 开始迭代
        for generation in range(1, self.max_generations + 1):
            logger.info("=" * 60)
            logger.info(f"开始第 {generation} 代进化")
            logger.info("=" * 60)
            
            # 1. 遗传操作
            crossover_file, mutation_file = self.run_genetic_operations(current_parents_file, generation)
            if not crossover_file or not mutation_file:
                logger.error(f"第{generation}代遗传操作失败，工作流终止")
                return False
            
            # 2. 子代评估
            offspring_docked_file = self.run_offspring_evaluation(crossover_file, mutation_file, generation)
            if not offspring_docked_file:
                logger.error(f"第{generation}代子代评估失败，工作流终止")
                return False
            
            # 3. 选择操作
            next_parents_file = self.run_selection(current_parents_file, offspring_docked_file, generation)
            if not next_parents_file:
                logger.error(f"第{generation}代选择操作失败，工作流终止")
                return False
            
            # 更新父代文件，准备下一代
            current_parents_file = next_parents_file
            
            logger.info(f"第 {generation} 代进化完成")
        
        logger.info("=" * 60)
        logger.info("GA工作流程全部完成!")
        logger.info(f"最终优化结果保存在: {current_parents_file}")
        logger.info(f"所有中间结果保存在: {self.output_dir}")
        logger.info("=" * 60)
        
        return True

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GA完整工作流执行器 (可独立运行)')
    parser.add_argument('--config', type=str, 
                       default='GA_gpt/config_example.json',
                       help='配置文件路径')
    parser.add_argument('--receptor', type=str, default=None,
                       help='(可选) 要运行的目标受体名称')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='(可选) 指定输出目录，覆盖配置文件中的设置')
    
    args = parser.parse_args()
    
    try:
        # 创建并运行GA工作流
        executor = GAWorkflowExecutor(args.config, args.receptor, args.output_dir)
        
        # 执行完整工作流
        success = executor.run_complete_workflow()
        
        if success:
            logger.info("GA工作流执行成功完成")
            return 0
        else:
            logger.error("GA工作流执行失败")
            return 1
            
    except Exception as e:
        logger.error(f"工作流执行过程中发生异常: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
