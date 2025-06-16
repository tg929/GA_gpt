#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA-GPT 混合分子生成主工作流程
============================
这是FragGPT-GA项目的核心脚本,实现了一个混合的生成工作流程:
- 使用基于片段的GPT模型扩展化学多样性
- 使用遗传算法优化分子以获得高质量候选物
"""

import os
import sys
import argparse
import logging
import time
import random
import tempfile
import shutil
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import numpy as np
from rdkit import Chem

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 导入各个模块
try:
    # 导入分解和掩码模块
    from datasets.decompose.demo_frags import batch_process as decompose_batch_process
    from datasets.decompose.demo_frags import break_into_fragments
    
    # 导入GPT生成模块
    from fragment_GPT.generate_all import main_test as gpt_generate
    
    # 导入评分模块
    from operations.scoring.scoring_demo import load_smiles_from_file, get_rdkit_mols
    from operations.scoring.scoring_demo import calculate_qed_scores, calculate_sa_scores
    
    # 导入对接模块
    from operations.docking.docking_demo_finetune import DockingWorkflow
    
    # 导入交叉模块
    from operations.crossover.crossover_demo_finetune import main as crossover_main
    
    # 导入变异模块  
    from operations.mutation.mutation_demo_finetune import main as mutation_main
    
    # 导入过滤模块
    from operations.filter.filter_demo import main as filter_main
    
    # 导入选择模块
    from operations.selecting.selecting_multi_demo import (
        load_molecules_with_scores, add_additional_scores, 
        select_molecules_nsga2, save_selected_molecules
    )
    
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保所有必需的模块都已正确安装")
    sys.exit(1)


class GAGPTWorkflow:
    """GA-GPT混合分子生成工作流程主类"""
    
    def __init__(self, config: Dict):
        """
        初始化工作流程
        
        Args:
            config: 配置参数字典
        """
        self.config = config
        self.logger = self._setup_logging()
        self.generation = 0
        self.population_size = config.get('population_size', 115)
        self.max_generations = config.get('max_generations', 10)
        self.output_dir = config.get('output_dir', 'output')
        self.temp_dir = config.get('temp_dir', 'temp')
        
        # 创建必要的目录
        self._setup_directories()
        
        # 初始化种群存储
        self.current_population = []
        self.population_scores = {}
        
        self.logger.info("GA-GPT工作流程初始化完成")
        self.logger.info(f"种群大小: {self.population_size}")
        self.logger.info(f"最大代数: {self.max_generations}")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志系统"""
        logger = logging.getLogger('GAGPTWorkflow')
        logger.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 文件处理器
        log_file = os.path.join(self.config.get('output_dir', 'output'), 'workflow.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_directories(self):
        """创建必要的目录结构"""
        dirs_to_create = [
            self.output_dir,
            self.temp_dir,
            os.path.join(self.output_dir, 'generations'),
            os.path.join(self.output_dir, 'fragments'),
            os.path.join(self.output_dir, 'gpt_outputs'),
            os.path.join(self.output_dir, 'docking_results'),
            os.path.join(self.output_dir, 'crossover_results'),
            os.path.join(self.output_dir, 'mutation_results'),
            os.path.join(self.output_dir, 'filtered_results'),
            os.path.join(self.output_dir, 'selection_results')
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
    
    def initialize_population(self, initial_smiles_file: str) -> List[str]:       
        self.logger.info("正在初始化起始种群...")        
        # 加载初始分子
        initial_smiles = load_smiles_from_file(initial_smiles_file)        
        if not initial_smiles:
            raise ValueError(f"无法从文件 {initial_smiles_file} 加载初始分子")
        
        # 如果分子数量超过期望的种群大小，随机采样
        if len(initial_smiles) > self.population_size:
            initial_smiles = random.sample(initial_smiles, self.population_size)
        # 如果分子数量不足，重复采样至目标大小
        elif len(initial_smiles) < self.population_size:
            while len(initial_smiles) < self.population_size:
                initial_smiles.extend(random.choices(initial_smiles, 
                                                   k=min(len(initial_smiles), 
                                                        self.population_size - len(initial_smiles))))
        
        self.current_population = initial_smiles[:self.population_size]
        
        # 保存初始种群
        init_pop_file = os.path.join(self.output_dir, 'generations', 'generation_0_initial.smi')
        with open(init_pop_file, 'w') as f:
            for smi in self.current_population:
                f.write(f"{smi}\n")
        
        self.logger.info(f"初始种群已创建，包含 {len(self.current_population)} 个分子")
        return self.current_population
    
    def evaluate_population(self, smiles_list: List[str], generation: int) -> Dict:
        """
        评估种群的适应度分数
        
        Args:
            smiles_list: SMILES列表
            generation: 当前代数
            
        Returns:
            评估结果字典
        """
        self.logger.info(f"正在评估第 {generation} 代种群 ({len(smiles_list)} 个分子)...")
        
        # 创建临时文件保存SMILES
        temp_smiles_file = os.path.join(self.temp_dir, f'gen_{generation}_for_eval.smi')
        with open(temp_smiles_file, 'w') as f:
            for i, smi in enumerate(smiles_list):
                f.write(f"{smi}\tnaphthalene_{i}\n")
        
        # 1. 计算对接分数
        self.logger.info("正在计算对接分数...")
        docking_results = self._run_docking(temp_smiles_file, generation)
        
        # 2. 计算QED和SA分数
        self.logger.info("正在计算QED和SA分数...")
        mols, valid_smiles = get_rdkit_mols(smiles_list)
        qed_scores = calculate_qed_scores(mols)
        sa_scores = calculate_sa_scores(mols)
        
        # 3. 整合评估结果
        evaluation_results = {
            'smiles': valid_smiles,
            'docking_scores': docking_results,
            'qed_scores': qed_scores,
            'sa_scores': sa_scores,
            'generation': generation
        }
        
        # 保存评估结果
        self._save_evaluation_results(evaluation_results, generation)
        
        self.logger.info(f"第 {generation} 代种群评估完成")
        return evaluation_results
    
    def _run_docking(self, smiles_file: str, generation: int) -> List[float]:
        """
        运行分子对接
        
        Args:
            smiles_file: SMILES文件路径
            generation: 当前代数
            
        Returns:
            对接分数列表
        """
        # 配置对接参数
        docking_config = {
            'output_directory': os.path.join(self.output_dir, 'docking_results', f'gen_{generation}'),
            'filename_of_receptor': self.config.get('receptor_pdb_file', 'receptor.pdb'),
            'center_x': self.config.get('center_x', 0.0),
            'center_y': self.config.get('center_y', 0.0), 
            'center_z': self.config.get('center_z', 0.0),
            'size_x': self.config.get('size_x', 20.0),
            'size_y': self.config.get('size_y', 20.0),
            'size_z': self.config.get('size_z', 20.0),
            'ligand_dir': os.path.join(self.temp_dir, f'ligands_gen_{generation}'),
            'sdf_dir': os.path.join(self.temp_dir, f'sdf_gen_{generation}'),
            'conversion_choice': 'mgltools',
            'docking_executable': self.config.get('vina_executable', 'vina'),
            'mgl_python': self.config.get('mgl_python', 'python'),
            'prepare_receptor4.py': self.config.get('prepare_receptor4', 'prepare_receptor4.py'),
            'prepare_ligand4.py': self.config.get('prepare_ligand4', 'prepare_ligand4.py'),
            'number_of_processors': self.config.get('num_processors', 4),
            'gypsum_timeout_limit': 120.0
        }
        
        # 创建对接工作流程并运行
        try:
            docking_workflow = DockingWorkflow(docking_config)            
            # 准备受体
            receptor_pdbqt = docking_workflow.prepare_receptor()            
            # 准备配体
            ligand_dir = docking_workflow.prepare_ligands(smiles_file)            
            # 运行对接
            results_file = docking_workflow.run_docking(receptor_pdbqt, ligand_dir)            
            # 解析对接结果
            docking_scores = []
            try:
                with open(results_file, 'r') as f:
                    next(f)  # 跳过标题行
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            try:
                                score = float(parts[1])
                                docking_scores.append(score)
                            except ValueError:
                                docking_scores.append(0.0)  # 默认分数
            except FileNotFoundError:
                self.logger.warning(f"对接结果文件未找到: {results_file}")
                # 返回默认分数
                num_molecules = len(load_smiles_from_file(smiles_file))
                docking_scores = [0.0] * num_molecules
            
            return docking_scores
            
        except Exception as e:
            self.logger.error(f"对接过程出错: {e}")
            # 返回默认分数
            num_molecules = len(load_smiles_from_file(smiles_file))
            return [0.0] * num_molecules
    
    def decompose_and_mask(self, smiles_list: List[str], generation: int, 
                          n_fragments_to_mask: int = 1) -> str:
        """
        分解分子并应用掩码
        
        Args:
            smiles_list: SMILES列表
            generation: 当前代数
            n_fragments_to_mask: 要掩码的片段数量
            
        Returns:
            掩码后的片段文件路径
        """
        self.logger.info(f"正在分解第 {generation} 代分子并应用掩码...")
        
        # 创建输入文件
        input_file = os.path.join(self.temp_dir, f'gen_{generation}_for_decompose.smi')
        with open(input_file, 'w') as f:
            for smi in smiles_list:
                f.write(f"{smi}\n")
        
        # 设置输出文件路径
        output_dir = os.path.join(self.output_dir, 'fragments', f'gen_{generation}')
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, 'decomposed_results.smi')
        formatted_file = os.path.join(output_dir, 'formatted_fragments.smi')
        masked_file = os.path.join(output_dir, 'masked_fragments.smi')
        original_file = os.path.join(output_dir, 'original_smiles.smi')        
        # 运行分解
        try:
            decompose_batch_process(input_file, output_file, formatted_file, 
                                  masked_file, original_file)
            
            # 实现灵活的掩码逻辑（根据n_fragments_to_mask参数）
            self._apply_flexible_masking(masked_file, n_fragments_to_mask)
            
            self.logger.info(f"分子分解和掩码完成，结果保存在 {output_dir}")
            return masked_file
            
        except Exception as e:
            self.logger.error(f"分子分解过程出错: {e}")
            # 创建一个空的掩码文件作为备用
            with open(masked_file, 'w') as f:
                for smi in smiles_list:
                    f.write(f"[BOS]{smi}[SEP]\n")
            return masked_file
    
    def _apply_flexible_masking(self, masked_file: str, n_fragments: int):
        """
        应用灵活的片段掩码逻辑
        
        Args:
            masked_file: 掩码文件路径
            n_fragments: 要掩码的片段数量
        """
        if n_fragments <= 1:
            return  # 默认掩码已经是掩码最后1个片段
        
        # 读取现有的掩码文件
        with open(masked_file, 'r') as f:
            lines = f.readlines()
        
        # 重新应用掩码逻辑
        new_lines = []
        for line in lines:
            line = line.strip()
            if line and '[SEP]' in line:
                # 移除BOS和EOS标记
                content = line.replace('[BOS]', '').replace('[EOS]', '')
                fragments = content.split('[SEP]')
                
                # 移除空片段
                fragments = [f for f in fragments if f.strip()]
                
                if len(fragments) > n_fragments:
                    # 掩码最后n个片段
                    masked_fragments = fragments[:-n_fragments]
                    new_line = f"[BOS]{('[SEP]'.join(masked_fragments))}[SEP]\n"
                else:
                    # 如果片段数不足，保留原始格式
                    new_line = line + "\n"
                
                new_lines.append(new_line)
            else:
                new_lines.append(line + "\n")
        
        # 写回文件
        with open(masked_file, 'w') as f:
            f.writelines(new_lines)
    
    def generate_with_gpt(self, masked_fragments_file: str, generation: int) -> str:
        """
        使用GPT生成新分子
        
        Args:
            masked_fragments_file: 掩码片段文件路径
            generation: 当前代数
            
        Returns:
            生成的分子文件路径
        """
        self.logger.info(f"正在使用GPT生成第 {generation} 代新分子...")
        
        # 创建输出目录
        gpt_output_dir = os.path.join(self.output_dir, 'gpt_outputs', f'gen_{generation}')
        os.makedirs(gpt_output_dir, exist_ok=True)
        
        try:
            # 模拟调用GPT生成函数的参数
            class GPTArgs:
                def __init__(self):
                    self.device = self.config.get('gpu_device', '0')
                    self.seed = str(generation)  # 使用代数作为随机种子
                    self.input_file = masked_fragments_file
            
            args = GPTArgs()
            
            # 调用GPT生成
            # 注意：这里需要根据实际的generate_all.py接口进行调整
            gpt_generate(args)
            
            # GPT生成的结果文件路径（根据generate_all.py的输出格式）
            generated_file = os.path.join(PROJECT_ROOT, "fragment_GPT/output", 
                                        f"crossovered0_frags_new_{generation}.smi")
            
            # 复制到我们的输出目录
            target_file = os.path.join(gpt_output_dir, f'gpt_generated_{generation}.smi')
            if os.path.exists(generated_file):
                shutil.copy2(generated_file, target_file)
            else:
                # 创建空文件作为备用
                with open(target_file, 'w') as f:
                    f.write("")
                self.logger.warning(f"GPT生成文件未找到: {generated_file}")
            
            self.logger.info(f"GPT生成完成,结果保存在 {target_file}")
            return target_file
            
        except Exception as e:
            self.logger.error(f"GPT生成过程出错: {e}")
            # 创建空的生成文件作为备用
            empty_file = os.path.join(gpt_output_dir, f'gpt_generated_{generation}.smi')
            with open(empty_file, 'w') as f:
                f.write("")
            return empty_file
    
    def perform_crossover(self, parent_population: List[str], 
                         gpt_generated_file: str, generation: int) -> str:
        """
        执行交叉操作
        
        Args:
            parent_population: 父代种群
            gpt_generated_file: GPT生成的分子文件
            generation: 当前代数
            
        Returns:
            交叉结果文件路径
        """
        self.logger.info(f"正在执行第 {generation} 代交叉操作...")
        
        # 创建父代种群文件
        parent_file = os.path.join(self.temp_dir, f'gen_{generation}_parents.smi')
        with open(parent_file, 'w') as f:
            for smi in parent_population:
                f.write(f"{smi}\n")
        
        # 设置交叉结果输出目录
        crossover_output_dir = os.path.join(self.output_dir, 'crossover_results', f'gen_{generation}')
        os.makedirs(crossover_output_dir, exist_ok=True)
        
        crossover_output_file = os.path.join(crossover_output_dir, f'crossover_gen_{generation}.smi')
        
        try:
            # 模拟调用交叉函数的参数
            class CrossoverArgs:
                def __init__(self):
                    self.source_compound_file = parent_file
                    self.llm_generation_file = gpt_generated_file
                    self.output_file = crossover_output_file
                    self.crossover_rate = self.config.get('crossover_rate', 0.8)
                    self.crossover_attempts = self.config.get('crossover_attempts', 
                                                            max(1, len(parent_population) // 4))
                    self.output_dir = crossover_output_dir
            
            args = CrossoverArgs()
            
            # 调用交叉函数
            crossover_main(args)
            
            self.logger.info(f"交叉操作完成，结果保存在 {crossover_output_file}")
            return crossover_output_file
            
        except Exception as e:
            self.logger.error(f"交叉操作出错: {e}")
            # 创建空的交叉文件作为备用
            with open(crossover_output_file, 'w') as f:
                f.write("")
            return crossover_output_file
    
    def perform_mutation(self, parent_population: List[str], 
                        gpt_generated_file: str, generation: int) -> str:
        """
        执行变异操作
        
        Args:
            parent_population: 父代种群
            gpt_generated_file: GPT生成的分子文件
            generation: 当前代数
            
        Returns:
            变异结果文件路径
        """
        self.logger.info(f"正在执行第 {generation} 代变异操作...")
        
        # 创建父代种群文件
        parent_file = os.path.join(self.temp_dir, f'gen_{generation}_parents.smi')
        with open(parent_file, 'w') as f:
            for smi in parent_population:
                f.write(f"{smi}\n")
        
        # 设置变异结果输出目录
        mutation_output_dir = os.path.join(self.output_dir, 'mutation_results', f'gen_{generation}')
        os.makedirs(mutation_output_dir, exist_ok=True)
        
        mutation_output_file = os.path.join(mutation_output_dir, f'mutation_gen_{generation}.smi')
        
        try:
            # 模拟调用变异函数的参数
            class MutationArgs:
                def __init__(self):
                    self.input_file = parent_file
                    self.llm_generation_file = gpt_generated_file
                    self.output_file = mutation_output_file
                    self.num_mutations = self.config.get('mutation_attempts', 
                                                       max(1, len(parent_population) // 4))
                    self.max_mutations = self.config.get('max_mutations_per_parent', 2)
                    self.output_dir = mutation_output_dir
            
            args = MutationArgs()
            
            # 调用变异函数
            mutation_main(args)
            
            self.logger.info(f"变异操作完成，结果保存在 {mutation_output_file}")
            return mutation_output_file
            
        except Exception as e:
            self.logger.error(f"变异操作出错: {e}")
            # 创建空的变异文件作为备用
            with open(mutation_output_file, 'w') as f:
                f.write("")
            return mutation_output_file
    
    def filter_population(self, child_population_file: str, generation: int) -> str:
        """
        过滤子代种群
        
        Args:
            child_population_file: 子代种群文件路径
            generation: 当前代数
            
        Returns:
            过滤后的种群文件路径
        """
        self.logger.info(f"正在过滤第 {generation} 代子代种群...")
        
        # 设置过滤结果输出目录
        filter_output_dir = os.path.join(self.output_dir, 'filtered_results', f'gen_{generation}')
        os.makedirs(filter_output_dir, exist_ok=True)
        
        filtered_output_file = os.path.join(filter_output_dir, f'filtered_gen_{generation}.smi')
        
        try:
            # 模拟调用过滤函数的参数
            class FilterArgs:
                def __init__(self):
                    self.input = child_population_file
                    self.output = filtered_output_file
                    # 启用基本的药物化学过滤器
                    self.LipinskiLenientFilter = True
                    self.PAINSFilter = True
                    self.No_Filters = False
            
            args = FilterArgs()
            
            # 调用过滤函数
            filter_main(args)
            
            self.logger.info(f"种群过滤完成，结果保存在 {filtered_output_file}")
            return filtered_output_file
            
        except Exception as e:
            self.logger.error(f"种群过滤出错: {e}")
            # 复制原文件作为备用
            shutil.copy2(child_population_file, filtered_output_file)
            return filtered_output_file
    
    def select_next_generation(self, parent_scores: Dict, child_scores: Dict, 
                              generation: int) -> List[str]:
        """
        选择下一代种群
        
        Args:
            parent_scores: 父代评估结果
            child_scores: 子代评估结果
            generation: 当前代数
            
        Returns:
            下一代种群的SMILES列表
        """
        self.logger.info(f"正在选择第 {generation+1} 代种群...")
        
        # 合并父代和子代数据
        all_molecules = []
        
        # 添加父代分子
        for i, smi in enumerate(parent_scores.get('smiles', [])):
            mol_data = {
                'smiles': smi,
                'docking_score': parent_scores['docking_scores'][i] if i < len(parent_scores['docking_scores']) else 0.0,
                'qed_score': parent_scores['qed_scores'][i] if i < len(parent_scores['qed_scores']) else 0.0,
                'sa_score': parent_scores['sa_scores'][i] if i < len(parent_scores['sa_scores']) else 5.0
            }
            all_molecules.append(mol_data)
        
        # 添加子代分子
        for i, smi in enumerate(child_scores.get('smiles', [])):
            mol_data = {
                'smiles': smi,
                'docking_score': child_scores['docking_scores'][i] if i < len(child_scores['docking_scores']) else 0.0,
                'qed_score': child_scores['qed_scores'][i] if i < len(child_scores['qed_scores']) else 0.0,
                'sa_score': child_scores['sa_scores'][i] if i < len(child_scores['sa_scores']) else 5.0
            }
            all_molecules.append(mol_data)
        
        # 去重
        unique_molecules = {}
        for mol in all_molecules:
            smi = mol['smiles']
            if smi not in unique_molecules:
                unique_molecules[smi] = mol
        
        all_molecules = list(unique_molecules.values())
        
        try:
            # 使用NSGA-II多目标选择
            selected_molecules = select_molecules_nsga2(
                all_molecules,
                n_select_fitness=max(50, self.population_size // 2),
                n_select_diversity=max(25, self.population_size // 4),
                population_size=100,
                generations=50
            )
            
            # 提取SMILES
            selected_smiles = [mol['smiles'] for mol in selected_molecules]
            
            # 如果选择的分子数量不足，用最佳分子补充
            if len(selected_smiles) < self.population_size:
                # 按对接分数排序
                sorted_molecules = sorted(all_molecules, key=lambda x: x['docking_score'])
                for mol in sorted_molecules:
                    if mol['smiles'] not in selected_smiles:
                        selected_smiles.append(mol['smiles'])
                        if len(selected_smiles) >= self.population_size:
                            break
            
            # 截取到目标种群大小
            selected_smiles = selected_smiles[:self.population_size]
            
            # 保存选择结果
            selection_output_dir = os.path.join(self.output_dir, 'selection_results', f'gen_{generation}')
            os.makedirs(selection_output_dir, exist_ok=True)
            
            selection_file = os.path.join(selection_output_dir, f'selected_gen_{generation+1}.smi')
            with open(selection_file, 'w') as f:
                for smi in selected_smiles:
                    f.write(f"{smi}\n")
            
            self.logger.info(f"选择完成，第 {generation+1} 代种群包含 {len(selected_smiles)} 个分子")
            return selected_smiles
            
        except Exception as e:
            self.logger.error(f"种群选择出错: {e}")
            # 备用选择策略：按对接分数选择最佳分子
            sorted_molecules = sorted(all_molecules, key=lambda x: x['docking_score'])
            selected_smiles = [mol['smiles'] for mol in sorted_molecules[:self.population_size]]
            return selected_smiles
    
    def _save_evaluation_results(self, results: Dict, generation: int):
        """保存评估结果"""
        results_dir = os.path.join(self.output_dir, 'generations')
        
        # 保存详细结果
        results_file = os.path.join(results_dir, f'generation_{generation}_scores.txt')
        with open(results_file, 'w') as f:
            f.write("SMILES\tDocking_Score\tQED_Score\tSA_Score\n")
            for i, smi in enumerate(results['smiles']):
                docking = results['docking_scores'][i] if i < len(results['docking_scores']) else 'NA'
                qed = results['qed_scores'][i] if i < len(results['qed_scores']) else 'NA'
                sa = results['sa_scores'][i] if i < len(results['sa_scores']) else 'NA'
                f.write(f"{smi}\t{docking}\t{qed}\t{sa}\n")
    
    def run_workflow(self, initial_smiles_file: str):
        """
        运行完整的GA-GPT工作流程
        
        Args:
            initial_smiles_file: 初始SMILES文件路径
        """
        self.logger.info("="*60)
        self.logger.info("开始GA-GPT混合分子生成工作流程")
        self.logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # 1. 初始化种群
            self.current_population = self.initialize_population(initial_smiles_file)
            
            # 2. 评估初始种群
            parent_scores = self.evaluate_population(self.current_population, 0)
            
            # 3. 迭代进化
            for generation in range(self.max_generations):
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"开始第 {generation+1} 代进化")
                self.logger.info(f"{'='*50}")
                
                # 3.1 分解和掩码
                masked_file = self.decompose_and_mask(
                    self.current_population, 
                    generation, 
                    n_fragments_to_mask=self.config.get('n_fragments_to_mask', 1)
                )
                
                # 3.2 GPT生成
                gpt_generated_file = self.generate_with_gpt(masked_file, generation)
                
                # 3.3 交叉操作
                crossover_file = self.perform_crossover(
                    self.current_population, 
                    gpt_generated_file, 
                    generation
                )
                
                # 3.4 变异操作
                mutation_file = self.perform_mutation(
                    self.current_population, 
                    gpt_generated_file, 
                    generation
                )
                
                # 3.5 合并子代种群
                child_population = []
                
                # 加载GPT生成的分子
                if os.path.exists(gpt_generated_file):
                    child_population.extend(load_smiles_from_file(gpt_generated_file))
                
                # 加载交叉生成的分子
                if os.path.exists(crossover_file):
                    child_population.extend(load_smiles_from_file(crossover_file))
                
                # 加载变异生成的分子
                if os.path.exists(mutation_file):
                    child_population.extend(load_smiles_from_file(mutation_file))
                
                # 去重
                child_population = list(set(child_population))
                
                if not child_population:
                    self.logger.warning(f"第 {generation+1} 代没有生成新分子，跳过...")
                    continue
                
                # 保存子代种群
                child_file = os.path.join(self.temp_dir, f'gen_{generation+1}_children.smi')
                with open(child_file, 'w') as f:
                    for smi in child_population:
                        f.write(f"{smi}\n")
                
                # 3.6 过滤子代种群
                filtered_child_file = self.filter_population(child_file, generation+1)
                
                # 加载过滤后的子代
                filtered_children = load_smiles_from_file(filtered_child_file)
                
                # 3.7 评估子代种群
                child_scores = self.evaluate_population(filtered_children, generation+1)
                
                # 3.8 选择下一代
                self.current_population = self.select_next_generation(
                    parent_scores, 
                    child_scores, 
                    generation
                )
                
                # 3.9 更新父代分数为下次迭代做准备
                parent_scores = self.evaluate_population(self.current_population, generation+1)
                
                self.logger.info(f"第 {generation+1} 代进化完成")
            
            # 4. 保存最终结果
            final_file = os.path.join(self.output_dir, 'final_population.smi')
            with open(final_file, 'w') as f:
                for smi in self.current_population:
                    f.write(f"{smi}\n")
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"\n{'='*60}")
            self.logger.info("GA-GPT工作流程完成！")
            self.logger.info(f"总运行时间: {elapsed_time:.2f} 秒")
            self.logger.info(f"最终种群大小: {len(self.current_population)}")
            self.logger.info(f"最终结果保存在: {final_file}")
            self.logger.info(f"{'='*60}")
            
        except Exception as e:
            self.logger.error(f"工作流程出错: {e}")
            raise


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='GA-GPT混合分子生成主工作流程',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument('--initial_smiles', '-i', required=True,
                       help='初始SMILES文件路径')
    parser.add_argument('--receptor_pdb', '-r', required=True,
                       help='受体蛋白PDB文件路径')
    
    # 工作流程参数
    parser.add_argument('--population_size', '-p', type=int, default=115,
                       help='种群大小')
    parser.add_argument('--max_generations', '-g', type=int, default=10,
                       help='最大进化代数')
    parser.add_argument('--output_dir', '-o', default='output',
                       help='输出目录')
    parser.add_argument('--temp_dir', default='temp',
                       help='临时文件目录')
    
    # 分子分解参数
    parser.add_argument('--n_fragments_to_mask', type=int, default=1,
                       help='要掩码的片段数量')
    
    # GA操作参数
    parser.add_argument('--crossover_rate', type=float, default=0.8,
                       help='交叉率')
    parser.add_argument('--crossover_attempts', type=int, default=20,
                       help='交叉尝试次数')
    parser.add_argument('--mutation_attempts', type=int, default=20,
                       help='变异尝试次数')
    parser.add_argument('--max_mutations_per_parent', type=int, default=2,
                       help='每个父代的最大变异尝试次数')
    
    # 对接参数
    parser.add_argument('--center_x', type=float, default=0.0,
                       help='对接中心X坐标')
    parser.add_argument('--center_y', type=float, default=0.0,
                       help='对接中心Y坐标')
    parser.add_argument('--center_z', type=float, default=0.0,
                       help='对接中心Z坐标')
    parser.add_argument('--size_x', type=float, default=20.0,
                       help='对接空间X大小')
    parser.add_argument('--size_y', type=float, default=20.0,
                       help='对接空间Y大小')
    parser.add_argument('--size_z', type=float, default=20.0,
                       help='对接空间Z大小')
    
    # 软件路径参数
    parser.add_argument('--vina_executable', default='vina',
                       help='AutoDock Vina可执行文件路径')
    parser.add_argument('--mgl_python', default='python',
                       help='MGLTools Python路径')
    parser.add_argument('--prepare_receptor4', default='prepare_receptor4.py',
                       help='prepare_receptor4.py脚本路径')
    parser.add_argument('--prepare_ligand4', default='prepare_ligand4.py',
                       help='prepare_ligand4.py脚本路径')
    
    # 计算资源参数
    parser.add_argument('--gpu_device', default='0',
                       help='GPU设备ID')
    parser.add_argument('--num_processors', type=int, default=4,
                       help='CPU处理器数量')
    
    return parser.parse_args()


def main():
    """主函数"""
    print("GA-GPT混合分子生成系统")
    print("作者: AI Assistant")
    print("版本: 1.0.0")
    print()
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 验证输入文件
    if not os.path.exists(args.initial_smiles):
        print(f"错误: 初始SMILES文件不存在: {args.initial_smiles}")
        sys.exit(1)
    
    if not os.path.exists(args.receptor_pdb):
        print(f"错误: 受体PDB文件不存在: {args.receptor_pdb}")
        sys.exit(1)
    
    # 创建配置字典
    config = {
        'population_size': args.population_size,
        'max_generations': args.max_generations,
        'output_dir': args.output_dir,
        'temp_dir': args.temp_dir,
        'n_fragments_to_mask': args.n_fragments_to_mask,
        'crossover_rate': args.crossover_rate,
        'crossover_attempts': args.crossover_attempts,
        'mutation_attempts': args.mutation_attempts,
        'max_mutations_per_parent': args.max_mutations_per_parent,
        'receptor_pdb_file': args.receptor_pdb,
        'center_x': args.center_x,
        'center_y': args.center_y,
        'center_z': args.center_z,
        'size_x': args.size_x,
        'size_y': args.size_y,
        'size_z': args.size_z,
        'vina_executable': args.vina_executable,
        'mgl_python': args.mgl_python,
        'prepare_receptor4': args.prepare_receptor4,
        'prepare_ligand4': args.prepare_ligand4,
        'gpu_device': args.gpu_device,
        'num_processors': args.num_processors
    }
    
    try:
        # 创建并运行工作流程
        workflow = GAGPTWorkflow(config)
        workflow.run_workflow(args.initial_smiles)
        
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n运行出错: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 