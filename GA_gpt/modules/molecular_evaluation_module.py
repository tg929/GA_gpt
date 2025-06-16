#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分子评估模块
===========

负责分子的多指标评估（对接、QED、SA）和种群选择操作。
这是GA-GPT混合系统中的评估和决策组件。
"""

import os
import sys
import logging
from typing import List, Dict, Optional, Tuple
import tempfile

# 添加项目路径
current_file = os.path.abspath(__file__)
# 从 /data1/ytg/GA_gpt/GA_gpt/modules/molecular_evaluation_module.py 到 /data1/ytg/GA_gpt
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.insert(0, PROJECT_ROOT)

# 导入相关模块
try:
    from operations.scoring.scoring_demo import load_smiles_from_file, get_rdkit_mols
    from operations.scoring.scoring_demo import calculate_qed_scores, calculate_sa_scores
    from operations.docking.docking_demo_finetune import DockingWorkflow
    from operations.selecting.selecting_multi_demo import (
        load_molecules_with_scores, add_additional_scores, 
        select_molecules_nsga2, save_selected_molecules
    )
    from operations.selecting.selecting_single_demo import select_molecules_single_top_k
except ImportError as e:
    print(f"评估模块导入失败: {e}")
    raise


class MolecularEvaluationModule:
    """分子评估模块类"""
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        初始化分子评估模块
        
        Args:
            config: 配置参数字典
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger or self._setup_default_logger()
        self.temp_dir = config.get('temp_dir', 'temp')
        self.output_dir = config.get('output_dir', 'output')
        
        # 对接配置
        self.receptor_pdb_file = config.get('receptor_pdb_file', 'receptor.pdb')
        self.docking_config = self._setup_docking_config()
        
        # 选择配置
        self.population_size = config.get('population_size', 115)
        self.selection_strategy = config.get('selection_strategy', 'multi_objective')  # 'multi_objective' 或 'single_objective'
        
        self.logger.info("分子评估模块初始化完成")
        self.logger.info(f"目标种群大小: {self.population_size}")
        self.logger.info(f"选择策略: {self.selection_strategy}")
    
    def _setup_default_logger(self) -> logging.Logger:
        """设置默认日志记录器"""
        logger = logging.getLogger('MolecularEvaluation')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_docking_config(self) -> Dict:
        """设置对接配置"""
        return {
            'output_directory': os.path.join(self.output_dir, 'docking_results'),
            'filename_of_receptor': self.receptor_pdb_file,
            'center_x': self.config.get('center_x', 0.0),
            'center_y': self.config.get('center_y', 0.0), 
            'center_z': self.config.get('center_z', 0.0),
            'size_x': self.config.get('size_x', 20.0),
            'size_y': self.config.get('size_y', 20.0),
            'size_z': self.config.get('size_z', 20.0),
            'ligand_dir': os.path.join(self.temp_dir, 'ligands'),
            'sdf_dir': os.path.join(self.temp_dir, 'sdf'),
            'conversion_choice': 'mgltools',
            'docking_executable': self.config.get('vina_executable', 'vina'),
            'mgl_python': self.config.get('mgl_python', 'python'),
            'prepare_receptor4.py': self.config.get('prepare_receptor4', 'prepare_receptor4.py'),
            'prepare_ligand4.py': self.config.get('prepare_ligand4', 'prepare_ligand4.py'),
            'number_of_processors': self.config.get('num_processors', 4),
            'gypsum_timeout_limit': 120.0
        }
    
    def evaluate_docking_scores(self, smiles_list: List[str], generation: int) -> List[float]:
        """
        计算对接分数
        
        Args:
            smiles_list: SMILES列表
            generation: 当前代数
            
        Returns:
            对接分数列表
        """
        self.logger.info(f"正在计算第 {generation} 代的对接分数...")
        
        # 创建临时SMILES文件
        temp_smiles_file = os.path.join(self.temp_dir, f'eval_gen_{generation}_for_docking.smi')
        with open(temp_smiles_file, 'w') as f:
            for i, smi in enumerate(smiles_list):
                f.write(f"{smi}\tmol_{i}\n")
        
        try:
            # 更新对接配置中的生成特定路径
            current_docking_config = self.docking_config.copy()
            current_docking_config['output_directory'] = os.path.join(
                self.output_dir, 'docking_results', f'gen_{generation}'
            )
            current_docking_config['ligand_dir'] = os.path.join(
                self.temp_dir, f'ligands_gen_{generation}'
            )
            current_docking_config['sdf_dir'] = os.path.join(
                self.temp_dir, f'sdf_gen_{generation}'
            )
            
            # 创建对接工作流程
            docking_workflow = DockingWorkflow(current_docking_config)
            
            # 准备受体
            receptor_pdbqt = docking_workflow.prepare_receptor()
            
            # 准备配体
            ligand_dir = docking_workflow.prepare_ligands(temp_smiles_file)
            
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
                docking_scores = [0.0] * len(smiles_list)
            
            self.logger.info(f"对接评估完成，处理了 {len(docking_scores)} 个分子")
            return docking_scores
            
        except Exception as e:
            self.logger.error(f"对接评估过程出错: {e}")
            # 返回默认分数
            return [0.0] * len(smiles_list)
    
    def evaluate_molecular_properties(self, smiles_list: List[str]) -> Tuple[List[float], List[float]]:
        """
        计算分子性质（QED和SA分数）
        
        Args:
            smiles_list: SMILES列表
            
        Returns:
            QED分数列表和SA分数列表的元组
        """
        self.logger.info(f"正在计算 {len(smiles_list)} 个分子的QED和SA分数...")
        
        try:
            # 获取RDKit分子对象
            mols, valid_smiles = get_rdkit_mols(smiles_list)
            
            # 计算QED分数
            qed_scores = calculate_qed_scores(mols)
            
            # 计算SA分数
            sa_scores = calculate_sa_scores(mols)
            
            # 为无效分子填充默认值
            final_qed_scores = []
            final_sa_scores = []
            valid_idx = 0
            
            for smi in smiles_list:
                if smi in valid_smiles:
                    if valid_idx < len(qed_scores):
                        final_qed_scores.append(qed_scores[valid_idx])
                    else:
                        final_qed_scores.append(0.0)
                    
                    if valid_idx < len(sa_scores):
                        final_sa_scores.append(sa_scores[valid_idx])
                    else:
                        final_sa_scores.append(5.0)
                    
                    valid_idx += 1
                else:
                    final_qed_scores.append(0.0)
                    final_sa_scores.append(5.0)
            
            self.logger.info(f"分子性质评估完成")
            return final_qed_scores, final_sa_scores
            
        except Exception as e:
            self.logger.error(f"分子性质评估出错: {e}")
            # 返回默认分数
            return [0.0] * len(smiles_list), [5.0] * len(smiles_list)
    
    def comprehensive_evaluation(self, smiles_list: List[str], generation: int) -> Dict:
        """
        综合评估分子（对接、QED、SA）
        
        Args:
            smiles_list: SMILES列表
            generation: 当前代数
            
        Returns:
            评估结果字典
        """
        self.logger.info(f"开始第 {generation} 代分子综合评估")
        self.logger.info(f"评估分子数量: {len(smiles_list)}")
        
        # 1. 计算对接分数
        docking_scores = self.evaluate_docking_scores(smiles_list, generation)
        
        # 2. 计算分子性质
        qed_scores, sa_scores = self.evaluate_molecular_properties(smiles_list)
        
        # 3. 整合评估结果
        evaluation_results = {
            'smiles': smiles_list,
            'docking_scores': docking_scores,
            'qed_scores': qed_scores,
            'sa_scores': sa_scores,
            'generation': generation
        }
        
        # 4. 保存评估结果
        self._save_evaluation_results(evaluation_results, generation)
        
        # 5. 计算统计信息
        stats = self._calculate_evaluation_statistics(evaluation_results)
        self.logger.info(f"评估统计: {stats}")
        
        self.logger.info(f"第 {generation} 代分子综合评估完成")
        return evaluation_results
    
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
        self.logger.info(f"开始选择第 {generation+1} 代种群...")
        
        # 合并所有分子数据
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
        self.logger.info(f"合并去重后候选分子数量: {len(all_molecules)}")
        
        try:
            if self.selection_strategy == 'multi_objective':
                # 使用多目标NSGA-II选择
                selected_molecules = self._select_multi_objective(all_molecules)
            else:
                # 使用单目标选择
                selected_molecules = self._select_single_objective(all_molecules)
            
            # 提取SMILES
            selected_smiles = [mol['smiles'] for mol in selected_molecules]
            
            # 确保种群大小
            selected_smiles = self._ensure_population_size(selected_smiles, all_molecules)
            
            # 保存选择结果
            self._save_selection_results(selected_smiles, generation)
            
            self.logger.info(f"选择完成，第 {generation+1} 代种群包含 {len(selected_smiles)} 个分子")
            return selected_smiles
            
        except Exception as e:
            self.logger.error(f"种群选择出错: {e}")
            # 备用选择策略：按对接分数选择最佳分子
            sorted_molecules = sorted(all_molecules, key=lambda x: x['docking_score'])
            selected_smiles = [mol['smiles'] for mol in sorted_molecules[:self.population_size]]
            return selected_smiles
    
    def _select_multi_objective(self, molecules: List[Dict]) -> List[Dict]:
        """使用多目标优化选择分子"""
        self.logger.info("使用NSGA-II多目标优化选择分子")
        
        n_select_fitness = max(int(self.population_size * 0.7), 30)
        n_select_diversity = max(int(self.population_size * 0.3), 15)
        
        selected_molecules, pareto_info = select_molecules_nsga2(
            molecules,
            n_select_fitness=n_select_fitness,
            n_select_diversity=n_select_diversity,
            population_size=100,
            generations=50
        )
        
        return selected_molecules
    
    def _select_single_objective(self, molecules: List[Dict]) -> List[Dict]:
        """使用单目标优化选择分子"""
        self.logger.info("使用单目标优化选择分子（仅对接分数）")
        
        return select_molecules_single_top_k(molecules, self.population_size)
    
    def _ensure_population_size(self, selected_smiles: List[str], 
                               all_molecules: List[Dict]) -> List[str]:
        """确保种群大小符合要求"""
        if len(selected_smiles) < self.population_size:
            # 如果选择的分子不足，用最佳分子补充
            sorted_molecules = sorted(all_molecules, key=lambda x: x['docking_score'])
            for mol in sorted_molecules:
                if mol['smiles'] not in selected_smiles:
                    selected_smiles.append(mol['smiles'])
                    if len(selected_smiles) >= self.population_size:
                        break
        
        # 截取到目标种群大小
        return selected_smiles[:self.population_size]
    
    def _save_evaluation_results(self, results: Dict, generation: int):
        """保存评估结果"""
        results_dir = os.path.join(self.output_dir, 'generations')
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, f'generation_{generation}_scores.txt')
        with open(results_file, 'w') as f:
            f.write("SMILES\tDocking_Score\tQED_Score\tSA_Score\n")
            for i, smi in enumerate(results['smiles']):
                docking = results['docking_scores'][i] if i < len(results['docking_scores']) else 'NA'
                qed = results['qed_scores'][i] if i < len(results['qed_scores']) else 'NA'
                sa = results['sa_scores'][i] if i < len(results['sa_scores']) else 'NA'
                f.write(f"{smi}\t{docking}\t{qed}\t{sa}\n")
    
    def _save_selection_results(self, selected_smiles: List[str], generation: int):
        """保存选择结果"""
        selection_output_dir = os.path.join(self.output_dir, 'selection_results', f'gen_{generation}')
        os.makedirs(selection_output_dir, exist_ok=True)
        
        selection_file = os.path.join(selection_output_dir, f'selected_gen_{generation+1}.smi')
        with open(selection_file, 'w') as f:
            for smi in selected_smiles:
                f.write(f"{smi}\n")
    
    def _calculate_evaluation_statistics(self, results: Dict) -> Dict:
        """计算评估统计信息"""
        stats = {}
        
        if results['docking_scores']:
            docking_scores = [score for score in results['docking_scores'] if isinstance(score, (int, float))]
            if docking_scores:
                stats['docking'] = {
                    'best': min(docking_scores),
                    'mean': sum(docking_scores) / len(docking_scores),
                    'worst': max(docking_scores),
                    'count': len(docking_scores)
                }
        
        if results['qed_scores']:
            qed_scores = [score for score in results['qed_scores'] if isinstance(score, (int, float))]
            if qed_scores:
                stats['qed'] = {
                    'best': max(qed_scores),
                    'mean': sum(qed_scores) / len(qed_scores),
                    'worst': min(qed_scores),
                    'count': len(qed_scores)
                }
        
        if results['sa_scores']:
            sa_scores = [score for score in results['sa_scores'] if isinstance(score, (int, float))]
            if sa_scores:
                stats['sa'] = {
                    'best': min(sa_scores),
                    'mean': sum(sa_scores) / len(sa_scores),
                    'worst': max(sa_scores),
                    'count': len(sa_scores)
                }
        
        return stats
    
    def switch_selection_strategy(self, strategy: str):
        """切换选择策略"""
        if strategy in ['multi_objective', 'single_objective']:
            self.selection_strategy = strategy
            self.config['selection_strategy'] = strategy
            self.logger.info(f"选择策略已切换为: {strategy}")
        else:
            self.logger.warning(f"未知的选择策略: {strategy}")


def create_evaluation_module(config: Dict, logger: Optional[logging.Logger] = None) -> MolecularEvaluationModule:
    """
    工厂函数：创建分子评估模块实例
    
    Args:
        config: 配置参数字典
        logger: 日志记录器
        
    Returns:
        分子评估模块实例
    """
    return MolecularEvaluationModule(config, logger)


def test_evaluation_module():
    """测试评估模块功能"""
    # 测试配置
    config = {
        'temp_dir': 'temp_test',
        'output_dir': 'output_test',
        'receptor_pdb_file': 'test_receptor.pdb',
        'population_size': 10,
        'selection_strategy': 'single_objective',
        'center_x': 0.0,
        'center_y': 0.0,
        'center_z': 0.0,
        'size_x': 20.0,
        'size_y': 20.0,
        'size_z': 20.0
    }
    
    # 创建测试目录
    os.makedirs(config['temp_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 测试SMILES
    test_smiles = [
        'CCO',
        'c1ccccc1',
        'CC(C)O',
        'CCN',
        'CCC'
    ]
    
    try:
        # 创建评估模块
        eval_module = create_evaluation_module(config)
        
        # 测试分子性质评估
        qed_scores, sa_scores = eval_module.evaluate_molecular_properties(test_smiles)
        
        print("评估模块测试完成")
        print(f"QED分数: {qed_scores}")
        print(f"SA分数: {sa_scores}")
        
        return True
        
    except Exception as e:
        print(f"评估模块测试失败: {e}")
        return False
    
    finally:
        # 清理测试文件
        import shutil
        if os.path.exists(config['temp_dir']):
            shutil.rmtree(config['temp_dir'])
        if os.path.exists(config['output_dir']):
            shutil.rmtree(config['output_dir'])


if __name__ == '__main__':
    # 运行测试
    test_evaluation_module() 