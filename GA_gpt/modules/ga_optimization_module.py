#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA遗传算法优化模块
================

负责分子的交叉、变异和过滤操作，这是混合生成系统中负责精细优化的组件。
在GA-GPT对抗体系中，GA模块主要负责exploitation（利用）。

重构版本：使用统一配置和函数调用，避免参数冗余
"""

import os
import sys
import shutil
import logging
from typing import List, Dict, Optional
import tempfile

# 添加项目路径
current_file = os.path.abspath(__file__)
# 从 /data1/ytg/GA_gpt/GA_gpt/modules/ga_optimization_module.py 到 /data1/ytg/GA_gpt
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.insert(0, PROJECT_ROOT)

# 导入重构后的操作模块
try:
    from operations.crossover.crossover_demo_finetune import run_crossover_simple
    from operations.mutation.mutation_demo_finetune import run_mutation_simple  
    from operations.filter.filter_demo import run_filter_simple
    from operations.scoring.scoring_demo import load_smiles_from_file
except ImportError as e:
    print(f"GA模块导入失败: {e}")
    raise


class GAOptimizationModule:
    """GA遗传算法优化模块类"""
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        初始化GA优化模块
        
        Args:
            config: 配置参数字典
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger or self._setup_default_logger()
        
        # 从配置中获取路径信息
        paths_config = config.get('paths', {})
        self.temp_dir = paths_config.get('temp_dir', 'temp')
        self.output_dir = paths_config.get('output_dir', 'output')
        
        # 确保目录存在
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # GA特定配置
        ga_config = config.get('ga', {})
        self.crossover_rate = ga_config.get('crossover_rate', 0.8)
        self.crossover_attempts = ga_config.get('crossover_attempts', 20)
        self.mutation_attempts = ga_config.get('mutation_attempts', 20)
        self.max_mutations_per_parent = ga_config.get('max_mutations_per_parent', 2)
        
        # 过滤器配置
        filter_config = config.get('filter', {})
        self.enable_lipinski_lenient = filter_config.get('enable_lipinski_lenient', True)
        self.enable_pains_filter = filter_config.get('enable_pains', True)
        
        self.logger.info("GA优化模块初始化完成")
        self.logger.info(f"交叉率: {self.crossover_rate}")
        self.logger.info(f"交叉尝试次数: {self.crossover_attempts}")
        self.logger.info(f"变异尝试次数: {self.mutation_attempts}")
    
    def _setup_default_logger(self) -> logging.Logger:
        """设置默认日志记录器"""
        logger = logging.getLogger('GAOptimization')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def perform_crossover(self, parent_population: List[str], 
                         additional_population: List[str], 
                         generation: int) -> List[str]:
        """
        执行交叉操作
        
        Args:
            parent_population: 父代种群
            additional_population: 额外的种群（如GPT生成的分子）
            generation: 当前代数
            
        Returns:
            List[str]: 交叉结果分子列表
        """
        self.logger.info(f"正在执行第 {generation} 代交叉操作...")
        self.logger.info(f"父代种群: {len(parent_population)} 个分子")
        self.logger.info(f"额外种群: {len(additional_population)} 个分子")
        
        try:
            # 使用重构后的交叉函数
            crossover_results = run_crossover_simple(
                self.config, 
                parent_population, 
                additional_population
            )
            
            self.logger.info(f"交叉操作完成，生成 {len(crossover_results)} 个新分子")
            return crossover_results
            
        except Exception as e:
            self.logger.error(f"交叉操作出错: {e}")
            return []
    
    def perform_mutation(self, parent_population: List[str], 
                        additional_population: List[str], 
                        generation: int) -> List[str]:
        """
        执行变异操作
        
        Args:
            parent_population: 父代种群
            additional_population: 额外的种群（如GPT生成的分子）
            generation: 当前代数
            
        Returns:
            List[str]: 变异结果分子列表
        """
        self.logger.info(f"正在执行第 {generation} 代变异操作...")
        self.logger.info(f"父代种群: {len(parent_population)} 个分子")
        self.logger.info(f"额外种群: {len(additional_population)} 个分子")
        
        try:
            # 使用重构后的变异函数
            mutation_results = run_mutation_simple(
                self.config,
                parent_population,
                additional_population
            )
            
            self.logger.info(f"变异操作完成，生成 {len(mutation_results)} 个新分子")
            return mutation_results
            
        except Exception as e:
            self.logger.error(f"变异操作出错: {e}")
            return []
    
    def apply_filters(self, child_population: List[str], generation: int) -> List[str]:
        """
        应用药物化学过滤器
        
        Args:
            child_population: 子代种群分子列表
            generation: 当前代数
            
        Returns:
            List[str]: 过滤后的分子列表
        """
        self.logger.info(f"正在过滤第 {generation} 代子代种群...")
        self.logger.info(f"过滤前分子数量: {len(child_population)}")
        
        try:
            # 使用重构后的过滤函数
            filtered_molecules = run_filter_simple(self.config, child_population)
            
            filter_rate = len(filtered_molecules) / len(child_population) if child_population else 0
            self.logger.info(f"过滤完成，保留 {len(filtered_molecules)}/{len(child_population)} 个分子 ({filter_rate:.2%})")
            
            return filtered_molecules
            
        except Exception as e:
            self.logger.error(f"种群过滤出错: {e}")
            return child_population  # 返回原列表作为备用
    
    def run_ga_optimization_pipeline(self, parent_population: List[str], 
                                   additional_population: List[str], 
                                   generation: int) -> List[str]:
        """
        运行完整的GA优化流水线：交叉 -> 变异 -> 过滤
        
        Args:
            parent_population: 父代种群
            additional_population: 额外种群（如GPT生成的分子）
            generation: 当前代数
            
        Returns:
            List[str]: 优化后的分子列表
        """
        self.logger.info(f"启动GA优化流水线 - 第 {generation} 代")
        self.logger.info(f"父代种群: {len(parent_population)} 个分子")
        self.logger.info(f"额外种群: {len(additional_population)} 个分子")
        
        try:
            # 步骤1: 交叉操作
            crossover_results = self.perform_crossover(parent_population, additional_population, generation)
            
            # 步骤2: 变异操作
            mutation_results = self.perform_mutation(parent_population, additional_population, generation)
            
            # 步骤3: 合并GA操作结果
            child_population = []
            
            # 加载交叉生成的分子
            child_population.extend(crossover_results)
            
            # 加载变异生成的分子
            child_population.extend(mutation_results)
            
            # 加载额外种群（GPT生成的分子）
            child_population.extend(additional_population)
            
            # 去重
            child_population = list(set(child_population))
            
            self.logger.info(f"GA操作总共生成 {len(child_population)} 个候选分子")
            
            # 步骤4: 应用过滤器
            final_population = self.apply_filters(child_population, generation)
            
            self.logger.info(f"GA优化流水线完成")
            self.logger.info(f"最终输出 {len(final_population)} 个优化分子")
            
            return final_population
            
        except Exception as e:
            self.logger.error(f"GA优化流水线出错: {e}")
            raise
    
    def get_optimization_statistics(self, input_parent: List[str], 
                                  input_additional: List[str], 
                                  output_molecules: List[str]) -> Dict:
        """
        获取GA优化的统计信息
        
        Args:
            input_parent: 输入的父代分子
            input_additional: 输入的额外分子
            output_molecules: 输出分子
            
        Returns:
            Dict: 统计信息字典
        """
        try:
            all_inputs = set(input_parent + input_additional)
            output_set = set(output_molecules)
            
            # 计算优化指标
            novel_molecules = output_set - all_inputs
            retained_molecules = output_set & all_inputs
            
            stats = {
                'input_parent_count': len(input_parent),
                'input_additional_count': len(input_additional),
                'total_input_count': len(all_inputs),
                'output_count': len(output_molecules),
                'novel_molecules': len(novel_molecules),
                'retained_molecules': len(retained_molecules),
                'novelty_rate': len(novel_molecules) / len(output_set) if output_set else 0,
                'retention_rate': len(retained_molecules) / len(all_inputs) if all_inputs else 0,
                'optimization_rate': len(output_molecules) / len(all_inputs) if all_inputs else 0
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"统计GA优化信息出错: {e}")
            return {}
    
    def adjust_optimization_intensity(self, current_performance: Dict, target_exploitation: float = 0.3):
        """
        根据当前性能调整GA优化强度
        这实现了GA在对抗系统中的自适应控制
        
        Args:
            current_performance: 当前性能指标
            target_exploitation: 目标利用率
        """
        current_retention = current_performance.get('retention_rate', 0.3)
        
        if current_retention < target_exploitation * 0.8:
            # 利用不足，增加GA的作用
            self.crossover_attempts = min(self.crossover_attempts + 5, 50)
            self.mutation_attempts = min(self.mutation_attempts + 5, 50)
            self.logger.info(f"增强GA优化强度：交叉尝试 {self.crossover_attempts}，变异尝试 {self.mutation_attempts}")
        elif current_retention > target_exploitation * 1.2:
            # 利用过度，减少GA的作用
            self.crossover_attempts = max(self.crossover_attempts - 5, 5)
            self.mutation_attempts = max(self.mutation_attempts - 5, 5)
            self.logger.info(f"降低GA优化强度：交叉尝试 {self.crossover_attempts}，变异尝试 {self.mutation_attempts}")
        
        # 更新配置
        self.config['ga']['crossover_attempts'] = self.crossover_attempts
        self.config['ga']['mutation_attempts'] = self.mutation_attempts
    
    def enable_aggressive_optimization(self):
        """启用激进优化模式（增强GA作用）"""
        self.crossover_attempts = min(self.crossover_attempts * 2, 100)
        self.mutation_attempts = min(self.mutation_attempts * 2, 100)
        self.max_mutations_per_parent = min(self.max_mutations_per_parent + 1, 5)
        
        self.logger.info("启用激进GA优化模式")
        self.logger.info(f"交叉尝试: {self.crossover_attempts}")
        self.logger.info(f"变异尝试: {self.mutation_attempts}")
        self.logger.info(f"每个父代最大变异: {self.max_mutations_per_parent}")
        
        # 更新配置
        self.config['ga']['crossover_attempts'] = self.crossover_attempts
        self.config['ga']['mutation_attempts'] = self.mutation_attempts
        self.config['ga']['max_mutations_per_parent'] = self.max_mutations_per_parent
    
    def enable_conservative_optimization(self):
        """启用保守优化模式（减弱GA作用）"""
        self.crossover_attempts = max(self.crossover_attempts // 2, 5)
        self.mutation_attempts = max(self.mutation_attempts // 2, 5)
        self.max_mutations_per_parent = max(self.max_mutations_per_parent - 1, 1)
        
        self.logger.info("启用保守GA优化模式")
        self.logger.info(f"交叉尝试: {self.crossover_attempts}")
        self.logger.info(f"变异尝试: {self.mutation_attempts}")
        self.logger.info(f"每个父代最大变异: {self.max_mutations_per_parent}")
        
        # 更新配置
        self.config['ga']['crossover_attempts'] = self.crossover_attempts
        self.config['ga']['mutation_attempts'] = self.mutation_attempts
        self.config['ga']['max_mutations_per_parent'] = self.max_mutations_per_parent


def create_ga_module(config: Dict, logger: Optional[logging.Logger] = None) -> GAOptimizationModule:
    """
    工厂函数：创建GA优化模块实例
    
    Args:
        config: 配置参数字典
        logger: 日志记录器
        
    Returns:
        GAOptimizationModule: GA优化模块实例
    """
    return GAOptimizationModule(config, logger)


def test_ga_module():
    """测试GA模块功能"""
    # 测试配置
    config = {
        'paths': {
            'temp_dir': 'temp_test',
            'output_dir': 'output_test'
        },
        'ga': {
            'crossover_rate': 0.8,
            'crossover_attempts': 10,
            'mutation_attempts': 10,
            'max_mutations_per_parent': 2,
            'min_atom_match_mcs': 4,
            'max_time_mcs_prescreen': 1,
            'max_time_mcs_thorough': 1,
            'protanate_step': True,
            'max_variants_per_compound': 1,
            'debug_mode': False,
            'gypsum_timeout_limit': 120.0,
            'gypsum_thoroughness': 3,
            'rxn_library': 'all_rxns',
            'rxn_library_file': '/data1/ytg/GA_gpt/autogrow/operators/mutation/smiles_click_chem/reaction_libraries/all_rxns/All_Rxns_rxn_library.json',
            'function_group_library': '/data1/ytg/GA_gpt/autogrow/operators/mutation/smiles_click_chem/reaction_libraries/all_rxns/All_Rxns_functional_groups.json',
            'complementary_mol_directory': '/data1/ytg/GA_gpt/autogrow/operators/mutation/smiles_click_chem/reaction_libraries/all_rxns/complementary_mol_dir'
        },
        'filter': {
            'enable_lipinski_strict': False,
            'enable_lipinski_lenient': True,
            'enable_pains': True,
            'no_filters': False
        }
    }
    
    # 创建测试目录
    os.makedirs(config['paths']['temp_dir'], exist_ok=True)
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    
    # 测试SMILES
    test_parent = [
        'CCO',
        'c1ccccc1'
    ]
    
    test_additional = [
        'CC(C)O',
        'CCN'
    ]
    
    try:
        # 创建GA模块
        ga_module = create_ga_module(config)
        
        # 测试优化流水线
        result_molecules = ga_module.run_ga_optimization_pipeline(test_parent, test_additional, 0)
        
        # 获取统计信息
        stats = ga_module.get_optimization_statistics(test_parent, test_additional, result_molecules)
        
        print("GA模块测试完成")
        print(f"统计信息: {stats}")
        
        return True
        
    except Exception as e:
        print(f"GA模块测试失败: {e}")
        return False
    
    finally:
        # 清理测试文件
        import shutil
        if os.path.exists(config['paths']['temp_dir']):
            shutil.rmtree(config['paths']['temp_dir'])
        if os.path.exists(config['paths']['output_dir']):
            shutil.rmtree(config['paths']['output_dir'])


if __name__ == '__main__':
    # 运行测试
    test_ga_module() 