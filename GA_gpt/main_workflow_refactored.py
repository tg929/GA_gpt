#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA-GPT 混合分子生成主工作流程 (重构版)
===================================

这是 FragGPT-GA 项目的核心工作流程脚本的重构版本，采用了模块化设计：
- GPT模块:负责exploration(探索)
- GA模块:负责exploitation(利用)
- 评估模块：负责分子评估和选择
重构版
"""

import os
import sys
import argparse
import logging
import time
from typing import List, Dict
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 导入模块化组件
try:
    from modules.gpt_generation_module import create_gpt_module
    from modules.ga_optimization_module import create_ga_module
    from modules.molecular_evaluation_module import create_evaluation_module
    from operations.scoring.scoring_demo import load_smiles_from_file
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保所有模块文件都在正确位置")
    sys.exit(1)


class GAGPTWorkflowRefactored:
    """GA-GPT混合分子生成工作流程主类(重构版）"""
    
    def __init__(self, config: Dict):
        """
        初始化工作流程
        
        Args:
            config: 配置参数字典
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # 基本配置
        self.population_size = config.get('population_size', 115)
        self.max_generations = config.get('max_generations', 10)
        self.output_dir = config.get('output_dir', 'output')
        
        # 创建必要的目录
        self._setup_directories()
        
        # 初始化各个模块
        self.logger.info("正在初始化各个模块...")
        self.gpt_module = create_gpt_module(config, self.logger)
        self.ga_module = create_ga_module(config, self.logger)
        self.evaluation_module = create_evaluation_module(config, self.logger)
        
        # 当前种群状态
        self.current_population = []
        self.generation = 0
        
        # 对抗系统参数
        self.gpt_strength = config.get('gpt_strength', 0.5)  # GPT在系统中的作用强度
        self.ga_strength = config.get('ga_strength', 0.5)    # GA在系统中的作用强度
        
        self.logger.info("GA-GPT工作流程初始化完成")
        self.logger.info(f"种群大小: {self.population_size}")
        self.logger.info(f"最大代数: {self.max_generations}")
        self.logger.info(f"GPT作用强度: {self.gpt_strength}")
        self.logger.info(f"GA作用强度: {self.ga_strength}")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志系统"""
        logger = logging.getLogger('GAGPTWorkflowRefactored')
        logger.setLevel(logging.INFO)
        
        # 避免重复添加处理器
        if not logger.handlers:
            # 创建格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # 文件处理器
            log_file = os.path.join(self.config.get('output_dir', 'output'), 'workflow_refactored.log')
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _setup_directories(self):
        """创建必要的目录结构"""
        dirs_to_create = [
            self.output_dir,
            self.config.get('temp_dir', 'temp'),
            os.path.join(self.output_dir, 'generations'),
            os.path.join(self.output_dir, 'final_results')
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
    
    def initialize_population(self, initial_smiles_file: str) -> List[str]:
        """
        初始化起始种群
        
        Args:
            initial_smiles_file: 初始SMILES文件路径
            
        Returns:
            初始种群的SMILES列表
        """
        self.logger.info("正在初始化起始种群...")
        
        # 加载初始分子
        initial_smiles = load_smiles_from_file(initial_smiles_file)
        
        if not initial_smiles:
            raise ValueError(f"无法从文件 {initial_smiles_file} 加载初始分子")
        
        # 调整种群大小
        if len(initial_smiles) > self.population_size:
            import random
            initial_smiles = random.sample(initial_smiles, self.population_size)
        elif len(initial_smiles) < self.population_size:
            import random
            while len(initial_smiles) < self.population_size:
                initial_smiles.extend(random.choices(
                    initial_smiles, 
                    k=min(len(initial_smiles), self.population_size - len(initial_smiles))
                ))
        
        self.current_population = initial_smiles[:self.population_size]
        
        # 保存初始种群
        init_pop_file = os.path.join(self.output_dir, 'generations', 'generation_0_initial.smi')
        with open(init_pop_file, 'w') as f:
            for smi in self.current_population:
                f.write(f"{smi}\n")
        
        self.logger.info(f"初始种群已创建，包含 {len(self.current_population)} 个分子")
        return self.current_population
    
    def run_single_generation(self, generation: int) -> List[str]:
        """
        运行单代进化
        
        Args:
            generation: 当前代数
            
        Returns:
            下一代种群
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"开始第 {generation+1} 代进化")
        self.logger.info(f"{'='*60}")
        
        # 1. 评估当前种群（父代）
        self.logger.info("步骤1: 评估父代种群")
        parent_scores = self.evaluation_module.comprehensive_evaluation(
            self.current_population, generation
        )
        
        # 2. GPT生成新分子（exploration）
        self.logger.info("步骤2: GPT多样性生成")
        gpt_generated_file = self.gpt_module.run_gpt_generation_pipeline(
            self.current_population, generation
        )
        gpt_generated_molecules = load_smiles_from_file(gpt_generated_file)
        
        # 获取GPT生成统计
        gpt_stats = self.gpt_module.get_generation_statistics(
            self.current_population, gpt_generated_file
        )
        self.logger.info(f"GPT生成统计: {gpt_stats}")
        
        # 3. GA优化操作（exploitation）
        self.logger.info("步骤3: GA优化操作")
        ga_optimized_molecules = self.ga_module.run_ga_optimization_pipeline(
            self.current_population, gpt_generated_molecules, generation
        )
        
        # 获取GA优化统计
        ga_stats = self.ga_module.get_optimization_statistics(
            self.current_population, gpt_generated_molecules, ga_optimized_molecules
        )
        self.logger.info(f"GA优化统计: {ga_stats}")
        
        # 4. 评估子代种群
        self.logger.info("步骤4: 评估子代种群")
        child_scores = self.evaluation_module.comprehensive_evaluation(
            ga_optimized_molecules, generation+1
        )
        
        # 5. 种群选择
        self.logger.info("步骤5: 选择下一代种群")
        next_generation = self.evaluation_module.select_next_generation(
            parent_scores, child_scores, generation
        )
        
        # 6. 对抗系统自适应调整
        self._adjust_adversarial_system(gpt_stats, ga_stats, generation)
        
        self.logger.info(f"第 {generation+1} 代进化完成，新种群大小: {len(next_generation)}")
        return next_generation
    
    def _adjust_adversarial_system(self, gpt_stats: Dict, ga_stats: Dict, generation: int):
        """
        调整对抗系统中GPT和GA的相对作用
        
        Args:
            gpt_stats: GPT生成统计
            ga_stats: GA优化统计
            generation: 当前代数
        """
        self.logger.info("正在调整GPT-GA对抗系统参数...")
        
        # 获取关键指标
        gpt_novelty = gpt_stats.get('novelty_rate', 0.5)
        ga_retention = ga_stats.get('retention_rate', 0.3)
        
        # 根据当前表现调整作用强度
        if gpt_novelty < 0.4:  # GPT多样性不足
            self.gpt_module.adjust_generation_intensity(gpt_stats, target_diversity=0.6)
            self.logger.info("检测到多样性不足,增强GPT生成能力")
            
        if ga_retention < 0.2:  # GA优化不足
            self.ga_module.adjust_optimization_intensity(ga_stats, target_exploitation=0.4)
            self.logger.info("检测到优化不足,增强GA优化能力")
        
        # 动态平衡GPT和GA的作用
        if generation < self.max_generations // 3:
            # 早期：偏重GPT探索
            if gpt_novelty < 0.5:
                self.gpt_module.enable_aggressive_optimization()
            if ga_retention > 0.4:
                self.ga_module.enable_conservative_optimization()
                
        elif generation > 2 * self.max_generations // 3:
            # 后期：偏重GA优化
            if ga_retention < 0.3:
                self.ga_module.enable_aggressive_optimization()
            if gpt_novelty > 0.7:
                self.gpt_module.enable_conservative_optimization()
    
    def run_complete_workflow(self, initial_smiles_file: str):
        """
        运行完整的GA-GPT工作流程
        
        Args:
            initial_smiles_file: 初始SMILES文件路径
        """
        self.logger.info("="*80)
        self.logger.info("开始GA-GPT混合分子生成工作流程 (重构版)")
        self.logger.info("="*80)
        
        start_time = time.time()
        
        try:
            # 1. 初始化种群
            self.current_population = self.initialize_population(initial_smiles_file)
            
            # 2. 迭代进化
            for generation in range(self.max_generations):
                self.current_population = self.run_single_generation(generation)
                self.generation = generation + 1
            
            # 3. 保存最终结果
            self._save_final_results()
            
            # 4. 生成报告
            self._generate_final_report(start_time)
            
        except KeyboardInterrupt:
            self.logger.info("用户中断操作，正在保存当前结果...")
            self._save_final_results()
            raise
            
        except Exception as e:
            self.logger.error(f"工作流程出错: {e}")
            self._save_final_results()
            raise
    
    def _save_final_results(self):
        """保存最终结果"""
        final_results_dir = os.path.join(self.output_dir, 'final_results')
        
        # 保存最终种群
        final_file = os.path.join(final_results_dir, 'final_population.smi')
        with open(final_file, 'w') as f:
            for smi in self.current_population:
                f.write(f"{smi}\n")
        
        # 最终评估
        if self.current_population:
            final_scores = self.evaluation_module.comprehensive_evaluation(
                self.current_population, self.generation
            )
            
            # 保存最终评估结果
            final_scores_file = os.path.join(final_results_dir, 'final_scores.txt')
            with open(final_scores_file, 'w') as f:
                f.write("SMILES\tDocking_Score\tQED_Score\tSA_Score\n")
                for i, smi in enumerate(final_scores['smiles']):
                    docking = final_scores['docking_scores'][i]
                    qed = final_scores['qed_scores'][i] 
                    sa = final_scores['sa_scores'][i]
                    f.write(f"{smi}\t{docking}\t{qed}\t{sa}\n")
        
        self.logger.info(f"最终结果已保存到 {final_results_dir}")
    
    def _generate_final_report(self, start_time: float):
        """生成最终报告"""
        elapsed_time = time.time() - start_time
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info("GA-GPT工作流程完成!")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"总运行时间: {elapsed_time:.2f} 秒")
        self.logger.info(f"完成代数: {self.generation}")
        self.logger.info(f"最终种群大小: {len(self.current_population)}")
        
        if self.current_population:
            # 最终种群统计
            final_scores = self.evaluation_module.comprehensive_evaluation(
                self.current_population, self.generation
            )
            
            docking_scores = [s for s in final_scores['docking_scores'] if isinstance(s, (int, float))]
            qed_scores = [s for s in final_scores['qed_scores'] if isinstance(s, (int, float))]
            sa_scores = [s for s in final_scores['sa_scores'] if isinstance(s, (int, float))]
            
            if docking_scores:
                self.logger.info(f"最佳对接分数: {min(docking_scores):.4f}")
                self.logger.info(f"平均对接分数: {sum(docking_scores)/len(docking_scores):.4f}")
            
            if qed_scores:
                self.logger.info(f"最佳QED分数: {max(qed_scores):.4f}")
                self.logger.info(f"平均QED分数: {sum(qed_scores)/len(qed_scores):.4f}")
            
            if sa_scores:
                self.logger.info(f"最佳SA分数: {min(sa_scores):.4f}")
                self.logger.info(f"平均SA分数: {sum(sa_scores)/len(sa_scores):.4f}")
        
        self.logger.info(f"详细结果保存在: {self.output_dir}")
        self.logger.info(f"{'='*80}")


def create_config_from_args(args) -> Dict:
    """
    从命令行参数创建配置字典
    
    Args:
        args: 解析后的命令行参数
        
    Returns:
        配置字典
    """
    config = {
        # 基本参数
        'population_size': args.population_size,
        'max_generations': args.max_generations,
        'output_dir': args.output_dir,
        'temp_dir': args.temp_dir,
        
        # 对接参数
        'receptor_pdb_file': args.receptor_pdb,
        'center_x': args.center_x,
        'center_y': args.center_y,
        'center_z': args.center_z,
        'size_x': args.size_x,
        'size_y': args.size_y,
        'size_z': args.size_z,
        
        # 软件路径（使用默认值，减少参数冗余）
        'vina_executable': getattr(args, 'vina_executable', 'vina'),
        'mgl_python': getattr(args, 'mgl_python', 'python'),
        'prepare_receptor4': getattr(args, 'prepare_receptor4', 'prepare_receptor4.py'),
        'prepare_ligand4': getattr(args, 'prepare_ligand4', 'prepare_ligand4.py'),
        
        # 计算资源
        'gpu_device': getattr(args, 'gpu_device', '0'),
        'num_processors': getattr(args, 'num_processors', 4),
        
        # GPT参数（简化）
        'n_fragments_to_mask': getattr(args, 'n_fragments_to_mask', 1),
        
        # GA参数（简化）
        'crossover_rate': getattr(args, 'crossover_rate', 0.8),
        'crossover_attempts': getattr(args, 'crossover_attempts', 20),
        'mutation_attempts': getattr(args, 'mutation_attempts', 20),
        'max_mutations_per_parent': getattr(args, 'max_mutations_per_parent', 2),
        
        # 过滤器参数
        'enable_lipinski_filter': True,
        'enable_pains_filter': True,
        
        # 选择策略
        'selection_strategy': getattr(args, 'selection_strategy', 'multi_objective'),
        
        # 对抗系统参数
        'gpt_strength': getattr(args, 'gpt_strength', 0.5),
        'ga_strength': getattr(args, 'ga_strength', 0.5)
    }
    
    return config


def parse_arguments():
    """解析命令行参数（简化版）"""
    parser = argparse.ArgumentParser(
        description='GA-GPT混合分子生成主工作流程 (重构版)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument('--initial_smiles', '-i', required=True,
                       help='初始SMILES文件路径')
    parser.add_argument('--receptor_pdb', '-r', required=True,
                       help='受体蛋白PDB文件路径')
    
    # 核心工作流程参数
    parser.add_argument('--population_size', '-p', type=int, default=115,
                       help='种群大小')
    parser.add_argument('--max_generations', '-g', type=int, default=10,
                       help='最大进化代数')
    parser.add_argument('--output_dir', '-o', default='output',
                       help='输出目录')
    parser.add_argument('--temp_dir', default='temp',
                       help='临时文件目录')
    
    # 对接参数（必需）
    parser.add_argument('--center_x', type=float, required=True,
                       help='对接中心X坐标')
    parser.add_argument('--center_y', type=float, required=True,
                       help='对接中心Y坐标')
    parser.add_argument('--center_z', type=float, required=True,
                       help='对接中心Z坐标')
    parser.add_argument('--size_x', type=float, default=20.0,
                       help='对接空间X大小')
    parser.add_argument('--size_y', type=float, default=20.0,
                       help='对接空间Y大小')
    parser.add_argument('--size_z', type=float, default=20.0,
                       help='对接空间Z大小')
    
    # 可选的高级参数
    parser.add_argument('--selection_strategy', choices=['multi_objective', 'single_objective'],
                       default='multi_objective', help='选择策略')
    parser.add_argument('--n_fragments_to_mask', type=int, default=1,
                       help='要掩码的片段数量')
    parser.add_argument('--gpu_device', default='0',
                       help='GPU设备ID')
    parser.add_argument('--num_processors', type=int, default=4,
                       help='CPU处理器数量')
    
    # 对抗系统参数
    parser.add_argument('--gpt_strength', type=float, default=0.5,
                       help='GPT在对抗系统中的作用强度 (0.0-1.0)')
    parser.add_argument('--ga_strength', type=float, default=0.5,
                       help='GA在对抗系统中的作用强度 (0.0-1.0)')
    
    return parser.parse_args()


def main():
    """主函数"""
    print("GA-GPT混合分子生成系统 (重构版)")
    print("作者: AI Assistant")
    print("版本: 2.0.0")
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
    
    # 创建配置
    config = create_config_from_args(args)
    
    try:
        # 创建并运行工作流程
        workflow = GAGPTWorkflowRefactored(config)
        workflow.run_complete_workflow(args.initial_smiles)
        
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n运行出错: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 