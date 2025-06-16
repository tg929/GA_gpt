#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT分子生成模块
==============

负责分子的分解、掩码和GPT生成,这是混合生成系统中负责扩展化学多样性的组件。
在GA-GPT对抗体系中,GPT模块主要负责exploration(探索）。
"""

import os
import sys
import shutil
import logging
from typing import List, Dict, Optional
import tempfile

# 添加项目路径
current_file = os.path.abspath(__file__)
# 从 /data1/ytg/GA_gpt/GA_gpt/modules/gpt_generation_module.py 到 /data1/ytg/GA_gpt
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'fragment_GPT'))

# 导入相关模块
try:
    from datasets.decompose.demo_frags import batch_process as decompose_batch_process
    from fragment_GPT.generate_all import main_test as gpt_generate
    from operations.scoring.scoring_demo import load_smiles_from_file
except ImportError as e:
    print(f"GPT模块导入失败: {e}")
    raise


class GPTGenerationModule:
    """GPT分子生成模块类"""
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        初始化GPT生成模块
        
        Args:
            config: 配置参数字典
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger or self._setup_default_logger()
        self.temp_dir = config.get('temp_dir', 'temp')
        self.output_dir = config.get('output_dir', 'output')
        
        # GPT特定配置
        self.gpu_device = config.get('gpu_device', '0')
        self.n_fragments_to_mask = config.get('n_fragments_to_mask', 1)
        
        self.logger.info("GPT生成模块初始化完成")
        self.logger.info(f"GPU设备: {self.gpu_device}")
        self.logger.info(f"掩码片段数: {self.n_fragments_to_mask}")
    
    def _setup_default_logger(self) -> logging.Logger:
        """设置默认日志记录器"""
        logger = logging.getLogger('GPTGeneration')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def decompose_molecules(self, smiles_list: List[str], generation: int) -> str:
        """
        分解分子为片段
        
        Args:
            smiles_list: SMILES列表
            generation: 当前代数
            
        Returns:
            分解结果文件路径
        """
        self.logger.info(f"正在分解第 {generation} 代的 {len(smiles_list)} 个分子...")
        
        # 创建输入文件
        input_file = os.path.join(self.temp_dir, f'gpt_gen_{generation}_for_decompose.smi')
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
        
        try:
            # 运行分解
            decompose_batch_process(input_file, output_file, formatted_file, 
                                  masked_file, original_file)
            
            self.logger.info(f"分子分解完成，结果保存在 {output_dir}")
            return masked_file
            
        except Exception as e:
            self.logger.error(f"分子分解过程出错: {e}")
            # 创建备用掩码文件
            with open(masked_file, 'w') as f:
                for smi in smiles_list:
                    f.write(f"[BOS]{smi}[SEP]\n")
            return masked_file
    
    def apply_fragment_masking(self, masked_file: str, n_fragments: int = None) -> str:
        """
        应用灵活的片段掩码
        
        Args:
            masked_file: 掩码文件路径
            n_fragments: 要掩码的片段数量，如果为None则使用配置中的值
            
        Returns:
            处理后的掩码文件路径
        """
        if n_fragments is None:
            n_fragments = self.n_fragments_to_mask
        
        if n_fragments <= 1:
            # 默认掩码已经是掩码最后1个片段
            return masked_file
        
        self.logger.info(f"应用灵活掩码: 掩码最后 {n_fragments} 个片段")
        
        try:
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
            
            self.logger.info(f"片段掩码应用完成")
            return masked_file
            
        except Exception as e:
            self.logger.error(f"应用片段掩码出错: {e}")
            return masked_file
    
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
            # 创建GPT生成参数
            class GPTArgs:
                def __init__(self, device, seed, input_file):
                    self.device = device
                    self.seed = seed
                    self.input_file = input_file
            
            args = GPTArgs(
                device=self.gpu_device,
                seed=str(generation),
                input_file=masked_fragments_file
            )
            
            # 调用GPT生成
            gpt_generate(args)
            
            # GPT生成的结果文件路径
            generated_file = os.path.join(PROJECT_ROOT, "fragment_GPT/output", 
                                        f"crossovered0_frags_new_{generation}.smi")
            
            # 复制到输出目录
            target_file = os.path.join(gpt_output_dir, f'gpt_generated_{generation}.smi')
            if os.path.exists(generated_file):
                shutil.copy2(generated_file, target_file)
                
                # 统计生成的分子数量
                generated_smiles = load_smiles_from_file(target_file)
                self.logger.info(f"GPT成功生成 {len(generated_smiles)} 个新分子")
            else:
                # 创建空文件作为备用
                with open(target_file, 'w') as f:
                    f.write("")
                self.logger.warning(f"GPT生成文件未找到: {generated_file}")
            
            self.logger.info(f"GPT生成完成，结果保存在 {target_file}")
            return target_file
            
        except Exception as e:
            self.logger.error(f"GPT生成过程出错: {e}")
            # 创建空的生成文件作为备用
            empty_file = os.path.join(gpt_output_dir, f'gpt_generated_{generation}.smi')
            with open(empty_file, 'w') as f:
                f.write("")
            return empty_file
    
    def run_gpt_generation_pipeline(self, smiles_list: List[str], generation: int) -> str:
        """
        运行完整的GPT生成流水线：分解 -> 掩码 -> GPT生成
        
        Args:
            smiles_list: 输入的SMILES列表
            generation: 当前代数
            
        Returns:
            生成的分子文件路径
        """
        self.logger.info(f"启动GPT生成流水线 - 第 {generation} 代")
        self.logger.info(f"输入分子数量: {len(smiles_list)}")
        
        try:
            # 步骤1: 分解分子
            masked_file = self.decompose_molecules(smiles_list, generation)
            
            # 步骤2: 应用掩码
            masked_file = self.apply_fragment_masking(masked_file)
            
            # 步骤3: GPT生成
            generated_file = self.generate_with_gpt(masked_file, generation)
            
            # 统计最终结果
            generated_smiles = load_smiles_from_file(generated_file)
            self.logger.info(f"GPT生成流水线完成")
            self.logger.info(f"最终生成 {len(generated_smiles)} 个新分子")
            
            return generated_file
            
        except Exception as e:
            self.logger.error(f"GPT生成流水线出错: {e}")
            raise
    
    def get_generation_statistics(self, input_smiles: List[str], output_file: str) -> Dict:
        """
        获取GPT生成的统计信息
        
        Args:
            input_smiles: 输入的SMILES列表
            output_file: 输出文件路径
            
        Returns:
            统计信息字典
        """
        try:
            output_smiles = load_smiles_from_file(output_file)
            
            # 计算多样性指标
            unique_outputs = set(output_smiles)
            novel_molecules = unique_outputs - set(input_smiles)
            
            stats = {
                'input_count': len(input_smiles),
                'output_count': len(output_smiles),
                'unique_outputs': len(unique_outputs),
                'novel_molecules': len(novel_molecules),
                'novelty_rate': len(novel_molecules) / len(unique_outputs) if unique_outputs else 0,
                'generation_rate': len(output_smiles) / len(input_smiles) if input_smiles else 0
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"统计GPT生成信息出错: {e}")
            return {}
    
    def adjust_generation_intensity(self, current_performance: Dict, target_diversity: float = 0.7):
        """
        根据当前性能调整GPT生成强度
        这实现了GPT在对抗系统中的自适应控制
        
        Args:
            current_performance: 当前性能指标
            target_diversity: 目标多样性水平
        """
        current_novelty = current_performance.get('novelty_rate', 0.5)
        
        if current_novelty < target_diversity * 0.8:
            # 多样性不足，增加GPT的作用
            self.n_fragments_to_mask = min(self.n_fragments_to_mask + 1, 3)
            self.logger.info(f"增强GPT多样性生成：掩码片段数调整为 {self.n_fragments_to_mask}")
        elif current_novelty > target_diversity * 1.2:
            # 多样性过度，减少GPT的作用
            self.n_fragments_to_mask = max(self.n_fragments_to_mask - 1, 1)
            self.logger.info(f"收敛GPT生成：掩码片段数调整为 {self.n_fragments_to_mask}")
        
        # 更新配置
        self.config['n_fragments_to_mask'] = self.n_fragments_to_mask


def create_gpt_module(config: Dict, logger: Optional[logging.Logger] = None) -> GPTGenerationModule:
    """
    工厂函数:创建GPT生成模块实例
    
    Args:
        config: 配置参数字典
        logger: 日志记录器
        
    Returns:
        GPT生成模块实例
    """
    return GPTGenerationModule(config, logger)


def test_gpt_module():
    """测试GPT模块功能"""
    # 测试配置
    config = {
        'temp_dir': 'temp_test',
        'output_dir': 'output_test',
        'gpu_device': '0',
        'n_fragments_to_mask': 1
    }
    
    # 创建测试目录
    os.makedirs(config['temp_dir'], exist_ok=True)
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 测试SMILES
    test_smiles = [
        'CCO',
        'c1ccccc1',
        'CC(C)O'
    ]
    
    try:
        # 创建GPT模块
        gpt_module = create_gpt_module(config)
        
        # 测试生成流水线
        result_file = gpt_module.run_gpt_generation_pipeline(test_smiles, 0)
        
        # 获取统计信息
        stats = gpt_module.get_generation_statistics(test_smiles, result_file)
        
        print("GPT模块测试完成")
        print(f"统计信息: {stats}")
        
        return True
        
    except Exception as e:
        print(f"GPT模块测试失败: {e}")
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
    test_gpt_module() 