#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA_gpt.modules 包初始化文件
=========================

包含GA-GPT系统的核心模块:
- GPT生成模块
- GA优化模块  
- 分子评估模块
"""

# 导入核心模块类
try:
    from .gpt_generation_module import GPTGenerationModule, create_gpt_module
    from .ga_optimization_module import GAOptimizationModule, create_ga_module
    from .molecular_evaluation_module import MolecularEvaluationModule, create_evaluation_module
    
    __all__ = [
        'GPTGenerationModule', 'create_gpt_module',
        'GAOptimizationModule', 'create_ga_module', 
        'MolecularEvaluationModule', 'create_evaluation_module'
    ]
    
except ImportError as e:
    print(f"警告: 模块导入失败: {e}")
    __all__ = [] 