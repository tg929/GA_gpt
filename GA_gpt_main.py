#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FragGPT-GA: 混合分子生成项目主入口
"""
import os
import sys
import argparse
import logging
import json

# --- 项目根目录设置 ---
# 此脚本位于项目根目录下，直接获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, PROJECT_ROOT)

# --- 模块导入 ---
try:
    from operations.operations_execute_GAgpt_demo import GAGPTWorkflowExecutor
except ImportError as e:
    print(f"错误: 无法导入核心工作流模块。请确保脚本从正确的项目目录运行。")
    print(f"Python 搜索路径: {sys.path}")
    print(f"详细错误: {e}")
    sys.exit(1)

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GA_GPT_MAIN")

def main():
    """
    主函数:解析参数,启动GA-GPT工作流。
    """
    parser = argparse.ArgumentParser(
        description="GA-GPT 混合分子生成项目主入口",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default='GA_gpt/config_GA_gpt.json',
                        help='主配置文件的路径。')
    parser.add_argument('--receptor', type=str, default=None,
                        help='(可选) 指定要运行的目标受体名称。如果未提供，将使用默认受体。')
    parser.add_argument('--all_receptors', action='store_true',
                        help='(可选) 运行配置文件中`target_list`的所有受体。如果使用此选项，将忽略--receptor参数。')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='(可选) 指定输出总目录，每个受体的结果将在此目录下创建子文件夹。')

    args = parser.parse_args()

    # --- 1. 加载配置并确定要运行的受体列表 ---
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.critical(f"配置文件未找到: {args.config}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.critical(f"配置文件解析失败: {args.config}")
        sys.exit(1)
        
    receptors_to_run = []
    if args.all_receptors:
        logger.info("检测到 --all_receptors 标志，将为配置文件中的所有受体运行工作流。")
        receptors_to_run = list(config.get('receptors', {}).get('target_list', {}).keys())
        if not receptors_to_run:
            logger.error("配置文件中未找到`receptors.target_list`，无法执行 --all_receptors。")
            sys.exit(1)
        logger.info(f"计划运行的受体列表: {receptors_to_run}")
    else:
        # 如果不运行全部，则只运行指定的受体（或默认受体）
        receptors_to_run.append(args.receptor)

    successful_runs = []
    failed_runs = []

    # --- 2. 循环为每个受体启动工作流 ---
    for receptor_name in receptors_to_run:
        # 使用显示名称，方便日志中识别默认受体
        default_name = config.get('receptors', {}).get('default_receptor', {}).get('name', 'default')
        receptor_display_name = receptor_name if receptor_name else default_name

        logger.info("=" * 80)
        logger.info(f"🚀 开始为受体 '{receptor_display_name}' 运行GA-GPT混合工作流")
        logger.info(f"使用配置文件: {args.config}")
        logger.info("=" * 80)
    
        try:
            # 初始化工作流执行器
            executor = GAGPTWorkflowExecutor(
                config_path=args.config, 
                receptor_name=receptor_name,
                output_dir_override=args.output_dir
            )
            
            # 运行完整的工作流
            success = executor.run_complete_workflow()
            
            if success:
                logger.info("-" * 60)
                logger.info(f"✅ 针对受体 '{receptor_display_name}' 的GA-GPT工作流成功完成!")
                logger.info("-" * 60)
                successful_runs.append(receptor_display_name)
            else:
                logger.error("=" * 60)
                logger.error(f"❌ 针对受体 '{receptor_display_name}' 的GA-GPT工作流执行失败。")
                logger.error("=" * 60)
                failed_runs.append(receptor_display_name)
                
        except Exception as e:
            logger.critical(f"为受体 '{receptor_display_name}' 运行主流程时发生未捕获的严重异常: {e}", exc_info=True)
            failed_runs.append(receptor_display_name)

    # --- 3. 最终总结报告 ---
    logger.info("=" * 80)
    logger.info("所有GA-GPT工作流执行完毕。")
    logger.info(f"🟢 成功运行的受体 ({len(successful_runs)}): {successful_runs}")
    logger.info(f"🔴 失败的受体 ({len(failed_runs)}): {failed_runs}")
    logger.info("=" * 80)

    if failed_runs:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    sys.exit(main())
