#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA优化流程主入口
================
该脚本是整个GA优化实验的顶层控制器。
它负责解析命令行参数（如配置文件和目标受体），
然后调用核心工作流执行器来完成针对单个或多个受体的完整GA流程。
通过修改配置文件中的 'selection_mode'，可以切换单目标或多目标优化。

  - 针对默认受体运行 (使用配置文件中的默认模式):
    python GA_gpt/GA_main.py --config GA_gpt/config_example.json
  - 针对特定受体运行:
    python GA_gpt/GA_main.py --config GA_gpt/config_example.json --receptor 4r6e
  - 为所有受体运行:
    python GA_gpt/GA_main.py --config GA_gpt/config_example.json --all_receptors
"""
import os
import sys
import argparse
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# 将项目根目录添加到Python路径
# 假设GA_gpt/GA_main.py位于项目根目录下的GA_gpt文件夹中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# 从重构后的执行器模块中导入核心类
from operations.operations_execute_demo import GAWorkflowExecutor

def run_workflow_for_receptor(config_path_str: str, receptor_name: str, output_dir_override: str):
    """
    为单个受体运行完整GA工作流的包装函数。
    设计为可在单独的进程中安全执行。
    """
    receptor_display_name = receptor_name or '默认受体'
    try:
        logger.info(f"[进程 {os.getpid()}] 开始为受体 '{receptor_display_name}' 运行GA工作流")
        
        executor = GAWorkflowExecutor(
            config_path=config_path_str, 
            receptor_name=receptor_name,
            output_dir_override=output_dir_override
        )
        
        success = executor.run_complete_workflow()
        
        if success:
            logger.info(f"[进程 {os.getpid()}] 针对受体 '{receptor_display_name}' 的GA工作流成功完成!")
            return receptor_display_name, True
        else:
            logger.error(f"[进程 {os.getpid()}] 针对受体 '{receptor_display_name}' 的GA工作流执行失败。")
            return receptor_display_name, False
            
    except Exception as e:
        logger.error(f"[进程 {os.getpid()}] 为受体 '{receptor_display_name}' 运行主流程时发生未处理的异常: {e}", exc_info=True)
        return receptor_display_name, False

def get_available_workers() -> int:
    """
    动态计算可用的工作进程数。

    该函数通过以下步骤确定最佳工作进程数：
    1. 使用 `os.sched_getaffinity(0)` 获取当前进程被允许使用的CPU核心数。
       这在容器化环境(如Docker)或通过任务调度器(如Slurm)运行时特别有用，
       因为它能准确反映资源限制。
    2. 如果 `sched_getaffinity` 不可用(例如在非Linux系统上),则回退到使用 `os.cpu_count()`
       获取系统总的逻辑CPU核心数。
    3. 获取系统的1分钟平均负载(使用 `os.getloadavg()`),它反映了近期CPU的繁忙程度。
    4. 可用工作进程数估算为 `允许的核心数 - 平均负载`。
    5. 结果向下取整,并确保至少返回1,以保证总有进程能运行。

    Returns:
        int: 估算出的可用工作进程数。
    """
    try:
        # 在Linux上，这能正确处理cgroup中的CPU限制
        allowed_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        logger.warning("`os.sched_getaffinity` 在当前系统不可用，将回退到 `os.cpu_count()`。")
        allowed_cores = os.cpu_count()

    if allowed_cores is None:
        logger.warning("无法确定CPU核心数,将默认使用1个工作进程。")
        return 1

    try:
        # getloadavg() 在Unix-like系统上可用
        load_avg_1min = os.getloadavg()[0]
    except (AttributeError, OSError):
        logger.warning("`os.getloadavg` 在当前系统不可用，无法基于负载调整。")
        load_avg_1min = 0.0

    # 核心计算逻辑：从允许的核心数中减去负载
    # 向下取整，因为负载是浮点数，而worker数必须是整数
    # 至少保留一个worker
    available_workers = max(1, int(allowed_cores - load_avg_1min))
    
    logger.info(
        f"动态计算可用工作进程: "
        f"允许的核心数 = {allowed_cores}, "
        f"1分钟平均负载 = {load_avg_1min:.2f}, "
        f"估算可用工作数 = {available_workers}"
    )
    return available_workers

def main():
    """主函数:解析参数并启动GA工作流"""
    parser = argparse.ArgumentParser(description='GA分子优化流程主入口')
    parser.add_argument('--config', type=str, default='GA_gpt/config_example.json')
    parser.add_argument('--receptor', type=str, default=None,help='(可选) 指定要运行的目标受体名称。如果未提供，将使用默认受体。')
    parser.add_argument('--all_receptors', action='store_true',help='(可选) 运行配置文件中`target_list`的所有受体。如果使用此选项，将忽略--receptor参数。')
    parser.add_argument('--output_dir', type=str, default=None)   #在此设置的话就会覆盖掉参数配置json文件中的设置
    parser.add_argument('--num_workers', type=int, default=None, help='并行运行的工作进程数。如果未指定，将动态地根据系统当前负载和可用核心数计算。')
    args = parser.parse_args()
    
    # --- 参数验证和准备 ---
    config_path = Path(args.config)
    if not config_path.is_file():
        logger.error(f"配置文件不存在: {config_path}")
        sys.exit(1)
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # --- 确定工作进程数 ---
    if args.num_workers:
        num_workers = args.num_workers
        logger.info(f"用户指定了工作进程数: {num_workers}")
    else:
        num_workers = get_available_workers()
        
    receptors_to_run = []
    if args.all_receptors:
        logger.info("检测到 --all_receptors 标志，将为配置文件中的所有受体运行工作流。")
        # 确保 target_list 存在
        target_list = config.get('receptors', {}).get('target_list', {})
        if not target_list:
            logger.warning("配置文件中未找到 'receptors.target_list' 或列表为空，无受体可运行。")
        else:
            receptors_to_run = list(target_list.keys())
        
        logger.info(f"计划运行的受体列表: {receptors_to_run}")
    else:
        # 如果未指定 --all_receptors，则运行单个受体（可以是指定的或默认的）
        receptors_to_run.append(args.receptor)

    # --- 启动工作流 ---
    if not receptors_to_run:
        logger.info("没有需要运行的受体，程序退出。")
        sys.exit(0)
        
    successful_runs = []
    failed_runs = []

    # --- 并行执行 ---
    with ProcessPoolExecutor(max_workers=num_workers) as process_executor:
        futures = {
            process_executor.submit(
                run_workflow_for_receptor, 
                str(config_path), 
                receptor_name, 
                args.output_dir
            ): receptor_name 
            for receptor_name in receptors_to_run
        }
        
        num_tasks = len(futures)
        logger.info(f"已提交 {num_tasks} 个任务到进程池，使用最多 {process_executor._max_workers} 个工作进程。")

        for i, future in enumerate(as_completed(futures), 1):
            receptor_name_from_future = futures[future]
            try:
                receptor_display_name, success = future.result()
                if success:
                    successful_runs.append(receptor_display_name)
                else:
                    failed_runs.append(receptor_display_name)
            except Exception as exc:
                logger.error(f"受体 '{receptor_name_from_future}' 的任务在执行中产生异常: {exc}")
                failed_runs.append(receptor_name_from_future or '默认受体')
            
            logger.info(f"--- 进度: {i}/{num_tasks} 个任务已完成 ---")

    # --- 最终总结 ---
    logger.info("=" * 80)
    logger.info("所有GA工作流执行完毕。")
    if successful_runs:
        logger.info(f"成功运行的受体 ({len(successful_runs)}): {sorted(successful_runs)}")
    if failed_runs:
        logger.error(f"失败的受体 ({len(failed_runs)}): {sorted(failed_runs)}")
    logger.info("=" * 80)
    
    if failed_runs:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main() 