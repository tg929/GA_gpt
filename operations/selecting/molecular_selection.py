#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多种分子选择算法脚本
====================
支持三种基于对接分数的分子选择算法：
- Rank_Selector: 排名选择（精英选择，无替换）
- Roulette_Selector: 轮盘赌选择（随机加权选择，无替换）
- Tournament_Selector: 锦标赛选择（随机锦标赛选择，无替换）
支持从父代+子代合并池中进行选择，符合遗传算法"适者生存"原则。
"""
import logging
import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import autogrow.docking.ranking.selecting.rank_selection as Rank_Sel
import autogrow.docking.ranking.selecting.roulette_selection as Roulette_Sel
import autogrow.docking.ranking.selecting.tournament_selection as Tournament_Sel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_molecules_with_scores(docked_file: str) -> List[List[str]]:
    """
    从对接结果文件中加载分子及其分数,转换为autogrow格式。
    
    Args:
        docked_file: 对接结果文件路径，格式为 "SMILES score"。
    
    Returns:
        autogrow格式的分子数据列表,每个元素格式为 [SMILES, name, docking_score]。
    """
    molecules = []
    try:
        with open(docked_file, 'r') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        smiles = parts[0]
                        try:
                            docking_score = float(parts[1])
                            # autogrow格式: [SMILES, name, docking_score]
                            molecules.append([smiles, f"mol_{idx}", str(docking_score)])
                        except ValueError:
                            logger.warning(f"无法解析分数 {parts[1]} for SMILES {smiles}")
    except FileNotFoundError:
        logger.error(f"找不到文件 {docked_file}")
        return []    
    logger.info(f"成功加载 {len(molecules)} 个分子")
    return molecules

def merge_and_deduplicate_populations(child_molecules: List[List[str]], 
                                    parent_molecules: List[List[str]] = None) -> List[List[str]]:
    """
    合并父代和子代分子群体，并去重。
    
    Args:
        child_molecules: 子代分子列表
        parent_molecules: 父代分子列表（可选）
    
    Returns:
        合并并去重后的分子列表
    """
    # 使用字典来去重，key为SMILES，value为完整的分子信息
    # 如果同一个SMILES有不同分数，保留分数更好的（更小的）
    merged_dict = {}
    
    # 先添加子代分子
    for mol in child_molecules:
        smiles = mol[0]
        score = float(mol[2])
        if smiles not in merged_dict or score < float(merged_dict[smiles][2]):
            merged_dict[smiles] = mol
    
    # 如果有父代分子，添加父代分子（保持去重逻辑）
    if parent_molecules:
        for mol in parent_molecules:
            smiles = mol[0]
            score = float(mol[2])
            if smiles not in merged_dict or score < float(merged_dict[smiles][2]):
                merged_dict[smiles] = mol
    
    merged_list = list(merged_dict.values())
    
    if parent_molecules:
        logger.info(f"合并完成: 子代({len(child_molecules)}) + 父代({len(parent_molecules)}) = 总计{len(merged_list)}个独特分子")
    else:
        logger.info(f"仅使用子代分子: {len(merged_list)}个独特分子")
    
    return merged_list

def select_molecules_by_algorithm(molecules_data: List[List[str]], 
                                 n_select: int, 
                                 selector_choice: str, 
                                 tourn_size: float = 0.1) -> List[str]:
    """
    使用指定算法选择分子。    
    Args:
        molecules_data: autogrow格式的分子数据列表
        n_select: 要选择的分子数量
        selector_choice: 选择算法（"Rank_Selector", "Roulette_Selector", "Tournament_Selector"）
        tourn_size: 锦标赛大小(仅用于Tournament_Selector)
    
    Returns:
        选中的SMILES列表
    """
    if not molecules_data:
        logger.warning("分子数据为空，无法执行选择。")
        return []
    
    if n_select <= 0:
        logger.warning("选择数量必须大于0。")
        return []
    
    if len(molecules_data) < n_select:
        logger.warning(f"候选分子数量({len(molecules_data)})少于要选择的数量({n_select})，返回所有分子。")
        return [mol[0] for mol in molecules_data]
    
    logger.info(f"使用 {selector_choice} 从 {len(molecules_data)} 个候选中选择 {n_select} 个分子")
    
    try:
        if selector_choice == "Rank_Selector":
            # 排名选择：基于对接分数排名，选择最优的N个
            # column_idx_to_select=-1表示使用最后一列(docking_score)
            # reverse_sort=False表示升序排序（对接分数越小越好）
            selected_smiles = Rank_Sel.run_rank_selector(
                molecules_data, n_select, -1, False
            )
            
        elif selector_choice == "Roulette_Selector":
            # 轮盘赌选择：基于对接分数的加权随机选择
            numpy_result = Roulette_Sel.spin_roulette_selector(
                molecules_data, n_select, "docking"
            )
            # spin_roulette_selector返回numpy数组，需要转换为列表
            selected_smiles = numpy_result.tolist()
            
        elif selector_choice == "Tournament_Selector":
            # 锦标赛选择：随机分组竞赛选择
            # idx_to_sel=-1表示使用最后一列(docking_score)
            # favor_most_negative=True表示对接分数越小越好
            selected_molecules = Tournament_Sel.run_Tournament_Selector(
                molecules_data, n_select, tourn_size, -1, True
            )
            # Tournament_Selector返回完整的分子信息，需要提取SMILES
            selected_smiles = [mol[0] for mol in selected_molecules]
            
        else:
            raise ValueError(f"不支持的选择算法: {selector_choice}")
        
        logger.info(f"{selector_choice} 选择完成：成功选出 {len(selected_smiles)} 个分子")
        return selected_smiles
        
    except Exception as e:
        logger.error(f"选择算法执行失败: {e}")
        return []

def save_selected_molecules(selected_smiles: List[str], output_file: str):
    """将选中的分子SMILES保存到文件。"""
    with open(output_file, 'w') as f:
        for smiles in selected_smiles:
            f.write(f"{smiles}\n")
    logger.info(f"已保存 {len(selected_smiles)} 个选中的分子到 {output_file}")

def print_selection_statistics(molecules_data: List[List[str]], selected_smiles: List[str]):
    """打印选择统计信息。"""
    if not selected_smiles:
        logger.warning("没有选择任何分子用于统计。")
        return
    
    # 获取选中分子的对接分数
    selected_scores = []
    smiles_to_score = {mol[0]: float(mol[-1]) for mol in molecules_data}
    
    for smiles in selected_smiles:
        if smiles in smiles_to_score:
            selected_scores.append(smiles_to_score[smiles])
    
    if selected_scores:
        print("\n========== 选择统计信息 ==========")
        print(f"选中分子数量: {len(selected_smiles)}")
        print(f"对接分数范围: {min(selected_scores):.4f} 至 {max(selected_scores):.4f}")
        print(f"平均对接分数: {sum(selected_scores)/len(selected_scores):.4f}")
        print("=" * 40)

def main():
    parser = argparse.ArgumentParser(description='多种分子选择算法 - 支持父代+子代合并选择')
    parser.add_argument('--docked_file', type=str, required=True,
                        help='子代对接结果文件路径（格式: SMILES score)')
    parser.add_argument('--parent_file', type=str, default=None,
                        help='父代对接结果文件路径（可选，格式: SMILES score)')
    parser.add_argument('--output_file', type=str, required=True,
                        help='输出的选中分子文件路径')
    parser.add_argument('--config_file', type=str, default='GA_gpt/config_example.json',
                        help='配置文件路径')
    
    args = parser.parse_args()

    # 从配置文件加载参数
    try:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        selection_config = config.get("molecular_selection", {})
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"无法加载或解析配置文件 {args.config_file}: {e}")
        return

    n_select = selection_config.get('n_select', 100)
    selector_choice = selection_config.get('selector_choice', 'Rank_Selector')
    tourn_size = selection_config.get('tourn_size', 0.1)

    logger.info(f"开始 {selector_choice} 分子选择...")
    logger.info(f"子代文件: {args.docked_file}")
    if args.parent_file:
        logger.info(f"父代文件: {args.parent_file}")
    logger.info(f"输出文件: {args.output_file}")
    logger.info(f"选择数量 (来自配置): {n_select}")
    if selector_choice == "Tournament_Selector":
        logger.info(f"锦标赛大小 (来自配置): {tourn_size}")

    # 1. 加载子代分子
    child_molecules = load_molecules_with_scores(args.docked_file)
    if not child_molecules:
        logger.error("无法加载子代分子数据，程序终止。")
        return

    # 2. 加载父代分子（如果提供）
    parent_molecules = None
    if args.parent_file:
        parent_molecules = load_molecules_with_scores(args.parent_file)
        if not parent_molecules:
            logger.warning("无法加载父代分子数据，将仅使用子代分子进行选择。")

    # 3. 合并并去重分子群体
    merged_molecules = merge_and_deduplicate_populations(child_molecules, parent_molecules)
    if not merged_molecules:
        logger.error("合并后的分子群体为空，程序终止。")
        return

    # 4. 使用指定算法选择分子
    selected_smiles = select_molecules_by_algorithm(
        merged_molecules, n_select, selector_choice, tourn_size
    )

    # 5. 保存并打印结果
    if selected_smiles:
        save_selected_molecules(selected_smiles, args.output_file)
        print_selection_statistics(merged_molecules, selected_smiles)
    else:
        logger.error("未选择任何分子。")

if __name__ == "__main__":
    main()
