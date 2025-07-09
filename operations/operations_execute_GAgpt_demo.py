#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GA-GPT 混合工作流执行脚本
==========================
1. 种群初始化和评估
2. 基于父代的分子分解与掩码
3. 使用GPT模型生成新的、多样化的分子
4. 对父代和GPT生成的分子进行遗传算法操作(交叉、突变)
5. 对新生成的子代进行评估
6. 通过选择策略（单目标或多目标）筛选出下一代种群
"""
import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from operations.stating.config_snapshot_generator import save_config_snapshot
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class GAGPTWorkflowExecutor:    
    def __init__(self, config_path: str, receptor_name: Optional[str] = None, output_dir_override: Optional[str] = None):
        """
        初始化GA-GPT工作流执行器。        
        Args:
            config_path (str): 配置文件路径。
            receptor_name (Optional[str]): 目标受体名称。如果为None, 则使用默认受体。
            output_dir_override (Optional[str]): 覆盖配置文件中的输出目录。
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.run_params = {}
        self._setup_parameters_and_paths(receptor_name, output_dir_override)
        self._save_run_parameters()        
        logger.info(f"GA-GPT工作流初始化完成, 输出目录: {self.output_dir}")
        logger.info(f"最大迭代代数: {self.max_generations}")

    def _load_config(self) -> dict:
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)       

    def _setup_parameters_and_paths(self, receptor_name: Optional[str], output_dir_override: Optional[str]):        
        self.project_root = Path(self.config.get('paths', {}).get('project_root', PROJECT_ROOT))
        workflow_config = self.config.get('workflow', {})
        gpt_config = self.config.get('gpt', {})
        self.dynamic_masking_config = gpt_config.get('dynamic_masking', {'enable': False})
        # 记录配置和根目录
        self.run_params['config_file_path'] = self.config_path
        self.run_params['project_root'] = str(self.project_root)
        # 确定输出目录
        if output_dir_override:
            output_dir_name = output_dir_override
        else:
            output_dir_name = workflow_config.get('output_directory', 'GA_GPT_output')
        base_output_dir = self.project_root / output_dir_name
        self.run_params['base_output_dir'] = str(base_output_dir)
        # 根据受体确定最终运行目录
        self.receptor_name = receptor_name
        if self.receptor_name:
            self.output_dir = base_output_dir / self.receptor_name
            self.run_params['receptor_name'] = self.receptor_name
        else:
            # 如果没有指定受体，使用默认或创建一个通用运行目录
            default_receptor_info = self.config.get('receptors', {}).get('default_receptor', {})
            default_receptor_name = default_receptor_info.get('name', 'default_run')
            self.output_dir = base_output_dir / default_receptor_name
            self.run_params['receptor_name'] = default_receptor_name
        self.run_params['run_specific_output_dir'] = str(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 加载GA和GPT的核心参数
        self.max_generations = workflow_config.get('max_generations', 10)
        self.initial_population_file = workflow_config.get('initial_population_file')
        self.population_size = workflow_config.get('population_size', 115)        
        self.run_params['max_generations'] = self.max_generations
        self.run_params['initial_population_file'] = self.initial_population_file
        self.run_params['population_size'] = self.population_size
        # 记录选择模式
        selection_config = self.config.get('selection', {})
        self.run_params['selection_mode'] = selection_config.get('selection_mode', 'single_objective')
    def _get_dynamic_mask_count(self, generation: int) -> int:
        """
        根据当前代数计算动态掩码片段的数量。
        如果未启用动态掩码，则返回配置中的固定值。
        Args:
            generation (int): 当前的进化代数。            
        Returns:
            int: 应该用于掩码的片段数量。
        """
        if not self.dynamic_masking_config.get('enable', False) or self.max_generations <= 1:
            # 如果不启用或总代数只有1代，则使用固定的值
            return self.config.get('gpt', {}).get('n_fragments_to_mask', 1)        
        initial_mask = self.dynamic_masking_config.get('initial_mask_fragments', 2)
        final_mask = self.dynamic_masking_config.get('final_mask_fragments', 1)        
        # 使用线性插值计算当前代数的掩码数
        # y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        # 这里 x=generation, x1=1, y1=initial_mask, x2=max_generations, y2=final_mask        
        # 防止除以零
        if self.max_generations == 1:
            return initial_mask            
        progress = (generation - 1) / (self.max_generations - 1)
        mask_count = initial_mask + progress * (final_mask - initial_mask)        
        # 四舍五入到最近的整数，并确保结果在[final_mask, initial_mask]范围内
        return int(round(max(min(mask_count, initial_mask), final_mask)))
    def _save_run_parameters(self):
        """保存本次运行的完整参数快照。"""
        snapshot_file_path = self.output_dir / "execution_config_snapshot.json"
        success = save_config_snapshot(
            original_config=self.config,
            execution_context=self.run_params,
            output_file_path=str(snapshot_file_path)
        )
        if success:
            logger.info(f"完整的执行配置快照已保存到: {snapshot_file_path}")
        else:
            logger.error("保存执行配置快照失败")

    def _run_script(self, script_path: str, args: List[str]) -> bool:
        """
        统一的脚本执行函数。        
        Args:
            script_path (str): 相对于项目根目录的脚本路径。
            args (List[str]): 脚本的命令行参数列表。            
        Returns:
            bool: 脚本是否执行成功。
        """
        full_script_path = self.project_root / script_path
        cmd = ['python', str(full_script_path)] + args        
        logger.debug(f"执行命令: {' '.join(cmd)}")        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.project_root), check=False)            
            if result.returncode == 0:
                logger.info(f"脚本 {script_path} 执行成功")
                return True
            else:
                logger.error(f"脚本 {script_path} 执行失败")
                logger.error(f"错误输出 (stderr):\n{result.stderr}")
                if result.stdout:
                    logger.error(f"标准输出 (stdout):\n{result.stdout}")
                return False
        except Exception as e:
            logger.error(f"执行脚本 {script_path} 时发生异常: {e}")
            return False
    def _count_molecules(self, file_path: str) -> int:
        """统计SMILES文件中的分子数量"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                count = sum(1 for line in f if line.strip())
            return count
        except FileNotFoundError:
            return 0
    def _remove_duplicates_from_smiles_file(self, input_file: str, output_file: str) -> int:
        """
        去除SMILES文件中的重复分子,并为每个分子添加唯一ID。
        输出格式: SMILES  ligand_id_X
        """
        try:
            unique_smiles = set()
            with open(input_file, 'r') as f:
                for line in f:
                    smiles = line.strip().split()[0]
                    if smiles:
                        unique_smiles.add(smiles)            
            unique_smiles_list = sorted(list(unique_smiles))            
            with open(output_file, 'w') as f:
                for i, smiles in enumerate(unique_smiles_list):
                    f.write(f"{smiles}\tligand_id_{i}\n")            
            logger.info(f"去重完成: {len(unique_smiles_list)} 个独特分子保存到 {output_file}")
            return len(unique_smiles_list)
        except Exception as e:
            logger.error(f"去重过程中发生错误: {e}")
            return 0

    def _extract_smiles_from_docked_file(self, docked_file: str, output_smiles_file: str) -> bool:
        """从带对接分数的文件中提取纯SMILES,用于遗传操作或分解"""
        try:
            with open(docked_file, 'r') as infile, open(output_smiles_file, 'w') as outfile:
                for line in infile:
                    line = line.strip()
                    if line:
                        smiles = line.split()[0]
                        outfile.write(f"{smiles}\n")
            return True
        except Exception as e:
            logger.error(f"从 {docked_file} 提取SMILES时出错: {e}")
            return False

    def run_decomposition_and_masking(self, parent_smiles_file: str, generation: int) -> Optional[str]:
        """
        执行分子分解和掩码操作。        
        Args:
            parent_smiles_file (str): 父代SMILES文件路径。
            generation (int): 当前代数。            
        Returns:
            Optional[str]: 成功则返回掩码后片段文件的路径，失败则返回None。
        """
        logger.info(f"第 {generation} 代: 开始分解和掩码...")
        gen_dir = self.output_dir / f"generation_{generation}"
        gen_dir.mkdir(exist_ok=True)
        masked_fragments_file = gen_dir / "masked_fragments.smi"        
        # -- 动态计算掩码数 --
        n_mask = self._get_dynamic_mask_count(generation)
        logger.info(f"第 {generation} 代: 使用动态掩码数 n_mask = {n_mask}")        
        decompose_args = [
            '--input', parent_smiles_file,
            '--output3', str(masked_fragments_file), # 使用output3作为掩码文件的输出
            '--mask_fragments', str(n_mask)]
        if not self._run_script('datasets/decompose/demo_frags.py', decompose_args):
            logger.error(f"第 {generation} 代: 分解和掩码失败。")
            return None        
        if self._count_molecules(str(masked_fragments_file)) == 0:
            logger.warning(f"第 {generation} 代: 未生成任何有效的掩码片段。")
            return None
        logger.info(f"第 {generation} 代: 分解和掩码完成，结果保存至 {masked_fragments_file}")
        return str(masked_fragments_file)
    def run_gpt_generation(self, masked_fragments_file: str, generation: int) -> Optional[str]:
        """
        使用GPT模型生成新分子。        
        Args:
            masked_fragments_file (str): 掩码片段文件路径。
            generation (int): 当前代数。            
        Returns:
            Optional[str]: 成功则返回GPT生成的新分子文件路径，失败则返回None。
        """
        logger.info(f"第 {generation} 代: 开始GPT生成...")
        gen_dir = self.output_dir / f"generation_{generation}"
        gpt_output_dir = gen_dir / "gpt_generated"
        gpt_output_dir.mkdir(exist_ok=True)        
        # 注意: fragment_GPT/generate_all.py的输出是固定的，需要找到它
        # 这里我们假设它输出到 'fragment_GPT/output/crossovered0_frags_new_{seed}.smi'
        gpt_config = self.config.get('gpt', {})
        seed = gpt_config.get('seed', generation) # 使用代数作为种子以保证可复现性        
        # 实际的GPT脚本输出路径
        # TODO: 强烈建议将GPT脚本的输出路径改为通过命令行参数传递
        gpt_script_output_dir = self.project_root / "fragment_GPT/output"
        # 假设输出文件名格式，需要根据`generate_all.py`的实现来确定
        expected_output_filename = f'crossovered0_frags_new_{seed}.smi'
        expected_output_path = gpt_script_output_dir / expected_output_filename
        # 执行前删除旧的输出文件，以确保我们得到的是本次运行的结果
        if expected_output_path.exists():
            expected_output_path.unlink()
        gpt_args = [
            '--input_file', masked_fragments_file,
            '--seed', str(seed)]
        if not self._run_script('fragment_GPT/generate_all.py', gpt_args):
            logger.error(f"第 {generation} 代: GPT生成脚本执行失败。")
            return None        
        # 检查预期的输出文件是否已生成
        if not expected_output_path.exists():
            logger.error(f"第 {generation} 代: GPT生成失败，未找到预期的输出文件: {expected_output_path}")
            logger.error(f"请检查fragment_GPT脚本是否正常执行，以及输出目录 {gpt_script_output_dir} 是否存在")
            return None            
        # 将生成的文件移动到代数目录中并重命名
        gpt_generated_file = gpt_output_dir / "gpt_generated_molecules.smi"
        try:
            import shutil
            shutil.move(str(expected_output_path), str(gpt_generated_file))
        except Exception as e:
            logger.error(f"无法移动GPT生成的文件: {e}")
            return None
        generated_count = self._count_molecules(str(gpt_generated_file))
        if generated_count == 0:
            logger.warning(f"第 {generation} 代: GPT生成了0个有效分子。")
            # 不认为是致命错误，可以继续执行GA
            return None
            
        logger.info(f"第 {generation} 代: GPT生成完成,产出 {generated_count} 个新分子。")
        return str(gpt_generated_file)
    def _combine_files(self, file_list: List[str], output_file: str) -> bool:
        """合并多个SMILES文件到一个文件"""
        try:
            with open(output_file, 'w') as outf:
                for file_path in file_list:
                    if not file_path or not Path(file_path).exists():
                        continue
                    with open(file_path, 'r') as inf:
                        for line in inf:
                            line = line.strip()
                            if line:
                                outf.write(line + '\n')
            return True
        except Exception as e:
            logger.error(f"合并文件时发生错误: {e}")
            return False
    def run_ga_operations(self, parent_smiles_file: str, gpt_generated_file: Optional[str], generation: int) -> Optional[Tuple[str, str]]:
        """
        执行遗传算法操作（交叉和突变）。
        
        Args:
            parent_smiles_file (str): 父代SMILES文件路径。
            gpt_generated_file (Optional[str]): GPT生成的SMILES文件路径。
            generation (int): 当前代数。
            
        Returns:
            Optional[Tuple[str, str]]: 成功则返回(交叉后代文件, 突变后代文件)，失败则返回None。
        """
        logger.info(f"第 {generation} 代: 开始遗传算法操作...")
        gen_dir = self.output_dir / f"generation_{generation}"

        # 1. 合并父代和GPT产出，作为GA操作的输入
        ga_input_pool_file = gen_dir / "ga_input_pool.smi"
        files_to_combine = [parent_smiles_file]
        if gpt_generated_file:
            files_to_combine.append(gpt_generated_file)
        
        if not self._combine_files(files_to_combine, str(ga_input_pool_file)):
            logger.error(f"第 {generation} 代: 合并父代和GPT产出失败。")
            return None
        logger.info(f"第 {generation} 代: GA操作输入池已创建，共 {self._count_molecules(str(ga_input_pool_file))} 个分子。")

        # 2. 交叉操作
        crossover_raw_file = gen_dir / "crossover_raw.smi"
        crossover_filtered_file = gen_dir / "crossover_filtered.smi"
        
        crossover_succeeded = self._run_script('operations/crossover/crossover_demo_finetune.py', [
            '--smiles_file', str(ga_input_pool_file),
            '--output_file', str(crossover_raw_file),
            '--config_file', self.config_path
        ])
        if not crossover_succeeded:
            logger.error(f"第 {generation} 代: 交叉脚本执行失败。")
            return None

        filter_crossover_succeeded = self._run_script('operations/filter/filter_demo.py', [
            '--smiles_file', str(crossover_raw_file),
            '--output_file', str(crossover_filtered_file)
        ])
        if not filter_crossover_succeeded:
            logger.error(f"第 {generation} 代: 交叉结果过滤失败。")
            return None
            
        logger.info(f"交叉完成，生成 {self._count_molecules(str(crossover_filtered_file))} 个过滤后的分子。")

        # 3. 突变操作
        mutation_raw_file = gen_dir / "mutation_raw.smi"
        mutation_filtered_file = gen_dir / "mutation_filtered.smi"
        
        mutation_succeeded = self._run_script('operations/mutation/mutation_demo_finetune.py', [
            '--smiles_file', str(ga_input_pool_file),
            '--output_file', str(mutation_raw_file),
            '--config_file', self.config_path
        ])
        if not mutation_succeeded:
            logger.error(f"第 {generation} 代: 突变脚本执行失败。")
            return None

        filter_mutation_succeeded = self._run_script('operations/filter/filter_demo.py', [
            '--smiles_file', str(mutation_raw_file),
            '--output_file', str(mutation_filtered_file)
        ])
        if not filter_mutation_succeeded:
            logger.error(f"第 {generation} 代: 突变结果过滤失败。")
            return None

        logger.info(f"突变完成，生成 {self._count_molecules(str(mutation_filtered_file))} 个过滤后的分子。")
        
        return str(crossover_filtered_file), str(mutation_filtered_file)

    def run_offspring_evaluation(self, crossover_file: str, mutation_file: str, generation: int) -> Optional[str]:
        """
        执行子代种群的评估（对接）。
        
        Args:
            crossover_file (str): 交叉后代文件路径。
            mutation_file (str): 突变后代文件路径。
            generation (int): 当前代数。
            
        Returns:
            Optional[str]: 成功则返回子代对接结果文件路径，失败则返回None。
        """
        logger.info(f"第 {generation} 代: 开始子代评估...")
        gen_dir = self.output_dir / f"generation_{generation}"

        # 1. 合并交叉和突变结果
        offspring_raw_file = gen_dir / "offspring_combined_raw.smi"
        if not self._combine_files([crossover_file, mutation_file], str(offspring_raw_file)):
            logger.error(f"第 {generation} 代: 子代合并失败。")
            return None
        
        # 2. 对子代进行去重和格式化（为对接做准备）
        offspring_formatted_file = gen_dir / "offspring_formatted_for_docking.smi"
        offspring_count = self._remove_duplicates_from_smiles_file(
            str(offspring_raw_file), 
            str(offspring_formatted_file)
        )
        if offspring_count == 0:
            logger.warning(f"第 {generation} 代: 经过滤和去重后，无有效子代分子。")
            # 创建一个空的对接文件，避免后续对接步骤
            offspring_docked_file = gen_dir / "offspring_docked.smi"
            open(offspring_docked_file, 'a').close()  # 创建一个空文件
            return str(offspring_docked_file)

        logger.info(f"子代格式化完成: 共 {offspring_count} 个独特分子准备对接。")

        # 3. 对子代进行对接
        offspring_docked_file = gen_dir / "offspring_docked.smi"
        docking_args = [
            '--smiles_file', str(offspring_formatted_file),
            '--output_file', str(offspring_docked_file),
            '--generation_dir', str(gen_dir),
            '--config_file', self.config_path
        ]
        if self.receptor_name:
            docking_args.extend(['--receptor', self.receptor_name])
            
        if not self._run_script('operations/docking/docking_demo_finetune.py', docking_args):
            logger.error(f"第 {generation} 代: 子代对接失败。")
            return None

        docked_count = self._count_molecules(str(offspring_docked_file))
        logger.info(f"第 {generation} 代: 子代评估完成，{docked_count} 个分子已对接。")

        return str(offspring_docked_file)

    def run_selection(self, parent_docked_file: str, offspring_docked_file: str, generation: int) -> Optional[str]:
        """
        执行选择操作，从父代和子代中选出下一代。
        
        Args:
            parent_docked_file (str): 父代对接结果文件路径。
            offspring_docked_file (str): 子代对接结果文件路径。
            generation (int): 当前代数。
            
        Returns:
            Optional[str]: 成功则返回下一代父代文件路径，失败则返回None。
        """
        logger.info(f"第 {generation} 代: 开始选择操作...")
        gen_dir = self.output_dir / f"generation_{generation}"
        next_parents_file = self.output_dir / f"generation_{generation+1}" / "initial_population_docked.smi"
        next_parents_file.parent.mkdir(exist_ok=True)

        selection_config = self.config.get('selection', {})
        selection_mode = selection_config.get('selection_mode', 'single_objective')

        selection_succeeded = False
        if selection_mode == 'single_objective':
            logger.info("执行单目标选择...")
            selection_args = [
                '--docked_file', offspring_docked_file,
                '--parent_file', parent_docked_file,
                '--output_file', str(next_parents_file),
                '--config_file', self.config_path
            ]
            selection_succeeded = self._run_script('operations/selecting/molecular_selection.py', selection_args)
        
        elif selection_mode == 'multi_objective':
            logger.info("执行多目标选择...")
            multi_obj_config = selection_config.get('multi_objective_settings', {})
            n_select = multi_obj_config.get('n_select', self.population_size)
            selection_args = [
                '--docked_file', offspring_docked_file,
                '--parent_file', parent_docked_file,
                '--output_file', str(next_parents_file),
                '--n_select', str(n_select)
            ]
            selection_succeeded = self._run_script('operations/selecting/selecting_multi_demo.py', selection_args)
        
        else:
            logger.error(f"不支持的选择模式: {selection_mode}")
            return None

        if not selection_succeeded or self._count_molecules(str(next_parents_file)) == 0:
            logger.error(f"第 {generation} 代: 选择操作失败或未选出任何分子。")
            return None
        
        selected_count = self._count_molecules(str(next_parents_file))
        logger.info(f"选择操作完成 ({selection_mode}): 选出 {selected_count} 个分子作为下一代父代。")
        
        return str(next_parents_file)

    def run_selected_population_evaluation(self, selected_parents_file: str, generation: int) -> bool:
        """
        对选择后的精英种群（下一代父代）进行评分分析        
        Args:
            selected_parents_file (str): 选择后的下一代父代文件路径
            generation (int): 当前代数            
        Returns:
            bool: 评分分析是否成功
        """
        logger.info(f"第 {generation} 代: 开始对选择后的精英种群进行评分分析")
        
        gen_dir = self.output_dir / f"generation_{generation}"
        scoring_report_file = gen_dir / f"generation_{generation}_evaluation.txt"
        
        scoring_succeeded = self._run_script('operations/scoring/scoring_demo.py', [
            '--current_population_docked_file', str(selected_parents_file),
            '--initial_population_file', self.initial_population_file,
            '--output_file', str(scoring_report_file)
        ])        
        if scoring_succeeded:
            logger.info(f"第 {generation} 代: 精英种群评分分析完成，报告保存到 {scoring_report_file}")
        else:
            logger.warning(f"第 {generation} 代: 精英种群评分分析失败，但不影响主流程")            
        return scoring_succeeded

    def run_complete_workflow(self):
        """
        执行完整的GA-GPT工作流。
        """
        logger.info(f"开始执行完整的GA-GPT工作流程 (输出目录: {self.output_dir})")
        
        # 第0步：初代种群处理
        current_parents_docked_file = self.run_initial_generation()
        if not current_parents_docked_file:
            logger.error("初代种群处理失败，工作流终止。")
            return False
        
        logger.info(f"初代种群处理成功，结果文件: {current_parents_docked_file}")

        # 开始迭代
        for generation in range(1, self.max_generations + 1):
            next_parents_file = self.run_generation_step(generation, current_parents_docked_file)
            if not next_parents_file:
                logger.error(f"第 {generation} 代处理失败，工作流终止。")
                return False
            current_parents_docked_file = next_parents_file
        
        logger.info("=" * 60)
        logger.info("GA-GPT工作流程全部完成!")
        logger.info(f"最终优化种群保存在: {current_parents_docked_file}")
        logger.info("=" * 60)
        
        return True

    def run_initial_generation(self) -> Optional[str]:
        """
        执行初代种群的处理，包括去重、格式化和对接。
        
        Returns:
            Optional[str]: 成功则返回初代种群对接结果文件的路径，失败则返回None。
        """
        logger.info("开始处理初代种群 (Generation 0)...")
        
        gen_dir = self.output_dir / "generation_0"
        gen_dir.mkdir(exist_ok=True)
        
        # 1. 检查初始种群文件是否存在
        if not Path(self.initial_population_file).exists():
            logger.error(f"初始种群文件未找到: {self.initial_population_file}")
            return None
        
        # 2. 去重并格式化初始种群
        initial_formatted_file = gen_dir / "initial_population_formatted.smi"
        unique_count = self._remove_duplicates_from_smiles_file(
            self.initial_population_file, 
            str(initial_formatted_file)
        )
        if unique_count == 0:
            logger.error("初始种群文件为空或处理失败。")
            return None
        
        # 3. 对初代种群进行对接
        initial_docked_file = gen_dir / "initial_population_docked.smi"
        docking_args = [
            '--smiles_file', str(initial_formatted_file),
            '--output_file', str(initial_docked_file),
            '--generation_dir', str(gen_dir),
            '--config_file', self.config_path
        ]
        if self.receptor_name:
            docking_args.extend(['--receptor', self.receptor_name])
            
        docking_succeeded = self._run_script('operations/docking/docking_demo_finetune.py', docking_args)
        
        docked_count = self._count_molecules(str(initial_docked_file))
        if not docking_succeeded or docked_count == 0:
            logger.error("初代种群对接失败或未生成任何有效对接结果。")
            return None
        
        logger.info(f"初代种群对接完成: {docked_count} 个分子已评分。")
        return str(initial_docked_file)

    def run_generation_step(self, generation: int, current_parents_docked_file: str):
        """
        执行单代GA-GPT的完整流程。
        """
        logger.info(f"========== 开始第 {generation} 代进化 ==========")
        gen_dir = self.output_dir / f"generation_{generation}"
        gen_dir.mkdir(exist_ok=True)
        # 1. 从父代对接文件中提取纯SMILES
        parent_smiles_file = gen_dir / "current_parent_smiles.smi"
        if not self._extract_smiles_from_docked_file(current_parents_docked_file, str(parent_smiles_file)):
            logger.error(f"第{generation}代: 无法从父代文件提取SMILES,工作流终止")
            return None

        # 2. 分解与掩码
        masked_file = self.run_decomposition_and_masking(str(parent_smiles_file), generation)
        if not masked_file:
            # 如果分解失败，可以决定是终止还是跳过GPT步骤
            logger.warning(f"第{generation}代: 分解掩码步骤失败,将跳过GPT生成。")
            gpt_generated_file = None
        else:
            # 3. GPT生成
            gpt_generated_file = self.run_gpt_generation(masked_file, generation)

        # 4. 遗传操作
        ga_children_files = self.run_ga_operations(str(parent_smiles_file), gpt_generated_file, generation)
        if not ga_children_files:
            logger.error(f"第{generation}代: 遗传操作失败，工作流终止。")
            return None
        
        crossover_file, mutation_file = ga_children_files

        # 5. 子代评估（对接，但不进行评分分析）
        offspring_docked_file = self.run_offspring_evaluation(crossover_file, mutation_file, generation)
        if offspring_docked_file is None:
             logger.error(f"第{generation}代: 子代评估失败，工作流终止。")
             return None

        # 6. 选择
        next_parents_docked_file = self.run_selection(
            current_parents_docked_file, 
            offspring_docked_file, 
            generation
        )
        if not next_parents_docked_file:
            logger.error(f"第{generation}代: 选择操作失败，工作流终止。")
            return None

        # 7. 对选择后的精英种群进行评分分析（这是新的逻辑）
        self.run_selected_population_evaluation(next_parents_docked_file, generation)

        logger.info(f"========== 第 {generation} 代进化完成 ==========")
        return next_parents_docked_file

# --- 主函数入口 ---
def main():
    """主函数，用于解析命令行参数和启动工作流"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GA-GPT混合工作流执行器')
    parser.add_argument('--config', type=str, 
                       default='GA_gpt/config_example.json',
                       help='配置文件路径')
    parser.add_argument('--receptor', type=str, default=None,
                       help='(可选) 要运行的目标受体名称')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='(可选) 指定输出目录，覆盖配置文件中的设置')
    
    args = parser.parse_args()
    
    try:
        executor = GAGPTWorkflowExecutor(args.config, args.receptor, args.output_dir)
        success = executor.run_complete_workflow()
        if not success:
            logger.error("GA-GPT工作流执行失败。")
            return 1
        
        logger.info("GA-GPT工作流成功完成!")

    except Exception as e:
        logger.critical(f"工作流执行过程中发生严重错误: {e}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
