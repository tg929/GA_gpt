#我需要对比的的指标计算
import argparse
import os
import numpy as np
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import QED
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
def calculate_sa_scores(mols: list) -> list: #SA计算   1-10：越小越好（合成难易）
    sa_scores = []
    sa_calculator = None    
    
    # 尝试多种SA计算器，按优先级顺序
    # 1. 首先尝试使用项目自带的sascorer
    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'fragment_GPT', 'utils'))
        import sascorer
        sa_calculator = sascorer.calculateScore
        print("使用项目自带的SA计算器 (fragment_GPT/utils/sascorer)")
    except ImportError:
        # 2. 尝试使用autogrow的SA计算器
        try:
            from autogrow.operators.filter.execute_filters import sascorer as autogrow_sascorer
            sa_calculator = autogrow_sascorer.calculateScore
            print("使用AutoGrow SA计算器")
        except ImportError:
            # 3. 最后尝试rdkit.Contrib中的SA_score
            try:
                from rdkit.Contrib.SA_score import sascorer as rdkit_sascorer
                sa_calculator = rdkit_sascorer.calculateScore
                print("使用RDKit Contrib SA计算器")
            except ImportError:
                print("警告: 所有SA计算器都无法导入，SA得分将不会被计算。")
                return sa_scores # 如果无法导入任何计算器，直接返回空列表
    
    # 2. 遍历分子并计算分数
    for mol in mols:
        try:
            sa_score = sa_calculator(mol)
            sa_scores.append(sa_score)
        except Exception as e:
            # 如果单个分子计算失败，打印警告并继续
            print(f"警告: 无法为某个分子计算SA分数。错误: {e}")
            sa_scores.append(None) # 添加None作为占位符
    
    print(f"成功为 {sum(s is not None for s in sa_scores)}/{len(mols)} 个分子计算了SA分数。")
    return sa_scores
def calculate_qed_scores(mols):  #QED  0-1
    qed_scores = []    
    for mol in mols:
        try:            
            qed_scores.append(QED.qed(mol))
        except Exception as e:
            print(f"Warning: Could not calculate QED for a molecule. Error: {str(e)}")    
    print(f"Successfully calculated QED scores for {len(qed_scores)} out of {len(mols)} molecules.")
    return qed_scores
def load_smiles_from_file(filepath):   #加载smile
    smiles_list = []    
    with open(filepath, 'r') as f:
        for line in f:
            smiles = line.strip().split()[0] 
            if smiles:
                smiles_list.append(smiles)    
    return smiles_list
def load_smiles_and_scores_from_file(filepath):   #加载smile和score：对接之后输出文件（带分数）
    molecules = []
    scores = []
    smiles_list = []    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                smiles = parts[0]
                try:
                    score = float(parts[1])
                    molecules.append(smiles)
                    scores.append(score)
                    smiles_list.append(smiles)
                except ValueError:
                    print(f"Warning: Could not parse score for SMILES: {smiles}")
            elif len(parts) == 1: # If only SMILES is present, no score
                smiles_list.append(parts[0])    
    return smiles_list, molecules, scores
def get_rdkit_mols(smiles_list): #smiles-----mol
    mols = []
    valid_smiles = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            mols.append(mol)
            valid_smiles.append(s)
        else:
            print(f"Warning: Could not parse SMILES: {s}")
    return mols, valid_smiles
def calculate_docking_stats(scores):
    """Calculates Top-1, Top-10 mean, Top-100 mean docking scores."""    
    sorted_scores = sorted(scores) # Docking scores, lower is better
    top1_score = sorted_scores[0] if len(sorted_scores) >= 1 else np.nan    #top1
    top10_scores = sorted_scores[:10]
    top10_mean = np.mean(top10_scores) if top10_scores else np.nan           #top10
    top100_scores = sorted_scores[:100]
    top100_mean = np.mean(top100_scores) if top100_scores else np.nan        #top100
    return top1_score, top10_mean, top100_mean
def calculate_novelty(current_smiles, initial_smiles_path): #Nov
    """Calculates novelty against an initial set of SMILES."""
    if not current_smiles:
        return 0.0
    initial_smiles = load_smiles_from_file(initial_smiles_path)           
    set_initial_smiles = set(initial_smiles) #有去重
    set_current_smiles = set(current_smiles)    
    new_molecules = set_current_smiles - set_initial_smiles    
    novelty = len(new_molecules) / len(set_current_smiles) if len(set_current_smiles) > 0 else 0.0
    return novelty
def calculate_top100_diversity(mols):    #Div   :分子指纹和Tanimoto similarity     #top100      
    # 取前100个分子（如果不足100个则取全部）
    top_mols = mols[:min(100, len(mols))]    
    fps = [GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in top_mols]    
    sum_similarity = 0
    num_pairs = 0    
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sum_similarity += similarity
            num_pairs += 1            
    if num_pairs == 0:
        return 0.0         
    average_similarity = sum_similarity / num_pairs
    diversity = 1.0 - average_similarity
    return diversity
def print_calculation_results(results):    
    print("Calculation Results:")
    print(results)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a generation of molecules.")
    parser.add_argument("--current_population_docked_file", type=str, required=True,
                        help="Path to the SMILES file of the current population with docking scores (SMILES score per line).")
    parser.add_argument("--initial_population_file", type=str, required=True,
                        help="Path to the SMILES file of the initial population (for novelty calculation).")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output file to save calculated metrics (e.g., results.txt or results.csv).")
    
    args = parser.parse_args()
    print(f"Processing population file: {args.current_population_docked_file}")
    print(f"Using initial population for novelty: {args.initial_population_file}")
    print(f"Saving results to: {args.output_file}")
    # Load current population SMILES and scores
    current_smiles_list, scored_molecules_smiles, docking_scores = load_smiles_and_scores_from_file(args.current_population_docked_file)
    # 首先根据对接分数对分子进行排序（分数越低越好）
    # 创建(SMILES, score)对，按分数排序
    if scored_molecules_smiles and docking_scores:
        molecules_with_scores = list(zip(scored_molecules_smiles, docking_scores))
        # 按对接分数排序（越小越好）
        molecules_with_scores.sort(key=lambda x: x[1])        
        # 获取top 100分子（如果不足100个则取全部）
        top_molecules_count = min(100, len(molecules_with_scores))
        top_molecules = molecules_with_scores[:top_molecules_count]#top100/top<100
        top_smiles = [item[0] for item in top_molecules]        #top100中只取smile
        print(f"Selected top {top_molecules_count} molecules for QED and SA calculation")     #top100/top<100   
        # 将top分子转换为RDKit Mol对象
        top_mols, valid_top_smiles = get_rdkit_mols(top_smiles)#top100/top<100:smile-->mol
    else:
        print("Warning: No molecules with docking scores found. Using all molecules for calculations.")
        top_mols = []
    # Convert all SMILES to RDKit Mol objects for general metrics
    all_mols, valid_smiles_for_props = get_rdkit_mols(current_smiles_list)
    # 1. Docking Score Metrics
    top1_score, top10_mean_score, top100_mean_score = calculate_docking_stats(docking_scores)    
    # 2. Novelty   
    novelty = calculate_novelty(list(set(current_smiles_list)), args.initial_population_file)    
    # 3. Diversity (Top 100 molecules)
    diversity = calculate_top100_diversity(top_mols if top_mols else all_mols)    
    # 4. QED & SA Scores (for Top 100 or all molecules if no scores)
    mols_for_scoring = top_mols if top_mols else all_mols
    qed_scores = calculate_qed_scores(mols_for_scoring)
    sa_scores = calculate_sa_scores(mols_for_scoring)    
    mean_qed = np.mean([s for s in qed_scores if s is not None]) if qed_scores else np.nan
    mean_sa = np.mean([s for s in sa_scores if s is not None]) if sa_scores else np.nan    
    score_description = f"Top {len(mols_for_scoring)}" if top_mols else "All Molecules"

    # 安全地处理可能包含特殊字符的文件名
    population_filename = os.path.basename(args.current_population_docked_file)
    initial_population_filename = os.path.basename(args.initial_population_file)    
    # 为了避免f-string格式化问题，使用传统的字符串格式化
    results = "Metrics for Population: {}\n".format(population_filename)
    results += "--------------------------------------------------\n"
    results += "Total molecules processed: {}\n".format(len(current_smiles_list))
    results += "Valid RDKit molecules for properties: {}\n".format(len(all_mols))
    results += "Molecules with docking scores: {}\n".format(len(docking_scores))
    results += "--------------------------------------------------\n"    
    # 处理浮点数格式化，注意处理NaN情况
    if np.isnan(top1_score): #top1
        results += "Docking Score - Top 1: N/A\n"
    else:
        results += "Docking Score - Top 1: {:.4f}\n".format(top1_score)
        
    if np.isnan(top10_mean_score): #top10
        results += "Docking Score - Top 10 Mean: N/A\n"
    else:
        results += "Docking Score - Top 10 Mean: {:.4f}\n".format(top10_mean_score)    

    if np.isnan(top100_mean_score): #top100
        results += "Docking Score - Top 100 Mean: N/A\n"
    else:
        results += "Docking Score - Top 100 Mean: {:.4f}\n".format(top100_mean_score)    
    results += "--------------------------------------------------\n"
    results += "Novelty (vs {}): {:.4f}\n".format(initial_population_filename, novelty)
    results += "Diversity (Top 100): {:.4f}\n".format(diversity)
    results += "--------------------------------------------------\n"    
    if np.isnan(mean_qed):
        results += "QED - {} Mean: N/A\n".format(score_description)
    else:
        results += "QED - {} Mean: {:.4f}\n".format(score_description, mean_qed)        
    if np.isnan(mean_sa):
        results += "SA Score - {} Mean: N/A\n".format(score_description)
    else:
        results += "SA Score - {} Mean: {:.4f}\n".format(score_description, mean_sa)    
    results += "--------------------------------------------------\n"    
    print_calculation_results(results)
    
    # Save results to output file
    try:
        with open(args.output_file, 'w') as f:
            f.write(results)
        print(f"Results successfully saved to {args.output_file}")
    except IOError:
        print(f"Error: Could not write results to {args.output_file}")
if __name__ == "__main__":
    main()
