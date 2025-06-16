# GA-GPT 混合分子生成主工作流程

## 简介

这是 FragGPT-GA 项目的核心工作流程脚本，实现了一个混合的生成工作流程：
- 使用基于片段的 GPT 模型扩展化学多样性
- 使用遗传算法优化分子以获得高质量候选物

## 核心特性

### 🧬 混合生成策略
- **GPT生成**: 扩展化学多样性，特别是在早期代数
- **遗传算法**: 基于特定适应度标准进行细粒度优化 (对接分数、QED、SA)

### 🔄 完整迭代循环
1. **初始化**: 从约115个父代分子开始
2. **评估**: 使用对接分数、QED和SA评估种群
3. **分解与掩码**: 将分子分解为片段并应用掩码
4. **GPT生成**: 从掩码片段生成新分子
5. **GA操作**: 对合并池 (GPT生成+父代) 执行交叉和变异
6. **过滤**: 确保化学有效性并移除不良结构
7. **选择**: 使用多目标优化选择下一代

## 安装要求

### 必需的Python包
```bash
# 基础科学计算
numpy
scipy

# 化学信息学
rdkit-pypi
tdc

# 机器学习
torch
transformers

# 多目标优化
pymoo

# 其他工具
tqdm
argparse
logging
```

### 外部软件
- **AutoDock Vina**: 分子对接
- **MGLTools**: 分子文件格式转换
- **GPU**: CUDA兼容的GPU (用于GPT生成)

## 使用方法

### 1. 准备输入文件

#### 初始SMILES文件
创建一个文本文件，每行包含一个SMILES字符串：
```
CCO
c1ccccc1
CC(C)O
```

#### 受体蛋白PDB文件
准备目标蛋白的PDB格式文件。

### 2. 运行工作流程

#### 基本用法
```bash
python main_workflow.py \
    --initial_smiles path/to/initial.smi \
    --receptor_pdb path/to/receptor.pdb \
    --population_size 115 \
    --max_generations 10
```

#### 使用示例脚本
```bash
# 修改 run_example.sh 中的参数
./run_example.sh
```

### 3. 完整参数说明

#### 必需参数
- `--initial_smiles, -i`: 初始SMILES文件路径
- `--receptor_pdb, -r`: 受体蛋白PDB文件路径

#### 工作流程参数
- `--population_size, -p`: 种群大小 (默认: 115)
- `--max_generations, -g`: 最大进化代数 (默认: 10)
- `--output_dir, -o`: 输出目录 (默认: output)
- `--temp_dir`: 临时文件目录 (默认: temp)

#### 分子分解参数
- `--n_fragments_to_mask`: 要掩码的片段数量 (默认: 1)

#### 遗传算法参数
- `--crossover_rate`: 交叉率 (默认: 0.8)
- `--crossover_attempts`: 交叉尝试次数 (默认: 20)
- `--mutation_attempts`: 变异尝试次数 (默认: 20)
- `--max_mutations_per_parent`: 每个父代的最大变异尝试次数 (默认: 2)

#### 分子对接参数
- `--center_x/y/z`: 对接中心坐标 (默认: 0.0)
- `--size_x/y/z`: 对接空间大小 (默认: 20.0)

#### 软件路径参数
- `--vina_executable`: AutoDock Vina可执行文件路径
- `--mgl_python`: MGLTools Python路径
- `--prepare_receptor4`: prepare_receptor4.py脚本路径
- `--prepare_ligand4`: prepare_ligand4.py脚本路径

#### 计算资源参数
- `--gpu_device`: GPU设备ID (默认: 0)
- `--num_processors`: CPU处理器数量 (默认: 4)

## 输出结果

### 目录结构
```
output/
├── workflow.log                 # 主日志文件
├── final_population.smi         # 最终种群
├── generations/                 # 每代的评估结果
├── fragments/                   # 分解片段文件
├── gpt_outputs/                 # GPT生成结果
├── docking_results/             # 对接结果
├── crossover_results/           # 交叉结果
├── mutation_results/            # 变异结果
├── filtered_results/            # 过滤结果
└── selection_results/           # 选择结果
```

### 关键输出文件
- `final_population.smi`: 最终进化得到的分子种群
- `workflow.log`: 详细的运行日志
- `generations/generation_X_scores.txt`: 每代的分子评分详情

## 工作流程详解

### 第1步: 初始化种群
从提供的SMILES文件中加载初始分子，调整到目标种群大小。

### 第2步: 分子评估
使用三个关键指标评估每个分子：
- **对接分数**: 与目标蛋白的结合亲和力 (越低越好)
- **QED分数**: 药物相似性评分 (0-1，越高越好)
- **SA分数**: 合成可达性评分 (1-10，越低越好)

### 第3步: 分子分解与掩码
使用BRICS规则将分子分解为片段，然后掩码最后n个片段以供GPT生成。

### 第4步: GPT多样性生成
基于掩码片段，使用训练好的Fragment-GPT模型生成新的分子候选物。

### 第5步: 遗传算法操作
- **交叉**: 在父代和GPT生成分子的混合池中进行交叉操作
- **变异**: 对混合池中的分子进行化学变异

### 第6步: 分子过滤
应用药物化学过滤器移除不良分子：
- Lipinski规则过滤器
- PAINS过滤器
- 基本结构有效性检查

### 第7步: 多目标选择
使用NSGA-II算法进行多目标优化选择，平衡：
- 对接分数优化 (exploitation)
- 化学多样性维持 (exploration)

## 高级用法

### 自定义分解策略
调整 `--n_fragments_to_mask` 参数来控制GPT生成的多样性：
- `n=1`: 保守生成，较小变化
- `n=2`: 中等变化
- `n=3`: 更大的结构变化

### 调整进化策略
- 增加 `--crossover_attempts` 和 `--mutation_attempts` 可以产生更多子代
- 调整 `--population_size` 来平衡计算成本和多样性
- 修改 `--max_generations` 来控制进化时间

### 对接参数优化
根据目标蛋白调整对接参数：
```bash
# 例：针对特定结合位点的对接
--center_x 25.5 --center_y -10.2 --center_z 8.7 \
--size_x 15.0 --size_y 15.0 --size_z 15.0
```

## 故障排除

### 常见问题

1. **模块导入失败**
   - 确保所有必需的Python包都已安装
   - 检查项目路径设置是否正确

2. **对接失败**
   - 验证AutoDock Vina和MGLTools的安装
   - 检查受体PDB文件格式
   - 确认对接参数设置合理

3. **GPT生成失败**
   - 确认GPU可用性和CUDA环境
   - 检查Fragment-GPT模型文件是否存在
   - 验证片段掩码文件格式

4. **内存不足**
   - 减少种群大小 (`--population_size`)
   - 减少处理器数量 (`--num_processors`)
   - 清理临时文件

### 性能优化

1. **并行化**
   - 增加 `--num_processors` 来加速对接计算
   - 使用更强大的GPU来加速GPT生成

2. **存储优化**
   - 定期清理临时文件
   - 压缩中间结果文件

3. **计算资源管理**
   - 监控GPU内存使用
   - 适当调整批处理大小

## 引用

如果您在研究中使用了此工作流程，请引用：

```
FragGPT-GA: A Hybrid Fragment-based GPT and Genetic Algorithm Approach for 
De Novo Molecular Generation and Optimization
```

## 许可证

本项目遵循相应的开源许可证。详情请查看项目根目录的LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 在使用前请确保已正确安装所有依赖并配置好计算环境。建议先在小规模数据上测试工作流程。 