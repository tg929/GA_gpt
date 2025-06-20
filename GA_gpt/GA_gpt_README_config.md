# GA-GPT混合工作流配置文件说明

## 配置文件概览

本项目提供两个独立的配置文件，分别用于不同的工作流：

### 1. `config_example.json` - 纯GA工作流配置
- **用途**: 传统的遗传算法分子优化工作流
- **特点**: 仅使用交叉、突变、选择等GA操作
- **适用场景**: 基于现有分子库的优化

### 2. `config_GA_gpt.json` - GA-GPT混合工作流配置 ⭐️
- **用途**: 结合GPT生成和GA优化的混合工作流
- **特点**: 在每代中加入GPT生成的新分子，增强化学多样性
- **适用场景**: 需要探索更大化学空间的分子发现

## 主要配置差异

| 配置项 | 纯GA配置 | GA-GPT混合配置 | 说明 |
|--------|----------|----------------|------|
| `workflow_type` | - | `"GA_GPT_hybrid"` | 工作流类型标识 |
| `gpt` 模块 | ✅ (基础配置) | ✅ (增强配置) | GPT生成参数配置 |
| `selection_mode` | `"multi_objective"` | `"single_objective"` | 默认选择策略差异 |
| `n_select` | 50 | 115 | 选择的分子数量 |
| 其他GA参数 | ✅ | ✅ | **完全一致** |

**重要更新**: 除了GPT相关配置和默认选择模式外，两个配置文件的其他参数现已完全保持一致，包括：
- 对接参数 (`docking`)
- 受体配置 (`receptors`) 
- 交叉参数 (`crossover_finetune`)
- 突变参数 (`mutation_finetune`)
- 过滤参数 (`filter`)
- 工作流基础参数 (`workflow`)

## 使用方法

### 运行GA-GPT混合工作流（推荐）
```bash
# 使用默认配置
python GA_gpt_main.py

# 指定受体
python GA_gpt_main.py --receptor 3pbl

# 指定输出目录
python GA_gpt_main.py --output_dir my_results
```

### 运行纯GA工作流
```bash
# 使用纯GA配置
python GA_gpt_main.py --config GA_gpt/config_example.json
```

## 关键参数说明

### GPT生成参数 (`gpt` 配置块)
```json
{
  "gpt": {
    "n_fragments_to_mask": 1,        // 要掩码的片段数量
    "seed": 42,                      // 随机种子
    "device": "1",                   // GPU设备ID
    "temperature": 1.0               // 生成温度
  }
}
```

### 受体配置 (`receptors` 配置块)
```json
{
  "receptors": {
    "default_receptor": {
      "name": "parp1"                // 默认受体名称
    },
    "available_receptors": {         // 可用受体列表
      "parp1": { /* 受体详细配置 */ },
      "3pbl": { /* 受体详细配置 */ }
    }
  }
}
```

### 工作流控制 (`workflow` 配置块)
```json
{
  "workflow": {
    "max_generations": 10,           // 最大进化代数
    "population_size": 115,          // 种群大小
    "enable_gpt_generation": true,   // 是否启用GPT生成
    "gpt_integration_mode": "additive" // GPT集成模式
  }
}
```

## 性能调优建议

1. **GPU内存不足时**: 调整 `gpt.device` 或减小 `performance.batch_size`
2. **加速训练**: 设置 `performance.parallel_processing: true`
3. **节省存储空间**: 设置 `performance.cleanup_intermediate_files: true`
4. **调试模式**: 设置 `logging.log_level: "DEBUG"`

## 注意事项

- GA-GPT混合工作流需要更多计算资源（GPU用于GPT生成）
- 首次运行时会自动创建输出目录结构
- 配置参数变更后会自动保存执行快照到输出目录
- 建议根据具体硬件配置调整 `crossover_attempts` 和 `mutation_attempts` 参数 