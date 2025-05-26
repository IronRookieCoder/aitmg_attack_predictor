# Web 攻击结果识别系统

基于大语言模型的 Web 攻击结果识别系统，支持对 WAF 攻击结果进行自动分类和分析。

## 功能特点

- 支持多个大语言模型的并行调用
- 基于模式匹配的响应分析
- 灵活的投票策略和评估机制
- 批量处理支持
- 增量保存和结果合并

## 系统架构

系统由以下核心模块组成：

1. **LLM 接口模块** (llm_interface.py)

   - 提供统一的模型调用接口
   - 支持错误重试和并行调用
   - 内置多个模型支持

2. **RAG 辅助模块** (rag_helper.py)

   - 基于规则的响应分析
   - 动态 Prompt 增强
   - 结果建议生成

3. **数据加载模块** (data_loader.py)

   - 数据加载和验证
   - Few-shot 示例选择
   - 批量处理支持

4. **评估策略模块** (evaluate_strategies.py)
   - 多种投票策略实现
   - 模型组合评估
   - 性能指标计算

## 安装说明

1. 克隆项目代码：

```bash
git clone [repository_url]
cd aitmg_attack_predictor
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 配置模型访问：
   - 在 config.json 中配置工号和模型池
   - 确保有必要的 API 访问权限

## 使用方法

### 1. 基本使用

```bash
# 单次预测
python main.py --input single.csv --output results.csv

# 批量处理
python main.py --input batch/ --output results/ --batch
```

### 2. 评估模式

```bash
# 运行评估
python evaluate_strategies.py --data validation.csv

# 运行测试
bash run_tests.sh
```

### 3. 结果合并

```bash
# 合并批处理结果
python merge_batches.py --input results/ --output final.csv
```

## 项目结构

```
├── config.json              # 系统配置文件
├── main.py                  # 主程序入口
├── requirements.txt         # 项目依赖
├── return_result.sh         # 结果返回脚本
├── run_tests.sh             # 测试运行脚本
├── run.sh                   # 主程序运行脚本
├── docs/                    # 文档目录
├── rag_data/                # RAG 数据文件
│   └── waf_patterns.json    # WAF 模式匹配规则
├── src/                     # 源代码目录
│   ├── data_loader.py       # 数据加载模块
│   ├── evaluate_strategies.py # 评估策略模块
│   ├── llm_interface.py     # LLM 接口模块
│   ├── merge_batches.py     # 批处理结果合并
│   ├── prompt_templates.py  # 提示词模板
│   └── rag_helper.py        # RAG 辅助模块
└── tests/                   # 测试目录
    ├── __init__.py
    ├── test_batch_processing.py
    ├── test_evaluator.py
    └── test_rag_helper.py
```

## 配置说明

### config.json

```json
{
  "model_pool": [
    "secgpt7b",
    "qwen3-8b",
    "qwen7b",
    "deepseek-r1-distill-qwen7b"
  ],
  "labels": ["SUCCESS", "FAILURE", "UNKNOWN"],
  "batch_processing": {
    "enabled": true,
    "batch_size": 32
  },
  "verification": {
    "allow_override": true,
    "default_verify_model": "secgpt7b"
  },
  "single_model_strategies": {
    "qwen3-8b_review": {
      "name": "SingleModel_qwen3_8b_review",
      "type": "single_model",
      "model_name": "qwen3-8b",
      "enabled": true,
      "prompt_template": "v3",
      "use_review_filter": true,
      "verify_model": "secgpt7b"
    }
  }
}
```

### 单一模型策略验证配置

在单一模型策略中，支持配置其他模型进行验证。配置方法如下：

1. 在 `config.json` 的 `single_model_strategies` 部分，为每个单一模型策略定义：

   - `model_name`: 主要分析模型
   - `use_review_filter`: 设为 `true` 以启用验证
   - `verify_model`: 指定用于验证的模型

2. 验证模型选择逻辑：
   - 如果找到匹配的策略且配置了 `verify_model`，则使用该模型进行验证
   - 如果未找到匹配策略或未配置 `verify_model`，则使用全局默认验证模型（在 `verification.default_verify_model` 中配置）
   - 如果全局默认验证模型也未配置，则使用主分析模型自身进行验证

示例配置：

```json
"single_model_strategies": {
  "secgpt7b_review": {
    "model_name": "secgpt7b",
    "use_review_filter": true,
    "verify_model": "qwen3-8b"
  },
  "qwen3-8b_review": {
    "model_name": "qwen3-8b",
    "use_review_filter": true,
    "verify_model": "secgpt7b"
  },
  "qwen3-8b_self_review": {
    "model_name": "qwen3-8b",
    "use_review_filter": true,
    "verify_model": null   // 使用默认验证模型或自身
  }
}
```

### 单一模型验证示例

项目提供了一个专门用于测试单一模型策略及其验证模型配置的示例脚本：`single_model_verify_example.py`

使用方法：

```bash
python single_model_verify_example.py --model qwen3-8b --verify secgpt7b --req test_req.txt --rsp test_rsp.txt
```

参数说明：

- `--model`: 主分析模型名称
- `--verify`: [可选] 验证模型名称，不指定则使用配置文件中的设置
- `--req`: 测试请求内容或文件路径
- `--rsp`: 测试响应内容或文件路径
- `--config`: [可选] 配置文件路径，默认为`config.json`

### 模式文件 (rag_data/waf_patterns.json)

```json
{
  "waf_block_patterns": [
    {
      "pattern": "Access Denied.*Security",
      "type": "WAF_BLOCK",
      "result": "FAILURE"
    }
  ]
}
```

## 输入输出格式

### 输入 CSV 格式

```csv
uuid,req,rsp
1,"GET /admin","403 Forbidden"
2,"POST /login","Access Denied"
```

### 输出 CSV 格式

```csv
uuid,predict
1,"FAILURE"
2,"FAILURE"
```

## 性能优化

1. **批处理机制**

   - 自动分批处理大数据集
   - 支持增量保存结果
   - 断点续传支持

2. **并行处理**

   - 多模型并行调用
   - 异步 IO 处理
   - 线程池优化

3. **错误处理**
   - 智能重试机制
   - 异常恢复
   - 详细的日志记录

## 开发说明

### 添加新模型

1. 在 `config.json` 中添加模型配置
2. 在 `llm_interface.py` 中实现模型调用方法
3. 更新单元测试

### 自定义评估策略

1. 在 `evaluate_strategies.py` 中添加新的投票策略
2. 实现评估方法
3. 在配置中启用新策略

## 测试覆盖

运行测试套件：

```bash
# 运行所有测试
bash run_tests.sh

# 运行单个测试模块
python -m unittest tests/test_rag_helper.py
```

## 注意事项

1. 确保模型 API 的稳定访问
2. 定期更新模式文件
3. 监控系统资源使用
4. 及时处理错误日志

## 问题排查

### 常见问题

1. 模型调用超时

   - 检查网络连接
   - 确认 API 限流设置
   - 调整超时参数

2. 内存使用过高

   - 减小批处理大小
   - 启用增量保存
   - 清理缓存数据

3. 结果不一致
   - 检查模型权重设置
   - 验证投票策略配置
   - 更新模式文件

### 日志说明

```python
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

系统使用 Python 标准日志模块，支持不同级别的日志记录：

- INFO: 正常操作信息
- WARNING: 需要注意的情况
- ERROR: 错误但可恢复
- CRITICAL: 严重错误
