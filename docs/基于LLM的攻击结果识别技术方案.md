## 基于LLM的攻击结果识别技术方案

**1. 目标概述**

本方案旨在根据提供的HTTP请求（`req`）和响应（`rsp`）数据，利用指定的大语言模型（LLM）池，通过结合评估器-优化器选定的最优策略（可能包含多模型投票），准确、高效地判断攻击尝试的结果是成功（`SUCCESS`）、失败（`FAILURE`）还是未知（`UNKNOWN`），并最终输出符合比赛格式要求的CSV文件。

**2. 核心技术栈**

*   **语言:** Python 3.x
*   **核心库:** `pandas` (数据处理), `requests`/模型官方SDK (LLM调用), `scikit-learn` (评估指标计算), `concurrent.futures` (可选，用于并行LLM调用), `conda-pack` (环境打包)
*   **LLM模型池:** `secgpt7b`, `qwen3-8b`, `qwen7b`, `deepseek-r1-distill-qwen7b`
*   **方法论:** Workflows, 评估器-优化器, 多模型投票, Few-Shot Prompt Engineering, (可选) RAG

**3. Workflows 构建方案**

**Workflow 1: 环境准备与数据加载 (Setup & Data Ingestion)**

1.  **环境初始化:**
    *   基于官方提供的基础Conda环境。
    *   创建 `requirements.txt` 文件，列出所需依赖：`pandas`, `scikit-learn`, `requests` (或特定模型SDK)。
    *   **(如果修改环境)** 在开发环境中安装依赖 (`pip install -r requirements.txt`)。完成后，使用 `conda-pack --name <your_env_name> --output aitmg_2025_{选手工号}.tar.gz` 打包环境。若不修改，则跳过打包。
2.  **数据加载:**
    *   编写Python脚本 (`data_loader.py` 或在主脚本中实现)。
    *   使用 `pandas.read_csv()` 加载少量标注数据 (`labeled_demo_data_aitmg_202504.csv`) 到 DataFrame `df_labeled`。此数据用于：
        *   Few-Shot Prompt示例选取。
        *   评估器-优化器流程。
    *   使用 `pandas.read_csv()` 加载大量未标注数据 (`unlabeled_demo_data_aitmg_202504.csv`) 到 DataFrame `df_unlabeled`。此数据用于最终推理。
3.  **常量定义:**
    *   定义模型名称列表: `MODEL_POOL = ['secgpt7b', 'qwen3-8b', 'qwen7b', 'deepseek-r1-distill-qwen7b']`
    *   定义输出标签: `LABELS = ["SUCCESS", "FAILURE", "UNKNOWN"]`
    *   定义选手工号和时间格式。

**Workflow 2: LLM接口封装与Prompt模板设计**

1.  **LLM 统一接口:**
    *   创建 `llm_interface.py` 或类似模块。
    *   定义一个核心函数 `call_llm(model_name: str, prompt: str, max_retries: int = 3, timeout: int = 120) -> str:`
        *   根据 `model_name` 选择调用相应的API或SDK。
        *   实现请求发送逻辑。
        *   包含错误处理（如API错误、超时）和重试机制。
        *   返回LLM的原始文本输出。如果多次重试失败，返回特定错误标识（如 `LLM_ERROR`）。
2.  **Prompt模板设计:**
    *   创建 `prompt_templates.py` 或将模板存储在文本文件中。
    *   设计 **多个** 候选Prompt模板（`prompt_template_v1`, `prompt_template_v2`, ...）。
    *   **模板结构 (示例 - 需优化):**
        ```python
        PROMPT_TEMPLATE_V1 = """
        # Role
        You are a cybersecurity analyst. Your task is to classify the outcome of a potential web attack based on the HTTP request and response.
        
        # Task Description
        Analyze the following Request and Response. Determine if the implied attack attempt resulted in SUCCESS, FAILURE, or if the outcome is UNKNOWN.
        
        # Output Format
        Respond ONLY with one word: SUCCESS, FAILURE, or UNKNOWN.
        
        # Definitions & Guidelines
        - SUCCESS: Attack likely achieved its goal (e.g., data leak in response, command output visible, SQL errors confirming injection, WAF bypassed AND malicious action confirmed).
        - FAILURE: Attack blocked or ineffective (e.g., WAF block page, 403 Forbidden, input validation error without sensitive info leak, expected vulnerable behavior not observed, successful login required but failed).
        - UNKNOWN: Insufficient evidence in req/rsp pair to determine outcome (e.g., generic errors, ambiguous responses, request is not clearly an attack). Focus on clear indicators.
        
        # Examples
        {few_shot_examples}
        
        # Current Task
        Request:
        \"\"\"
        {req_data}
        \"\"\"
        
        Response:
        \"\"\"
        {rsp_data}
        \"\"\"
        
        Outcome:"""
        ```
    *   **Few-Shot示例生成:** 编写函数从 `df_labeled` 中选取3-5个多样化、高质量的示例，并格式化为字符串 `few_shot_examples_str`，用于填充模板。
    *   **数据填充函数:** `format_prompt(template: str, req: str, rsp: str, few_shot_examples: str) -> str:` 用于将实际数据填入模板。注意处理 `req`/`rsp` 中的特殊字符（如引号、换行符）。

**Workflow 3: 评估器-优化器 (Evaluator-Optimizer - 开发阶段)**

1.  **执行脚本:** `evaluate_strategies.py`
2.  **输入:** `df_labeled`, `MODEL_POOL`, 候选Prompt模板列表, (可选) RAG组件。
3.  **策略定义:**
    *   定义一系列待评估的策略，例如：
        *   `{'name': 'SecGPT_Prompt1', 'type': 'single', 'model': 'secgpt7b', 'prompt_template': PROMPT_TEMPLATE_V1}`
        *   `{'name': 'Qwen3_Prompt1', 'type': 'single', 'model': 'qwen3-8b', 'prompt_template': PROMPT_TEMPLATE_V1}`
        *   `{'name': 'SecGPT_Prompt2', 'type': 'single', 'model': 'secgpt7b', 'prompt_template': PROMPT_TEMPLATE_V2}`
        *   `{'name': 'Vote_SQD_P1_Majority', 'type': 'vote', 'models': ['secgpt7b', 'qwen3-8b', 'deepseek-r1-distill-qwen7b'], 'prompt_template': PROMPT_TEMPLATE_V1, 'vote_method': 'majority'}`
        *   `{'name': 'Vote_SQ_P1_Weighted', 'type': 'vote', 'models': ['secgpt7b', 'qwen3-8b'], 'prompt_template': PROMPT_TEMPLATE_V1, 'vote_method': 'weighted', 'weights': {'secgpt7b': 0.6, 'qwen3-8b': 0.4}}`
4.  **评估循环:**
    *   对每个策略：
        *   遍历 `df_labeled` 中的每一行 (`uuid`, `req`, `rsp`, `label`)。
        *   格式化Prompt。
        *   **执行预测:**
            *   **单模型策略:** 调用 `call_llm()` 一次。
            *   **投票策略:** 调用 `call_llm()` 多次（可使用 `concurrent.futures.ThreadPoolExecutor` 加速），收集各模型结果。
        *   **解析结果:** 从LLM输出中提取预测标签 (`SUCCESS`, `FAILURE`, `UNKNOWN`)。处理 `LLM_ERROR` 或非标准输出（计为 `UNKNOWN` 或特定错误类别）。
        *   **应用投票 (如果需要):** 实现 `majority_vote` 和 `weighted_vote` 函数，处理平票（如默认 `UNKNOWN` 或优先级）。
        *   存储该策略对所有标注样本的预测结果 `y_pred`。
5.  **计算指标:**
    *   对每个策略的 `y_pred` 和真实的 `y_true` (来自 `df_labeled['label']`)：
        *   使用 `sklearn.metrics.accuracy_score` 计算准确率。
        *   使用 `sklearn.metrics.f1_score(average='macro')` 计算宏F1分数（对所有类别同等重要）。
        *   使用 `sklearn.metrics.f1_score(average='weighted')` 计算加权F1分数（考虑类别样本数）。
        *   使用 `sklearn.metrics.classification_report` 获取详细报告（Precision, Recall, F1 per class）。
        *   使用 `sklearn.metrics.confusion_matrix` 生成混淆矩阵。
6.  **优化决策:**
    *   比较所有策略的评估指标。
    *   **选择最优策略:** 基于F1分数（特别是宏F1）和准确率，同时考虑混淆矩阵反映的弱点，选择表现最佳且最鲁棒的策略。
    *   记录选定的 `best_prompt_template` 和 `best_strategy_config` (包含模型名称/列表、投票方法等)。
7.  **输出:** 将最优策略配置保存到文件（如 `config.json`）或直接硬编码到主推理脚本中。

**Workflow 4: 最终推理执行 (Inference Execution)**

1.  **主执行脚本:** `main.py` (或由 `run.sh` 调用的脚本)。
2.  **输入:** `df_unlabeled`, `best_prompt_template`, `best_strategy_config` (来自Workflow 3)。
3.  **初始化:** 加载最优策略配置。准备Few-Shot示例字符串（基于 `df_labeled`）。
4.  **推理循环:**
    *   遍历 `df_unlabeled` 中的每一行 (`uuid`, `req`, `rsp`)。
    *   **格式化Prompt:** 使用 `best_prompt_template` 和当前行的 `req`, `rsp`, 以及准备好的Few-Shot示例字符串。
    *   **执行推理 (根据最优策略):**
        *   **单模型:** 调用 `call_llm(best_strategy_config['model'], formatted_prompt)`。
        *   **多模型投票:**
            *   使用 `concurrent.futures.ThreadPoolExecutor` (推荐) 或串行方式，为 `best_strategy_config['models']` 中的每个模型调用 `call_llm(model_name, formatted_prompt)`。
            *   收集结果列表 `model_predictions`。
    *   **解析与投票:**
        *   **解析:** 对单模型结果或 `model_predictions` 列表中的每个结果，调用一个 `parse_llm_output(output: str) -> str` 函数，该函数负责：
            *   清理输出文本（去除多余空格、换行）。
            *   严格匹配 `LABELS`。
            *   处理 `LLM_ERROR` 或无法识别的输出（返回 `UNKNOWN`）。
        *   **投票 (如果需要):** 将解析后的预测列表传入对应的投票函数 (`majority_vote` 或 `weighted_vote`)，得到最终预测 `final_prediction`。
    *   **存储结果:** 将 (`uuid`, `final_prediction`) 添加到结果列表 `results_list`。
    *   **(可选) 进度显示:** 每处理N条记录打印一次进度。
5.  **输出:** `results_list` (包含所有未标注数据的预测结果)。

**Workflow 5: 结果整合与输出**

1.  **执行:** 在 `main.py` 的末尾或一个单独的输出函数中。
2.  **输入:** `results_list`。
3.  **数据整合:** `results_df = pd.DataFrame(results_list, columns=['uuid', 'predict'])`。
4.  **文件名生成:**
    *   获取选手工号 (e.g., from `config.json` or environment variable)。
    *   获取当前时间 `now = datetime.datetime.now()`。
    *   格式化时间 `timestamp = now.strftime('%Y%m%d%H%M')`。
    *   构建文件名 `output_filename = f"{worker_id}_{timestamp}.csv"`。
5.  **保存文件:**
    *   `results_df.to_csv(output_filename, index=False, encoding='utf-8')`。
    *   确保文件保存在 `run.sh` 脚本执行的当前目录下。

**Workflow 6: 打包与提交**

1.  **文件结构:**
    ```
    submission_package/
    ├── run.sh                  # Main execution script
    ├── return_result.sh        # Result transfer script
    ├── main.py                 # Main Python script (Workflows 4, 5)
    ├── llm_interface.py        # LLM API call wrapper
    ├── prompt_templates.py     # Prompt templates and formatting
    ├── evaluate_strategies.py  # Evaluator-Optimizer script (Workflow 3)
    ├── data_loader.py          # (Optional) Data loading logic
    ├── config.json             # (Optional) Stores best strategy config
    ├── requirements.txt        # Python dependencies
    ├── aitmg_2025_{worker_id}.tar.gz # (Optional) Packed Conda env
    ├── README.md               # (Optional) Documentation
    └── (Optional) rag_data/    # Directory for RAG external data
    ```
2.  **`run.sh` 示例:**
    ```bash
    #!/bin/bash
    WORKER_ID="12345" # Replace with your actual ID
    echo "Setting up environment..."
    # Activate conda environment (adjust path/name as needed)
    # Example: source /opt/conda/bin/activate your_env_name
    # If using packed env:
    # tar -xzf aitmg_2025_${WORKER_ID}.tar.gz -C /tmp/ # Unpack somewhere
    # source /tmp/aitmg_2025_${WORKER_ID}/bin/activate
    
    # (Optional) Run evaluation if needed, though typically done offline
    # echo "Running evaluation (if configured)..."
    # python evaluate_strategies.py --output-config config.json
    
    echo "Starting main inference process..."
    # Ensure demo data is accessible, e.g., at /home/jovyan/demo_data
    python main.py \
        --labeled-data /home/jovyan/demo_data/labeled_demo_data_aitmg_202504.csv \
        --unlabeled-data /home/jovyan/demo_data/unlabeled_demo_data_aitmg_202504.csv \
        --output-dir . \
        --worker-id ${WORKER_ID} \
        # --config config.json # (Optional) Load config from file
    
    # The python script will generate {WORKER_ID}_{timestamp}.csv in '.'
    echo "Inference finished. Checking output..."
    OUTPUT_FILE=$(ls -t ${WORKER_ID}_*.csv | head -n 1)
    if [ -f "$OUTPUT_FILE" ] && [ -s "$OUTPUT_FILE" ]; then
        echo "Output file '$OUTPUT_FILE' created successfully."
        exit 0
    else
        echo "Error: Output file not found or empty!"
        exit 1
    fi
    ```
3.  **`return_result.sh` 示例:** (与之前版本类似，确保能找到并拷贝最新生成的CSV)
    ```bash
    #!/bin/bash
    CONTESTANT_ID="12345" # Replace or get dynamically
    TARGET_DIR="/path/to/your/personal/submission/directory" # TARGET PATH
    RESULT_FILE=$(ls -t ${CONTESTANT_ID}_*.csv | head -n 1)
    if [ -z "$RESULT_FILE" ]; then
        echo "Error: No result file found for ${CONTESTANT_ID}." && exit 1
    fi
    echo "Copying '$RESULT_FILE' to '$TARGET_DIR'..."
    cp "$RESULT_FILE" "$TARGET_DIR/"
    if [ $? -eq 0 ]; then
        echo "Copy successful." && exit 0
    else
        echo "Error: Copy failed." && exit 1
    fi
    ```
4.  **打包:** 将 `submission_package/` 目录下的所有必需和可选文件打包成 `{选手工号}_{提交时间}_AI_TMG_202505.zip` 或 `.tar.gz`。
5.  **测试:** 在提交前，务必在本地（如果环境类似）或提供的开发环境中，使用 `sh run.sh` 完整运行一遍，确保能成功处理 `demo_data` 并生成格式正确的输出 CSV 文件。
6.  **提交:** 使用官方提供的 `submit.sh` 命令提交打包好的文件。

**7. (可选) RAG 增强集成**

*   **数据准备:** 收集或创建简单的知识库（文本文件、CSV、或向量数据库索引），包含：
    *   常见WAF拦截页面HTML片段或关键词。
    *   已知SQL注入成功/失败的响应模式。
    *   敏感信息关键词或Regex模式（如 `password`, `ssn`, `credit card`）。
    *   特定漏洞利用成功的指示器。
*   **检索实现:** 在 `main.py` 或单独模块中：
    *   对于每个 `req`/`rsp` 对，执行简单的关键词匹配或（如果使用向量库）相似性搜索。
    *   提取最相关的1-3条知识片段。
*   **Prompt注入:** 修改 `format_prompt` 函数，在Prompt中添加一个 `# Relevant Context from Knowledge Base` 部分，并将检索到的信息插入其中。
*   **评估:** 在 Workflow 3 中，评估带 RAG 和不带 RAG 的策略，看是否有提升，以及是否满足 "LLM 作用显著" 的要求。

**8. 风险与缓解**

*   **LLM幻觉/不遵循指令:** 强化Prompt中的输出格式约束，增加解析逻辑的健壮性，默认`UNKNOWN`。
*   **小样本评估偏差:** 承认30条样本的局限性，评估结果作为参考，可结合少量未标注数据的试运行进行定性判断。
*   **性能瓶颈:**
    *   **LLM调用:** 使用并行调用 (`ThreadPoolExecutor`)。
    *   **数据处理:** `pandas` 通常足够快，避免行级迭代，使用向量化操作。
    *   **整体时间:** 如果多模型投票太慢，考虑在评估阶段确定是否真的需要投票，或减少投票模型数量，或使用更快的基线模型（如果允许）。
*   **API稳定性:** 实现健壮的重试和错误处理逻辑。