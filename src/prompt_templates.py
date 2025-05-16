"""Prompt模板管理模块

本模块负责管理和维护所有的prompt模板，提供模板加载、格式化和验证功能。主要特点：
1. 支持多版本prompt模板
2. Few-shot示例动态选择和格式化
3. 模板参数验证
4. 结构化输出格式控制

使用示例:
    templates = PromptTemplates("config.json")
    prompt = templates.format_prompt(
        "v1",
        req="GET /admin",
        rsp="403 Forbidden",
        few_shot_examples=[{'req': '...', 'rsp': '...', 'label': 'FAILURE'}]
    )
"""

import json
from typing import List, Dict, Optional


from src.logger_module import get_logger

class PromptTemplates:
    """Prompt模板管理类
    
    该类负责管理多个版本的prompt模板，提供统一的格式化接口。
    每个模板都经过优化，确保生成结构化且一致的输出。
    
    主要功能：
    - 多版本模板管理
    - Few-shot示例格式化
    - 输出格式规范化
    - 模板参数验证
    
    Attributes:
        TEMPLATE_V1: str 第一版基础模板
        TEMPLATE_V2: str 第二版增强模板
        TEMPLATE_V3: str 第三版结论在前模板
        TEMPLATE_VERIFY: str 用于验证的模板
        config: Dict 配置信息
    """

    TEMPLATE_V1 = """
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

    TEMPLATE_V2 = """
# Role and Task
As a cybersecurity expert, analyze the HTTP request and response pair to determine if an attack attempt was SUCCESS, FAILURE, or UNKNOWN.

# Classification Guidelines
SUCCESS indicators:
- Data leakage in response
- Command execution output visible
- SQL errors revealing injection success
- WAF bypass confirmed with malicious action success

FAILURE indicators:
- WAF block page
- 403/401 errors
- Input validation blocks
- Expected vulnerable behavior absent
- Authentication failure

UNKNOWN when:
- Generic error messages
- Ambiguous responses
- Unclear attack intent
- Insufficient evidence

# Examples
{few_shot_examples}

# Analysis Target
[REQUEST]
{req_data}

[RESPONSE]
{rsp_data}

# Required Output Format
Return exactly one word (SUCCESS/FAILURE/UNKNOWN):"""

    TEMPLATE_V3 = """
# Role
You are a cybersecurity analyst specializing in attack outcome classification.

# Analysis Target
[REQUEST]
{req_data}

[RESPONSE]
{rsp_data}

# Task
First, give your PRELIMINARY CONCLUSION about whether this attack attempt was SUCCESS, FAILURE, or UNKNOWN.

Then, EXPLAIN your reasoning by analyzing:
1. What type of attack is being attempted
2. Whether the response indicates successful exploitation
3. Specific indicators in the request/response that support your conclusion

Finally, provide your FINAL CONCLUSION, which may confirm or revise your preliminary conclusion based on your analysis.

# Classification Criteria
SUCCESS: 
- Contains data leakage evidence (e.g., database dumps, file contents)
- Shows command execution results
- Displays SQL/injection errors revealing successful exploitation
- Contains sensitive information not normally accessible

FAILURE:
- WAF/security block page
- 403/401 Forbidden responses
- Input validation errors without sensitive data leak
- Expected exploitation behavior absent

UNKNOWN:
- Generic error messages without clear outcome indicators
- Ambiguous responses that don't clearly show success or failure
- Request doesn't contain clear attack signatures
- Insufficient evidence to determine outcome

# Examples
{few_shot_examples}

# Output Format
PRELIMINARY CONCLUSION: [SUCCESS/FAILURE/UNKNOWN]

ANALYSIS:
[Your detailed reasoning and evidence analysis]

FINAL CONCLUSION: [SUCCESS/FAILURE/UNKNOWN]"""

    TEMPLATE_VERIFY = """
# Role
You are a security review expert who verifies attack classification decisions.

# Context
An initial assessment has classified an attack attempt as: {initial_classification}

# Evidence
[REQUEST]
{req_data}

[RESPONSE]
{rsp_data}

# Task
Review the evidence and decide if you AGREE or DISAGREE with the initial classification.

# Classification Criteria
SUCCESS indicators:
- Clear data leak in response (database records, sensitive files)
- Visible command execution output
- Error messages revealing successful exploitation
- Access to protected resources

FAILURE indicators:
- WAF/firewall block messages
- 403/401 access denied responses
- Input sanitization errors
- No evidence of successful exploitation

UNKNOWN when:
- Evidence is ambiguous or insufficient
- Generic error messages without clear indicators
- Response doesn't clearly indicate success or failure

# Output Format
VERIFICATION: [AGREE/DISAGREE]

REASONING:
[Brief explanation of why you agree or disagree]

CORRECT CLASSIFICATION: [SUCCESS/FAILURE/UNKNOWN]"""

    def __init__(self, config_path: str = "config.json"):
        """初始化Prompt模板管理器
        
        Args:
            config_path: 配置文件路径，用于加载相关配置
        """
        # 获取日志记录器
        self.logger = get_logger(__name__)
        self.logger.info("初始化PromptTemplates")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
            
        self.logger.debug("Prompt模板管理器初始化完成")
        
        # 记录可用模板版本
        available_templates = ["v1", "v2", "v3", "verify"]
        self.logger.debug(f"可用模板版本: {available_templates}")

    def get_template(self, template_version: str = "v1") -> str:
        """获取指定版本的模板
        
        Args:
            template_version: 模板版本，支持"v1"、"v2"、"v3"和"verify"
            
        Returns:
            str: 模板字符串
            
        Raises:
            ValueError: 当指定了未知的模板版本时
            
        Note:
            - v1: 基础版本，适合大多数场景
            - v2: 增强版本，提供更详细的分类指南
            - v3: 结论在前模式，先给出初步结论，再提供分析，最后确认结论
            - verify: 用于二次验证的模板，检查初始分类是否正确
        """
        self.logger.debug(f"获取模板版本: {template_version}")
        
        if template_version == "v1":
            return self.TEMPLATE_V1
        elif template_version == "v2":
            return self.TEMPLATE_V2
        elif template_version == "v3":
            return self.TEMPLATE_V3
        elif template_version == "verify":
            return self.TEMPLATE_VERIFY
        else:
            self.logger.error(f"请求了未知的模板版本: {template_version}")
            raise ValueError(f"未知的模板版本: {template_version}")

    def format_few_shot_examples(self, examples: List[Dict[str, str]]) -> str:
        """格式化few-shot示例
        
        将示例列表转换为格式化的字符串，用于插入模板中。
        每个示例都包含请求、响应和标签。
        
        Args:
            examples: 示例列表，每个示例是包含req, rsp, label的字典
            
        Returns:
            str: 格式化后的示例字符串
            
        Note:
            输出格式示例:
            Example 1:
            Request:
            \"\"\"
            GET /admin
            \"\"\"
            
            Response:
            \"\"\"
            403 Forbidden
            \"\"\"
            
            Outcome: FAILURE
        """
        self.logger.debug(f"格式化 {len(examples)} 个few-shot示例")
        
        formatted_examples = []
        for i, example in enumerate(examples, 1):
            formatted_example = f"Example {i}:\n"
            formatted_example += f"Request:\n\"\"\"\n{example['req']}\n\"\"\"\n\n"
            formatted_example += f"Response:\n\"\"\"\n{example['rsp']}\n\"\"\"\n\n"
            formatted_example += f"Outcome: {example['label']}\n"
            formatted_examples.append(formatted_example)
            
            self.logger.debug(f"示例 {i} 格式化完成，标签: {example['label']}")
        
        result = "\n".join(formatted_examples)
        self.logger.debug(f"所有示例格式化完成，总字符数: {len(result)}")
        
        return result

    def format_prompt(self, template_version: str, req: str, rsp: str, 
                     few_shot_examples: Optional[List[Dict[str, str]]] = None,
                     initial_classification: Optional[str] = None) -> str:
        """格式化完整的prompt
        
        将所有必要的信息组合成完整的prompt，包括示例、请求和响应数据。
        该方法确保输出的prompt结构清晰，便于模型理解和生成一致的输出。
        
        Args:
            template_version: 模板版本
            req: 请求数据
            rsp: 响应数据
            few_shot_examples: few-shot示例列表（可选）
            initial_classification: 初始分类结果（仅用于verify模板）
            
        Returns:
            str: 格式化后的完整prompt
            
        Note:
            1. 自动处理特殊字符和格式
            2. 确保示例数据和目标数据格式一致
            3. 提供清晰的分隔和标记
        """
        self.logger.info(f"格式化prompt，使用模板版本: {template_version}")
        self.logger.debug(f"请求数据长度: {len(req)}, 响应数据长度: {len(rsp)}")
        
        template = self.get_template(template_version)
        examples_str = ""
        
        if few_shot_examples:
            self.logger.debug(f"使用 {len(few_shot_examples)} 个few-shot示例")
            examples_str = self.format_few_shot_examples(few_shot_examples)
        else:
            self.logger.debug("未提供few-shot示例")
        
        # 根据不同模板类型格式化prompt
        if template_version == "verify" and initial_classification:
            prompt = template.format(
                initial_classification=initial_classification,
                req_data=req,
                rsp_data=rsp
            )
        else:
            prompt = template.format(
                few_shot_examples=examples_str,
                req_data=req,
                rsp_data=rsp
            )
        
        self.logger.debug(f"格式化完成，prompt总长度: {len(prompt)}")
        
        return prompt