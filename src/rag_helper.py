"""RAG辅助模块

本模块实现了检索增强生成（RAG）相关功能，主要用于：
1. Web攻击特征模式匹配
2. 响应文本智能分析
3. 结果建议生成
4. Prompt动态增强

主要功能：
- 基于预定义模式的响应分析
- 智能结果建议
- 动态Prompt增强
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Union

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RagHelper:
    """RAG辅助工具类
    
    该类提供了一系列基于模式匹配和知识检索的辅助功能，用于：
    - 分析HTTP响应中的攻击特征
    - 生成结果建议
    - 增强LLM的输入prompt
    
    特点：
    1. 支持多种攻击类型的模式匹配
    2. 可配置的模式规则
    3. 实时的模式编译和缓存
    4. 智能的结果建议策略
    
    属性：
        patterns: Dict 原始模式定义
        compiled_patterns: Dict 编译后的正则表达式缓存
    """
    
    def __init__(self, patterns_file: str = "rag_data/waf_patterns.json"):
        """初始化RAG辅助器
        
        加载并编译模式文件，准备用于响应分析。
        
        Args:
            patterns_file: JSON格式的模式文件路径
            
        Raises:
            FileNotFoundError: 当模式文件不存在时
            json.JSONDecodeError: 当模式文件格式不正确时
            
        Note:
            模式文件结构示例：
            {
                "waf_block_patterns": [{
                    "pattern": "Access Denied.*Security",
                    "type": "WAF_BLOCK",
                    "result": "FAILURE"
                }],
                ...
            }
        """
        try:
            with open(patterns_file, 'r', encoding='utf-8') as f:
                self.patterns = json.load(f)
            
            # 编译所有正则表达式并缓存
            self.compiled_patterns = {}
            for category, patterns in self.patterns.items():
                self.compiled_patterns[category] = [
                    {
                        'regex': re.compile(p['pattern'], re.IGNORECASE),
                        'type': p['type'],
                        'result': p['result']
                    }
                    for p in patterns
                ]
            
            logger.info(f"Successfully loaded {sum(len(p) for p in self.patterns.values())} patterns")
            
        except Exception as e:
            logger.error(f"Error loading patterns: {str(e)}")
            raise

    def analyze_response(self, response: str) -> List[Dict[str, str]]:
        """分析响应文本，查找匹配的模式
        
        对响应文本进行全面分析，识别所有匹配的攻击特征模式。
        支持多个模式同时匹配。
        
        Args:
            response: HTTP响应文本
            
        Returns:
            List[Dict[str, str]]: 匹配结果列表，每个结果包含：
                - type: 模式类型（如WAF_BLOCK, SQL_ERROR等）
                - result: 建议结果（SUCCESS/FAILURE/UNKNOWN）
            
        Note:
            1. 使用预编译的正则表达式提高性能
            2. 所有匹配都忽略大小写
            3. 支持多个模式同时匹配
            4. 返回所有匹配的模式，而不是仅返回第一个匹配
        """
        matches = []
        
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern['regex'].search(response):
                    matches.append({
                        'type': pattern['type'],
                        'result': pattern['result']
                    })
        
        return matches

    def get_suggested_result(self, matches: List[Dict[str, str]]) -> Optional[str]:
        """根据匹配结果建议最终结果
        
        基于所有匹配的模式，使用预定义的优先级规则生成最终建议。
        
        Args:
            matches: 匹配结果列表
            
        Returns:
            Optional[str]: 建议的结果（SUCCESS/FAILURE/UNKNOWN）
                         如果没有匹配结果则返回None
            
        Note:
            结果优先级：
            1. SUCCESS（最高优先级）
            2. FAILURE
            3. UNKNOWN（最低优先级）
            
            这是因为：
            - 成功的攻击通常有明确的特征
            - 失败的攻击可能有多种表现形式
            - UNKNOWN用作默认的降级选项
        """
        if not matches:
            return None
        
        # 结果优先级：SUCCESS > FAILURE > UNKNOWN
        if any(m['result'] == 'SUCCESS' for m in matches):
            return 'SUCCESS'
        elif any(m['result'] == 'FAILURE' for m in matches):
            return 'FAILURE'
        elif any(m['result'] == 'UNKNOWN' for m in matches):
            return 'UNKNOWN'
        
        return None

    def enhance_prompt_with_matches(self, prompt: str, matches: List[Dict[str, str]]) -> str:
        """使用匹配结果增强prompt
        
        将识别到的攻击特征信息注入到原始prompt中，
        帮助LLM做出更准确的判断。
        
        Args:
            prompt: 原始prompt文本
            matches: 匹配结果列表
            
        Returns:
            str: 增强后的prompt
            
        Note:
            1. 保持原始prompt的结构
            2. 在适当位置插入匹配信息
            3. 使用清晰的格式展示匹配结果
            4. 如果没有匹配结果，返回原始prompt
        """
        if not matches:
            return prompt
        
        # 在prompt中添加匹配到的模式信息
        patterns_info = "\n# Matched Patterns\n"
        for match in matches:
            patterns_info += f"- Found {match['type']} pattern suggesting {match['result']}\n"
        
        # 在Current Task之前插入匹配信息
        task_index = prompt.find("# Current Task")
        if task_index != -1:
            return prompt[:task_index] + patterns_info + prompt[task_index:]
        
        return prompt + patterns_info