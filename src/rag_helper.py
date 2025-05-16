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
import re
from pathlib import Path
from typing import List, Dict, Optional, Union


from src.logger_module import get_logger

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
        # 获取日志记录器
        self.logger = get_logger(__name__)
        self.logger.info("初始化RagHelper")
        self.logger.debug(f"使用模式文件: {patterns_file}")
        
        try:
            with open(patterns_file, 'r', encoding='utf-8') as f:
                self.patterns = json.load(f)
            
            # 编译所有正则表达式并缓存
            self.compiled_patterns = {}
            pattern_count = 0
            
            for category, patterns in self.patterns.items():
                self.logger.debug(f"编译 {category} 类别的 {len(patterns)} 个模式")
                pattern_count += len(patterns)
                
                self.compiled_patterns[category] = [
                    {
                        'regex': re.compile(p['pattern'], re.IGNORECASE),
                        'type': p['type'],
                        'result': p['result']
                    }
                    for p in patterns
                ]
            
            self.logger.info(f"成功加载 {pattern_count} 个模式")
            
        except FileNotFoundError:
            self.logger.error(f"模式文件不存在: {patterns_file}")
            raise
            
        except json.JSONDecodeError as e:
            self.logger.error(f"模式文件格式不正确: {patterns_file}, 错误: {str(e)}")
            raise
            
        except Exception as e:
            self.logger.error(f"加载模式时出错: {str(e)}", exc_info=True)
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
        self.logger.debug(f"分析响应文本 (长度: {len(response)} 字符)")
        matches = []
        
        for category, patterns in self.compiled_patterns.items():
            category_matches = 0
            for pattern in patterns:
                if pattern['regex'].search(response):
                    matches.append({
                        'type': pattern['type'],
                        'result': pattern['result']
                    })
                    category_matches += 1
                    self.logger.debug(f"匹配到 {pattern['type']} 模式，建议结果: {pattern['result']}")
            
            if category_matches > 0:
                self.logger.debug(f"在 {category} 类别中找到 {category_matches} 个匹配")
        
        if matches:
            self.logger.info(f"分析完成，找到 {len(matches)} 个匹配")
        else:
            self.logger.info("分析完成，未找到匹配")
        
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
            self.logger.debug("没有匹配结果，无法提供建议")
            return None
        
        # 统计不同结果的出现次数
        result_counts = {
            'SUCCESS': sum(1 for m in matches if m['result'] == 'SUCCESS'),
            'FAILURE': sum(1 for m in matches if m['result'] == 'FAILURE'),
            'UNKNOWN': sum(1 for m in matches if m['result'] == 'UNKNOWN')
        }
        
        self.logger.debug(f"匹配结果统计: {result_counts}")
        
        # 结果优先级：SUCCESS > FAILURE > UNKNOWN
        if result_counts['SUCCESS'] > 0:
            self.logger.info(f"建议结果: SUCCESS (基于 {result_counts['SUCCESS']} 个匹配)")
            return 'SUCCESS'
        elif result_counts['FAILURE'] > 0:
            self.logger.info(f"建议结果: FAILURE (基于 {result_counts['FAILURE']} 个匹配)")
            return 'FAILURE'
        elif result_counts['UNKNOWN'] > 0:
            self.logger.info(f"建议结果: UNKNOWN (基于 {result_counts['UNKNOWN']} 个匹配)")
            return 'UNKNOWN'
        
        # 正常情况下不会执行到这里
        self.logger.warning("匹配结果列表中没有有效的结果建议")
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
            self.logger.debug("无匹配结果，返回原始prompt")
            return prompt
        
        self.logger.info(f"使用 {len(matches)} 个匹配结果增强prompt")
        
        # 在prompt中添加匹配到的模式信息
        patterns_info = "\n# Matched Patterns\n"
        for match in matches:
            patterns_info += f"- Found {match['type']} pattern suggesting {match['result']}\n"
        
        # 在Current Task之前插入匹配信息
        task_index = prompt.find("# Current Task")
        if task_index != -1:
            self.logger.debug("在'# Current Task'之前插入匹配信息")
            enhanced_prompt = prompt[:task_index] + patterns_info + prompt[task_index:]
        else:
            self.logger.debug("在prompt末尾添加匹配信息")
            enhanced_prompt = prompt + patterns_info
        
        # 记录prompt增强
        self.logger.debug(f"原始prompt长度: {len(prompt)}, 增强后: {len(enhanced_prompt)}")
        
        return enhanced_prompt