"""RAG辅助模块的单元测试

本模块包含了对RAG辅助功能的完整测试用例，包括：
1. 模式加载和编译
2. 响应分析功能
3. 结果建议生成
4. Prompt增强功能
"""

import unittest
from pathlib import Path
import json
import tempfile
from typing import Dict

from src.rag_helper import RagHelper

class TestRagHelper(unittest.TestCase):
    """RAG辅助工具测试类
    
    测试RAG辅助工具的所有核心功能，确保：
    1. 正确加载和处理模式文件
    2. 准确识别响应中的特征
    3. 正确生成结果建议
    4. 正确增强prompt
    """

    @classmethod
    def setUpClass(cls):
        """测试类初始化
        
        创建临时的测试模式文件，包含多种测试场景。
        """
        # 创建临时的测试模式文件
        cls.test_patterns = {
            "waf_block_patterns": [
                {
                    "pattern": "Access Denied.*Security",
                    "type": "WAF_BLOCK",
                    "result": "FAILURE"
                },
                {
                    "pattern": "Forbidden.*WAF",
                    "type": "WAF_BLOCK",
                    "result": "FAILURE"
                }
            ],
            "sql_error_patterns": [
                {
                    "pattern": "MySQL Error.*syntax",
                    "type": "SQL_ERROR",
                    "result": "SUCCESS"
                }
            ],
            "data_leak_patterns": [
                {
                    "pattern": "password:.*[a-f0-9]{32}",
                    "type": "DATA_LEAK",
                    "result": "SUCCESS"
                }
            ]
        }
        
        # 将测试模式保存到临时文件
        cls.temp_dir = tempfile.mkdtemp()
        cls.patterns_file = Path(cls.temp_dir) / "test_patterns.json"
        with open(cls.patterns_file, 'w', encoding='utf-8') as f:
            json.dump(cls.test_patterns, f)

    def setUp(self):
        """每个测试用例的初始化
        
        创建RagHelper实例，准备测试数据。
        """
        self.rag_helper = RagHelper(str(self.patterns_file))
        self.test_prompt = """
# Role
You are a cybersecurity analyst...

# Current Task
Request:
\"\"\"
GET /admin
\"\"\"

Response:
\"\"\"
403 Forbidden
\"\"\"

Outcome:"""

    def test_pattern_loading(self):
        """测试模式加载功能
        
        验证：
        1. 正确加载所有模式
        2. 正确编译正则表达式
        3. 正确处理模式类别
        """
        # 验证所有模式都被加载
        self.assertEqual(
            len(self.rag_helper.patterns),
            len(self.test_patterns)
        )
        
        # 验证所有模式都被正确编译
        for category in self.test_patterns:
            self.assertIn(category, self.rag_helper.compiled_patterns)
            self.assertEqual(
                len(self.rag_helper.compiled_patterns[category]),
                len(self.test_patterns[category])
            )

    def test_response_analysis_waf_block(self):
        """测试WAF拦截响应分析
        
        验证：
        1. 正确识别WAF拦截模式
        2. 返回正确的结果类型
        3. 正确处理大小写
        """
        response = "Access Denied by Security Module"
        matches = self.rag_helper.analyze_response(response)
        
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]['type'], 'WAF_BLOCK')
        self.assertEqual(matches[0]['result'], 'FAILURE')

    def test_response_analysis_sql_error(self):
        """测试SQL错误响应分析
        
        验证：
        1. 正确识别SQL错误模式
        2. 返回正确的结果类型
        """
        response = "MySQL Error 1064: You have an error in your SQL syntax"
        matches = self.rag_helper.analyze_response(response)
        
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]['type'], 'SQL_ERROR')
        self.assertEqual(matches[0]['result'], 'SUCCESS')

    def test_response_analysis_multiple_matches(self):
        """测试多模式匹配
        
        验证：
        1. 能够识别多个匹配模式
        2. 正确返回所有匹配结果
        """
        response = """
        Access Denied by Security Module
        MySQL Error 1064: You have an error in your SQL syntax
        """
        matches = self.rag_helper.analyze_response(response)
        
        self.assertEqual(len(matches), 2)
        results = {m['type'] for m in matches}
        self.assertEqual(results, {'WAF_BLOCK', 'SQL_ERROR'})

    def test_get_suggested_result_priority(self):
        """测试结果建议优先级
        
        验证：
        1. 正确处理优先级（SUCCESS > FAILURE > UNKNOWN）
        2. 在有多个匹配时返回正确的结果
        """
        matches = [
            {'type': 'WAF_BLOCK', 'result': 'FAILURE'},
            {'type': 'SQL_ERROR', 'result': 'SUCCESS'}
        ]
        
        result = self.rag_helper.get_suggested_result(matches)
        self.assertEqual(result, 'SUCCESS')  # SUCCESS应该优先

    def test_get_suggested_result_empty(self):
        """测试空匹配结果处理
        
        验证：
        1. 正确处理空匹配列表
        2. 返回None而不是默认值
        """
        result = self.rag_helper.get_suggested_result([])
        self.assertIsNone(result)

    def test_enhance_prompt(self):
        """测试Prompt增强功能
        
        验证：
        1. 正确注入匹配信息
        2. 保持原始prompt结构
        3. 在正确位置插入信息
        """
        matches = [
            {'type': 'WAF_BLOCK', 'result': 'FAILURE'},
            {'type': 'SQL_ERROR', 'result': 'SUCCESS'}
        ]
        
        enhanced_prompt = self.rag_helper.enhance_prompt_with_matches(
            self.test_prompt, matches
        )
        
        # 验证匹配信息被正确插入
        self.assertIn('# Matched Patterns', enhanced_prompt)
        self.assertIn('WAF_BLOCK', enhanced_prompt)
        self.assertIn('SQL_ERROR', enhanced_prompt)
        
        # 验证原始内容保持不变
        self.assertIn('# Role', enhanced_prompt)
        self.assertIn('# Current Task', enhanced_prompt)

    def test_enhance_prompt_no_matches(self):
        """测试无匹配时的Prompt增强
        
        验证：
        1. 当没有匹配时返回原始prompt
        2. 不添加多余的信息
        """
        enhanced_prompt = self.rag_helper.enhance_prompt_with_matches(
            self.test_prompt, []
        )
        self.assertEqual(enhanced_prompt, self.test_prompt)

if __name__ == '__main__':
    unittest.main()