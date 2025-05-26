"""单一模型策略验证模型配置测试模块"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_interface import LLMInterface


class TestSingleModelVerify(unittest.TestCase):
    """测试单一模型策略验证模型配置"""

    def setUp(self):
        """设置测试环境"""
        # 创建临时配置文件
        self.temp_config = tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False)
        
        # 配置内容
        config_content = {
            "model_pool": ["secgpt7b", "qwen3-8b", "qwen7b"],
            "verification": {
                "allow_override": True,
                "default_verify_model": "secgpt7b"
            },
            "single_model_strategies": {
                "strategy1": {
                    "name": "SingleModel_strategy1",
                    "type": "single_model",
                    "model_name": "qwen3-8b",
                    "use_review_filter": True,
                    "verify_model": "secgpt7b"
                },
                "strategy2": {
                    "name": "SingleModel_strategy2",
                    "type": "single_model",
                    "model_name": "qwen7b",
                    "use_review_filter": True,
                    "verify_model": "qwen3-8b"
                },
                "strategy3": {
                    "name": "SingleModel_strategy3",
                    "type": "single_model",
                    "model_name": "secgpt7b",
                    "use_review_filter": True,
                    # 没有配置verify_model，应该使用默认验证模型
                }
            }
        }
        
        json.dump(config_content, self.temp_config)
        self.temp_config.flush()
        self.config_path = self.temp_config.name

    def tearDown(self):
        """清理测试环境"""
        # 删除临时文件
        self.temp_config.close()
        os.unlink(self.temp_config.name)

    @patch('src.llm_interface.LLMInterface.call_llm')
    @patch('src.llm_interface.LLMInterface.call_multiple_models')
    def test_single_model_with_verify(self, mock_call_multiple, mock_call_llm):
        """测试使用单一模型且配置了验证模型的情况"""
        # 设置模拟返回值
        mock_call_llm.return_value = "SUCCESS"
        
        # 创建LLMInterface实例
        llm = LLMInterface(self.config_path)
        
        # 调用analyze_attack方法，指定单一模型qwen3-8b
        result = llm.analyze_attack(
            req="测试请求",
            rsp="测试响应",
            models_primary=["qwen3-8b"],
            use_verify=True
        )
        
        # 验证是否调用了call_llm，而不是call_multiple_models
        mock_call_llm.assert_called()
        mock_call_multiple.assert_not_called()
        
        # 验证是否用secgpt7b模型进行验证
        verify_calls = [call for call in mock_call_llm.call_args_list if 'secgpt7b' in call[0]]
        self.assertGreaterEqual(len(verify_calls), 1, "应该使用secgpt7b进行验证")

    @patch('src.llm_interface.LLMInterface.call_llm')
    @patch('src.llm_interface.LLMInterface.call_multiple_models')
    def test_single_model_with_different_verify(self, mock_call_multiple, mock_call_llm):
        """测试使用单一模型配置不同的验证模型"""
        # 设置模拟返回值
        mock_call_llm.return_value = "FAILURE"
        
        # 创建LLMInterface实例
        llm = LLMInterface(self.config_path)
        
        # 调用analyze_attack方法，指定单一模型qwen7b
        result = llm.analyze_attack(
            req="测试请求",
            rsp="测试响应",
            models_primary=["qwen7b"],
            use_verify=True
        )
        
        # 验证是否用qwen3-8b模型进行验证
        verify_calls = [call for call in mock_call_llm.call_args_list if 'qwen3-8b' in call[0]]
        self.assertGreaterEqual(len(verify_calls), 1, "应该使用qwen3-8b进行验证")

    @patch('src.llm_interface.LLMInterface.call_llm')
    @patch('src.llm_interface.LLMInterface.call_multiple_models')
    def test_single_model_default_verify(self, mock_call_multiple, mock_call_llm):
        """测试使用单一模型但未配置验证模型，应使用默认验证模型"""
        # 设置模拟返回值
        mock_call_llm.return_value = "UNKNOWN"
        
        # 创建LLMInterface实例
        llm = LLMInterface(self.config_path)
        
        # 调用analyze_attack方法，指定单一模型secgpt7b
        result = llm.analyze_attack(
            req="测试请求",
            rsp="测试响应",
            models_primary=["secgpt7b"],
            use_verify=True
        )
        
        # 我们期望使用默认验证模型(secgpt7b)
        # 由于验证模型与主模型相同，所以应该有至少两次secgpt7b调用
        secgpt_calls = [call for call in mock_call_llm.call_args_list if 'secgpt7b' in call[0]]
        self.assertGreaterEqual(len(secgpt_calls), 2, "应该至少调用secgpt7b两次")

    @patch('src.llm_interface.LLMInterface.call_llm')
    @patch('src.llm_interface.LLMInterface.call_multiple_models')
    def test_overwrite_verify_model(self, mock_call_multiple, mock_call_llm):
        """测试通过参数覆盖配置的验证模型"""
        # 设置模拟返回值
        mock_call_llm.return_value = "SUCCESS"
        
        # 创建LLMInterface实例
        llm = LLMInterface(self.config_path)
        
        # 调用analyze_attack方法，指定单一模型qwen3-8b，并覆盖验证模型
        result = llm.analyze_attack(
            req="测试请求",
            rsp="测试响应",
            models_primary=["qwen3-8b"],
            use_verify=True,
            model_verify="qwen7b"  # 明确指定验证模型
        )
        
        # 验证是否用qwen7b模型进行验证
        verify_calls = [call for call in mock_call_llm.call_args_list if 'qwen7b' in call[0]]
        self.assertGreaterEqual(len(verify_calls), 1, "应该使用qwen7b进行验证")


if __name__ == "__main__":
    unittest.main()
