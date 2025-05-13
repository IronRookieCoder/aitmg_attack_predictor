"""评估策略模块的单元测试"""

import unittest
import json
import tempfile
from pathlib import Path
from typing import Dict, List

from src.evaluate_strategies import StrategyEvaluator

class TestEvaluator(unittest.TestCase):
    def setUp(self):
        """测试前准备"""
        self.config = {
            "worker_id": "12345",
            "model_pool": ["secgpt7b", "qwen3-8b"],
            "labels": ["SUCCESS", "FAILURE", "UNKNOWN"],
            "best_strategy": {
                "name": "Vote_SQ_P1_Majority",
                "type": "vote",
                "models": ["secgpt7b", "qwen3-8b"],
                "prompt_template": "v1",
                "vote_method": "majority"
            }
        }
        
        # 创建临时配置文件
        with open('test_config.json', 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4)
        
        self.evaluator = StrategyEvaluator('test_config.json')

    def test_majority_vote(self):
        """测试多数投票"""
        # 测试有效的多数投票
        predictions = ['SUCCESS', 'SUCCESS', 'FAILURE']
        result = self.evaluator.majority_vote(predictions)
        self.assertEqual(result, 'SUCCESS')

        # 测试平局
        predictions = ['SUCCESS', 'FAILURE']
        result = self.evaluator.majority_vote(predictions)
        self.assertEqual(result, 'UNKNOWN')

        # 测试包含错误结果
        predictions = ['SUCCESS', 'LLM_ERROR', 'SUCCESS']
        result = self.evaluator.majority_vote(predictions)
        self.assertEqual(result, 'SUCCESS')

        # 测试全部错误
        predictions = ['LLM_ERROR', 'LLM_ERROR']
        result = self.evaluator.majority_vote(predictions)
        self.assertEqual(result, 'UNKNOWN')

    def test_weighted_vote(self):
        """测试加权投票"""
        predictions = {
            'secgpt7b': 'SUCCESS',
            'qwen3-8b': 'FAILURE'
        }
        weights = {
            'secgpt7b': 0.6,
            'qwen3-8b': 0.4
        }
        result = self.evaluator.weighted_vote(predictions, weights)
        self.assertEqual(result, 'SUCCESS')

        # 测试权重相等的情况
        predictions = {
            'secgpt7b': 'SUCCESS',
            'qwen3-8b': 'FAILURE'
        }
        weights = {
            'secgpt7b': 0.5,
            'qwen3-8b': 0.5
        }
        result = self.evaluator.weighted_vote(predictions, weights)
        self.assertEqual(result, 'UNKNOWN')

        # 测试包含错误结果
        predictions = {
            'secgpt7b': 'SUCCESS',
            'qwen3-8b': 'LLM_ERROR'
        }
        weights = {
            'secgpt7b': 0.6,
            'qwen3-8b': 0.4
        }
        result = self.evaluator.weighted_vote(predictions, weights)
        self.assertEqual(result, 'SUCCESS')

    def test_define_strategies(self):
        """测试策略定义"""
        strategies = self.evaluator.define_strategies()
        
        # 验证单模型策略
        single_model_strategies = [s for s in strategies if s.type == 'single']
        self.assertTrue(len(single_model_strategies) > 0)
        
        # 验证投票策略
        vote_strategies = [s for s in strategies if s.type == 'vote']
        self.assertTrue(len(vote_strategies) > 0)
        
        # 验证策略内容
        for strategy in strategies:
            self.assertIn(strategy.prompt_template, ['v1', 'v2'])
            if strategy.type == 'single':
                self.assertIn(strategy.model, self.config['model_pool'])
            else:
                self.assertIn(strategy.vote_method, ['majority', 'weighted'])
                for model in strategy.models:
                    self.assertIn(model, self.config['model_pool'])

    def tearDown(self):
        """测试后清理"""
        # 删除临时配置文件
        Path('test_config.json').unlink(missing_ok=True)

if __name__ == '__main__':
    unittest.main()