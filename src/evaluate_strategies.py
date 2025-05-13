"""评估策略模块

本模块实现了结果评估和策略优化功能，包括：
1. 模型评估和策略选择
2. 投票机制实现
3. 性能评估指标计算
4. 最佳策略优化

主要功能：
- 支持多种投票策略
- 模型组合评估
- 动态策略优化
- 详细的性能报告
"""

import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyEvaluator:
    """策略评估器类
    
    该类负责评估不同的预测策略，找出最佳的模型组合和投票方法。
    支持多种投票策略和评估指标。
    
    主要功能：
    - 多种投票策略实现
    - 模型组合评估
    - 性能指标计算
    - 结果可视化
    
    属性：
        config: Dict 配置信息
        labels: List[str] 有效的标签列表
        metrics: Dict 评估指标缓存
    """
    
    def __init__(self, config_path: str = "config.json"):
        """初始化策略评估器
        
        Args:
            config_path: 配置文件路径，包含可用模型、标签等信息
            
        Note:
            配置文件应包含：
            - model_pool: 可用模型列表
            - labels: 有效标签列表
            - best_strategy: 最佳策略配置（可选）
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.labels = self.config['labels']
        self.metrics = {}  # 用于缓存评估结果

    def majority_vote(self, predictions: List[Dict[str, str]]) -> str:
        """多数投票策略
        
        基于多个模型的预测结果，采用简单多数投票决定最终结果。
        
        Args:
            predictions: 预测结果列表，每个元素为模型名到预测结果的映射
            
        Returns:
            str: 投票结果（SUCCESS/FAILURE/UNKNOWN）
            
        Note:
            1. 忽略LLM_ERROR结果
            2. 如果出现平票，优先选择：SUCCESS > FAILURE > UNKNOWN
            3. 如果所有模型都返回LLM_ERROR，返回UNKNOWN
        """
        # 过滤掉LLM_ERROR
        valid_predictions = [
            pred for pred in predictions
            if pred != "LLM_ERROR"
        ]
        
        if not valid_predictions:
            return "UNKNOWN"
        
        # 统计各结果的票数
        vote_count = Counter(valid_predictions)
        
        # 如果有明显的多数，直接返回
        if vote_count.most_common(1)[0][1] > len(valid_predictions) / 2:
            return vote_count.most_common(1)[0][0]
        
        # 处理平票情况
        max_votes = vote_count.most_common(1)[0][1]
        candidates = [
            result for result, votes in vote_count.items()
            if votes == max_votes
        ]
        
        # 按优先级选择
        for result in ['SUCCESS', 'FAILURE', 'UNKNOWN']:
            if result in candidates:
                return result
        
        return 'UNKNOWN'

    def weighted_vote(self, predictions: List[Dict[str, str]], 
                     weights: Dict[str, float]) -> str:
        """加权投票策略
        
        基于模型的可信度权重进行加权投票。
        
        Args:
            predictions: 预测结果列表
            weights: 模型权重映射
            
        Returns:
            str: 加权投票结果
            
        Note:
            1. 权重应该在[0, 1]范围内
            2. 忽略LLM_ERROR结果
            3. 如果所有模型都返回LLM_ERROR，返回UNKNOWN
        """
        if not predictions:
            return 'UNKNOWN'
        
        # 初始化票数统计
        weighted_votes = {label: 0.0 for label in self.labels}
        
        # 统计加权票数
        for pred in predictions:
            if pred != 'LLM_ERROR':
                model_name = list(pred.keys())[0]
                result = pred[model_name]
                weight = weights.get(model_name, 1.0)
                weighted_votes[result] += weight
        
        # 如果没有有效票数，返回UNKNOWN
        if sum(weighted_votes.values()) == 0:
            return 'UNKNOWN'
        
        # 返回得票最高的结果
        return max(weighted_votes.items(), key=lambda x: x[1])[0]

    def evaluate_strategy(self, strategy: Dict[str, Any], 
                        true_labels: List[str], 
                        predictions: List[Dict[str, str]]) -> Dict[str, float]:
        """评估单个策略的性能
        
        计算策略的准确率、精确率、召回率和F1分数。
        
        Args:
            strategy: 策略配置
            true_labels: 真实标签列表
            predictions: 预测结果列表
            
        Returns:
            Dict[str, float]: 包含各项评估指标的字典
            
        Note:
            1. 支持多分类评估
            2. 使用宏平均计算全局指标
            3. 保存每个类别的详细指标
        """
        # 应用策略获取预测结果
        if strategy['type'] == 'majority':
            pred_labels = [
                self.majority_vote(pred_list)
                for pred_list in predictions
            ]
        elif strategy['type'] == 'weighted':
            pred_labels = [
                self.weighted_vote(pred_list, strategy['weights'])
                for pred_list in predictions
            ]
        else:
            raise ValueError(f"Unknown strategy type: {strategy['type']}")
        
        # 计算评估指标
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, labels=self.labels, average='macro'
        )
        
        # 计算每个类别的指标
        class_metrics = {}
        for i, label in enumerate(self.labels):
            p, r, f, _ = precision_recall_fscore_support(
                true_labels, pred_labels, labels=[label], average='binary'
            )
            class_metrics[label] = {
                'precision': p[0],
                'recall': r[0],
                'f1': f[0]
            }
        
        return {
            'accuracy': accuracy,
            'macro_precision': precision,
            'macro_recall': recall,
            'macro_f1': f1,
            'class_metrics': class_metrics
        }

    def find_best_strategy(self, validation_data: pd.DataFrame, 
                          all_predictions: List[Dict[str, str]]) -> Dict[str, Any]:
        """寻找最佳策略
        
        通过网格搜索找到最佳的模型组合和投票策略。
        
        Args:
            validation_data: 验证数据集
            all_predictions: 所有模型的预测结果
            
        Returns:
            Dict[str, Any]: 最佳策略配置
            
        Note:
            1. 考虑不同的模型组合
            2. 评估不同的投票策略
            3. 基于F1分数选择最佳策略
            4. 保存详细的评估结果
        """
        best_strategy = None
        best_score = 0.0
        true_labels = validation_data['label'].tolist()
        
        # 尝试不同的策略配置
        strategies = [
            {
                'type': 'majority',
                'models': models
            }
            for models in self._get_model_combinations()
        ]
        
        # 如果配置了模型权重，也尝试加权投票
        if 'model_weights' in self.config:
            strategies.extend([
                {
                    'type': 'weighted',
                    'models': models,
                    'weights': self.config['model_weights']
                }
                for models in self._get_model_combinations()
            ])
        
        # 评估每个策略
        for strategy in strategies:
            metrics = self.evaluate_strategy(
                strategy, true_labels, all_predictions
            )
            
            # 使用F1分数作为主要指标
            if metrics['macro_f1'] > best_score:
                best_score = metrics['macro_f1']
                best_strategy = {
                    **strategy,
                    'metrics': metrics
                }
        
        return best_strategy

    def _get_model_combinations(self) -> List[List[str]]:
        """生成模型组合
        
        生成所有可能的模型组合，用于策略搜索。
        
        Returns:
            List[List[str]]: 模型组合列表
            
        Note:
            1. 考虑2个以上的模型组合
            2. 排除性能特别差的模型
            3. 限制组合的最大规模
        """
        models = self.config['model_pool']
        result = []
        
        # 生成2到len(models)个模型的组合
        for i in range(2, len(models) + 1):
            from itertools import combinations
            result.extend(list(combinations(models, i)))
        
        return [list(combo) for combo in result]

    def save_evaluation_results(self, results: Dict[str, Any], 
                              output_path: str = "evaluation_results.json"):
        """保存评估结果
        
        将评估结果保存到文件，包括：
        - 最佳策略配置
        - 详细的评估指标
        - 各个类别的性能
        
        Args:
            results: 评估结果
            output_path: 输出文件路径
            
        Note:
            1. 使用JSON格式保存
            2. 包含时间戳
            3. 保存配置信息
        """
        from datetime import datetime
        
        # 添加评估时间
        results['evaluation_time'] = datetime.now().isoformat()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Evaluation results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")
            raise