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
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
from pathlib import Path
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


from src.logger_module import get_logger, create_persistent_record

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
        # 获取日志记录器
        self.logger = get_logger(__name__)
        self.logger.info("初始化StrategyEvaluator")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.labels = self.config['labels']
        self.metrics = {}  # 用于缓存评估结果
        
        self.logger.debug(f"可用标签: {self.labels}")
        self.logger.debug(f"可用模型: {self.config.get('model_pool', [])}")
        
        # 持久化记录初始化信息
        create_persistent_record({
            'operation': 'init_strategy_evaluator',
            'config_path': config_path,
            'labels': self.labels,
            'models': self.config.get('model_pool', [])
        })

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
        self.logger.debug(f"执行多数投票，预测数量: {len(predictions)}")
        
        # 过滤掉LLM_ERROR
        valid_predictions = [
            pred for pred in predictions
            if pred != "LLM_ERROR"
        ]
        
        if not valid_predictions:
            self.logger.debug("所有预测都是LLM_ERROR，返回UNKNOWN")
            return "UNKNOWN"
        
        # 统计各结果的票数
        vote_count = Counter(valid_predictions)
        self.logger.debug(f"投票统计: {dict(vote_count)}")
        
        # 如果有明显的多数，直接返回
        most_common = vote_count.most_common(1)[0]
        if most_common[1] > len(valid_predictions) / 2:
            self.logger.debug(f"明显多数结果: {most_common[0]}, 票数: {most_common[1]}/{len(valid_predictions)}")
            return most_common[0]
        
        # 处理平票情况
        max_votes = most_common[1]
        candidates = [
            result for result, votes in vote_count.items()
            if votes == max_votes
        ]
        
        self.logger.debug(f"平票情况，候选结果: {candidates}")
        
        # 按优先级选择
        for result in ['SUCCESS', 'FAILURE', 'UNKNOWN']:
            if result in candidates:
                self.logger.debug(f"根据优先级选择结果: {result}")
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
        self.logger.debug(f"执行加权投票，预测数量: {len(predictions)}")
        self.logger.debug(f"使用权重: {weights}")
        
        if not predictions:
            self.logger.debug("没有预测，返回UNKNOWN")
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
                self.logger.debug(f"模型 {model_name} 预测 {result}，权重 {weight}")
        
        self.logger.debug(f"加权投票统计: {weighted_votes}")
        
        # 如果没有有效票数，返回UNKNOWN
        if sum(weighted_votes.values()) == 0:
            self.logger.debug("没有有效投票，返回UNKNOWN")
            return 'UNKNOWN'
        
        # 返回得票最高的结果
        result = max(weighted_votes.items(), key=lambda x: x[1])[0]
        self.logger.debug(f"加权投票结果: {result}")
        return result

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
        strategy_name = strategy.get('name', f"{strategy['type']}_{','.join(strategy['models'])}")
        self.logger.info(f"评估策略: {strategy_name}")
        
        # 应用策略获取预测结果
        start_time = datetime.now()
        
        if strategy['type'] == 'majority':
            self.logger.debug("使用多数投票策略")
            pred_labels = [
                self.majority_vote(pred_list)
                for pred_list in predictions
            ]
        elif strategy['type'] == 'weighted':
            self.logger.debug("使用加权投票策略")
            pred_labels = [
                self.weighted_vote(pred_list, strategy['weights'])
                for pred_list in predictions
            ]
        else:
            self.logger.error(f"未知的策略类型: {strategy['type']}")
            raise ValueError(f"未知的策略类型: {strategy['type']}")
        
        # 计算评估指标
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, labels=self.labels, average='macro'
        )
        
        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=self.labels)
        
        # 计算每个类别的指标
        class_metrics = {}
        for i, label in enumerate(self.labels):
            p, r, f, _ = precision_recall_fscore_support(
                true_labels, pred_labels, labels=[label], average='binary'
            )
            class_metrics[label] = {
                'precision': float(p[0]),
                'recall': float(r[0]),
                'f1': float(f[0])
            }
        
        # 记录评估耗时
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        metrics = {
            'accuracy': float(accuracy),
            'macro_precision': float(precision),
            'macro_recall': float(recall),
            'macro_f1': float(f1),
            'class_metrics': class_metrics,
            'confusion_matrix': cm.tolist(),
            'evaluation_time': elapsed_time
        }
        
        self.logger.info(f"策略 {strategy_name} 评估完成")
        self.logger.info(f"准确率: {accuracy:.4f}, F1分数: {f1:.4f}")
        self.logger.debug(f"每类指标: {class_metrics}")
        
        # 持久化记录评估结果
        create_persistent_record({
            'operation': 'evaluate_strategy',
            'strategy': strategy,
            'metrics': metrics,
            'samples_count': len(true_labels),
            'evaluation_time': elapsed_time
        })
        
        return metrics

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
        self.logger.info(f"开始寻找最佳策略, 验证数据大小: {len(validation_data)}")
        
        best_strategy = None
        best_score = 0.0
        true_labels = validation_data['label'].tolist()
        
        # 记录所有标签分布
        label_dist = Counter(true_labels)
        self.logger.info(f"验证集标签分布: {dict(label_dist)}")
        
        # 尝试不同的策略配置
        model_combinations = self._get_model_combinations()
        self.logger.info(f"共生成 {len(model_combinations)} 种模型组合")
        
        strategies = []
        
        # 多数投票策略
        for models in model_combinations:
            strategy_name = f"Majority_{''.join([m[0] for m in models])}"
            strategies.append({
                'name': strategy_name,
                'type': 'majority',
                'models': models
            })
        
        # 如果配置了模型权重，也尝试加权投票
        if 'model_weights' in self.config:
            for models in model_combinations:
                strategy_name = f"Weighted_{''.join([m[0] for m in models])}"
                strategies.append({
                    'name': strategy_name,
                    'type': 'weighted',
                    'models': models,
                    'weights': self.config['model_weights']
                })
        
        self.logger.info(f"共生成 {len(strategies)} 种策略")
        
        # 记录策略搜索开始时间
        search_start_time = datetime.now()
        
        # 评估每个策略
        all_results = []
        for i, strategy in enumerate(strategies):
            self.logger.info(f"评估策略 {i+1}/{len(strategies)}: {strategy['name']}")
            
            metrics = self.evaluate_strategy(
                strategy, true_labels, all_predictions
            )
            
            strategy_result = {
                **strategy,
                'metrics': metrics
            }
            all_results.append(strategy_result)
            
            # 使用F1分数作为主要指标
            if metrics['macro_f1'] > best_score:
                best_score = metrics['macro_f1']
                best_strategy = strategy_result
                self.logger.info(f"发现新的最佳策略: {strategy['name']}, F1分数: {best_score:.4f}")
        
        # 计算策略搜索耗时
        search_time = (datetime.now() - search_start_time).total_seconds()
        
        # 记录搜索结果
        self.logger.info(f"策略搜索完成，耗时: {search_time:.2f}秒")
        self.logger.info(f"最佳策略: {best_strategy['name']}, F1分数: {best_score:.4f}")
        
        # 持久化记录
        create_persistent_record({
            'operation': 'find_best_strategy',
            'validation_samples': len(validation_data),
            'strategies_count': len(strategies),
            'search_time': search_time,
            'best_strategy': best_strategy['name'],
            'best_score': float(best_score),
            'all_results': all_results
        })
        
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
        self.logger.debug("生成模型组合")
        models = self.config['model_pool']
        result = []
        
        # 生成2到len(models)个模型的组合
        for i in range(2, len(models) + 1):
            self.logger.debug(f"生成{i}个模型的组合")
            result.extend(list(combinations(models, i)))
        
        model_combinations = [list(combo) for combo in result]
        self.logger.debug(f"生成了 {len(model_combinations)} 种模型组合")
        
        return model_combinations

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
        self.logger.info(f"保存评估结果到 {output_path}")
        
        # 添加评估时间
        results['evaluation_time'] = datetime.now().isoformat()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"评估结果已保存到 {output_path}")
            
            # 持久化记录
            create_persistent_record({
                'operation': 'save_evaluation_results',
                'output_path': output_path,
                'best_strategy': results.get('name', '未知'),
                'best_score': results.get('metrics', {}).get('macro_f1', 0)
            })
            
        except Exception as e:
            self.logger.error(f"保存评估结果时出错: {str(e)}", exc_info=True)
            
            # 记录错误
            create_persistent_record({
                'operation': 'save_evaluation_results',
                'error': str(e),
                'output_path': output_path
            }, level="error")
            
            raise