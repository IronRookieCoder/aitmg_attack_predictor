"""主执行脚本，实现推理逻辑"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import pandas as pd
from tqdm import tqdm

from src.data_loader import DataLoader
from src.llm_interface import LLMInterface
from src.prompt_templates import PromptTemplates
from src.rag_helper import RagHelper
from src.logger_module import init_logger, get_logger

class AttackPredictor:
    def __init__(self, config_path: str = "config.json"):
        """初始化攻击预测器"""
        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 初始化日志模块
        init_logger(config_path)
        self.logger = get_logger(__name__)
        self.logger.info("AttackPredictor 初始化中...")
        
        self.llm_interface = LLMInterface(config_path)
        self.prompt_templates = PromptTemplates(config_path)
        self.data_loader = DataLoader(config_path)
        self.rag_helper = RagHelper()
        
        self.best_strategy = self.config['best_strategy']
        self.single_model_strategies = self.config.get('single_model_strategies', {})
        self.logger.info(f"加载单一模型策略: {', '.join(self.single_model_strategies.keys()) if self.single_model_strategies else '无'}")
        
        self.batch_config = self.config.get('batch_processing', {
            'enabled': False,
            'batch_size': 32,
            'max_workers': 4,
            'max_concurrent_batches': 2,
            'save_interval': 100
        })
        
        self.logger.info("AttackPredictor 初始化完成")

    def process_single_item(self, row: pd.Series, few_shot_examples: List[Dict[str, str]]) -> Dict[str, str]:
        """处理单个数据项
        
        Args:
            row: 数据行
            few_shot_examples: few-shot示例列表
            
        Returns:
            Dict[str, str]: 预测结果
        """
        try:
            logger = get_logger(f"{__name__}.process_item")
            logger.debug(f"开始处理数据项: {row['uuid']}")
            
            # 使用RAG分析响应
            matches = self.rag_helper.analyze_response(row['rsp'])
            rag_suggestion = self.rag_helper.get_suggested_result(matches)
            
            # 获取当前策略是默认最佳策略还是单一模型策略
            current_strategy = self.best_strategy
            
            # 如果是单一模型策略类型
            if self.best_strategy['type'] == 'single_model':
                model_name = self.best_strategy.get('model_name')
                logger.debug(f"使用单一模型策略: {model_name}")
            
            # 格式化并增强prompt
            prompt_template = current_strategy.get('prompt_template', 'v1')
            prompt = self.prompt_templates.format_prompt(
                prompt_template,
                row['req'],
                row['rsp'],
                few_shot_examples
            )
            enhanced_prompt = self.rag_helper.enhance_prompt_with_matches(prompt, matches)
            
            # 根据策略类型进行预测
            if current_strategy['type'] == 'single_model':
                model_name = current_strategy.get('model_name')
                logger.debug(f"使用单一模型策略: {model_name}")
                prediction = self.llm_interface.call_llm(
                    model_name,
                    enhanced_prompt
                )
            elif current_strategy['type'] == 'single':  # 兼容旧版本
                model_name = current_strategy.get('model')
                logger.debug(f"使用旧版单模型策略: {model_name}")
                prediction = self.llm_interface.call_llm(
                    model_name,
                    enhanced_prompt
                )
            else:  # vote
                logger.debug(f"使用多模型投票策略: {','.join(current_strategy['models'])}")
                model_predictions = self.llm_interface.call_multiple_models(
                    current_strategy['models'],
                    enhanced_prompt
                )
                if current_strategy['vote_method'] == 'majority':
                    prediction = self.majority_vote(list(model_predictions.values()))
                else:  # weighted
                    prediction = self.weighted_vote(
                        model_predictions,
                        current_strategy.get('weights', self.config.get('model_weights', {}))
                    )
            
            # 如果LLM预测失败，使用RAG建议
            if prediction == "LLM_ERROR" and rag_suggestion:
                logger.warning(f"LLM预测失败，使用RAG建议: {rag_suggestion}")
                prediction = rag_suggestion
            elif prediction == "LLM_ERROR":
                logger.warning(f"LLM预测失败，无RAG建议，使用UNKNOWN")
                prediction = "UNKNOWN"
            
            logger.debug(f"数据项处理完成: {row['uuid']}, 预测结果: {prediction}")
            return {
                'uuid': row['uuid'],
                'predict': prediction
            }
            
        except Exception as e:
            logger = get_logger(f"{__name__}.process_item")
            logger.error(f"处理数据项出错: {row['uuid']}, 错误: {str(e)}", exc_info=True)
            
            return {
                'uuid': row['uuid'],
                'predict': 'UNKNOWN'
            }

    def process_batch(self, batch_df: pd.DataFrame, few_shot_examples: List[Dict[str, str]], 
                     batch_index: int = 0) -> List[Dict[str, str]]:
        """处理数据批次
        
        Args:
            batch_df: 批次数据
            few_shot_examples: few-shot示例列表
            batch_index: 批次索引
            
        Returns:
            List[Dict[str, str]]: 批次预测结果
        """
        logger = get_logger(f"{__name__}.batch")
        logger.info(f"开始处理批次 {batch_index}, 数据项数量: {len(batch_df)}")
        
        predictions = []
        max_workers = min(self.batch_config['max_workers'], len(batch_df))
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_row = {
                executor.submit(self.process_single_item, row, few_shot_examples): row
                for _, row in batch_df.iterrows()
            }
            
            for future in tqdm(as_completed(future_to_row), 
                             total=len(future_to_row),
                             desc=f"处理批次 {batch_index}"):
                try:
                    result = future.result()
                    predictions.append(result)
                except Exception as e:
                    row = future_to_row[future]
                    logger.error(f"批次 {batch_index} 中处理行 {row['uuid']} 出错: {str(e)}", 
                                exc_info=True)
                    predictions.append({
                        'uuid': row['uuid'],
                        'predict': 'UNKNOWN'
                    })
        
        # 记录批处理统计信息
        batch_time = time.time() - start_time
        items_per_second = len(batch_df) / batch_time
        logger.info(f"批次 {batch_index} 统计信息:")
        logger.info(f"- 处理时间: {batch_time:.2f}s")
        logger.info(f"- 每秒处理项数: {items_per_second:.2f}")
        logger.info(f"- 成功率: {sum(1 for p in predictions if p['predict'] != 'UNKNOWN')}/{len(predictions)}")
        
        return predictions

    def predict(self, df_unlabeled: pd.DataFrame, few_shot_examples: List[Dict[str, str]], 
               output_path: str) -> List[Dict[str, str]]:
        """处理未标注数据
        
        Args:
            df_unlabeled: 未标注数据DataFrame
            few_shot_examples: few-shot示例列表
            output_path: 输出文件路径
            
        Returns:
            List[Dict[str, str]]: 预测结果列表
        """
        logger = get_logger(f"{__name__}.predict")
        logger.info(f"开始预测处理, 数据项总数: {len(df_unlabeled)}")
        
        all_predictions = []
        batch_generator = self.data_loader.get_batch_generator(df_unlabeled)
        
        for batch_index, batch_df in enumerate(batch_generator):
            start_time = time.time()

            # 处理当前批次
            batch_predictions = self.process_batch(
                batch_df, 
                few_shot_examples,
                batch_index
            )
            all_predictions.extend(batch_predictions)

            # 计算批处理统计信息
            batch_time = time.time() - start_time
            items_per_second = len(batch_df) / batch_time
            logger.info(f"Batch {batch_index} stats:")
            logger.info(f"- Processing time: {batch_time:.2f}s")
            logger.info(f"- Items per second: {items_per_second:.2f}")
            
            # 定期保存中间结果
            if self.batch_config['enabled'] and \
               len(all_predictions) >= self.batch_config['save_interval']:
                intermediate_path = f"{output_path}.intermediate.{batch_index}"
                logger.info(f"保存中间结果到: {intermediate_path}")
                self.data_loader.save_batch_predictions(
                    all_predictions,
                    intermediate_path,
                    batch_index
                )
        
        logger.info(f"预测处理完成, 总预测项数: {len(all_predictions)}")
        return all_predictions

    def majority_vote(self, predictions: List[str]) -> str:
        """多数投票"""
        logger = get_logger(f"{__name__}.vote")
        
        valid_predictions = [p for p in predictions if p != "LLM_ERROR"]
        if not valid_predictions:
            logger.warning("所有模型预测都失败, 返回UNKNOWN")
            return "UNKNOWN"
        
        votes = {}
        for pred in valid_predictions:
            votes[pred] = votes.get(pred, 0) + 1
        
        max_votes = max(votes.values())
        winners = [label for label, count in votes.items() if count == max_votes]
        
        if len(winners) == 1:
            logger.debug(f"多数投票结果: {winners[0]}, 得票: {max_votes}/{len(valid_predictions)}")
            return winners[0]
        else:
            logger.warning(f"多数投票平局: {winners}, 返回UNKNOWN")
            return "UNKNOWN"

    def weighted_vote(self, predictions: Dict[str, str], weights: Dict[str, float]) -> str:
        """加权投票"""
        logger = get_logger(f"{__name__}.vote")
        
        scores = {label: 0.0 for label in self.config['labels']}
        
        for model, pred in predictions.items():
            if pred != "LLM_ERROR" and model in weights:
                scores[pred] += weights[model]
        
        max_score = max(scores.values())
        winners = [label for label, score in scores.items() if score == max_score]
        
        if len(winners) == 1:
            logger.debug(f"加权投票结果: {winners[0]}, 得分: {max_score}")
            return winners[0]
        else:
            logger.warning(f"加权投票平局: {winners}, 返回UNKNOWN")
            return "UNKNOWN"

    def generate_output_filename(self) -> str:
        """生成输出文件名"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        return f"{self.config['worker_id']}_{timestamp}.csv"

def main():
    """主函数"""
    # 初始化日志模块
    init_logger()
    logger = get_logger("main")
    
    parser = argparse.ArgumentParser(description="攻击结果识别主程序")
    parser.add_argument(
        '--labeled-data',
        required=True,
        help='标注数据文件路径'
    )
    parser.add_argument(
        '--unlabeled-data',
        required=True,
        help='未标注数据文件路径'
    )
    parser.add_argument(
        '--output-dir',
        help='输出目录路径，如未指定则使用配置文件中的路径'
    )
    parser.add_argument(
        '--config',
        default='config.json',
        help='配置文件路径'
    )
    args = parser.parse_args()
    
    try:
        # 初始化预测器
        logger.info("初始化攻击预测器...")
        predictor = AttackPredictor(args.config)
        
        # 从配置文件中获取输出路径，如果命令行参数中指定了则优先使用命令行参数
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        output_dir = args.output_dir if args.output_dir else config.get('output', {}).get('path', './result')
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"输出目录: {output_dir}")
        
        # 加载数据
        logger.info("正在加载数据...")
        df_labeled = predictor.data_loader.load_labeled_data(args.labeled_data)
        df_unlabeled = predictor.data_loader.load_unlabeled_data(args.unlabeled_data)
        
        logger.info(f"已加载标注数据: {len(df_labeled)}行")
        logger.info(f"已加载未标注数据: {len(df_unlabeled)}行")
        
        # 获取few-shot示例
        few_shot_examples = predictor.data_loader.select_few_shot_examples(df_labeled)
        logger.info(f"已选择few-shot示例: {len(few_shot_examples)}个")
        
        # 记录运行参数
        logger.info(f"运行参数: labeled={args.labeled_data}, unlabeled={args.unlabeled_data}, config={args.config}")
        logger.info(f"使用策略: {predictor.best_strategy['name']}")
        
        # 进行预测
        logger.info("开始预测...")
        output_path = output_dir / predictor.generate_output_filename()
        predictions = predictor.predict(df_unlabeled, few_shot_examples, str(output_path))
        
        # 保存结果
        predictor.data_loader.save_predictions(predictions, output_path)
        logger.info(f"结果已保存到: {output_path}")
        
        # 统计预测结果
        success_count = sum(1 for p in predictions if p['predict'] != 'UNKNOWN')
        unknown_count = sum(1 for p in predictions if p['predict'] == 'UNKNOWN')
        logger.info(f"预测统计: 总数={len(predictions)}, 成功={success_count}, 未知={unknown_count}")
        
    except Exception as e:
        logger.error(f"主程序执行错误: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    import traceback  # 移到这里以确保它被导入
    main()