"""主执行脚本，实现推理逻辑"""

import argparse
import json
import logging
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('attack_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AttackPredictor:
    def __init__(self, config_path: str = "config.json"):
        """初始化攻击预测器"""
        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.llm_interface = LLMInterface(config_path)
        self.prompt_templates = PromptTemplates(config_path)
        self.data_loader = DataLoader(config_path)
        self.rag_helper = RagHelper()
        
        self.best_strategy = self.config['best_strategy']
        self.batch_config = self.config.get('batch_processing', {
            'enabled': False,
            'batch_size': 32,
            'max_workers': 4,
            'max_concurrent_batches': 2,
            'save_interval': 100
        })

    def process_single_item(self, row: pd.Series, few_shot_examples: List[Dict[str, str]]) -> Dict[str, str]:
        """处理单个数据项
        
        Args:
            row: 数据行
            few_shot_examples: few-shot示例列表
            
        Returns:
            Dict[str, str]: 预测结果
        """
        try:
            # 使用RAG分析响应
            matches = self.rag_helper.analyze_response(row['rsp'])
            rag_suggestion = self.rag_helper.get_suggested_result(matches)
            
            # 格式化并增强prompt
            prompt = self.prompt_templates.format_prompt(
                self.best_strategy['prompt_template'],
                row['req'],
                row['rsp'],
                few_shot_examples
            )
            enhanced_prompt = self.rag_helper.enhance_prompt_with_matches(prompt, matches)
            
            # 根据策略类型进行预测
            if self.best_strategy['type'] == 'single':
                prediction = self.llm_interface.call_llm(
                    self.best_strategy['model'],
                    enhanced_prompt
                )
            else:  # vote
                model_predictions = self.llm_interface.call_multiple_models(
                    self.best_strategy['models'],
                    enhanced_prompt
                )
                if self.best_strategy['vote_method'] == 'majority':
                    prediction = self.majority_vote(list(model_predictions.values()))
                else:  # weighted
                    prediction = self.weighted_vote(
                        model_predictions,
                        self.best_strategy['weights']
                    )
            
            # 如果LLM预测失败，使用RAG建议
            if prediction == "LLM_ERROR" and rag_suggestion:
                prediction = rag_suggestion
            elif prediction == "LLM_ERROR":
                prediction = "UNKNOWN"
            
            return {
                'uuid': row['uuid'],
                'predict': prediction
            }
            
        except Exception as e:
            logger.error(f"Error processing row {row['uuid']}: {str(e)}")
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
        predictions = []
        max_workers = min(self.batch_config['max_workers'], len(batch_df))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_row = {
                executor.submit(self.process_single_item, row, few_shot_examples): row
                for _, row in batch_df.iterrows()
            }
            
            for future in tqdm(as_completed(future_to_row), 
                             total=len(future_to_row),
                             desc=f"Processing batch {batch_index}"):
                try:
                    result = future.result()
                    predictions.append(result)
                except Exception as e:
                    row = future_to_row[future]
                    logger.error(f"Error in batch {batch_index}, row {row['uuid']}: {str(e)}")
                    predictions.append({
                        'uuid': row['uuid'],
                        'predict': 'UNKNOWN'
                    })
        
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
                self.data_loader.save_batch_predictions(
                    all_predictions,
                    output_path,
                    batch_index
                )
        
        return all_predictions

    def majority_vote(self, predictions: List[str]) -> str:
        """多数投票"""
        valid_predictions = [p for p in predictions if p != "LLM_ERROR"]
        if not valid_predictions:
            return "UNKNOWN"
        
        votes = {}
        for pred in valid_predictions:
            votes[pred] = votes.get(pred, 0) + 1
        
        max_votes = max(votes.values())
        winners = [label for label, count in votes.items() if count == max_votes]
        
        return winners[0] if len(winners) == 1 else "UNKNOWN"

    def weighted_vote(self, predictions: Dict[str, str], weights: Dict[str, float]) -> str:
        """加权投票"""
        scores = {label: 0.0 for label in self.config['labels']}
        
        for model, pred in predictions.items():
            if pred != "LLM_ERROR" and model in weights:
                scores[pred] += weights[model]
        
        max_score = max(scores.values())
        winners = [label for label, score in scores.items() if score == max_score]
        
        return winners[0] if len(winners) == 1 else "UNKNOWN"

    def generate_output_filename(self) -> str:
        """生成输出文件名"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M')
        return f"{self.config['worker_id']}_{timestamp}.csv"

def main():
    """主函数"""
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
        default='.',
        help='输出目录路径'
    )
    parser.add_argument(
        '--config',
        default='config.json',
        help='配置文件路径'
    )
    args = parser.parse_args()
    
    try:
        # 初始化预测器
        predictor = AttackPredictor(args.config)
        
        # 加载数据
        logger.info("Loading data...")
        df_labeled = predictor.data_loader.load_labeled_data(args.labeled_data)
        df_unlabeled = predictor.data_loader.load_unlabeled_data(args.unlabeled_data)
        
        # 获取few-shot示例
        few_shot_examples = predictor.data_loader.select_few_shot_examples(df_labeled)
        
        # 进行预测
        logger.info("Starting prediction...")
        output_path = Path(args.output_dir) / predictor.generate_output_filename()
        predictions = predictor.predict(df_unlabeled, few_shot_examples, str(output_path))
        
        # 保存结果
        predictor.data_loader.save_predictions(predictions, output_path)
        logger.info(f"Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    main()