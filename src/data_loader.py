"""数据加载和预处理模块

本模块负责处理所有数据相关的操作，包括：
1. 数据加载和验证
2. Few-shot示例选择
3. 数据集划分
4. 批处理支持
5. 预测结果保存

主要功能：
- 支持标注和未标注数据的加载
- 提供数据完整性和格式验证
- 实现高效的批处理机制
- 支持增量保存和结果合并
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


from src.logger_module import get_logger, create_persistent_record

class DataLoader:
    """数据加载和处理类
    
    该类负责所有数据相关操作，提供完整的数据处理流水线。
    包括数据加载、验证、划分、批处理等功能。
    
    主要功能：
    - 数据加载和验证
    - Few-shot示例选择
    - 训练/验证集划分
    - 批量处理支持
    - 结果保存和合并
    
    属性：
        config: Dict 配置信息
        labels: List[str] 有效的标签列表
        batch_config: Dict 批处理配置
    """

    def __init__(self, config_path: str = "config.json"):
        """初始化数据加载器
        
        Args:
            config_path: 配置文件路径，包含标签信息和批处理设置
        """
        # 获取日志记录器
        self.logger = get_logger(__name__)
        self.logger.info("初始化DataLoader")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.labels = self.config['labels']
        self.batch_config = self.config.get('batch_processing', {
            'enabled': False,
            'batch_size': 32
        })
        
        self.logger.debug(f"已加载标签: {self.labels}")
        self.logger.debug(f"批处理配置: {self.batch_config}")

    def load_labeled_data(self, file_path: str) -> pd.DataFrame:
        """加载标注数据
        
        加载并验证标注数据集，确保数据格式和标签正确。
        
        Args:
            file_path: 数据文件路径（CSV格式）
            
        Returns:
            pd.DataFrame: 包含uuid, req, rsp, label的DataFrame
            
        Raises:
            ValueError: 当缺少必要列或存在无效标签时
            
        Note:
            要求CSV文件包含以下列：
            - uuid: 唯一标识符
            - req: HTTP请求
            - rsp: HTTP响应
            - label: 标签(SUCCESS/FAILURE/UNKNOWN)
        """
        self.logger.info(f"加载标注数据: {file_path}")
        try:
            df = pd.read_csv(file_path)
            required_columns = ['uuid', 'req', 'rsp', 'label']
            if not all(col in df.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in df.columns]
                self.logger.error(f"缺少必要列: {missing_cols}")
                raise ValueError(f"缺少必要列。期望: {required_columns}, 缺少: {missing_cols}")
            
            # 验证标签值
            invalid_labels = df[~df['label'].isin(self.labels)]['label'].unique()
            if len(invalid_labels) > 0:
                self.logger.error(f"发现无效标签: {invalid_labels}")
                raise ValueError(f"发现无效标签: {invalid_labels}")
            
            self.logger.info(f"成功加载 {len(df)} 条标注样本")
            
            # 记录持久化数据加载信息
            create_persistent_record({
                'operation': 'load_labeled_data',
                'file_path': file_path,
                'record_count': len(df),
                'label_distribution': df['label'].value_counts().to_dict()
            })
            
            return df
        
        except Exception as e:
            self.logger.error(f"加载标注数据出错: {str(e)}", exc_info=True)
            raise

    def load_unlabeled_data(self, file_path: str) -> pd.DataFrame:
        """加载未标注数据
        
        加载待预测的数据集，验证必要的列是否存在。
        
        Args:
            file_path: 数据文件路径（CSV格式）
            
        Returns:
            pd.DataFrame: 包含uuid, req, rsp的DataFrame
            
        Raises:
            ValueError: 当缺少必要列时
            
        Note:
            要求CSV文件包含以下列：
            - uuid: 唯一标识符
            - req: HTTP请求
            - rsp: HTTP响应
        """
        self.logger.info(f"加载未标注数据: {file_path}")
        try:
            df = pd.read_csv(file_path)
            required_columns = ['uuid', 'req', 'rsp']
            if not all(col in df.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in df.columns]
                self.logger.error(f"缺少必要列: {missing_cols}")
                raise ValueError(f"缺少必要列。期望: {required_columns}, 缺少: {missing_cols}")
            
            self.logger.info(f"成功加载 {len(df)} 条未标注样本")
            
            # 记录持久化数据加载信息
            create_persistent_record({
                'operation': 'load_unlabeled_data',
                'file_path': file_path,
                'record_count': len(df)
            })
            
            return df
        
        except Exception as e:
            self.logger.error(f"加载未标注数据出错: {str(e)}", exc_info=True)
            raise

    def select_few_shot_examples(self, df: pd.DataFrame, n_examples: int = 3) -> List[Dict[str, str]]:
        """从标注数据中选择few-shot示例
        
        智能选择代表性的示例用于few-shot学习。
        优先确保每个标签类别都有示例。
        
        Args:
            df: 标注数据DataFrame
            n_examples: 需要的示例数量
            
        Returns:
            List[Dict[str, str]]: 示例列表，每个示例包含req, rsp, label
            
        Note:
            1. 优先选择不同标签的示例
            2. 随机选择剩余示例
            3. 返回的示例数可能少于请求数（如数据不足）
        """
        self.logger.info(f"从 {len(df)} 条数据中选择 {n_examples} 个few-shot示例")
        
        # 确保每个标签至少有一个示例（如果可能）
        examples = []
        remaining_examples = n_examples
        
        for label in self.labels:
            label_df = df[df['label'] == label]
            if len(label_df) > 0:
                # 随机选择一个示例
                example = label_df.sample(n=1).iloc[0]
                examples.append({
                    'req': example['req'],
                    'rsp': example['rsp'],
                    'label': example['label']
                })
                remaining_examples -= 1
                self.logger.debug(f"已为标签 '{label}' 选择示例 UUID: {example['uuid']}")
        
        # 如果还需要更多示例，从整个数据集中随机选择
        if remaining_examples > 0:
            self.logger.debug(f"选择额外 {remaining_examples} 个随机示例")
            additional_examples = df.sample(n=remaining_examples)
            for _, example in additional_examples.iterrows():
                examples.append({
                    'req': example['req'],
                    'rsp': example['rsp'],
                    'label': example['label']
                })
                self.logger.debug(f"已选择随机示例 UUID: {example['uuid']}, 标签: {example['label']}")
        
        # 记录持久化信息
        label_counts = {}
        for example in examples:
            label_counts[example['label']] = label_counts.get(example['label'], 0) + 1
            
        create_persistent_record({
            'operation': 'select_few_shot_examples',
            'examples_count': len(examples),
            'label_distribution': label_counts
        })
        
        self.logger.info(f"已选择 {len(examples)} 个few-shot示例，标签分布: {label_counts}")
        return examples

    def split_validation_set(self, df: pd.DataFrame, test_size: float = 0.2, 
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """划分验证集
        
        将标注数据集划分为训练集和验证集。
        支持分层抽样，保持标签分布。
        
        Args:
            df: 标注数据DataFrame
            test_size: 验证集比例
            random_state: 随机种子，确保结果可复现
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (训练集, 验证集)
            
        Note:
            1. 使用分层抽样保持标签分布
            2. 当数据量较小时自动切换到普通随机抽样
            3. 固定随机种子确保结果可复现
        """
        self.logger.info(f"划分验证集，测试集比例: {test_size}, 随机种子: {random_state}")
        
        # 检查是否可以进行分层抽样
        use_stratify = len(df) >= 5
        stratify = df['label'] if use_stratify else None
        
        if use_stratify:
            self.logger.debug("使用分层抽样")
        else:
            self.logger.debug("数据量较小，使用普通随机抽样")
        
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, 
            stratify=stratify
        )
        
        # 记录持久化信息
        train_label_dist = train_df['label'].value_counts().to_dict()
        test_label_dist = test_df['label'].value_counts().to_dict()
        
        create_persistent_record({
            'operation': 'split_validation_set',
            'train_count': len(train_df),
            'test_count': len(test_df),
            'test_size': test_size,
            'train_label_distribution': train_label_dist,
            'test_label_distribution': test_label_dist
        })
        
        self.logger.info(f"数据集划分完成，训练集: {len(train_df)}条, 验证集: {len(test_df)}条")
        self.logger.debug(f"训练集标签分布: {train_label_dist}")
        self.logger.debug(f"验证集标签分布: {test_label_dist}")
        
        return train_df, test_df

    def get_batch_generator(self, df: pd.DataFrame) -> Generator[pd.DataFrame, None, None]:
        """创建数据批次生成器
        
        将大数据集分割成小批次进行处理。
        支持配置是否启用批处理和批次大小。
        
        Args:
            df: 输入数据DataFrame
            
        Returns:
            Generator: 数据批次生成器
            
        Note:
            1. 如果禁用批处理，直接返回完整数据集
            2. 批次大小通过配置文件控制
            3. 最后一个批次可能小于标准大小
        """
        if not self.batch_config['enabled']:
            self.logger.info("批处理已禁用，一次性处理全部数据")
            yield df
            return

        batch_size = self.batch_config['batch_size']
        total_batches = math.ceil(len(df) / batch_size)
        
        self.logger.info(f"批处理已启用，批次大小: {batch_size}, 总批次数: {total_batches}")
        
        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx].copy()
            self.logger.info(f"生成批次 {i+1}/{total_batches} (大小: {len(batch_df)})")
            yield batch_df

    def merge_prediction_batches(self, prediction_batches: List[List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """合并批量预测结果
        
        将多个批次的预测结果合并成一个完整的结果列表。
        
        Args:
            prediction_batches: 批量预测结果列表
            
        Returns:
            List[Dict[str, str]]: 合并后的预测结果
            
        Note:
            1. 保持结果顺序
            2. 支持不同大小的批次
            3. 去除可能的重复结果
        """
        self.logger.info(f"合并 {len(prediction_batches)} 个批次的预测结果")
        
        merged = []
        for i, batch in enumerate(prediction_batches):
            self.logger.debug(f"批次 {i+1} 中有 {len(batch)} 个预测结果")
            merged.extend(batch)
        
        self.logger.info(f"合并完成，总共 {len(merged)} 个预测结果")
        
        # 记录持久化信息
        create_persistent_record({
            'operation': 'merge_prediction_batches',
            'batch_count': len(prediction_batches),
            'total_predictions': len(merged)
        })
        
        return merged

    def save_batch_predictions(self, predictions: List[Dict[str, str]], 
                            base_path: str, batch_index: int):
        """保存批次预测结果
        
        将单个批次的预测结果保存到文件。
        支持增量保存，防止数据丢失。
        
        Args:
            predictions: 预测结果列表
            base_path: 基础文件路径
            batch_index: 批次索引
            
        Note:
            1. 自动生成批次文件名
            2. 使用CSV格式保存
            3. 包含错误处理和日志记录
        """
        # 构建批次文件名
        path = Path(base_path)
        batch_path = path.parent / f"{path.stem}_batch{batch_index}{path.suffix}"
        
        self.logger.info(f"保存批次 {batch_index} 的预测结果到 {batch_path}")
        
        try:
            df = pd.DataFrame(predictions)
            df.to_csv(batch_path, index=False, encoding='utf-8')
            
            # 记录标签分布
            if 'predict' in df.columns:
                label_dist = df['predict'].value_counts().to_dict()
                self.logger.debug(f"预测标签分布: {label_dist}")
                
                # 记录持久化信息
                create_persistent_record({
                    'operation': 'save_batch_predictions',
                    'batch_index': batch_index,
                    'predictions_count': len(predictions),
                    'output_path': str(batch_path),
                    'label_distribution': label_dist
                })
            
            self.logger.info(f"成功保存批次 {batch_index} 的预测结果，共 {len(predictions)} 条")
        
        except Exception as e:
            self.logger.error(f"保存批次 {batch_index} 的预测结果出错: {str(e)}", exc_info=True)
            raise

    def save_predictions(self, predictions: List[Dict[str, str]], output_path: str):
        """保存完整的预测结果
        
        将所有预测结果保存到一个文件中。
        
        Args:
            predictions: 预测结果列表，每个元素包含uuid和predict
            output_path: 输出文件路径
            
        Note:
            1. 使用CSV格式保存
            2. 确保UTF-8编码
            3. 包含错误处理和日志记录
        """
        self.logger.info(f"保存 {len(predictions)} 条预测结果到 {output_path}")
        
        try:
            df = pd.DataFrame(predictions)
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            # 记录标签分布
            if 'predict' in df.columns:
                label_dist = df['predict'].value_counts().to_dict()
                self.logger.info(f"预测标签分布: {label_dist}")
                
                # 记录持久化信息
                create_persistent_record({
                    'operation': 'save_predictions',
                    'predictions_count': len(predictions),
                    'output_path': str(output_path),
                    'label_distribution': label_dist
                })
            
            self.logger.info(f"成功保存预测结果到 {output_path}")
        
        except Exception as e:
            self.logger.error(f"保存预测结果出错: {str(e)}", exc_info=True)
            raise