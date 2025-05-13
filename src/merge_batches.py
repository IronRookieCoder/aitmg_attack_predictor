"""批次结果合并脚本"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict

import pandas as pd
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_batch_files(directory: str, worker_id: str) -> List[Path]:
    """查找所有批次结果文件
    
    Args:
        directory: 目录路径
        worker_id: 工号
        
    Returns:
        List[Path]: 批次文件路径列表
    """
    base_dir = Path(directory)
    # 查找格式为 {worker_id}_*_batch*.csv 的文件
    return sorted(base_dir.glob(f"{worker_id}_*_batch*.csv"))

def merge_batch_files(batch_files: List[Path], output_path: Path):
    """合并批次结果文件
    
    Args:
        batch_files: 批次文件路径列表
        output_path: 输出文件路径
    """
    all_predictions = []
    
    for batch_file in tqdm(batch_files, desc="Merging batches"):
        try:
            df = pd.read_csv(batch_file)
            all_predictions.append(df)
        except Exception as e:
            logger.error(f"Error reading {batch_file}: {str(e)}")
            continue
    
    if not all_predictions:
        raise ValueError("No valid batch files found")
    
    # 合并所有数据
    merged_df = pd.concat(all_predictions, ignore_index=True)
    
    # 按uuid去重（保留最后一次预测结果）
    merged_df.drop_duplicates(subset=['uuid'], keep='last', inplace=True)
    
    # 保存结果
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Successfully merged {len(batch_files)} batch files into {output_path}")
    logger.info(f"Total predictions: {len(merged_df)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="合并批次结果文件")
    parser.add_argument(
        '--worker-id',
        required=True,
        help='工号'
    )
    parser.add_argument(
        '--input-dir',
        default='.',
        help='批次文件目录'
    )
    parser.add_argument(
        '--output-file',
        help='输出文件路径'
    )
    args = parser.parse_args()
    
    try:
        # 查找批次文件
        batch_files = find_batch_files(args.input_dir, args.worker_id)
        if not batch_files:
            logger.error(f"No batch files found for worker {args.worker_id}")
            return
        
        logger.info(f"Found {len(batch_files)} batch files")
        
        # 如果没有指定输出文件，生成默认文件名
        if not args.output_file:
            base_name = batch_files[0].stem.split('_batch')[0]
            output_path = Path(args.input_dir) / f"{base_name}_merged.csv"
        else:
            output_path = Path(args.output_file)
        
        # 合并文件
        merge_batch_files(batch_files, output_path)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    main()