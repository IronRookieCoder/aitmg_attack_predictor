"""批处理功能的单元测试"""

import unittest
import tempfile
from pathlib import Path
import pandas as pd
import json
from typing import List, Dict

from src.data_loader import DataLoader
from src.llm_interface import LLMInterface


class TestBatchProcessing(unittest.TestCase):
    def setUp(self):
        """测试前准备"""
        self.test_config = {
            "worker_id": "test_worker",
            "labels": ["SUCCESS", "FAILURE", "UNKNOWN"],
            "batch_processing": {
                "enabled": True,
                "batch_size": 2,
                "max_workers": 2,
                "max_concurrent_batches": 1,
                "save_interval": 4
            }
        }
        
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(self.test_config, f)
            self.config_path = f.name
        
        self.data_loader = DataLoader(self.config_path)

    def create_test_df(self, size: int = 5) -> pd.DataFrame:
        """创建测试数据"""
        return pd.DataFrame({
            'uuid': [f'test_{i}' for i in range(size)],
            'req': [f'request_{i}' for i in range(size)],
            'rsp': [f'response_{i}' for i in range(size)]
        })

    def test_batch_generator(self):
        """测试批次生成器"""
        df = self.create_test_df(5)
        batches = list(self.data_loader.get_batch_generator(df))
        
        # 验证批次数量
        self.assertEqual(len(batches), 3)  # 5条数据，每批2条，应该生成3批
        
        # 验证前两批的大小
        self.assertEqual(len(batches[0]), 2)
        self.assertEqual(len(batches[1]), 2)
        
        # 验证最后一批的大小
        self.assertEqual(len(batches[2]), 1)

    def test_merge_prediction_batches(self):
        """测试批次结果合并"""
        batch1 = [
            {'uuid': 'test_1', 'predict': 'SUCCESS'},
            {'uuid': 'test_2', 'predict': 'FAILURE'}
        ]
        batch2 = [
            {'uuid': 'test_3', 'predict': 'UNKNOWN'}
        ]
        
        merged = self.data_loader.merge_prediction_batches([batch1, batch2])
        
        # 验证合并结果
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0]['uuid'], 'test_1')
        self.assertEqual(merged[2]['predict'], 'UNKNOWN')

    def test_save_batch_predictions(self):
        """测试批次结果保存"""
        predictions = [
            {'uuid': 'test_1', 'predict': 'SUCCESS'},
            {'uuid': 'test_2', 'predict': 'FAILURE'}
        ]
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            base_path = f.name
        
        try:
            # 保存批次结果
            self.data_loader.save_batch_predictions(predictions, base_path, 1)
            
            # 验证文件是否正确生成
            batch_path = Path(base_path).parent / f"{Path(base_path).stem}_batch1{Path(base_path).suffix}"
            self.assertTrue(batch_path.exists())
            
            # 验证文件内容
            df = pd.read_csv(batch_path)
            self.assertEqual(len(df), 2)
            self.assertEqual(df.iloc[0]['predict'], 'SUCCESS')
            
        finally:
            # 清理测试文件
            Path(base_path).unlink(missing_ok=True)
            batch_path.unlink(missing_ok=True)

    def test_batch_processing_disabled(self):
        """测试禁用批处理时的行为"""
        # 创建禁用批处理的配置
        config = self.test_config.copy()
        config['batch_processing']['enabled'] = False
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(config, f)
            config_path = f.name
        
        try:
            loader = DataLoader(config_path)
            df = self.create_test_df(5)
            batches = list(loader.get_batch_generator(df))
            
            # 验证只生成一个批次，包含所有数据
            self.assertEqual(len(batches), 1)
            self.assertEqual(len(batches[0]), 5)
            
        finally:
            Path(config_path).unlink(missing_ok=True)

    def tearDown(self):
        """测试后清理"""
        Path(self.config_path).unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()