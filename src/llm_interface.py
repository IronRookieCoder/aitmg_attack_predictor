"""LLM接口封装模块

本模块提供了与不同大语言模型交互的统一接口。主要功能包括:
1. 统一的模型调用接口
2. 错误重试和超时处理
3. 并行模型调用支持
4. 模型响应解析和验证

主要组件:
- LLMInterface: 核心接口类，处理所有模型调用
- 内置支持模型: SecGPT7B、Qwen3-8B、Qwen7B、Deepseek-r1-distill-qwen7b
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Union

import requests

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMInterface:
    """LLM接口封装类
    
    该类提供了统一的接口来调用不同的大语言模型，支持:
    - 单一模型调用
    - 多模型并行调用
    - 错误重试
    - 超时处理
    - 响应验证
    
    属性:
        config: Dict 配置信息，包含工号、模型池等
        max_retries: int 最大重试次数
        timeout: int API调用超时时间（秒）
    """
    
    def __init__(self, config_path: str = "config.json"):
        """初始化LLM接口
        
        Args:
            config_path: 配置文件路径，包含工号、模型池、超时设置等
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.max_retries = self.config.get('max_retries', 3)
        self.timeout = self.config.get('timeout', 120)

    def call_llm(self, model_name: str, prompt: str) -> str:
        """调用单个LLM模型
        
        该方法处理单个模型的调用，包括错误重试和响应验证。
        如果多次重试后仍然失败，返回"LLM_ERROR"。
        
        Args:
            model_name: 模型名称，必须在配置的模型池中
            prompt: 输入的prompt文本
            
        Returns:
            str: 模型的输出结果。正常返回"SUCCESS"/"FAILURE"/"UNKNOWN"，
                错误时返回"LLM_ERROR"
        
        Raises:
            ValueError: 当指定的模型名称不在支持列表中时
        """
        for attempt in range(self.max_retries):
            try:
                # 根据不同的模型调用对应的实现
                if model_name == 'secgpt7b':
                    return self._call_secgpt(prompt)
                elif model_name == 'qwen3-8b':
                    return self._call_qwen3(prompt)
                elif model_name == 'qwen7b':
                    return self._call_qwen7b(prompt)
                elif model_name == 'deepseek-r1-distill-qwen7b':
                    return self._call_deepseek(prompt)
                else:
                    raise ValueError(f"Unsupported model: {model_name}")
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to call {model_name} after {self.max_retries} attempts: {str(e)}")
                    return "LLM_ERROR"
                logger.warning(f"Attempt {attempt + 1} failed for {model_name}: {str(e)}")
                time.sleep(2 ** attempt)  # 指数退避重试
        
        return "LLM_ERROR"

    def call_multiple_models(self, models: List[str], prompt: str) -> Dict[str, str]:
        """并行调用多个LLM模型
        
        使用线程池实现并行调用多个模型，提高处理效率。
        每个模型的调用都有独立的错误处理。
        
        Args:
            models: 要调用的模型名称列表
            prompt: 输入的prompt文本
            
        Returns:
            Dict[str, str]: 模型名到输出结果的映射
                key: 模型名
                value: 预测结果或"LLM_ERROR"
                
        Note:
            - 使用ThreadPoolExecutor实现并行
            - 默认最多4个并行线程
            - 每个模型调用都有独立的重试机制
        """
        results = {}
        with ThreadPoolExecutor(max_workers=min(len(models), 4)) as executor:
            future_to_model = {
                executor.submit(self.call_llm, model, prompt): model
                for model in models
            }
            
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    result = future.result()
                    results[model] = result
                except Exception as e:
                    logger.error(f"Error calling {model}: {str(e)}")
                    results[model] = "LLM_ERROR"
        
        return results

    def _call_secgpt(self, prompt: str) -> str:
        """调用SecGPT模型的具体实现
        
        Args:
            prompt: 输入的prompt文本
            
        Returns:
            str: 模型预测结果
            
        Note:
            需要在实现中处理:
            1. API认证
            2. 请求格式化
            3. 响应解析
            4. 错误处理
        """
        # TODO: 实现具体的API调用
        pass

    def _call_qwen3(self, prompt: str) -> str:
        """调用Qwen3模型的具体实现
        
        Args:
            prompt: 输入的prompt文本
            
        Returns:
            str: 模型预测结果
            
        Note:
            需要在实现中处理:
            1. API认证
            2. 请求格式化
            3. 响应解析
            4. 错误处理
        """
        # TODO: 实现具体的API调用
        pass

    def _call_qwen7b(self, prompt: str) -> str:
        """调用Qwen7b模型的具体实现
        
        Args:
            prompt: 输入的prompt文本
            
        Returns:
            str: 模型预测结果
            
        Note:
            需要在实现中处理:
            1. API认证
            2. 请求格式化
            3. 响应解析
            4. 错误处理
        """
        # TODO: 实现具体的API调用
        pass

    def _call_deepseek(self, prompt: str) -> str:
        """调用Deepseek模型的具体实现
        
        Args:
            prompt: 输入的prompt文本
            
        Returns:
            str: 模型预测结果
            
        Note:
            需要在实现中处理:
            1. API认证
            2. 请求格式化
            3. 响应解析
            4. 错误处理
        """
        # TODO: 实现具体的API调用
        pass