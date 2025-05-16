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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Union
import traceback

import requests
from openai import OpenAI


from src.logger_module import get_logger, create_persistent_record

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
        # 获取日志记录器
        self.logger = get_logger(__name__)
        self.logger.info("初始化LLMInterface")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.max_retries = self.config.get('max_retries', 3)
        self.timeout = self.config.get('timeout', 120)
        
        # SGLANG服务配置
        self.SGLANG_ENDPOINT = "https://aip.sangfor.com.cn:12588/v1"
        self.api_key = "sk-FRdCi5yEKxUyDcRp2fBd67921aCa4bA391506cE7C3Dd7863"
        
        # 创建OpenAI兼容客户端
        self.client = OpenAI(
            base_url=self.SGLANG_ENDPOINT,
            api_key=self.api_key,
            timeout=self.timeout * 1000  # 转换为毫秒
        )

        # 检查模型池配置
        if 'model_pool' not in self.config:
            self.logger.error("配置文件中缺少model_pool")
            raise ValueError("配置文件中缺少model_pool")
            
        self.model_pool = set(self.config['model_pool'])  # 用于快速验证
        self.logger.info(f"可用模型池: {sorted(list(self.model_pool))}")
        self.logger.debug(f"最大重试次数: {self.max_retries}, 超时设置: {self.timeout}秒")
        
        # 记录初始化信息
        create_persistent_record({
            'operation': 'init_llm_interface',
            'model_pool': list(self.model_pool),
            'max_retries': self.max_retries,
            'timeout': self.timeout
        })

    def call_llm(self, model_name: str, prompt: str) -> str:
        """调用LLM模型
        
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
        # 验证模型名称
        if model_name not in self.model_pool:
            self.logger.error(f"模型 {model_name} 不在配置的模型池中")
            raise ValueError(f"模型 {model_name} 不在配置的模型池中")
        
        start_time = time.time()
        self.logger.info(f"开始调用模型 {model_name}")
        self.logger.debug(f"Prompt长度: {len(prompt)} 字符")

        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"尝试 {attempt + 1}/{self.max_retries}")
                
                # 根据不同的模型调用对应的实现
                if model_name == 'secgpt7b':
                    result = self._call_secgpt(prompt)
                elif model_name == 'qwen3-8b':
                    result = self._call_qwen3(prompt)
                elif model_name == 'qwen7b':
                    result = self._call_qwen7b(prompt)
                elif model_name == 'deepseek-r1-distill-qwen7b':
                    result = self._call_deepseek(prompt)
                else:
                    self.logger.error(f"不支持的模型: {model_name}")
                    raise ValueError(f"不支持的模型: {model_name}")
                
                # 计算耗时
                elapsed_time = time.time() - start_time
                
                # 记录成功调用信息
                self.logger.info(f"模型 {model_name} 调用成功，耗时 {elapsed_time:.2f}秒")
                self.logger.debug(f"模型 {model_name} 返回结果: {result}")
                
                # 持久化记录
                create_persistent_record({
                    'operation': 'call_llm',
                    'model': model_name,
                    'success': True,
                    'elapsed_time': elapsed_time,
                    'result': result,
                    'attempt': attempt + 1
                })
                
                return result
            
            except Exception as e:
                elapsed_time = time.time() - start_time
                # 最后一次尝试失败
                if attempt == self.max_retries - 1:
                    self.logger.error(f"模型 {model_name} 在尝试 {self.max_retries} 次后调用失败: {str(e)}", exc_info=True)
                    
                    # 持久化记录
                    create_persistent_record({
                        'operation': 'call_llm',
                        'model': model_name,
                        'success': False,
                        'elapsed_time': elapsed_time,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }, level="error")
                    
                    return "LLM_ERROR"
                
                # 中间尝试失败，准备重试
                self.logger.warning(f"模型 {model_name} 第 {attempt + 1} 次尝试失败: {str(e)}")
                # 指数退避重试
                retry_delay = 2 ** attempt
                self.logger.debug(f"将在 {retry_delay} 秒后重试")
                time.sleep(retry_delay)
        
        # 正常情况下不会执行到这里
        self.logger.error(f"模型 {model_name} 调用失败，未知错误")
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
        # 验证所有模型名称
        invalid_models = [m for m in models if m not in self.model_pool]
        if invalid_models:
            self.logger.error(f"模型 {invalid_models} 不在配置的模型池中")
            raise ValueError(f"模型 {invalid_models} 不在配置的模型池中")

        # 获取批处理配置
        max_workers = self.config.get('batch_processing', {}).get('max_workers', 4)
        parallel_count = min(len(models), max_workers)
        
        self.logger.info(f"开始并行调用 {len(models)} 个模型，最大并行数: {parallel_count}")
        self.logger.debug(f"要调用的模型: {models}")
        
        start_time = time.time()
        results = {}
        
        with ThreadPoolExecutor(max_workers=parallel_count) as executor:
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
                    self.logger.error(f"调用模型 {model} 时发生错误: {str(e)}", exc_info=True)
                    results[model] = "LLM_ERROR"
                    
                    # 持久化记录错误
                    create_persistent_record({
                        'operation': 'call_multiple_models',
                        'model': model,
                        'success': False,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }, level="error")
        
        # 计算总耗时
        elapsed_time = time.time() - start_time
        
        # 记录所有模型的调用结果
        self.logger.info(f"所有模型调用完成，总耗时: {elapsed_time:.2f}秒")
        for model, result in results.items():
            self.logger.debug(f"模型 {model} 结果: {result}")
        
        # 持久化记录结果
        create_persistent_record({
            'operation': 'call_multiple_models',
            'models': models,
            'results': results,
            'elapsed_time': elapsed_time
        })
        
        return results

    def _call_secgpt(self, prompt: str) -> str:
        """调用SecGPT模型"""
        self.logger.debug("调用SecGPT模型")
        start_time = time.time()
        
        try:
            completion = self.client.chat.completions.create(
                model="secgpt7b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            result = completion.choices[0].message.content
            
            elapsed_time = time.time() - start_time
            self.logger.debug(f"SecGPT调用成功，耗时: {elapsed_time:.2f}秒")
            
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"调用SecGPT出错: {str(e)}")
            
            # 持久化记录错误
            create_persistent_record({
                'operation': 'call_secgpt',
                'elapsed_time': elapsed_time,
                'error': str(e),
                'traceback': traceback.format_exc()
            }, level="error")
            
            raise

    def _call_qwen3(self, prompt: str) -> str:
        """调用Qwen3模型"""
        self.logger.debug("调用Qwen3-8B模型")
        start_time = time.time()
        
        try:
            completion = self.client.chat.completions.create(
                model="qwen3-8b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "top_k": 20
                }
            )
            result = completion.choices[0].message.content
            
            elapsed_time = time.time() - start_time
            self.logger.debug(f"Qwen3-8B调用成功，耗时: {elapsed_time:.2f}秒")
            
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"调用Qwen3-8B出错: {str(e)}")
            
            # 持久化记录错误
            create_persistent_record({
                'operation': 'call_qwen3',
                'elapsed_time': elapsed_time,
                'error': str(e),
                'traceback': traceback.format_exc()
            }, level="error")
            
            raise

    def _call_qwen7b(self, prompt: str) -> str:
        """调用Qwen7b模型"""
        self.logger.debug("调用Qwen7B模型")
        start_time = time.time()
        
        try:
            completion = self.client.chat.completions.create(
                model="qwen7b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            result = completion.choices[0].message.content
            
            elapsed_time = time.time() - start_time
            self.logger.debug(f"Qwen7B调用成功，耗时: {elapsed_time:.2f}秒")
            
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"调用Qwen7B出错: {str(e)}")
            
            # 持久化记录错误
            create_persistent_record({
                'operation': 'call_qwen7b',
                'elapsed_time': elapsed_time,
                'error': str(e),
                'traceback': traceback.format_exc()
            }, level="error")
            
            raise

    def _call_deepseek(self, prompt: str) -> str:
        """调用Deepseek模型"""
        self.logger.debug("调用Deepseek模型")
        start_time = time.time()
        
        try:
            completion = self.client.chat.completions.create(
                model="deepseek-r1-distill-qwen7b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            result = completion.choices[0].message.content
            
            elapsed_time = time.time() - start_time
            self.logger.debug(f"Deepseek调用成功，耗时: {elapsed_time:.2f}秒")
            
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"调用Deepseek出错: {str(e)}")
            
            # 持久化记录错误
            create_persistent_record({
                'operation': 'call_deepseek',
                'elapsed_time': elapsed_time,
                'error': str(e),
                'traceback': traceback.format_exc()
            }, level="error")
            
            raise