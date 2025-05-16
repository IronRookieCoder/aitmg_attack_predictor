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
import re  # 添加正则表达式库
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Union, Tuple
import traceback

import requests
from openai import OpenAI


from src.logger_module import get_logger
from src.prompt_templates import PromptTemplates

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
        
        # 从配置中获取SGLANG服务配置
        sglang_config = self.config.get('api', {}).get('sglang', {})
        self.SGLANG_ENDPOINT = sglang_config.get('endpoint', "https://aip.sangfor.com.cn:12588/v1")
        self.api_key = sglang_config.get('api_key', "sk-FRdCi5yEKxUyDcRp2fBd67921aCa4bA391506cE7C3Dd7863")
        
        self.logger.debug(f"SGLANG服务配置: {self.SGLANG_ENDPOINT}")
        
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
        
        # 为不同模型设置最佳参数配置
        self.model_configs = {
            'secgpt7b': {
                'temperature': 0,                 # 保持确定性输出
                'max_tokens': 2048,               # 允许较长输出
                'top_p': 1,                       # 保持全分布采样
                'description': "安全专用模型，通用能力对齐同等参数量开源模型，安全检测能力大幅提升"
            },
            'qwen7b': {
                'temperature': 0,                 # 保持确定性输出
                'max_tokens': 2048,               # 允许较长输出
                'top_p': 1,                       # 保持全分布采样
                'description': "通用基础模型，支持128K上下文，在知识、代码、推理等方面表现优秀"
            },
            'qwen3-8b': {
                'temperature': 0,                 # 保持确定性输出 
                'max_tokens': 2048,               # 允许较长输出
                'top_p': 1,                       # 保持全分布采样
                'top_k': 20,                      # 限制候选token数量
                'enable_thinking': False,         # 默认关闭思考模式
                'description': "最新Qwen系列模型，提供快慢思考模式，支持128K上下文"
            },
            'deepseek-r1-distill-qwen7b': {
                'temperature': 0,                 # 保持确定性输出
                'max_tokens': 2048,               # 允许较长输出
                'top_p': 0.9,                     # 稍微收缩分布，提高推理质量
                'description': "专注于数学、代码和逻辑推理任务的模型，通过知识蒸馏得到"
            }
        }

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
                
                return result
            
            except Exception as e:
                elapsed_time = time.time() - start_time
                # 最后一次尝试失败
                if attempt == self.max_retries - 1:
                    self.logger.error(f"模型 {model_name} 在尝试 {self.max_retries} 次后调用失败: {str(e)}", exc_info=True)
                    
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
        
        # 计算总耗时
        elapsed_time = time.time() - start_time
        
        # 记录所有模型的调用结果
        self.logger.info(f"所有模型调用完成，总耗时: {elapsed_time:.2f}秒")
        for model, result in results.items():
            self.logger.debug(f"模型 {model} 结果: {result}")
        
        return results

    def analyze_attack(self, req: str, rsp: str, 
                     few_shot_examples: Optional[List[Dict[str, str]]] = None,
                     use_verify: bool = True,
                     models_primary: Optional[List[str]] = None,
                     model_verify: Optional[str] = None) -> Dict[str, Union[str, Dict]]:
        """分析攻击请求和响应，判断攻击结果
        
        该方法实现了完整的攻击分析流程，包括:
        1. 使用指定模型进行初步分析
        2. 获取初步结果
        3. 可选地使用验证模型进行二次验证(ReviewFilter流程)
        4. 返回最终结果和详细分析信息
        
        Args:
            req: 请求数据
            rsp: 响应数据
            few_shot_examples: 可选的few-shot示例
            use_verify: 是否使用二次验证模型验证结果
            models_primary: 用于主要分析的模型列表，默认使用配置的模型池
            model_verify: 用于验证的模型，默认使用secgpt7b
            
        Returns:
            Dict: 包含分析结果和详细信息的字典
            {
                "result": "SUCCESS" | "FAILURE" | "UNKNOWN",  # 最终结果
                "confidence": 0.85,  # 置信度（如多数投票时的比例）
                "details": {
                    "primary_results": {"model1": "结果", ...},
                    "verification": {"model": "verify_model", "result": "结果", "agrees": True}
                }
            }
        """
        self.logger.info("开始分析攻击请求和响应")
        
        # 1. 设置默认值
        if not models_primary:
            models_primary = list(self.model_pool)
        if not model_verify and use_verify:
            model_verify = "secgpt7b"  # 默认使用secgpt7b作为验证模型
            
        self.logger.debug(f"主分析模型: {models_primary}, 验证模型: {model_verify if use_verify else '不使用'}")
        
        # 2. 创建prompt模板实例
        templates = PromptTemplates(config_path="config.json")
        
        # 3. 使用"结论在前"的V3模板进行初步分析
        primary_prompt = templates.format_prompt(
            template_version="v3",
            req=req,
            rsp=rsp,
            few_shot_examples=few_shot_examples
        )
        
        # 4. 调用多个模型进行初步分析
        self.logger.info("开始初步分析")
        primary_results = self.call_multiple_models(models_primary, primary_prompt)
        self.logger.debug(f"初步分析结果: {primary_results}")
        
        # 5. 处理初步分析结果
        initial_result, confidence = self._get_majority_vote(primary_results)
        self.logger.info(f"初步分析多数结果: {initial_result}, 置信度: {confidence:.2f}")
        
        # 6. 如果启用验证且初步结果不是LLM_ERROR，进行二次验证
        verification = {}
        if use_verify and initial_result != "LLM_ERROR" and model_verify in self.model_pool:
            self.logger.info(f"使用模型 {model_verify} 进行二次验证")
            
            # 创建验证prompt
            verify_prompt = templates.format_prompt(
                template_version="verify",
                req=req,
                rsp=rsp,
                initial_classification=initial_result
            )
            
            # 调用验证模型
            verification_response = self.call_llm(model_verify, verify_prompt)
            self.logger.debug(f"验证模型返回: {verification_response}")
            
            # 解析验证结果
            verification_result, agrees = self._parse_verification_response(verification_response, initial_result)
            verification = {
                "model": model_verify,
                "result": verification_result,
                "agrees": agrees
            }
            
            self.logger.info(f"验证结果: {verification_result}, 是否同意初步分析: {agrees}")
            
            # 如果验证不同意且配置允许覆盖，则采用验证结果
            if not agrees and self.config.get("verification", {}).get("allow_override", True):
                self.logger.info(f"验证模型 {model_verify} 覆盖了初步分析结果")
                final_result = verification_result
            else:
                final_result = initial_result
        else:
            final_result = initial_result
        
        # 7. 构建完整的结果信息
        result = {
            "result": final_result,
            "confidence": confidence,
            "details": {
                "primary_results": primary_results
            }
        }
        
        if verification:
            result["details"]["verification"] = verification
        
        self.logger.info(f"分析完成，最终结果: {final_result}")
        return result

    def _get_majority_vote(self, model_results: Dict[str, str]) -> Tuple[str, float]:
        """从多个模型结果中获取多数投票结果
        
        该方法使用简单多数投票，选出出现最多的结果。
        如果出现平局，优先选择按照SUCCESS > FAILURE > UNKNOWN > LLM_ERROR的顺序。
        
        Args:
            model_results: 模型名到预测结果的映射
            
        Returns:
            Tuple[str, float]: 多数结果和对应的置信度(占比)
        """
        if not model_results:
            self.logger.warning("模型结果为空，无法进行多数投票")
            return "UNKNOWN", 0.0
            
        self.logger.debug("进行多数投票")
        
        # 统计不同结果的数量
        result_counts = {}
        for result in model_results.values():
            result_counts[result] = result_counts.get(result, 0) + 1
            
        self.logger.debug(f"投票统计: {result_counts}")
        
        # 找出出现次数最多的结果
        max_count = max(result_counts.values())
        max_results = [r for r, c in result_counts.items() if c == max_count]
        self.logger.debug(f"最高票数: {max_count}, 候选结果: {max_results}")
        
        # 计算置信度
        confidence = max_count / len(model_results)
        
        # 如果只有一个最多的结果，直接返回
        if len(max_results) == 1:
            return max_results[0], confidence
            
        # 如果有平局，按照优先级顺序选择
        priority_order = ["SUCCESS", "FAILURE", "UNKNOWN", "LLM_ERROR"]
        for priority in priority_order:
            if priority in max_results:
                self.logger.warning(f"多数投票平局: {max_results}, 返回{priority}")
                return priority, confidence
                
        # 兜底情况
        self.logger.error(f"无法从 {max_results} 中决定多数投票结果")
        return "UNKNOWN", confidence

    def _parse_verification_response(self, response: str, initial_result: str) -> Tuple[str, bool]:
        """解析验证模型的响应
        
        从验证模型的响应中提取是否同意初步分析以及正确的分类结果。
        
        Args:
            response: 验证模型的响应
            initial_result: 初步分析的结果
            
        Returns:
            Tuple[str, bool]: 验证后的结果和是否同意初步分析
        """
        self.logger.debug("解析验证响应")
        
        # 查找VERIFICATION部分
        verification_match = re.search(r'VERIFICATION:\s*\[?(AGREE|DISAGREE)\]?', response, re.IGNORECASE)
        agrees = True
        if verification_match:
            agrees = verification_match.group(1).upper() == "AGREE"
            self.logger.debug(f"找到验证结果: {'同意' if agrees else '不同意'}")
        else:
            self.logger.warning("未在响应中找到明确的验证结果，假设同意初步分析")
        
        # 查找CORRECT CLASSIFICATION部分
        classification_match = re.search(r'CORRECT CLASSIFICATION:\s*\[?(SUCCESS|FAILURE|UNKNOWN)\]?', response, re.IGNORECASE)
        if classification_match:
            result = classification_match.group(1).upper()
            self.logger.debug(f"找到正确分类: {result}")
        else:
            self.logger.warning("未在响应中找到正确分类，使用初步分析结果")
            result = initial_result
        
        return result, agrees

    def _process_llm_response(self, response: str) -> str:
        """处理LLM的响应，提取最终答案
        
        该方法根据模型输出的格式不同，尝试提取其中的最终结论。
        
        Args:
            response: LLM原始响应文本
            
        Returns:
            str: 清理后的响应结果，通常应为"SUCCESS"/"FAILURE"/"UNKNOWN"
        """
        self.logger.debug(f"处理原始响应: {response[:100]}...")
        
        # 移除思考过程 (<think>...</think>)
        cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # 移除多余空白
        cleaned_response = cleaned_response.strip()
        
        # 尝试处理"结论在前"格式的响应
        final_conclusion = re.search(r'FINAL CONCLUSION:\s*\[?(SUCCESS|FAILURE|UNKNOWN)\]?', cleaned_response, re.IGNORECASE)
        if final_conclusion:
            result = final_conclusion.group(1).upper()
            self.logger.debug(f"从FINAL CONCLUSION提取结果: {result}")
            return result
            
        # 尝试处理"初步结论"格式
        preliminary_conclusion = re.search(r'PRELIMINARY CONCLUSION:\s*\[?(SUCCESS|FAILURE|UNKNOWN)\]?', cleaned_response, re.IGNORECASE)
        if preliminary_conclusion:
            result = preliminary_conclusion.group(1).upper()
            self.logger.debug(f"从PRELIMINARY CONCLUSION提取结果: {result}")
            return result
        
        # 提取最终答案
        valid_results = ["SUCCESS", "FAILURE", "UNKNOWN", "LLM_ERROR"]
        
        # 尝试完全匹配
        if cleaned_response in valid_results:
            return cleaned_response
            
        # 尝试在文本中查找这些关键词（大写）
        for result in valid_results:
            if result in cleaned_response.upper():
                self.logger.debug(f"从响应中提取结果: {result}")
                return result
                
        # 如果没有找到有效结果，但响应长度超过50字符，返回UNKNOWN
        if len(cleaned_response) > 50:
            self.logger.warning(f"响应格式不符合预期，返回UNKNOWN: {cleaned_response[:50]}...")
            return "UNKNOWN"
            
        # 否则返回清理后的响应
        return cleaned_response

    def _call_secgpt(self, prompt: str) -> str:
        """调用SecGPT模型"""
        self.logger.debug("调用SecGPT模型")
        start_time = time.time()
        
        try:
            config = self.model_configs['secgpt7b']
            completion = self.client.chat.completions.create(
                model="secgpt7b",
                messages=[{"role": "user", "content": prompt}],
                temperature=config['temperature'],
                max_tokens=config.get('max_tokens', None),
                top_p=config.get('top_p', None)
            )
            result = completion.choices[0].message.content
            
            elapsed_time = time.time() - start_time
            self.logger.debug(f"SecGPT调用成功，耗时: {elapsed_time:.2f}秒")
            
            # 处理响应结果
            return self._process_llm_response(result)
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"调用SecGPT出错: {str(e)}")
            raise

    def _call_qwen3(self, prompt: str) -> str:
        """调用Qwen3模型"""
        self.logger.debug("调用Qwen3-8B模型")
        start_time = time.time()
        
        try:
            config = self.model_configs['qwen3-8b']
            completion = self.client.chat.completions.create(
                model="qwen3-8b",
                messages=[{"role": "user", "content": prompt}],
                temperature=config['temperature'],
                max_tokens=config.get('max_tokens', None),
                top_p=config.get('top_p', None),
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": config.get('enable_thinking', False)},
                    "top_k": config.get('top_k', 20)
                }
            )
            result = completion.choices[0].message.content
            
            elapsed_time = time.time() - start_time
            self.logger.debug(f"Qwen3-8B调用成功，耗时: {elapsed_time:.2f}秒")
            
            # 处理响应结果
            return self._process_llm_response(result)
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"调用Qwen3-8B出错: {str(e)}")
            raise

    def _call_qwen7b(self, prompt: str) -> str:
        """调用Qwen7b模型"""
        self.logger.debug("调用Qwen7B模型")
        start_time = time.time()
        
        try:
            config = self.model_configs['qwen7b']
            completion = self.client.chat.completions.create(
                model="qwen7b",
                messages=[{"role": "user", "content": prompt}],
                temperature=config['temperature'],
                max_tokens=config.get('max_tokens', None),
                top_p=config.get('top_p', None)
            )
            result = completion.choices[0].message.content
            
            elapsed_time = time.time() - start_time
            self.logger.debug(f"Qwen7B调用成功，耗时: {elapsed_time:.2f}秒")
            
            # 处理响应结果
            return self._process_llm_response(result)
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"调用Qwen7B出错: {str(e)}")
            raise

    def _call_deepseek(self, prompt: str) -> str:
        """调用Deepseek模型"""
        self.logger.debug("调用Deepseek模型")
        start_time = time.time()
        
        try:
            config = self.model_configs['deepseek-r1-distill-qwen7b']
            completion = self.client.chat.completions.create(
                model="deepseek-r1-distill-qwen7b",
                messages=[{"role": "user", "content": prompt}],
                temperature=config['temperature'],
                max_tokens=config.get('max_tokens', None),
                top_p=config.get('top_p', None)
            )
            result = completion.choices[0].message.content
            
            elapsed_time = time.time() - start_time
            self.logger.debug(f"Deepseek调用成功，耗时: {elapsed_time:.2f}秒")
            
            # 处理响应结果
            return self._process_llm_response(result)
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"调用Deepseek出错: {str(e)}")
            raise