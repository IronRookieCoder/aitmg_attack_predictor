"""
日志模块: 提供统一的日志记录和持久化功能
"""

import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Dict, Optional, Union, Any


class LoggerModule:
    """
    日志模块类，提供统一的日志记录和持久化功能
    
    特性:
    - 同时支持控制台和文件输出
    - 可配置日志级别和格式
    - 支持自定义日志格式
    - 所有日志持久化到同一个文件
    """
    
    # 日志级别映射
    _LOG_LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    
    # 默认配置
    _DEFAULT_CONFIG = {
        "log_level": "info",
        "console_output": True,
        "file_output": True,
        "log_dir": "logs",
        "base_filename": "attack_predictor",
        "include_timestamp": True,
        "include_level": True,
        "include_process": False
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化日志模块
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config = self._DEFAULT_CONFIG.copy()
        
        # 如果提供了配置文件，从中加载日志配置
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                
                # 合并配置，仅更新logging相关部分
                if 'logging' in loaded_config:
                    self._update_config(self.config, loaded_config['logging'])
            except Exception as e:
                print(f"警告: 无法加载日志配置 ({str(e)}), 使用默认配置")
        
        # 创建日志目录
        if self.config['file_output']:
            log_dir = Path(self.config['log_dir'])
            log_dir.mkdir(exist_ok=True, parents=True)
        
        # 配置根日志记录器
        self._configure_root_logger()
        
        # 创建主日志记录器
        self.logger = self.get_logger("main")
        
        # 自动设置日志持久化
        self._setup_log_persistence()
        
        # 记录初始化消息
        self.logger.info("日志模块初始化完成")
        
    def _setup_log_persistence(self):
        """
        设置日志持久化，确保每次程序启动时写入同一个日志文件
        """
        if not self.config['file_output']:
            return
            
        # 创建日志目录
        log_dir = Path(self.config['log_dir'])
        log_dir.mkdir(exist_ok=True, parents=True)
        
        # 获取日志文件路径
        base_filename = self.config['base_filename']
        if not base_filename.endswith('.log'):
            base_filename += '.log'
        self.log_path = log_dir / base_filename
        
        self.logger.info(f"日志持久化已设置，将写入文件: {self.log_path}")
    
    def _update_config(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        递归更新配置字典
        
        Args:
            target: 目标字典
            source: 源字典
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_config(target[key], value)
            else:
                target[key] = value
    
    def _configure_root_logger(self) -> None:
        """配置根日志记录器"""
        # 清除现有处理器
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 设置日志级别
        log_level = self._LOG_LEVELS.get(
            self.config['log_level'].lower(),
            logging.INFO
        )
        root_logger.setLevel(log_level)
        
        # 创建格式化器
        formatter = self._create_text_formatter()
        
        # 添加控制台处理器
        if self.config['console_output']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # 添加文件处理器
        if self.config['file_output']:
            file_handler = self._create_file_handler(formatter)
            if file_handler:
                root_logger.addHandler(file_handler)
    
    def _create_text_formatter(self) -> logging.Formatter:
        """创建文本格式的格式化器"""
        format_parts = []
        
        if self.config['include_timestamp']:
            format_parts.append('%(asctime)s')
        
        if self.config['include_process']:
            format_parts.append('%(process)d-%(processName)s')
        
        format_parts.append('%(name)s')
        
        if self.config['include_level']:
            format_parts.append('%(levelname)s')
        
        format_parts.append('%(message)s')
        
        log_format = ' - '.join(format_parts)
        return logging.Formatter(log_format)
    
    def _create_file_handler(self, formatter: logging.Formatter) -> Optional[logging.Handler]:
        """
        创建文件处理器
        
        Args:
            formatter: 日志格式化器
            
        Returns:
            日志处理器或None
        """
        try:
            log_dir = Path(self.config['log_dir'])
            base_filename = self.config['base_filename']
            
            # 创建完整的日志文件路径
            if not base_filename.endswith('.log'):
                base_filename += '.log'
            log_path = log_dir / base_filename
            
            # 始终使用不轮转的文件处理器，确保写入同一个文件
            handler = logging.FileHandler(log_path, mode='a')  # 使用追加模式
            
            handler.setFormatter(formatter)
            return handler
        except Exception as e:
            print(f"警告: 无法创建文件处理器 ({str(e)})")
            return None
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        获取指定名称的日志记录器
        
        Args:
            name: 日志记录器名称
            
        Returns:
            日志记录器实例
        """
        return logging.getLogger(name)
    
    def set_log_level(self, level: Union[str, int]) -> None:
        """
        设置日志级别
        
        Args:
            level: 日志级别，可以是字符串('debug', 'info'等)或整数常量
        """
        if isinstance(level, str):
            level = self._LOG_LEVELS.get(level.lower(), logging.INFO)
        
        logging.getLogger().setLevel(level)
        self.logger.info(f"日志级别已设置为: {logging.getLevelName(level)}")

# 创建单例实例
logger_module = None

def init_logger(config_path: Optional[str] = None) -> LoggerModule:
    """
    初始化日志模块
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        LoggerModule实例
    """
    global logger_module
    if logger_module is None:
        logger_module = LoggerModule(config_path)
    return logger_module

def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器
    """
    if logger_module is None:
        init_logger()
    return logger_module.get_logger(name)

def create_persistent_record(log_data: Dict[str, Any], level: str = "info") -> None:
    """
    将日志数据以普通日志形式记录到主日志文件中
    
    Args:
        log_data: 日志数据
        level: 日志级别
    """
    if logger_module is None:
        init_logger()
    
    logger = logger_module.get_logger("persistent")
    
    # 转换为字符串并记录
    log_message = f"持久化记录: {json.dumps(log_data, ensure_ascii=False)}"
    
    # 根据级别调用相应的日志方法
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(log_message)