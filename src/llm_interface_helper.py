"""LLM接口辅助模块，提供共用功能"""

def extract_verify_model(config, primary_model, model_pool, logger):
    """为单一模型策略查找合适的验证模型
    
    查找逻辑:
    1. 遍历单一模型策略配置，查找匹配当前模型的策略
    2. 检查该策略是否启用了审核过滤并配置了验证模型
    3. 如果没有找到配置的验证模型，则使用全局默认验证模型
    4. 如果全局默认验证模型也未配置，则使用主分析模型自身
    
    Args:
        config: Dict 配置信息
        primary_model: str 主分析模型名称
        model_pool: set 可用模型集合
        logger: 日志记录器
        
    Returns:
        str: 验证模型名称
    """
    # 查找当前策略的配置
    strategy_found = False
    model_verify = None
    
    # 遍历单一模型策略配置，查找匹配当前模型的策略
    for strategy_key, strategy_config in config.get('single_model_strategies', {}).items():
        if strategy_config.get('model_name') == primary_model:
            # 检查策略是否启用了审核过滤
            if strategy_config.get('use_review_filter', False):
                # 检查是否配置了验证模型
                if strategy_config.get('verify_model') and strategy_config['verify_model'] in model_pool:
                    model_verify = strategy_config['verify_model']
                    logger.info(f"单一模型策略使用配置的验证模型: {model_verify}")
                    strategy_found = True
                    break
    
    # 如果未找到匹配的策略或策略中未指定验证模型
    if not strategy_found or not model_verify:
        # 检查是否有全局默认验证模型
        default_verify_model = config.get('verification', {}).get('default_verify_model')
        if default_verify_model and default_verify_model in model_pool:
            model_verify = default_verify_model
            logger.info(f"单一模型策略使用全局默认验证模型: {model_verify}")
        else:
            # 没有配置验证模型，则使用当前模型自身
            model_verify = primary_model
            logger.info(f"单一模型策略，将使用同一模型进行验证: {model_verify}")
    
    return model_verify
