from pydantic_settings import BaseSettings
from typing import List

class BrainConfig(BaseSettings):
    # 向量维度
    dim: int = 1024  # bge-m3 原生维度
    sdr_dim: int = 2048
    sdr_active_size: int = 60
    
    # 专家配置
    expert_names: List[str] = ["身份", "视觉", "概念", "空间", "抽象"]
    max_expert_dim: int = 8192
    
    # 检索配置
    top_k: int = 10
    min_similarity: float = 0.4
    target_expert_ratio: float = 0.5  # 目标专家比例
    other_expert_ratio: float = 0.1  # 其他专家比例
    
    # 记忆配置
    duplicate_threshold: float = 0.92  # 去重阈值
    permanent_importance_threshold: float = 0.9  # 永久记忆阈值
    forget_days: int = 365  # 遗忘天数
    forget_importance_threshold: float = 0.3  # 遗忘重要性阈值
    
    # LLM配置
    ollama_model_name: str = "gemma3:4b"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 256
    
    # 存储配置
    storage_dir: str = "brain_v7_demo"
    
    class Config:
        env_file = ".env"

# 全局配置实例
config = BrainConfig()