from pydantic_settings import BaseSettings
from typing import List, ClassVar, Dict, Any

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
    storage_dir: str = "brain_v9_demo"

    # ====================== 走神状态配置 ======================
    mind_wandering_enabled: bool = True          # 是否启用走神
    mind_wandering_idle_threshold: int = 30      # 空闲多少秒触发走神
    fatigue_sleep_threshold: float = 0.85         # 疲劳度达到多少触发睡眠
    mind_wandering_recall_prob: float = 0.5      # 记忆闪回概率
    mind_wandering_assoc_prob: float = 0.3       # 联想想象概率

    # ========== 修复：给局部连接偏置配置加上类型注解 ==========
    local_bias_enabled: bool = True          # 是否启用局部连接偏置
    local_bias_strength: float = 1.2          # 同一分区内的偏置强度（0.1-0.6）
    cross_partition_decay: float = 0.7        # 跨分区的衰减系数（0.3-0.9，越小越稀疏）
    spatial_decay_enabled: bool = False       # 是否启用空间位置衰减（可选）
    spatial_radius: int = 5                  # 空间局部半径（仅spatial_decay_enabled=True时生效）
    # ====================================================================
    
    entity_extraction_config: ClassVar[Dict[str, Any]] = {
        "min_entity_length": 2,           # 实体最小长度
        "use_quote_extraction": True,     # 是否提取引号中的内容
        "use_tag_extraction": True,        # 是否从标签中提取
        "use_triple_extraction": True,     # 是否从三元组中提取
        "exclude_tags": [],                # 排除的标签（空列表=不排除）
        "split_chars": ["，", "。", "！", "？", "；", "：", "、"],  # 分词字符
    }

    # ====================== 新增：专家专属差异化参数 ======================
    EXPERT_CONFIG: ClassVar[Dict] = {
        "身份": {
            "sparsity": 0.05,
            "local_radius": 0.1,
            "core_bias": 0.8,
            "sdr_active_count": 15
        },
        "概念": {
            "sparsity": 0.02,
            "local_radius": 0.2,
            "core_bias": 0.5,
            "sdr_active_count": 20
        },
        "空间": {
            "sparsity": 0.03,
            "local_radius": 0.35,
            "core_bias": 0.4,
            "sdr_active_count": 18
        },
        "抽象": {
            "sparsity": 0.025,
            "local_radius": 0.25,
            "core_bias": 0.45,
            "sdr_active_count": 16
        },
        "视觉": {
            "sparsity": 0.04,
            "local_radius": 0.15,
            "core_bias": 0.55,
            "sdr_active_count": 14
        }
    }
    

    class Config:
        env_file = ".env"

# 全局配置实例
config = BrainConfig()