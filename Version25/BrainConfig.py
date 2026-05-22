from pydantic_settings import BaseSettings
from typing import List, ClassVar, Dict, Any

class BrainConfig(BaseSettings):
    # ====================== 基础核心配置 ======================
    # 向量维度
    dim: int = 1024  # bge-m3 原生维度
    sdr_dim: int = 2048
    sdr_active_size: int = 60
    device: str = 'cpu'
    
    # 专家配置
    expert_names: List[str] = ["身份", "视觉", "概念", "空间", "抽象"]
    max_expert_dim: int = 8192
    
    # 检索配置
    top_k: int = 10
    min_similarity: float = 0.4
    target_expert_ratio: float = 0.5  # 目标专家结果占比
    other_expert_ratio: float = 0.1  # 其他专家结果占比
    
    # ====================== 🔴 实体中心核心配置（原记忆配置升级） ======================
    # 实体基础配置
    duplicate_entity_threshold: float = 0.92  # 实体去重阈值（余弦相似度）
    permanent_entity_threshold: float = 0.9  # 永久实体重要性阈值
    entity_forget_days: int = 365  # 普通实体自动遗忘天数
    entity_forget_importance_threshold: float = 0.3  # 低于此重要性的实体可被遗忘
    entity_access_count_weight: float = 0.2  # 访问次数对重要性的贡献权重
    
    # 实体关系配置
    relation_confidence_threshold: float = 0.6  # 关系置信度阈值
    relation_strength_decay: float = 0.01  # 关系强度每日衰减率
    max_relations_per_entity: int = 50  # 单个实体最多关联数
    
    # ====================== LLM与嵌入配置 ======================
    ollama_model_name: str = "gemma3:4b"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 256
    llm_timeout: int = 30  # LLM请求超时时间（秒）
    
    # ====================== 存储配置 ======================
    storage_dir: str = "brain_v13_demo"
    vae_model_path: str = r'D:\2250111005\HippoCortex\sd-vae-ft-mse'  # 重命名为规范格式
    auto_save_interval: int = 300  # 自动保存间隔（秒）

    # ====================== 认知状态配置 ======================
    # 走神状态
    mind_wandering_enabled: bool = True          # 是否启用走神
    mind_wandering_idle_threshold: int = 30      # 空闲多少秒触发走神
    fatigue_sleep_threshold: float = 0.85         # 疲劳度达到多少触发睡眠
    mind_wandering_recall_prob: float = 0.5      # 记忆闪回概率
    mind_wandering_assoc_prob: float = 0.3       # 联想想象概率
    
    # 局部连接偏置（修复注释与默认值不匹配问题）
    local_bias_enabled: bool = True          # 是否启用局部连接偏置
    local_bias_strength: float = 1.2          # 同一分区内的偏置强度（1.0-2.0）
    cross_partition_decay: float = 0.7        # 跨分区的衰减系数（0.3-0.9，越小越稀疏）
    spatial_decay_enabled: bool = False       # 是否启用空间位置衰减（可选）
    spatial_radius: int = 5                  # 空间局部半径（仅spatial_decay_enabled=True时生效）

    # ====================== 🔴 实体提取配置（原配置升级） ======================
    entity_extraction_config: ClassVar[Dict[str, Any]] = {
        "min_entity_length": 2,           # 实体最小长度
        "max_entity_length": 20,          # 实体最大长度
        "use_quote_extraction": True,     # 是否提取引号中的内容
        "use_tag_extraction": True,        # 是否从标签中提取
        "use_triple_extraction": True,     # 是否从三元组中提取
        "exclude_tags": [],                # 排除的标签（空列表=不排除）
        "split_chars": ["，", "。", "！", "？", "；", "：", "、"],  # 分词字符
        "confidence_threshold": 0.7        # 实体提取置信度阈值
    }

    # ====================== 🔴 专家STDP统一配置（全专家覆盖） ======================
    EXPERT_CONFIG: ClassVar[Dict] = {
        "身份": {
            "sparsity": 0.05,
            "local_radius": 0.1,
            "core_bias": 0.8,
            "sdr_active_count": 15,
            # STDP配置：身份专家需要强记忆保持
            "stdp_enabled": True,
            "tau_plus": 25.0,      # 突触前到突触后的时间窗口
            "tau_minus": 20.0,     # 突触后到突触前的时间窗口
            "A_plus": 0.015,        # LTP（长时程增强）幅度
            "A_minus": 0.01,        # LTD（长时程抑制）幅度
            "learning_rate": 0.001  # 专家专属学习率
        },
        "概念": {
            "sparsity": 0.02,
            "local_radius": 0.2,
            "core_bias": 0.5,
            "sdr_active_count": 20,
            # STDP配置：概念专家需要平衡可塑性和稳定性
            "stdp_enabled": True,
            "tau_plus": 20.0,
            "tau_minus": 20.0,
            "A_plus": 0.01,
            "A_minus": 0.012,
            "learning_rate": 0.001
        },
        "空间": {
            "sparsity": 0.03,
            "local_radius": 0.35,
            "core_bias": 0.4,
            "sdr_active_count": 18,
            # STDP配置：空间专家需要强关联学习
            "stdp_enabled": True,
            "tau_plus": 30.0,
            "tau_minus": 25.0,
            "A_plus": 0.012,
            "A_minus": 0.01,
            "learning_rate": 0.001
        },
        "抽象": {
            "sparsity": 0.025,
            "local_radius": 0.25,
            "core_bias": 0.45,
            "sdr_active_count": 16,
            # STDP配置：抽象专家需要高可塑性
            "stdp_enabled": True,
            "tau_plus": 15.0,
            "tau_minus": 15.0,
            "A_plus": 0.018,
            "A_minus": 0.015,
            "learning_rate": 0.0015
        },
        "视觉": {
            "sparsity": 0.04,
            "local_radius": 0.15,
            "core_bias": 0.55,
            "sdr_active_count": 14,
            # STDP配置：视觉专家需要快速特征学习
            "stdp_enabled": True,
            "tau_plus": 10.0,
            "tau_minus": 10.0,
            "A_plus": 0.02,
            "A_minus": 0.018,
            "learning_rate": 0.002
        }
    }

    # ====================== 🔴 对话记忆配置（升级为实体关联） ======================
    CONVERSATION_MEMORY_CONFIG: ClassVar[Dict] = {
        "normal_decay_lambda": 0.1,    # 普通对话衰减系数（半衰期≈7小时）
        "important_decay_lambda": 0.02, # 重要对话衰减系数（半衰期≈35小时）
        "forget_threshold": 0.1,        # 遗忘阈值（低于此值不再出现在上下文）
        "max_context_turns": 8,         # 最多返回多少轮上下文
        "auto_cleanup_interval": 10,    # 每添加多少轮对话自动清理一次
        "entity_association_strength": 0.8,  # 对话与实体的关联强度
        "important_conversation_threshold": 0.7,  # 重要对话阈值（自动关联永久实体）
        "max_conversation_entities": 5  # 单轮对话最多关联实体数
    }

    # ====================== 🔴 海马体配置（统一收敛硬编码参数） ======================
    HIPPOCAMPUS_CONFIG: ClassVar[Dict] = {
        "buffer_size": 20,              # 海马体临时实体缓冲区大小
        "consolidation_rate": 0.2,       # 单次睡眠巩固进度
        "separation_threshold": 0.85,    # 模式分离阈值
        "pattern_completion_threshold": 0.7,  # 模式补全阈值
        "replay_priority_weight": 0.5,   # 回放时重要性权重
        "replay_recency_weight": 0.5     # 回放时新鲜度权重
    }

    # ====================== 🔴 跨模态学习配置 ======================
    CROSS_MODAL_CONFIG: ClassVar[Dict] = {
        "enabled": True,                # 是否启用跨模态学习
        "learning_rate": 0.001,          # 跨模态脑桥学习率
        "loss_weight_text": 0.5,         # 文本模态损失权重
        "loss_weight_vision": 0.5,       # 视觉模态损失权重
        "association_threshold": 0.6,    # 跨模态关联阈值
        "max_cross_modal_pairs": 1000,   # 最多保存跨模态配对数
        "dream_visual_noise_level": 0.2  # 梦境视觉生成噪声水平
    }

    # ====================== 🔴 元认知与多巴胺配置 ======================
    METACOGNITION_CONFIG: ClassVar[Dict] = {
        "confidence_decay_rate": 0.001,  # 知识置信度每日衰减率
        "min_confidence": 0.1,           # 最低置信度
        "max_confidence": 0.99,          # 最高置信度
        "learning_confidence_boost": 0.1, # 学习后置信度提升
        "consolidation_confidence_boost": 0.05  # 巩固后置信度提升
    }

    DOPAMINE_CONFIG: ClassVar[Dict] = {
        "baseline_dopamine": 0.5,        # 基础多巴胺水平
        "curiosity_reward_scale": 1.0,    # 好奇心奖励系数
        "prediction_reward_scale": 1.0,   # 预测准确奖励系数
        "external_reward_scale": 1.5,     # 外部反馈奖励系数
        "rpe_decay": 0.9,                # 奖励预测误差衰减率
        "max_reward": 1.0,                # 最大奖励值
        "min_reward": -1.0                # 最小惩罚值
    }

    # ====================== 🔴 好奇心配置 ======================
    CURIOSITY_CONFIG: ClassVar[Dict] = {
        "enabled": True,                 # 是否启用好奇心
        "trigger_threshold": 0.3,         # 知识缺口触发阈值
        "max_questions_per_trigger": 3,   # 单次触发最多生成问题数
        "question_cooldown": 300,         # 提问冷却时间（秒）
        "novelty_weight": 0.6,            # 新颖度权重
        "complexity_weight": 0.4          # 复杂度权重
    }

    # ====================== 🔴 梦境与睡眠配置 ======================
    DREAM_CONFIG: ClassVar[Dict] = {
        "enabled": True,                 # 是否启用梦境生成
        "default_dream_length": 3,        # 默认梦境片段数
        "default_surrealism": 0.3,        # 默认梦境荒诞度
        "max_dream_depth": 3,             # 最大神经激活传播深度
        "dream_consolidation_rate": 0.1,  # 梦境记忆巩固率
        "visual_dream_enabled": True      # 是否启用视觉梦境
    }

    SLEEP_CONFIG: ClassVar[Dict] = {
        "default_epochs": 3,              # 默认睡眠轮数
        "manual_sleep_multiplier": 1.5,   # 手动睡眠轮数倍数
        "light_sleep_duration": 2,        # 浅睡阶段时长（秒）
        "deep_sleep_duration": 5,         # 深睡阶段时长（秒）
        "rem_sleep_duration": 3,          # REM阶段时长（秒）
        "optimal_forget_rate": 0.15,      # 最佳遗忘率（10-20%）
        "synapse_pruning_rate": 0.05      # 每次睡眠突触修剪率
    }

    class Config:
        env_file = ".env"
        extra = "forbid"  # 禁止未定义的配置项，防止拼写错误

# 全局配置实例
config = BrainConfig()