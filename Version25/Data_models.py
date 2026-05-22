from __future__ import annotations
from typing import List, Dict, Optional, Any, Set
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, computed_field
import uuid
import time
import torch

# ===================== 基础通用模型（保留并强化） =====================
class BaseDataModel(BaseModel):
    """所有数据契约的基类，统一配置与工具方法"""
    model_config = ConfigDict(
        extra="ignore",          # 全局自动忽略所有额外字段，彻底解决extra_forbidden
        frozen=False,
        populate_by_name=True,
        arbitrary_types_allowed=True,
        validate_assignment=True # 赋值时自动验证
    )

    def to_dict(self) -> dict:
        """统一序列化：递归处理所有张量/数组/集合类型 + 排除所有计算属性"""
        # ✅ 修复：手动列出所有计算属性（兼容所有Pydantic 2.x版本）
        # 注意：添加新的@computed_field时，需要在这里添加对应的字段名
        computed_fields = {
            # Entity 计算属性
            "latest_evidence", "semantic_vec",
            # EntityRelation 计算属性
            "weight",
            # ConversationTurn 计算属性
            "id", "activation",
            # SleepStageReport 计算属性
            "thalamus_consolidated", "expert_consolidated", 
            "cross_modal_consolidated", "consolidated", "forgotten",
            # SleepReport 计算属性
            "dream_content", "total_memories", "consolidation_rate"
        }
        
        data = self.model_dump(exclude=computed_fields)
        
        def _process_value(v):
            if hasattr(v, "tolist"):
                return v.tolist()
            elif isinstance(v, set):
                return list(v)
            elif isinstance(v, dict):
                return {k: _process_value(val) for k, val in v.items()}
            elif isinstance(v, list):
                return [_process_value(item) for item in v]
            return v
        
        return {k: _process_value(v) for k, v in data.items()}
    
    @classmethod
    def from_dict(cls, data: dict) -> "BaseDataModel":
        """统一反序列化入口：自动过滤无效字段 + 兼容旧数据"""
        # 只保留模型中实际定义的字段，提供双重保险
        valid_fields = cls.model_fields.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

# ===================== 🔴 核心：实体模型（唯一记忆单位） =====================
class Entity(BaseDataModel):
    """
    记忆系统的唯一基本单位
    所有知识、关系、证据都依附于实体存在
    ✅ 核心字段不可变 ✅ 自动生命周期管理 ✅ 统一神经表示
    """
    # ------------------------------
    # 🔒 核心不可变字段（创建后永不修改）
    # ------------------------------
    entity_id: str = Field(
        description="全局唯一ID：类型_名称哈希_时间戳，如 person_123456_1716000000",
        frozen=True,
        pattern=r"^[a-z]+_[0-9a-f]+_[0-9]+$"
    )
    name: str = Field(
        description="实体标准名称（唯一）",
        frozen=True,
        min_length=1,
        max_length=100
    )
    entity_type: str = Field(
        description="实体类型枚举",
        frozen=True,
        pattern=r"^(person|place|event|concept|object|skill|system|emotion|identity|visual)$"
    )
    created_at: float = Field(
        default_factory=lambda: datetime.now().timestamp(),
        description="创建时间戳（秒）",
        frozen=True
    )

    # ------------------------------
    # 🧠 神经表示（不可变，创建时生成）
    # ------------------------------
    sdr: torch.Tensor = Field(
        description="稀疏分布式表示（神经检索专用）",
        frozen=True
    )
    clip_vec: torch.Tensor = Field(
        description="CLIP语义向量（语义检索专用）",
        frozen=True
    )

    # ------------------------------
    # ⚙️ 可变核心字段（生命周期管理）
    # ------------------------------
    importance: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="重要性评分（多巴胺系统）"
    )
    is_permanent: bool = Field(
        False,
        description="永久记忆标记（免疫遗忘）"
    )
    is_obsolete: bool = Field(
        False,
        description="过时标记（软删除）"
    )
    consolidation_level: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="海马体巩固进度（0.0=未巩固，1.0=已写入皮层）"
    )

    # ------------------------------
    # 📦 结构化知识
    # ------------------------------
    attributes: Dict[str, str] = Field(
        default_factory=dict,
        description="属性键值对：{'年龄': '25', '职业': '程序员'}"
    )
    aliases: Set[str] = Field(
        default_factory=set,
        description="别名集合：{'小白', '白总'}"
    )
    tags: Set[str] = Field(
        default_factory=set,
        description="标签集合：{'主人', '喜好', '重要'}"
    )

    # ------------------------------
    # 📜 全文本证据（原全文本记忆）
    # ------------------------------
    evidences: List[Evidence] = Field(
        default_factory=list,
        description="支持该实体的全文本证据列表"
    )

    # ------------------------------
    # 📊 生命周期统计
    # ------------------------------
    last_accessed: float = Field(
        default_factory=lambda: datetime.now().timestamp(),
        description="最后访问时间戳"
    )
    access_count: int = Field(
        0,
        ge=0,
        description="总访问次数"
    )
    replay_count: int = Field(
        0,
        ge=0,
        description="睡眠回放次数"
    )
    relation_count: int = Field(
        0,
        ge=0,
        description="关联关系总数"
    )

    # ------------------------------
    # 🔧 扩展元数据
    # ------------------------------
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="扩展元数据"
    )

    expert: Optional[str] = Field(None, description="所属专家分类")

    # ------------------------------
    # ✅ 兼容属性（向后兼容）
    # ------------------------------
    @computed_field
    @property
    def latest_evidence(self) -> Optional[Evidence]:
        """获取最新的证据（兼容旧代码）"""
        return self.evidences[-1] if self.evidences else None

    @computed_field
    @property
    def semantic_vec(self) -> torch.Tensor:
        """语义向量别名（兼容新命名）"""
        return self.clip_vec

    # ------------------------------
    # ✅ 核心业务方法
    # ------------------------------
    def add_evidence(self, evidence: Evidence, max_keep: int = 10) -> None:
        """添加证据，自动保留最新的N条"""
        self.evidences.append(evidence)
        if len(self.evidences) > max_keep:
            self.evidences = self.evidences[-max_keep:]

    def update_attribute(self, key: str, value: str, overwrite: bool = True) -> bool:
        """更新属性，返回是否成功"""
        if key in self.attributes and not overwrite:
            return False
        self.attributes[key] = value
        return True

    def increment_access(self) -> None:
        """更新访问状态，自动提升重要性"""
        self.access_count += 1
        self.last_accessed = datetime.now().timestamp()
        # 访问越多越重要，上限0.95
        self.importance = min(0.95, self.importance + 0.02)

    # ✅ 兼容：旧代码的update_access方法
    def update_access(self) -> None:
        """更新访问状态（兼容旧代码）"""
        self.increment_access()

    def mark_obsolete(self) -> None:
        """标记为过时（软删除）"""
        self.is_obsolete = True
        self.metadata["obsolete_time"] = datetime.now().timestamp()

    # ------------------------------
    # ✅ 序列化/反序列化重写
    # ------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        # 自动删除所有计算属性（双重保险）
        data.pop("latest_evidence", None)
        data.pop("semantic_vec", None)
        
        # 兼容旧数据：semantic_vec → clip_vec
        if "semantic_vec" in data and "clip_vec" not in data:
            data["clip_vec"] = data.pop("semantic_vec")
        
        # 兼容旧数据：update_time → last_accessed
        if "update_time" in data and "last_accessed" not in data:
            data["last_accessed"] = data.pop("update_time")
        
        # 还原张量
        if isinstance(data.get("sdr"), list):
            data["sdr"] = torch.tensor(data["sdr"], dtype=torch.float32)
        if isinstance(data.get("clip_vec"), list):
            data["clip_vec"] = torch.tensor(data["clip_vec"], dtype=torch.float32)
        
        # 还原证据列表
        if "evidences" in data and isinstance(data["evidences"], list):
            data["evidences"] = [Evidence.from_dict(ev) for ev in data["evidences"]]
        
        # 还原集合类型
        if isinstance(data.get("aliases"), list):
            data["aliases"] = set(data["aliases"])
        if isinstance(data.get("tags"), list):
            data["tags"] = set(data["tags"])
        
        return super().from_dict(data)

# ===================== 🔴 核心：实体关系模型（神经突触） =====================
class EntityRelation(BaseDataModel):
    """
    实体间的关系 = 显式知识 + 神经突触权重
    同时支持：1. 结构化关系查询 2. 神经激活传播
    """
    # ------------------------------
    # 🔒 核心不可变字段
    # ------------------------------
    relation_id: str = Field(
        default_factory=lambda: f"rel_{uuid.uuid4().hex[:12]}",
        description="关系唯一ID",
        frozen=True
    )
    subject_id: str = Field(
        description="主体实体ID",
        frozen=True
    )
    predicate: str = Field(
        description="关系谓词：'住在'、'喜欢'、'是'、'拥有'、'认识'",
        frozen=True,
        min_length=1
    )
    object_id: str = Field(
        description="客体实体ID",
        frozen=True
    )
    created_at: float = Field(
        default_factory=lambda: datetime.now().timestamp(),
        description="创建时间戳",
        frozen=True
    )

    # ------------------------------
    # ⚙️ 可变字段（神经学习）
    # ------------------------------
    confidence: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="关系置信度"
    )
    synapse_weight: float = Field(
        0.1,
        ge=-1.0,
        le=1.0,
        description="神经突触权重（激活传播核心）"
    )
    access_count: int = Field(
        0,
        ge=0,
        description="关系被访问次数"
    )
    last_accessed: float = Field(
        default_factory=lambda: datetime.now().timestamp(),
        description="最后访问时间戳"
    )

    # ------------------------------
    # 🔧 元数据
    # ------------------------------
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="支持该关系的证据ID列表"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="扩展元数据"
    )

    # ------------------------------
    # ✅ 兼容属性
    # ------------------------------
    @computed_field
    @property
    def weight(self) -> float:
        """突触权重别名（兼容旧代码）"""
        return self.synapse_weight

    # ------------------------------
    # ✅ 核心方法
    # ------------------------------
    def update_synapse(self, delta: float) -> None:
        """更新突触权重，自动范围限制"""
        self.synapse_weight = max(-1.0, min(1.0, self.synapse_weight + delta))
        self.access_count += 1
        self.last_accessed = datetime.now().timestamp()

    # ------------------------------
    # ✅ 反序列化重写（兼容旧数据）
    # ------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> "EntityRelation":
        # 兼容旧数据：weight → synapse_weight
        if "weight" in data and "synapse_weight" not in data:
            data["synapse_weight"] = data.pop("weight")
        
        return super().from_dict(data)

# ===================== 🔴 核心：全文本证据模型 =====================
class Evidence(BaseDataModel):
    """
    全文本证据：所有原始文本的最终归宿
    不再是独立记忆，而是支持实体和关系存在的依据
    """
    # ------------------------------
    # 🔒 核心不可变字段
    # ------------------------------
    evidence_id: str = Field(
        default_factory=lambda: f"ev_{uuid.uuid4().hex[:12]}",
        description="证据唯一ID",
        frozen=True
    )
    content: str = Field(
        description="原始全文本内容",
        frozen=True
    )
    source: str = Field(
        description="来源：对话/导入/梦境/观察/系统/学习",
        frozen=True,
        pattern=r"^(对话|导入|梦境|观察|系统|学习)$"
    )
    created_at: float = Field(
        default_factory=lambda: datetime.now().timestamp(),
        description="创建时间戳",
        frozen=True
    )

    # ------------------------------
    # 🧠 神经表示（不可变）
    # ------------------------------
    sdr: torch.Tensor = Field(
        description="文本的SDR向量",
        frozen=True
    )
    clip_vec: torch.Tensor = Field(
        description="文本的CLIP向量",
        frozen=True
    )

    # ------------------------------
    # ⚙️ 可变字段
    # ------------------------------
    confidence: float = Field(
        0.95,
        ge=0.0,
        le=1.0,
        description="证据置信度"
    )
    emotion_valence: float = Field(
        0.0,
        ge=-1.0,
        le=1.0,
        description="情绪效价（-1负面 ~ 1正面）"
    )
    emotion_arousal: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="情绪唤醒度（0平静 ~ 1激动）"
    )

    # ------------------------------
    # 🔧 元数据
    # ------------------------------
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="扩展元数据"
    )

    # ------------------------------
    # ✅ 序列化/反序列化重写
    # ------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> "Evidence":
        if isinstance(data.get("sdr"), list):
            data["sdr"] = torch.tensor(data["sdr"], dtype=torch.float32)
        if isinstance(data.get("clip_vec"), list):
            data["clip_vec"] = torch.tensor(data["clip_vec"], dtype=torch.float32)
        return super().from_dict(data)

# ===================== 统一工厂类（唯一创建入口） =====================
class MemoryFactory:
    """
    所有记忆相关对象的唯一创建工厂
    集中处理ID生成、默认值、校验和初始化逻辑
    """
    @staticmethod
    def create_entity(
        name: str,
        entity_type: str,
        sdr: torch.Tensor,
        clip_vec: torch.Tensor,
        importance: float = 0.5,
        **kwargs
    ) -> Entity:
        """创建新实体"""
        # 生成全局唯一ID：类型_名称哈希_时间戳
        name_hash = format(hash(name) & 0xfffffff, 'x')
        timestamp = int(datetime.now().timestamp())
        entity_id = f"{entity_type}_{name_hash}_{timestamp}"

        return Entity(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            sdr=sdr,
            clip_vec=clip_vec,
            importance=importance,
            **kwargs
        )

    @staticmethod
    def create_relation(
        subject: Entity,
        predicate: str,
        object: Entity,
        confidence: float = 0.9,
        initial_weight: float = 0.1,
        evidence: Optional[Evidence] = None,
        **kwargs
    ) -> EntityRelation:
        """创建实体间关系"""
        relation = EntityRelation(
            subject_id=subject.entity_id,
            predicate=predicate,
            object_id=object.entity_id,
            confidence=confidence,
            synapse_weight=initial_weight,
            **kwargs
        )
        
        if evidence:
            relation.evidence_ids.append(evidence.evidence_id)
        
        # 更新实体的关系计数
        subject.relation_count += 1
        object.relation_count += 1
        
        return relation

    @staticmethod
    def create_evidence(
        content: str,
        source: str,
        sdr: torch.Tensor,
        clip_vec: torch.Tensor,
        confidence: float = 0.95,
        emotion_valence: float = 0.0,
        emotion_arousal: float = 0.5,
        **kwargs
    ) -> Evidence:
        """创建全文本证据"""
        return Evidence(
            content=content,
            source=source,
            sdr=sdr,
            clip_vec=clip_vec,
            confidence=confidence,
            emotion_valence=emotion_valence,
            emotion_arousal=emotion_arousal,
            **kwargs
        )

    # ✅ 兼容：旧代码的create_hippocampus_memory方法
    @staticmethod
    def create_hippocampus_memory(
        content: str,
        sdr: torch.Tensor,
        clip_vec: torch.Tensor,
        expert: str,
        **kwargs
    ) -> Entity:
        """创建海马体临时实体（兼容旧代码）"""
        # 从内容提取简单名称
        name = content.strip()[:20]
        # 自动映射专家到实体类型
        type_map = {
            "身份": "identity",
            "视觉": "visual",
            "概念": "concept",
            "空间": "place",
            "抽象": "abstract"
        }
        entity_type = type_map.get(expert, "concept")
        
        return MemoryFactory.create_entity(
            name=name,
            entity_type=entity_type,
            sdr=sdr,
            clip_vec=clip_vec,
            expert=expert,
            **kwargs
        )

    # ✅ 兼容：旧代码的create_from_dict方法
    @staticmethod
    def create_entity_from_dict(data: dict) -> Entity:
        """从字典创建实体（兼容旧代码）"""
        return Entity.from_dict(data)

# ===================== 业务模型（全部适配实体体系·100%兼容） =====================
class ConversationTurn(BaseDataModel):
    """对话轮次数据契约"""
    turn_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        frozen=True
    )
    user_input: str = Field(frozen=True)
    ai_response: str = Field(frozen=True)
    timestamp: float = Field(
        default_factory=time.time,
        frozen=True
    )
    
    # 实体关联
    extracted_entity_ids: List[str] = Field(
        default_factory=list,
        description="本轮对话提取的实体ID"
    )
    activated_entity_ids: List[str] = Field(
        default_factory=list,
        description="本轮对话激活的实体ID"
    )
    
    # 状态字段
    initial_activation: float = Field(1.0, ge=0.0, le=1.0)
    is_important: bool = Field(False)
    is_consolidated: bool = Field(False)
    
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # ✅ 兼容：旧代码的id和activation字段
    @computed_field
    @property
    def id(self) -> str:
        return self.turn_id

    @computed_field
    @property
    def activation(self) -> float:
        return self.initial_activation

    def __str__(self) -> str:
        return f"Turn({self.turn_id[:8]}: U='{self.user_input[:20]}...', A='{self.ai_response[:20]}...')"

    # ------------------------------
    # ✅ 反序列化重写（兼容旧数据）
    # ------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> "ConversationTurn":
        # 兼容旧数据：id → turn_id
        if "id" in data and "turn_id" not in data:
            data["turn_id"] = data.pop("id")
        
        # 兼容旧数据：activation → initial_activation
        if "activation" in data and "initial_activation" not in data:
            data["initial_activation"] = data.pop("activation")
        
        return super().from_dict(data)

class ThoughtResult(BaseDataModel):
    """思考结果数据契约"""
    thought_chain: str = Field("", frozen=True)
    core_ideas: List[str] = Field(default_factory=list, frozen=True)
    
    # 新架构标准字段
    activated_entity_ids: List[str] = Field(default_factory=list, frozen=True)
    activated_relation_ids: List[str] = Field(default_factory=list, frozen=True)
    
    # ✅ 普通可写字段（完全兼容旧代码数据结构）
    activated_memories: List[str] = Field(default_factory=list, frozen=True)
    activated_entities: List[Dict[str, str]] = Field(default_factory=list, frozen=True)
    
    expert: str = Field("", frozen=True)
    activation_strength: float = Field(0.0, frozen=True)
    prediction_error: float = Field(0.0, frozen=True)
    symbolic_context: str = Field("", frozen=True)
    energy_detail: Dict[str, float] = Field(default_factory=dict, frozen=True)
    error: Optional[str] = Field(None, frozen=True)
    predicted_memory: Optional[str] = Field(None, frozen=True)

    # ✅ 错误结果快速创建方法
    @classmethod
    def error_result(cls, error_msg: str) -> "ThoughtResult":
        return cls(
            thought_chain=f"思考出错：{error_msg}",
            core_ideas=["思考过程发生错误"],
            error=error_msg
        )

class Intention(BaseDataModel):
    """主动意图数据契约"""
    type: str = Field(frozen=True)
    priority: float = Field(ge=0.0, le=2.0, frozen=True)
    content: str = Field(frozen=True)
    action: str = Field(frozen=True)
    
    # 实体关联
    target_entity_ids: List[str] = Field(default_factory=list, frozen=True)
    
    context: Dict[str, Any] = Field(default_factory=dict, frozen=True)
    need_sleep: bool = Field(False, frozen=True)
    
    executed: bool = Field(False)
    result: Optional[str] = Field(None)

class SleepStageReport(BaseDataModel):
    """单个睡眠阶段报告"""
    entities_consolidated: int = Field(0, frozen=True)
    relations_updated: int = Field(0, frozen=True)
    evidences_archived: int = Field(0, frozen=True)
    entities_forgotten: int = Field(0, frozen=True)
    relations_pruned: int = Field(0, frozen=True)
    
    synapses_pruned: int = Field(0, frozen=True)
    synapses_created: int = Field(0, frozen=True)
    
    dream_generated: bool = Field(False, frozen=True)
    dream_content: Optional[str] = Field(None, frozen=True)
    dream_entity_count: int = Field(0, frozen=True)

    # ✅ 兼容：旧代码的统计字段
    @computed_field
    @property
    def thalamus_consolidated(self) -> int:
        return self.entities_consolidated

    @computed_field
    @property
    def expert_consolidated(self) -> int:
        return self.entities_consolidated

    @computed_field
    @property
    def cross_modal_consolidated(self) -> int:
        return self.relations_updated

    @computed_field
    @property
    def consolidated(self) -> int:
        return self.entities_consolidated

    @computed_field
    @property
    def forgotten(self) -> int:
        return self.entities_forgotten

class SleepReport(BaseDataModel):
    """完整睡眠报告"""
    stages: Dict[str, SleepStageReport] = Field(default_factory=dict, frozen=True)
    
    total_entities: int = Field(0, frozen=True)
    total_relations: int = Field(0, frozen=True)
    total_evidences: int = Field(0, frozen=True)
    
    consolidated_count: int = Field(0, frozen=True)
    forgotten_count: int = Field(0, frozen=True)
    
    energy_consumed: float = Field(0.0, frozen=True)
    sleep_duration: float = Field(0.0, frozen=True)
    quality_score: float = Field(0.0, ge=0.0, le=100.0)
    quality_rating: str = Field("")
    
    final_dream: str = Field("", frozen=True)
    is_manual: bool = Field(False, frozen=True)
    error: Optional[str] = Field(None, frozen=True)

    # ✅ 兼容：旧代码的dream_content和total_memories字段
    @computed_field
    @property
    def dream_content(self) -> str:
        return self.final_dream

    @computed_field
    @property
    def total_memories(self) -> int:
        return self.total_entities

    @property
    def consolidation_rate(self) -> float:
        return round(self.consolidated_count / max(self.total_entities, 1), 2)

class DreamFragment(BaseDataModel):
    """梦境片段数据契约"""
    content: str = Field(frozen=True)
    source_entity_id: str = Field(frozen=True)
    activation_score: float = Field(frozen=True)
    expert: str = Field(frozen=True)

class DreamResult(BaseDataModel):
    """梦境生成结果"""
    success: bool = Field(frozen=True)
    content: str = Field(frozen=True)
    fragments: List[DreamFragment] = Field(default_factory=list, frozen=True)
    
    dominant_entity_ids: List[str] = Field(default_factory=list, frozen=True)
    dominant_experts: List[str] = Field(default_factory=list, frozen=True)
    
    used_high_priority_entities: List[str] = Field(default_factory=list, frozen=True)
    error: Optional[str] = Field(None, frozen=True)

    # ✅ 兼容：旧代码的main_entity_id、visual、expert_dreams字段
    main_entity_id: Optional[str] = Field(None, frozen=True)
    visual: Optional[Dict[str, Any]] = Field(None, frozen=True)
    expert_dreams: Dict[str, str] = Field(default_factory=dict, frozen=True)