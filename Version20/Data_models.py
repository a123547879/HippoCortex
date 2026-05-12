# data_models.py
from __future__ import annotations
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

# ===================== 基础通用模型 =====================
class BaseDataModel(BaseModel):
    """所有数据契约的基类，统一配置"""
    model_config = ConfigDict(
        extra="forbid",  # 禁止传入未定义的字段，提前发现错误
        frozen=False,    # 允许修改字段值
        populate_by_name=True  # 支持通过别名赋值
    )

# ===================== 记忆相关模型 =====================
class MemoryPacket(BaseDataModel):
    """
    统一记忆数据包：所有模块间传递的记忆都必须使用这个结构
    替代原来的裸字典：{"id": "", "content": "", "metadata": {}, ...}
    """
    mem_id: int = Field(description="记忆唯一ID（整数）")
    content: str = Field(description="记忆文本内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="记忆元数据")
    sdr: Optional[Any] = Field(None, description="记忆的稀疏分布式表示（SDR向量）")
    clip_vec: Optional[Any] = Field(None, description="记忆的CLIP向量")
    created_at: float = Field(default_factory=lambda: datetime.now().timestamp(), description="创建时间戳")
    consolidation_level: float = Field(0.0, ge=0.0, le=1.0, description="巩固程度（0-1）")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="重要性评分")

    # 👇 新增配置：允许额外字段 + 支持张量（核心修复）
    class Config:
        extra = "allow"  # 允许额外输入，关闭严格校验
        arbitrary_types_allowed = True  # 支持 torch.Tensor 类型

    @property
    def expert(self) -> str:
        """快捷获取所属专家"""
        return self.metadata.get("expert", "概念")

    @property
    def access_count(self) -> int:
        """快捷获取访问次数"""
        return self.metadata.get("access_count", 0)

# ===================== 新增：对话轮次模型 =====================
class ConversationTurn(BaseDataModel):
    """
    对话轮次数据契约：所有对话历史都必须使用这个结构
    替代原来的裸字典：{"user_input": "", "ai_response": "", ...}
    """
    id: str = Field(description="对话轮次唯一ID")
    user_input: str = Field(description="用户输入内容")
    ai_response: str = Field(description="AI回复内容")
    timestamp: float = Field(description="时间戳（秒）")
    initial_activation: float = Field(1.0, description="初始激活值")
    is_important: bool = Field(False, description="是否为重要对话")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

# ===================== 思考相关模型 =====================
class ThoughtResult(BaseDataModel):
    """
    思考结果：PerceptionLoop.think() 方法的返回值
    替代原来的裸字典返回
    """
    thought_chain: str = Field(description="连贯的思考链")
    core_ideas: List[str] = Field(description="提取的核心思想")
    activated_memories: List[str] = Field(description="激活的记忆内容列表")
    expert: str = Field(description="负责处理的专家名称")
    activation_strength: float = Field(description="整体激活强度")
    predicted_memory: Optional[str] = Field(None, description="预测的下一个记忆")
    prediction_error: float = Field(0.0, description="预测误差")
    symbolic_context: str = Field("", description="符号检索上下文")
    energy_detail: Dict[str, float] = Field(default_factory=dict, description="能量消耗详情")
    error: Optional[str] = Field(None, description="错误信息（如果有）")

# ===================== 意图相关模型 =====================
class Intention(BaseDataModel):
    """
    意图对象：所有主动意图都必须使用这个结构
    替代原来的裸字典：{"type": "", "priority": 0.0, "content": "", ...}
    """
    type: str = Field(description="意图类型：physiological/cognitive/social/exploration")
    priority: float = Field(ge=0.0, le=2.0, description="意图优先级")
    content: str = Field(description="意图显示内容")
    action: str = Field(description="要执行的动作")
    context: Dict[str, Any] = Field(default_factory=dict, description="动作上下文")
    need_sleep: bool = Field(False, description="是否需要触发睡眠")
    executed: bool = Field(False, description="是否已执行")
    result: Optional[str] = Field(None, description="执行结果")

# ===================== 睡眠相关模型 =====================
class SleepStageReport(BaseDataModel):
    """单个睡眠阶段的报告"""
    important_conversations_consolidated: Optional[int] = Field(None, description="浅睡阶段：巩固的对话数")
    high_priority_count: Optional[int] = Field(None, description="深睡阶段：高优先级记忆数")
    thalamus_consolidated: Optional[int] = Field(None, description="深睡阶段：丘脑巩固数")
    expert_consolidated: Optional[int] = Field(None, description="深睡阶段：专家巩固数")
    consolidated: Optional[int] = Field(None, description="本阶段总巩固数")
    forgotten: Optional[int] = Field(None, description="本阶段总遗忘数")
    dream_generated: Optional[bool] = Field(None, description="REM阶段：是否生成梦境")
    dream_content: Optional[str] = Field(None, description="REM阶段：梦境内容")
    associations_created: Optional[int] = Field(None, description="REM阶段：创建的关联数")

class SleepReport(BaseDataModel):
    """
    完整睡眠报告：ConsolidationLoop.sleep_consolidate_all() 的返回值
    替代原来的裸字典返回
    """
    stages: Dict[str, SleepStageReport] = Field(default_factory=dict, description="各阶段报告")
    total_memories: int = Field(0, description="睡眠前总记忆数")
    consolidated_count: int = Field(0, description="总巩固记忆数")
    forgotten_count: int = Field(0, description="总遗忘记忆数")
    energy_consumed: float = Field(0.0, description="总能量消耗")
    sleep_duration: float = Field(0.0, description="睡眠时长（秒）")
    quality_score: float = Field(0.0, ge=0.0, le=100.0, description="睡眠质量评分（0-100）")
    quality_rating: str = Field("", description="质量评级：极佳/良好/一般/较差")
    dream_content: str = Field("", description="最终梦境内容")
    is_manual: bool = Field(False, description="是否为手动触发的睡眠")
    error: Optional[str] = Field(None, description="错误信息（如果有）")

    @property
    def consolidation_rate(self) -> float:
        """计算巩固率"""
        if self.total_memories == 0:
            return 0.0
        return round(self.consolidated_count / self.total_memories, 2)

# ===================== 梦境相关模型 =====================
class DreamFragment(BaseDataModel):
    """单个梦境片段数据契约"""
    content: str = Field(description="片段内容")
    expert: str = Field(description="来源专家名称")
    activation_score: float = Field(description="激活度评分")

class DreamResult(BaseDataModel):
    """
    梦境生成结果：DreamingLoop.generate_dream() 和 generate_global_dream() 的统一返回值
    替代原来的裸字典返回
    """
    success: bool = Field(description="是否生成成功")
    content: str = Field(description="完整梦境内容")
    fragments: List[DreamFragment] = Field(default_factory=list, description="梦境片段列表")
    dominant_experts: List[str] = Field(default_factory=list, description="主导脑区")
    expert_dreams: Dict[str, str] = Field(default_factory=dict, description="各专家的独立梦境")
    used_high_priority_memories: List[str] = Field(default_factory=list, description="使用的高优先级记忆")
    mem_id: Optional[int] = Field(None, description="梦境记忆的ID（整数）")
    error: Optional[str] = Field(None, description="错误信息（如果有）")