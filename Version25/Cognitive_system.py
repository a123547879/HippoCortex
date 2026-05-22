from BrainConfig import config
import os
import logging, time
from typing import Dict, List
from brain_core import BrainCore
from ILifecycle import ILifecycle
from event_system import EventBus, Event, EventType
from ComponentFactory import ComponentFactory

logger = logging.getLogger("CognitiveSystem")

class CognitiveSystem:
    """
    认知系统外观类
    不再持有组件引用，只作为服务容器的门面和对外统一接口
    """
    
    def __init__(self, embedding_model, llm, kg_enabled: bool = True):
        # 使用工厂构建服务容器
        self.container = ComponentFactory.build_container(embedding_model, llm, kg_enabled)
        
        # 兼容原有属性
        self.fatigue_sleep_threshold = 0.85
        self.needs_sleep_request = False
    
    # ===================== 🔥 新增：完整兼容属性区（彻底解决所有AttributeError） =====================
    @property
    def kg_enabled(self):
        """知识图谱开关兼容属性"""
        return self.container.kg_enabled
    
    @kg_enabled.setter
    def kg_enabled(self, value):
        self.container.kg_enabled = value
        # 初始化完成后同步更新cortex的kg_enabled
        if hasattr(self.container, '_initialized') and self.container._initialized:
            if "cortex" in self.container._services:
                self.container["cortex"].cortex.kg_enabled = value
    
    @property
    def cortex(self):
        """持久化皮层兼容属性"""
        return self.container["cortex"].cortex
    
    @property
    def experts(self):
        """专家网络兼容属性"""
        return self.container["experts"]
    
    @property
    def expert_names(self):
        """专家名称列表兼容属性（完全兼容原有代码）"""
        return list(self.container["experts"].keys())
    
    @property
    def sdr_encoders(self):
        """稀疏编码器兼容属性"""
        return self.container["sdr_encoders"]
    
    @property
    def hippocampus_router(self):
        """海马体路由兼容属性"""
        return self.container["hippocampus_router"].router
    
    @property
    def symbolic_core(self):
        """符号核心兼容属性（支持可选组件）"""
        if "symbolic_core" in self.container._services:
            return self.container["symbolic_core"].symbolic_core
        return None
    
    @property
    def vae_manager(self):
        """VAE管理器兼容属性（支持可选组件）"""
        if "vae_manager" in self.container._services:
            return self.container["vae_manager"].manager
        return None
    
    @property
    def thalamus(self):
        """丘脑兼容属性"""
        return self.container["thalamus"].thalamus
    
    @property
    def perception_loop(self):
        """感知循环兼容属性（完全兼容原有代码）"""
        return self.container["perception_loop"]
    
    @property
    def learning_loop(self):
        """学习循环兼容属性"""
        return self.container["learning_loop"]
    
    @property
    def dreaming_loop(self):
        """梦境循环兼容属性"""
        return self.container["dreaming_loop"]
    
    @property
    def consolidation_loop(self):
        """巩固循环兼容属性"""
        return self.container["consolidation_loop"]
    
    @property
    def cross_modal_bridge(self):
        """跨模态脑桥兼容属性（新增功能）"""
        return self.container["cross_modal_bridge"]
    
    @property
    def brain_core(self):
        """大脑核心兼容属性"""
        return self.container["brain_core"]
    # ================================================================================================
    
    def initialize(self, storage_dir: str):
        """初始化整个认知系统"""
        logger.info("🧠 初始化认知系统...")
        self.container.initialize_all(storage_dir)
        self.container.start_all()
        logger.info("✅ 认知系统初始化完成")
    
    def shutdown(self):
        """关闭整个认知系统"""
        logger.info("🔌 关闭认知系统...")
        self.container.stop_all()
        self.container.save_all()
        logger.info("✅ 认知系统已安全关闭")
    
    # ===================== 对外统一接口（保持与原代码完全一致） =====================
    def think(self, text: str, steps: int = 2, topk: int = 30, expert_last=None) -> Dict:
        return self.container["perception_loop"].think(text, steps, topk, expert_last)
    
    def learn(self, text: str, force_expert=None, external_reward=0.0):
        return self.container["learning_loop"].learn(text, force_expert, external_reward)
    
    def batch_learn(self, texts: List[str]) -> List[int]:
        return self.container["learning_loop"].batch_init_direct_to_cortex(texts)
    
    def sleep_consolidate_all(self, epochs=3, is_manual: bool = False):
        return self.container["consolidation_loop"].sleep_consolidate_all(epochs, is_manual)
    
    def generate_dream(self, dream_length: int = 3) -> dict:
        return self.container["dreaming_loop"].generate_dream(dream_length)
    
    def get_brain_status(self):
        return self.container["perception_loop"].get_brain_status()
    
    def bind_related_memories(self, new_mem_id, new_mem_vec, new_mem_text, target_expert, user_input):
        return self.container["learning_loop"].bind_related_memories(
            new_mem_id, new_mem_vec, new_mem_text, target_expert, user_input
        )
    
    def create_synapse(self, from_mem_id: str, to_mem_id: str, weight: float = 0.3):
        return self.container["learning_loop"].create_synapse(from_mem_id, to_mem_id, weight)
    
    def _update_interaction_time(self):
        # 发布交互更新事件
        EventBus().emit(Event(
            event_type=EventType.INTERACTION_UPDATED,
            data={},
            timestamp=time.time()
        ))
    
    def _check_mind_wandering_trigger(self):
        return self.container["mind_wandering_service"].check_mind_wandering_trigger()
    
    def _stop_mind_wandering(self):
        EventBus().emit(Event(
            event_type=EventType.MIND_WANDER_STOPPED,
            data={},
            timestamp=time.time()
        ))
    
    @property
    def is_mind_wandering(self):
        return self.container["brain_core"].is_mind_wandering
    
    @property
    def fatigue_level(self):
        return self.container["brain_core"].fatigue_level
    
    @property
    def pending_social_intention(self):
        return self.container["intention_service"].pending_social_intention
    
    @pending_social_intention.setter
    def pending_social_intention(self, value):
        self.container["intention_service"].pending_social_intention = value
        if value:
            self.container["intention_service"].pending_intention_created_at = time.time()
        else:
            self.container["intention_service"].pending_intention_created_at = None
    
    @property
    def last_dream(self):
        return self.container["dreaming_loop"].loop.last_dream
    
    @property
    def synapses(self):
        return self.container["learning_loop"].loop.synapses
    
    # ✅ 新增：文本编码兼容方法（DummyBrain需要）
    def encode_text(self, text: str):
        """文本编码兼容方法"""
        return self.container["think_engine"].encode_text(text)
    
    # Cognitive_system.py
    # @property
    # def last_dream(self):
    #     return self.container["dreaming_loop"].loop.last_dream

    # @last_dream.setter  # ✅ 添加setter
    # def last_dream(self, value):
    #     self.container["dreaming_loop"].loop.last_dream = value