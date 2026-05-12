import os
import logging
from typing import List, Dict

from brain_core import BrainCore
from event_system import EventBus
from DynamicExpertV8 import DynamicExpert
from PersistentCortexV16 import PersistentCortex
from LearnableSparseEncoderV2 import LearnableSparseEncoder
from HippocampusRouterV10 import HippocampusRouter
from SymbolicCore import SymbolicCore
from Thalamus import Thalamus
from VAEManager import VAEManager
from BrainConfig import config
from DopamineSystem import DopamineSystem
from Metacognition import Metacognition
from Curiosity import Curiosity

from Perception_loop import PerceptionLoop
from Learning_loop import LearningLoop
from Dreaming_loop import DreamingLoop
from Consolidation_loop import ConsolidationLoop

logger = logging.getLogger("CognitiveSystem")

class CognitiveSystem:
    def __init__(self, embedding_model, llm, kg_enabled: bool = True):
        self.core = BrainCore()
        self.embedding_model = embedding_model
        self.llm = llm
        self.kg_enabled = kg_enabled
        self.event_bus = EventBus()
        
        # 专家和编码器
        self.expert_names = ["身份", "概念", "空间", "抽象", "视觉"]
        self.experts = {}
        self.sdr_encoders = {}
        
        # 核心组件
        self.cortex = None
        self.hippocampus_router = None
        self.symbolic_core = None
        self.vae_manager = None
        self.thalamus = None
        
        # 类人学习核心模块
        self.dopamine = None
        self.metacognition = None
        self.curiosity = None
        
        # 四大循环模块
        self.perception_loop = PerceptionLoop(self.core, self.event_bus, embedding_model, llm)
        self.learning_loop = LearningLoop(self.core, self.event_bus, embedding_model, llm)
        self.dreaming_loop = DreamingLoop(self.core, self.event_bus, llm)
        self.consolidation_loop = ConsolidationLoop(self.core, self.event_bus, llm)
        
        # 兼容原有属性
        self.fatigue_sleep_threshold = 0.85
        self.needs_sleep_request = False

    def initialize(self, storage_dir: str):
        """初始化所有认知组件"""
        logger.info("🧠 初始化认知系统...")
        
        # 1. 初始化专家网络
        for name in self.expert_names:
            self.experts[name] = DynamicExpert(
                name, 
                initial_dim=config.sdr_dim, 
                max_dim=config.max_expert_dim,
                active_size=config.sdr_active_size
            )
        
        # 2. 初始化专家专属稀疏编码器
        for name in self.expert_names:
            self.sdr_encoders[name] = LearnableSparseEncoder(
                input_dim=self.core.config.dim,
                sdr_dim=config.sdr_dim, 
                active_size=config.sdr_active_size,
                expert_name=name
            )
            
            encoder_path = os.path.join(storage_dir, f"sdr_encoder_{name}.pt")
            if os.path.exists(encoder_path):
                try:
                    self.sdr_encoders[name].load(encoder_path)
                    logger.info(f"✅ [{name}] 专家历史稀疏编码器加载完成")
                except Exception as e:
                    logger.warning(f"⚠️  [{name}] 专家稀疏编码器加载失败: {e}")
        
        # 3. 初始化皮层
        self.cortex = PersistentCortex(
            storage_dir, 
            self.experts, 
            embedding_model=self.embedding_model, 
            llm=self.llm, 
            kg_enabled=self.kg_enabled
        )
        
        # 4. 初始化海马体路由
        self.hippocampus_router = HippocampusRouter(
            input_dim=self.core.config.dim,
            expert_names=self.expert_names,
            experts=self.experts
        )
        
        router_path = os.path.join(storage_dir, "hippocampus_router.pt")
        if os.path.exists(router_path):
            try:
                self.hippocampus_router.load(router_path)
                logger.info("✅ 海马体路由加载完成")
            except Exception as e:
                logger.warning(f"⚠️  海马体路由加载失败: {e}")
        
        if not hasattr(self.hippocampus_router, '_prototypes_initialized') or not self.hippocampus_router._prototypes_initialized:
            logger.info("🧭 首次运行，初始化全专家原型...")
            self.hippocampus_router._initialize_prototypes_with_embedding(self.embedding_model)
            self.hippocampus_router.save(router_path)
        
        # 5. 初始化符号核心
        try:
            self.symbolic_core = SymbolicCore(sdr_dim=config.sdr_dim)
            if hasattr(self, 'cortex'):
                self.cortex.symbolic_core = self.symbolic_core
            logger.info("✅ 符号语义核心初始化完成")
        except Exception as e:
            logger.warning(f"⚠️  符号语义核心初始化跳过: {e}")
            self.symbolic_core = None
        
        # 6. 初始化VAE管理器
        self.vae_manager = VAEManager(
            local_model_path=config.YOUR_VAE_MODEL_PATH,
            device="cpu"
        )
        
        # 7. 初始化丘脑
        self.thalamus = Thalamus(
            input_dim=self.core.config.dim,
            attention_threshold=0.3,
            consolidation_threshold=0.6,
            max_short_term_capacity=50
        )
        
        self.thalamus.bind_modules(
            hippocampus=self.hippocampus_router,
            cortex=self.cortex,
            energy_field=self.core.energy_field,
            experts=self.experts
        )
        
        # 8. 初始化类人学习模块
        try:
            self.dopamine = DopamineSystem()
            self.metacognition = Metacognition(self.cortex)
            self.curiosity = Curiosity(self.metacognition, self.dopamine)
            logger.info("✅ 类人学习核心模块初始化完成")
        except Exception as e:
            logger.warning(f"⚠️  类人学习模块初始化跳过: {e}")
            self.dopamine = None
            self.metacognition = None
            self.curiosity = None
        
        # 9. 执行每日记忆衰减
        self.cortex.decay_all_memories()
        
        # 10. 绑定所有循环模块的组件引用
        self._bind_loop_components(storage_dir)
        
        logger.info("✅ 认知系统初始化完成")

    def _bind_loop_components(self, storage_dir: str):
        """绑定所有循环模块的组件引用"""
        # 感知循环
        self.perception_loop.bind_components(
            thalamus=self.thalamus,
            hippocampus_router=self.hippocampus_router,
            symbolic_core=self.symbolic_core,
            experts=self.experts,
            sdr_encoders=self.sdr_encoders,
            cortex=self.cortex
        )
        
        # 学习循环
        self.learning_loop.bind_components(
            thalamus=self.thalamus,
            hippocampus_router=self.hippocampus_router,
            symbolic_core=self.symbolic_core,
            experts=self.experts,
            sdr_encoders=self.sdr_encoders,
            cortex=self.cortex,
            dopamine=self.dopamine,
            metacognition=self.metacognition,
            curiosity=self.curiosity
        )
        self.learning_loop.set_synapse_save_path(os.path.join(storage_dir, "synapses.json"))
        
        # 梦境循环
        self.dreaming_loop.bind_components(
            experts=self.experts,
            learning_loop=self.learning_loop
        )
        
        # 巩固循环
        self.consolidation_loop.bind_components(
            thalamus=self.thalamus,
            hippocampus_router=self.hippocampus_router,
            symbolic_core=self.symbolic_core,
            experts=self.experts,
            cortex=self.cortex,
            dopamine=self.dopamine,
            metacognition=self.metacognition,
            dreaming_loop=self.dreaming_loop,
            learning_loop=self.learning_loop
        )

    # ===================== 对外统一接口（保持与原代码完全一致） =====================
    def think(self, text: str, steps: int = 2, topk: int = 10, expert_last=None) -> Dict:
        return self.perception_loop.think(text, steps, topk, expert_last)
    
    def learn(self, text: str, force_expert=None, external_reward=0.0):
        return self.learning_loop.learn(text, force_expert, external_reward)
    
    def batch_learn(self, texts: List[str]) -> List[int]:
        """
        🔥 批量初始化导入知识库：直接写入皮层，跳过海马体/丘脑/学习链路
        仅用于首次批量灌数据，不触发任何类人学习逻辑
        """
        return self.learning_loop.batch_init_direct_to_cortex(texts)

    
    def sleep_consolidate_all(self, epochs=3, is_manual: bool = False):
        return self.consolidation_loop.sleep_consolidate_all(epochs, is_manual)
    
    def generate_dream(self, dream_length: int = 3) -> dict:
        return self.dreaming_loop.generate_dream(dream_length)
    
    def get_brain_status(self):
        return self.perception_loop.get_brain_status()
    
    def bind_related_memories(self, new_mem_id, new_mem_vec, new_mem_text, target_expert, user_input):
        return self.learning_loop.bind_related_memories(new_mem_id, new_mem_vec, new_mem_text, target_expert, user_input)
    
    def create_synapse(self, from_mem_id: str, to_mem_id: str, weight: float = 0.3):
        return self.learning_loop.create_synapse(from_mem_id, to_mem_id, weight)
    
    def _update_interaction_time(self):
        return self.perception_loop._update_interaction_time()
    
    def _check_mind_wandering_trigger(self):
        return self.perception_loop._check_mind_wandering_trigger()
    
    def _stop_mind_wandering(self):
        return self.perception_loop._stop_mind_wandering()
    
    @property
    def is_mind_wandering(self):
        return self.core.is_mind_wandering
    
    @property
    def fatigue_level(self):
        return self.core.fatigue_level
    
    @property
    def pending_social_intention(self):
        return self.perception_loop.pending_social_intention
    
    @pending_social_intention.setter
    def pending_social_intention(self, value):
        self.perception_loop.pending_social_intention = value
    
    @property
    def last_dream(self):
        return self.dreaming_loop.last_dream
    
    @property
    def synapses(self):
        return self.learning_loop.synapses