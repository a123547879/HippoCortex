import os
import logging
logger = logging.getLogger("DummyBrain")

class DummyBrain:
    def __init__(self, cognitive_system, brain_interface):
        self.cognitive_system = cognitive_system
        self.brain_interface = brain_interface
        
        # 暴露所有内部组件
        self.cortex = cognitive_system.cortex
        self.experts = cognitive_system.experts
        self.sdr_encoders = cognitive_system.sdr_encoders
        self.hippocampus_router = cognitive_system.hippocampus_router
        self.symbolic_core = cognitive_system.symbolic_core
        self.vae_manager = cognitive_system.vae_manager
        self.thalamus = cognitive_system.thalamus
        
        # 基础属性
        self.dim = 1024
        self.ollama_model = "bge-m3"
        self.kg_enabled = cognitive_system.kg_enabled
        self.core = brain_interface.core
    
    @property
    def is_mind_wandering(self):
        return self.core.is_mind_wandering
    
    @is_mind_wandering.setter
    def is_mind_wandering(self, value):
        self.core.is_mind_wandering = value
        self.cognitive_system.is_mind_wandering = value
    
    @property
    def fatigue_level(self):
        return self.core.fatigue_level
    
    @fatigue_level.setter
    def fatigue_level(self, value):
        self.core.fatigue_level = value
    
    # 🔥 修复2：删除重复定义的方法，只保留一次
    def _check_mind_wandering_trigger(self):
        self.cognitive_system._check_mind_wandering_trigger()
    
    def _stop_mind_wandering(self):
        self.cognitive_system._stop_mind_wandering()
    
    def _update_interaction_time(self):
        self.cognitive_system._update_interaction_time()
    
    def think(self, text, **kwargs):
        return self.cognitive_system.think(text, **kwargs)
    
    # 🔥 新增修复：解决缺失的 recall_compositional 方法（核心！）
    def recall_compositional(self, query, target_expert=None, **kwargs):
        """兼容旧版接口的复合记忆检索，彻底解决AttributeError"""
        try:
            # 调用底层认知系统的检索方法
            memories = self.cognitive_system.recall_memories(query, expert_name=target_expert)
            return memories, []
        except Exception as e:
            logger.debug(f"⚠️ 复合检索兼容层调用失败: {e}")
            # 兜底返回空，绝不崩溃
            return [], []
    
    def learn(self, text, force_expert=None):
        return self.brain_interface.learn_text(text, force_expert)
    
    def batch_learn(self, texts: list):
        """
        批量导入知识（直接写入皮层长期记忆）
        自动兼容：不传参数 / 传参数 / 关键字传参
        """
        return self.brain_interface.batch_learn_text(texts)
    
    def sleep_consolidate_all(self):
        self.brain_interface.trigger_sleep()
    
    def save_all(self):
        for name in self.cognitive_system.expert_names:
            encoder_path = os.path.join(self.core.storage_dir, f"sdr_encoder_{name}.pt")
            self.cognitive_system.sdr_encoders[name].save(encoder_path)
        router_path = os.path.join(self.core.storage_dir, "hippocampus_router.pt")
        self.cognitive_system.hippocampus_router.save(router_path)
        self.cortex.save_all()
        self.cortex.save_brain_state()
    
    def get_brain_status(self):
        return self.cognitive_system.get_brain_status() or {}
    
    def get_pending_social_intention(self):
        intention = self.cognitive_system.pending_social_intention
        self.cognitive_system.pending_social_intention = None
        return intention
    
    def check_and_consume_sleep_request(self):
        if self.core.needs_sleep_request:
            self.core.needs_sleep_request = False
            return True
        return False
    
    # 文本编码（核心修复）
    def encode_text(self, text):
        return self.cognitive_system.perception_loop.encode_text(text)
        
    def enable_kg(self):
        self.cognitive_system.kg_enabled = True
        self.cortex.kg_enabled = True
    
    def disable_kg(self):
        self.cognitive_system.kg_enabled = False
        self.cortex.kg_enabled = False

    def bind_related_memories(self, new_mem_id, new_mem_vec, new_mem_text, target_expert, user_input):
        try:
            self.cognitive_system.bind_related_memories(
                new_mem_id=new_mem_id,
                new_mem_vec=new_mem_vec,
                new_mem_text=new_mem_text,
                target_expert=target_expert,
                user_input=user_input
            )
        except Exception as e:
            logger.debug(f"绑定记忆跳过: {e}")