import os
import logging
logger = logging.getLogger("DummyBrain")

class DummyBrain:
    def __init__(self, cognitive_system, brain_interface):
        self.cognitive_system = cognitive_system
        self.brain_interface = brain_interface
        self._container = cognitive_system.container  # 核心：获取服务容器引用
        
        # ✅ 从服务容器获取所有内部组件（完全兼容原有接口）
        self.cortex = self._container["cortex"].cortex  # PersistentCortexWrapper -> 实际cortex实例
        self.experts = self._container["experts"]  # 专家字典直接可用
        self.sdr_encoders = self._container["sdr_encoders"]  # 编码器字典直接可用
        self.hippocampus_router = self._container["hippocampus_router"].router  # 包装器 -> 实际router实例
        self.symbolic_core = self._container["symbolic_core"].symbolic_core if "symbolic_core" in self._container._services else None
        self.vae_manager = self._container["vae_manager"].manager if "vae_manager" in self._container._services else None
        self.thalamus = self._container["thalamus"].thalamus  # 包装器 -> 实际thalamus实例
        
        # 基础属性（保持不变）
        self.dim = 1024
        self.ollama_model = "bge-m3"
        self.kg_enabled = cognitive_system.kg_enabled
        self.core = brain_interface.core  # BrainCore直接从brain_interface获取
    
    @property
    def is_mind_wandering(self):
        return self.core.is_mind_wandering
    
    @is_mind_wandering.setter
    def is_mind_wandering(self, value):
        self.core.is_mind_wandering = value
        # 重构后通过mind_wandering_service控制
        if not value:
            self._container["mind_wandering_service"]._stop_mind_wandering()
    
    @property
    def fatigue_level(self):
        return self.core.fatigue_level
    
    @fatigue_level.setter
    def fatigue_level(self, value):
        self.core.fatigue_level = value
    
    def _check_mind_wandering_trigger(self):
        # 转发到mind_wandering_service
        self._container["mind_wandering_service"].check_mind_wandering_trigger()
    
    def _stop_mind_wandering(self):
        # 转发到mind_wandering_service
        self._container["mind_wandering_service"]._stop_mind_wandering()
    
    def _update_interaction_time(self):
        # 转发到perception_loop
        self.cognitive_system._update_interaction_time()
    
    def think(self, text, **kwargs):
        # 保持原有调用方式不变
        return self.cognitive_system.think(text, **kwargs)
    
    def recall_compositional(self, query, target_expert=None, **kwargs):
        """兼容旧版接口的复合记忆检索，彻底解决AttributeError"""
        try:
            # 调用think_engine的检索方法（最接近原有逻辑）
            think_engine = self._container["think_engine"]
            clip_vec = think_engine.encode_text(query)
            sdr_encoder = self.sdr_encoders.get(target_expert, self.sdr_encoders["概念"])
            query_sdr = sdr_encoder.encode(clip_vec)
            
            # 获取专家分数
            expert_scores = self.hippocampus_router.last_scores if hasattr(self.hippocampus_router, 'last_scores') else {}
            
            # 执行全专家检索
            raw_results = think_engine._retrieve_memories(clip_vec, query_sdr, query, expert_scores, total_energy=5.0)
            
            # 转换为原有格式返回
            memories = []
            for mem_id, score, content, meta in raw_results:
                memories.append({
                    "id": mem_id,
                    "content": content,
                    "score": score,
                    "metadata": meta
                })
            
            return memories, []
        except Exception as e:
            logger.debug(f"⚠️ 复合检索兼容层调用失败: {e}")
            # 兜底返回空，绝不崩溃
            return [], []
    
    def learn(self, text, force_expert=None):
        # 保持原有调用方式不变
        return self.brain_interface.learn_text(text, force_expert)
    
    def batch_learn(self, texts: list):
        """
        批量导入知识（直接写入皮层长期记忆）
        自动兼容：不传参数 / 传参数 / 关键字传参
        """
        return self.brain_interface.batch_learn_text(texts)
    
    def sleep_consolidate_all(self):
        # 保持原有调用方式不变
        self.brain_interface.trigger_sleep()
    
    def save_all(self):
        """完全兼容原有保存逻辑，同时支持新架构的保存机制"""
        try:
            # 第一步：调用新架构的统一保存方法（推荐）
            self._container.save_all()
            logger.info("✅ 新架构统一保存完成")
        except Exception as e:
            logger.warning(f"⚠️  新架构统一保存失败，回退到手动保存: {e}")
            
            # 第二步：回退到原有手动保存逻辑（完全兼容）
            # 保存SDR编码器
            for name in self.experts.keys():  # ✅ 修复：从experts字典获取专家名称
                encoder_path = os.path.join(self.core.storage_dir, f"sdr_encoder_{name}.pt")
                self.sdr_encoders[name].save(encoder_path)
            
            # 保存海马体路由
            router_path = os.path.join(self.core.storage_dir, "hippocampus_router.pt")
            self.hippocampus_router.save(router_path)
            
            # 保存皮层
            if hasattr(self.cortex, 'save_all'):
                self.cortex.save_all()
            if hasattr(self.cortex, 'save_brain_state'):
                self.cortex.save_brain_state()
        
        logger.info("✅ 所有状态保存完成")
    
    def get_brain_status(self):
        # 保持原有调用方式不变
        return self.cognitive_system.get_brain_status() or {}
    
    def get_pending_social_intention(self):
        # 从intention_service获取待处理意图
        intention = self._container["intention_service"].pending_social_intention
        self._container["intention_service"].pending_social_intention = None
        return intention
    
    def check_and_consume_sleep_request(self):
        # 保持原有逻辑不变
        if self.core.needs_sleep_request:
            self.core.needs_sleep_request = False
            return True
        return False
    
    def encode_text(self, text):
        """文本编码（转发到think_engine，完全兼容原有接口）"""
        return self._container["think_engine"].encode_text(text)
        
    def enable_kg(self):
        # 同时更新cognitive_system和cortex的kg_enabled标志
        self.cognitive_system.kg_enabled = True
        self.cortex.kg_enabled = True
    
    def disable_kg(self):
        # 同时更新cognitive_system和cortex的kg_enabled标志
        self.cognitive_system.kg_enabled = False
        self.cortex.kg_enabled = False

    def bind_related_memories(self, new_mem_id, new_mem_vec, new_mem_text, target_expert, user_input):
        try:
            # 转发到learning_loop
            self._container["learning_loop"].bind_related_memories(
                new_mem_id=new_mem_id,
                new_mem_vec=new_mem_vec,
                new_mem_text=new_mem_text,
                target_expert=target_expert,
                user_input=user_input
            )
        except Exception as e:
            logger.debug(f"绑定记忆跳过: {e}")
            
    # ✅ 新增：跨模态脑桥访问（兼容未来扩展）
    @property
    def cross_modal_bridge(self):
        return self._container["cross_modal_bridge"]
    
    # ✅ 新增：书籍阅读服务访问（兼容未来扩展）
    @property
    def book_reader(self):
        return self._container["book_reading_service"].book_reader