import os, logging, time, logging
from typing import List
from ILifecycle import ILifecycle, IService
from ServiceContainer import ServiceContainer
from Learning_loop import LearningLoop
from event_system import EventBus, EventType, Event, on_event

logger = logging.getLogger("LearningLoopWrapper")

class LearningLoopWrapper(ILifecycle, IService):
    def __init__(self):
        self.loop = None
        self._container = None
        self._synapse_save_path = None
    
    # Wrappers/LearningLoopWrapper.py
    def initialize(self, container: 'ServiceContainer') -> None:
        self._container = container
        self.loop = LearningLoop(
            core=container["brain_core"],
            event_bus=EventBus(),
            embedding_model=container.embedding_model,
            llm=container.llm
        )
        
        # 绑定组件引用（原有代码保持不变）
        self.loop.bind_components(
            thalamus=container["thalamus"].thalamus,
            hippocampus_router=container["hippocampus_router"].router,
            symbolic_core=container["symbolic_core"].symbolic_core if "symbolic_core" in container._services else None,
            experts=container["experts"],
            sdr_encoders=container["sdr_encoders"],
            cortex=container["cortex"].cortex,
            dopamine=container["dopamine_system"].dopamine if "dopamine_system" in container._services else None,
            metacognition=container["metacognition"].metacognition if "metacognition" in container._services else None,
            curiosity=container["curiosity"].curiosity if "curiosity" in container._services else None
        )
        
        # ✅ 新增：手动订阅事件（替代@on_event装饰器）
        EventBus().subscribe(EventType.MEMORY_STORED, self._on_memory_stored)
    
    def start(self) -> None:
        pass
    
    def stop(self) -> None:
        pass
    
    def save(self, storage_dir: str) -> None:
        if self._synapse_save_path:
            self.loop.save_synapses(self._synapse_save_path)
    
    def load(self, storage_dir: str) -> None:
        self._synapse_save_path = os.path.join(storage_dir, "synapses.json")
        if os.path.exists(self._synapse_save_path):
            try:
                self.loop.load_synapses(self._synapse_save_path)
                logger.info("✅ 突触连接加载完成")
            except Exception as e:
                logger.warning(f"⚠️  突触连接加载失败: {e}")
    
    # 订阅记忆存储事件，自动创建关联
    # @on_event(EventType.MEMORY_STORED)
    def _on_memory_stored(self, event: Event):
        try:
            from event_system import safe_get_event_data
            
            # 支持多个备选键名，彻底解决KeyError
            mem_id = safe_get_event_data(event, 'mem_id', 'id')
            mem_text = safe_get_event_data(event, 'mem_text', 'text', 'content')
            mem_vec = safe_get_event_data(event, 'mem_vec', 'vector', 'embedding')
            expert_name = safe_get_event_data(event, 'expert_name', 'target_expert', 'expert')
            
            if all([mem_id, mem_text, mem_vec, expert_name]):
                self.loop._process_new_memory(mem_id, mem_text, mem_vec, expert_name)
        except Exception as e:
            logger.debug(f"LearningLoop事件处理跳过: {e}")
    
    # 对外接口
    def learn(self, text: str, force_expert=None, external_reward=0.0):
        result = self.loop.learn(text, force_expert, external_reward)
        
        EventBus().emit(Event(
            event_type=EventType.KNOWLEDGE_LEARNED,
            data={"text": text, "result": result},
            timestamp=time.time()
        ))
        
        return result
    
    def batch_init_direct_to_cortex(self, texts: List[str]) -> List[int]:
        result = self.loop.batch_init_direct_to_cortex(texts)
        
        EventBus().emit(Event(
            event_type=EventType.BATCH_LEARNING_COMPLETED,
            data={"count": len(texts), "result": result},
            timestamp=time.time()
        ))
        
        return result
    
    def bind_related_memories(self, new_mem_id, new_mem_vec, new_mem_text, target_expert, user_input):
        return self.loop.bind_related_memories(new_mem_id, new_mem_vec, new_mem_text, target_expert, user_input)
    
    def create_synapse(self, from_mem_id: str, to_mem_id: str, weight: float = 0.3):
        result = self.loop.create_synapse(from_mem_id, to_mem_id, weight)
        
        EventBus().emit(Event(
            event_type=EventType.SYNAPSE_CREATED,
            data={"from_id": from_mem_id, "to_id": to_mem_id, "weight": weight},
            timestamp=time.time()
        ))
        
        return result
    
    @property
    def synapses(self):
        return self.loop.synapses