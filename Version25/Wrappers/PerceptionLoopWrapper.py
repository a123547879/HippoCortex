import os, logging, time
from BrainConfig import config
from typing import Dict
from ILifecycle import ILifecycle, IService
from ServiceContainer import ServiceContainer
from event_system import EventBus, EventType, Event, on_event
import datetime
import logging
from typing import Dict, Any, Optional
from Data_models import ThoughtResult

logger = logging.getLogger("PerceptionLoop")

class PerceptionLoopWrapper(ILifecycle, IService):
    def __init__(self):
        self._container = None
    
    def initialize(self, container: 'ServiceContainer') -> None:
        self._container = container
        logger.info("✅ 感知循环包装器初始化完成")
    
    def start(self) -> None:
        pass
    
    def stop(self) -> None:
        pass
    
    def save(self, storage_dir: str) -> None:
        pass
    
    def load(self, storage_dir: str) -> None:
        pass
    
    def think(self, text: str, steps: int = 2, topk: int = 20, expert_last: Optional[str] = None) -> ThoughtResult:
        self._update_interaction_time()
        return self._container["think_engine"].think(text, steps, topk, expert_last)
    
    def process_image_with_caption(self, image_tensor, caption: str):
        return self._container["cross_modal_bridge"].process_image_with_caption(image_tensor, caption)
    
    def get_brain_status(self) -> Dict[str, Any]:
        from collections import defaultdict
        import numpy as np
        
        try:
            total_memories = len(self._container["cortex"].cortex.index.memories)
        except:
            total_memories = 0
        
        expert_counts = defaultdict(int)
        expert_access = defaultdict(list)
        expert_sparsity = {}
        
        try:
            cortex = self._container["cortex"].cortex
            for mem_id in cortex.index.memories.keys():
                mem = cortex.index.get_memory(mem_id)
                if mem:
                    expert = mem.expert
                    expert_counts[expert] += 1
                    expert_access[expert].append(mem.access_count)
        except:
            pass
        
        try:
            experts = self._container["experts"]
            for name in experts.keys():
                if hasattr(experts[name], 'get_sparsity'):
                    expert_sparsity[name] = experts[name].get_sparsity()
                else:
                    expert_sparsity[name] = 0.0
        except:
            pass
        
        cross_modal_bridge = self._container["cross_modal_bridge"]
        
        status = {
            "total_memories": total_memories,
            "ollama_model": "bge-m3",
            "embedding_dim": getattr(config, 'dim', 1024),
            "expert_distribution": {},
            "experts": {},
            "kg_enabled": getattr(self._container["cortex"].cortex, 'kg_enabled', True),
            "is_mind_wandering": getattr(self._container["brain_core"], 'is_mind_wandering', False),
            "fatigue_level": getattr(self._container["brain_core"], 'fatigue_level', 0.0),
            "intention_queue_size": len(self._container["intention_service"].intention_queue),
            "pending_social_intention": self._container["intention_service"].pending_social_intention is not None,
            "bridge_training": {
                "enabled": cross_modal_bridge.use_learnable_pons,
                "train_steps": cross_modal_bridge.bridge_train_steps,
                "buffer_size": len(cross_modal_bridge.bridge_pair_buffer),
                "batch_size": cross_modal_bridge.bridge_batch_size,
                "last_loss": cross_modal_bridge.bridge_loss_history["total"][-1] if cross_modal_bridge.bridge_loss_history["total"] else 0.0
            }
        }
        
        try:
            experts = self._container["experts"]
            for name in experts.keys():
                count = expert_counts.get(name, 0)
                access_list = expert_access.get(name, [0])
                avg_access = np.mean(access_list) if access_list else 0
                sparsity = expert_sparsity.get(name, 0.0)
                
                status["expert_distribution"][name] = count
                status["experts"][name] = {
                    "神经元": getattr(experts[name], 'dim', 2048),
                    "记忆数": count,
                    "平均访问": round(avg_access, 2),
                    "突触稀疏度": round(sparsity, 4)
                }
        except:
            pass
        
        return status
    
    def _update_interaction_time(self) -> None:
        self._container["brain_core"].update_interaction_time()
        EventBus().emit(Event(
            event_type=EventType.INTERACTION_UPDATED,
            data={},
            timestamp=time.time()
        ))
    
    def _check_mind_wandering_trigger(self) -> None:
        self._container["mind_wandering_service"].check_mind_wandering_trigger()
    
    def _stop_mind_wandering(self) -> None:
        self._container["mind_wandering_service"]._stop_mind_wandering()
    
    @property
    def pending_social_intention(self):
        return self._container["intention_service"].pending_social_intention
    
    @pending_social_intention.setter
    def pending_social_intention(self, value):
        self._container["intention_service"].pending_social_intention = value
        if value:
            self._container["intention_service"].pending_intention_created_at = datetime.datetime.now()
        else:
            self._container["intention_service"].pending_intention_created_at = None