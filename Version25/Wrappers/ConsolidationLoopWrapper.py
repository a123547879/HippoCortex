import os, logging, time
from typing import Dict
from ILifecycle import ILifecycle, IService
from ServiceContainer import ServiceContainer
from Consolidation_loop import ConsolidationLoop
from event_system import EventBus, EventType, Event

logger = logging.getLogger("ConsolidationLoopWrapper")

class ConsolidationLoopWrapper(ILifecycle, IService):
    def __init__(self):
        self.loop = None
        self._container = None
    
    def initialize(self, container: ServiceContainer) -> None:
        self._container = container
        self.loop = ConsolidationLoop(
            core=container["brain_core"],
            event_bus=EventBus(),
            llm=container.llm
        )
        
        # 绑定组件引用
        self.loop.bind_components(
            thalamus=container["thalamus"].thalamus,
            hippocampus_router=container["hippocampus_router"].router,
            symbolic_core=container["symbolic_core"].symbolic_core if "symbolic_core" in container._services else None,
            experts=container["experts"],
            cortex=container["cortex"].cortex,
            dopamine=container["dopamine_system"].dopamine if "dopamine_system" in container._services else None,
            metacognition=container["metacognition"].metacognition if "metacognition" in container._services else None,
            dreaming_loop=container["dreaming_loop"].loop,  # ✅ 修复：获取实际的DreamingLoop实例
            learning_loop=container["learning_loop"].loop   # ✅ 修复：获取实际的LearningLoop实例
        )
        
        # ✅ 新增：手动订阅事件（替代@on_event装饰器）
        EventBus().subscribe(EventType.SLEEP_COMPLETED, self._on_sleep_completed)
    
    def start(self) -> None:
        pass
    
    def stop(self) -> None:
        # ✅ 最佳实践：在stop方法中取消订阅，防止内存泄漏
        EventBus().unsubscribe(EventType.SLEEP_COMPLETED, self._on_sleep_completed)
    
    def save(self, storage_dir: str) -> None:
        pass
    
    def load(self, storage_dir: str) -> None:
        pass
    
    # ❌ 删除了@on_event装饰器
    def _on_sleep_completed(self, event: Event):
        logger.info(f"睡眠完成，共巩固了{event.data.get('consolidated_count', 0)}条记忆")
    
    # 对外接口保持完全不变
    def sleep_consolidate_all(self, epochs=3, is_manual: bool = False):
        EventBus().emit(Event(
            event_type=EventType.SLEEP_STARTED,
            data={"epochs": epochs, "is_manual": is_manual},
            timestamp=time.time()
        ))
        
        result = self.loop.sleep_consolidate_all(epochs, is_manual)
        
        EventBus().emit(Event(
            event_type=EventType.SLEEP_COMPLETED,
            # ✅ 修复：将Pydantic模型转换为字典
            data=result.model_dump() if hasattr(result, 'model_dump') else dict(result),
            timestamp=time.time()
        ))
        
        return result