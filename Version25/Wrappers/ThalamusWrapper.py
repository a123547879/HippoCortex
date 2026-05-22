import os, logging, time
from BrainConfig import config
from typing import Dict
from ILifecycle import ILifecycle, IService
from event_system import Event, EventType, EventBus
from ServiceContainer import ServiceContainer
from Thalamus import Thalamus

logger = logging.getLogger("ThalamusWrapper")


class ThalamusWrapper(ILifecycle, IService):
    def __init__(self):
        self.thalamus = None
        self._container = None
    
    def initialize(self, container: ServiceContainer) -> None:
        self._container = container
        self.thalamus = Thalamus(
            input_dim=container["brain_core"].config.dim,
            attention_threshold=0.3,
            consolidation_threshold=0.6,
            max_short_term_capacity=50
        )
        
        # 绑定模块引用
        self.thalamus.bind_modules(
            hippocampus=container["hippocampus_router"].router,
            cortex=container["cortex"].cortex,
            energy_field=container["brain_core"].energy_field,
            experts=container["experts"]
        )


        EventBus().subscribe(EventType.MEMORY_RETRIEVED, self._on_memory_retrieved)
        logger.info("✅ 丘脑初始化完成")
    
    def start(self) -> None:
        pass
    
    def stop(self) -> None:
        pass
    
    def save(self, storage_dir: str) -> None:
        thalamus_path = os.path.join(storage_dir, "thalamus_state.json")
        try:
            self.thalamus.save(thalamus_path)
            logger.info("✅ 丘脑状态保存完成")
        except Exception as e:
            logger.warning(f"⚠️  丘脑状态保存失败: {e}")
    
    def load(self, storage_dir: str) -> None:
        thalamus_path = os.path.join(storage_dir, "thalamus_state.json")
        if os.path.exists(thalamus_path):
            try:
                self.thalamus.load(thalamus_path)
                logger.info("✅ 丘脑状态加载完成")
            except Exception as e:
                logger.warning(f"⚠️  丘脑状态加载失败: {e}")
    
    # 事件订阅
    # @on_event(EventType.MEMORY_RETRIEVED)
    def _on_memory_retrieved(self, event: Event):
        # 当记忆被检索时，更新丘脑的注意力状态
        self.thalamus.update_attention(event.data["memory_id"], event.data["relevance_score"])