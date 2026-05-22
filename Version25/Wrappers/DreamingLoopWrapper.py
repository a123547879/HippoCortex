import os, logging, time
from BrainConfig import config
from typing import Dict
from ILifecycle import ILifecycle, IService
from event_system import Event, EventType, on_event, EventBus
from ServiceContainer import ServiceContainer
from Dreaming_loop import DreamingLoop

logger = logging.getLogger("DreamingLoopWrapper")

class DreamingLoopWrapper(ILifecycle, IService):
    def __init__(self):
        self.loop = None
        self._container = None
    
    def initialize(self, container: ServiceContainer) -> None:
        self._container = container
        self.loop = DreamingLoop(
            core=container["brain_core"],
            event_bus=EventBus(),
            llm=container.llm
        )
        
        # 绑定组件引用
        self.loop.bind_components(
            experts=container["experts"],
            learning_loop=container["learning_loop"]
        )
        
        EventBus().subscribe(EventType.SLEEP_PROGRESS_UPDATED, self._on_sleep_progress)
        logger.info("✅ 梦境循环初始化完成")

    
    def start(self) -> None:
        pass
    
    def stop(self) -> None:
        pass
    
    def save(self, storage_dir: str) -> None:
        # 梦境循环通常不需要持久化状态
        pass
    
    def load(self, storage_dir: str) -> None:
        pass
    
    # 事件订阅
    # @on_event(EventType.SLEEP_STARTED)
    def _on_sleep_started(self, event: Event):
        # 当睡眠开始时，准备梦境生成
        logger.info("🌙 进入睡眠状态，准备生成梦境...")
    
    # @on_event(EventType.SLEEP_PROGRESS_UPDATED)
    def _on_sleep_progress(self, event: Event):
        # 根据睡眠进度生成不同阶段的梦境
        progress = event.data["progress"]
        if 0.2 < progress < 0.8:  # REM睡眠阶段
            dream_length = event.data.get("dream_length", 3)
            self.generate_dream(dream_length)
    
    # 对外接口
    def generate_dream(self, dream_length: int = 3) -> dict:
        result = self.loop.generate_dream(dream_length)
        
        EventBus().emit(Event(
            event_type=EventType.DREAM_GENERATED,
            data={"dream": result, "length": dream_length},
            timestamp=time.time()
        ))
        
        logger.info(f"💭 生成梦境: {result.get('title', '无标题')}")
        return result
    
    @property
    def last_dream(self):
        return self.loop.last_dream