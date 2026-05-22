import os, logging, time
from typing import Dict
from ILifecycle import ILifecycle, IService
from ServiceContainer import ServiceContainer
from Metacognition import Metacognition
from event_system import EventBus, EventType, Event, on_event

logger = logging.getLogger("MetacognitionWrapper")

class MetacognitionWrapper(ILifecycle, IService):
    def __init__(self):
        self.metacognition = None
        self._container = None
    
    def initialize(self, container: 'ServiceContainer') -> None:
        self._container = container
        try:
            self.metacognition = Metacognition(container["cortex"].cortex)
            logger.info("✅ 元认知系统初始化完成")
        except Exception as e:
            logger.warning(f"⚠️  元认知系统初始化跳过: {e}")
            self.metacognition = None
        
        # ✅ 新增：手动订阅事件
        EventBus().subscribe(EventType.THOUGHT_GENERATED, self._on_thought_generated)
    
    def start(self) -> None:
        pass
    
    def stop(self) -> None:
        EventBus().unsubscribe(EventType.THOUGHT_GENERATED, self._on_thought_generated)
    
    def save(self, storage_dir: str) -> None:
        if self.metacognition:
            meta_path = os.path.join(storage_dir, "metacognition_state.json")
            try:
                self.metacognition.save(meta_path)
                logger.info("✅ 元认知系统状态保存完成")
            except Exception as e:
                logger.warning(f"⚠️  元认知系统保存失败: {e}")
    
    def load(self, storage_dir: str) -> None:
        if self.metacognition:
            meta_path = os.path.join(storage_dir, "metacognition_state.json")
            if os.path.exists(meta_path):
                try:
                    self.metacognition.load(meta_path)
                    logger.info("✅ 元认知系统状态加载完成")
                except Exception as e:
                    logger.warning(f"⚠️  元认知系统加载失败: {e}")
    
    # 事件订阅
    # @on_event(EventType.THOUGHT_GENERATED)
    def _on_thought_generated(self, event: Event):
        try:
            if self.metacognition:
                if hasattr(self.metacognition, 'on_thought_generated'):
                    self.metacognition.on_thought_generated(event.data)
                elif hasattr(self.metacognition, 'process_thought'):
                    thought = event.data.get("thought", "")
                    self.metacognition.process_thought(thought)
                elif hasattr(self.metacognition, 'evaluate'):
                    thought = event.data.get("thought", "")
                    self.metacognition.evaluate(thought)
                else:
                    logger.debug(f"元认知系统跳过思考生成事件：未找到匹配的方法")
        except Exception as e:
            logger.debug(f"元认知系统事件处理跳过: {e}")