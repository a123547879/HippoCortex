import os, logging, time
from BrainConfig import config
from typing import Dict
from ILifecycle import ILifecycle, IService
from event_system import Event, EventType, on_event, EventBus
from ServiceContainer import ServiceContainer
from Curiosity import Curiosity

logger = logging.getLogger("CuriosityWrapper")

class CuriosityWrapper(ILifecycle, IService):
    def __init__(self):
        self.curiosity = None
        self._container = None
    
    def initialize(self, container: 'ServiceContainer') -> None:
        self._container = container
        try:
            self.curiosity = Curiosity(
                container["metacognition"].metacognition,
                container["dopamine_system"].dopamine
            )
            logger.info("✅ 好奇心系统初始化完成")
        except Exception as e:
            logger.warning(f"⚠️  好奇心系统初始化跳过: {e}")
            self.curiosity = None
        
        # ✅ 新增：手动订阅事件
        EventBus().subscribe(EventType.MEMORY_STORED, self._on_memory_stored)
    
    def start(self) -> None:
        pass
    
    def stop(self) -> None:
        EventBus().unsubscribe(EventType.MEMORY_STORED, self._on_memory_stored)
    
    def save(self, storage_dir: str) -> None:
        if self.curiosity:
            curiosity_path = os.path.join(storage_dir, "curiosity_state.json")
            try:
                self.curiosity.save(curiosity_path)
                logger.info("✅ 好奇心系统状态保存完成")
            except Exception as e:
                logger.warning(f"⚠️  好奇心系统保存失败: {e}")
    
    def load(self, storage_dir: str) -> None:
        if self.curiosity:
            curiosity_path = os.path.join(storage_dir, "curiosity_state.json")
            if os.path.exists(curiosity_path):
                try:
                    self.curiosity.load(curiosity_path)
                    logger.info("✅ 好奇心系统状态加载完成")
                except Exception as e:
                    logger.warning(f"⚠️  好奇心系统加载失败: {e}")
    
    def _on_memory_stored(self, event: Event):
        try:
            if self.curiosity:
                mem_text = event.data.get("mem_text", "")
                target_expert = event.data.get("target_expert", "")
                
                if hasattr(self.curiosity, 'on_memory_stored'):
                    self.curiosity.on_memory_stored(event.data)
                elif hasattr(self.curiosity, 'calculate_novelty'):
                    novelty_score = self.curiosity.calculate_novelty(mem_text, target_expert)
                    # 如果新颖性高，触发探索意图
                    if novelty_score > 0.7:
                        EventBus().emit(Event(
                            event_type=EventType.INTENTION_GENERATED,
                            data={
                                "type": "exploration",
                                "novelty_score": novelty_score,
                                "topic": mem_text
                            },
                            timestamp=time.time()
                        ))
                elif hasattr(self.curiosity, 'update'):
                    self.curiosity.update(event.data)
                else:
                    logger.debug(f"好奇心系统跳过记忆存储事件：未找到匹配的方法")
        except Exception as e:
            logger.debug(f"好奇心系统事件处理跳过: {e}")