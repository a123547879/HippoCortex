import os, logging, time
from BrainConfig import config
from typing import Dict
from ILifecycle import ILifecycle, IService
from event_system import Event, EventType, on_event, EventBus
from ServiceContainer import ServiceContainer
from DopamineSystem import DopamineSystem

logger = logging.getLogger("DopamineSystemWrapper")

class DopamineSystemWrapper(ILifecycle, IService):
    def __init__(self):
        self.dopamine = None
        self._container = None
    
    def initialize(self, container: 'ServiceContainer') -> None:
        self._container = container
        try:
            self.dopamine = DopamineSystem()
            logger.info("✅ 多巴胺系统初始化完成")
        except Exception as e:
            logger.warning(f"⚠️  多巴胺系统初始化跳过: {e}")
            self.dopamine = None
        
        # ✅ 新增：手动订阅事件
        EventBus().subscribe(EventType.KNOWLEDGE_LEARNED, self._on_knowledge_learned)
        EventBus().subscribe(EventType.SYNAPSE_CREATED, self._on_synapse_created)
        
    def start(self) -> None:
        pass
    
    def stop(self) -> None:
        EventBus().unsubscribe(EventType.KNOWLEDGE_LEARNED, self._on_knowledge_learned)
        EventBus().unsubscribe(EventType.SYNAPSE_CREATED, self._on_synapse_created)
    
    def save(self, storage_dir: str) -> None:
        if self.dopamine:
            dopamine_path = os.path.join(storage_dir, "dopamine_state.json")
            try:
                self.dopamine.save(dopamine_path)
                logger.info("✅ 多巴胺系统状态保存完成")
            except Exception as e:
                logger.warning(f"⚠️  多巴胺系统保存失败: {e}")
    
    def load(self, storage_dir: str) -> None:
        if self.dopamine:
            dopamine_path = os.path.join(storage_dir, "dopamine_state.json")
            if os.path.exists(dopamine_path):
                try:
                    self.dopamine.load(dopamine_path)
                    logger.info("✅ 多巴胺系统状态加载完成")
                except Exception as e:
                    logger.warning(f"⚠️  多巴胺系统加载失败: {e}")
    
    def _on_knowledge_learned(self, event: Event):
        try:
            if self.dopamine:
                # ✅ 只调用实际存在的方法
                if hasattr(self.dopamine, 'on_knowledge_learned'):
                    self.dopamine.on_knowledge_learned(event.data)
                elif hasattr(self.dopamine, 'add_reward'):
                    # 兼容常见的替代方法名
                    reward = event.data.get("reward", 0.1)
                    self.dopamine.add_reward(reward)
                elif hasattr(self.dopamine, 'update'):
                    self.dopamine.update(event.data)
                # 如果都不存在，什么也不做，只记录日志
                else:
                    logger.debug(f"多巴胺系统跳过知识学习事件：未找到匹配的方法")
        except Exception as e:
            logger.debug(f"多巴胺系统事件处理跳过: {e}")
    
    def _on_synapse_created(self, event: Event):
        try:
            if self.dopamine:
                if hasattr(self.dopamine, 'on_synapse_created'):
                    self.dopamine.on_synapse_created(event.data)
                elif hasattr(self.dopamine, 'add_reward'):
                    reward = event.data.get("reward", 0.05)
                    self.dopamine.add_reward(reward)
                elif hasattr(self.dopamine, 'update'):
                    self.dopamine.update(event.data)
                else:
                    logger.debug(f"多巴胺系统跳过突触创建事件：未找到匹配的方法")
        except Exception as e:
            logger.debug(f"多巴胺系统事件处理跳过: {e}")