import os, logging, time
from BrainConfig import config
from typing import Dict
from ILifecycle import ILifecycle, IService
from ServiceContainer import ServiceContainer
from SymbolicCore import SymbolicCore
from event_system import EventBus, EventType, Event, on_event

logger = logging.getLogger("SymbolicCoreWrapper")

class SymbolicCoreWrapper(ILifecycle, IService):
    def __init__(self):
        self.symbolic_core = None
        self._container = None
    
    def initialize(self, container: ServiceContainer) -> None:
        self._container = container
        try:
            self.symbolic_core = SymbolicCore(sdr_dim=config.sdr_dim)
            logger.info("✅ 符号语义核心初始化完成")
        except Exception as e:
            logger.warning(f"⚠️  符号语义核心初始化跳过: {e}")
            self.symbolic_core = None
    
    def start(self) -> None:
        pass
    
    def stop(self) -> None:
        pass
    
    def save(self, storage_dir: str) -> None:
        if self.symbolic_core:
            symbolic_path = os.path.join(storage_dir, "symbolic_core.json")
            try:
                self.symbolic_core.save(symbolic_path)
                logger.info("✅ 符号核心状态保存完成")
            except Exception as e:
                logger.warning(f"⚠️  符号核心保存失败: {e}")
    
    def load(self, storage_dir: str) -> None:
        if self.symbolic_core:
            symbolic_path = os.path.join(storage_dir, "symbolic_core.json")
            if os.path.exists(symbolic_path):
                try:
                    self.symbolic_core.load(symbolic_path)
                    logger.info("✅ 符号核心状态加载完成")
                except Exception as e:
                    logger.warning(f"⚠️  符号核心加载失败: {e}")