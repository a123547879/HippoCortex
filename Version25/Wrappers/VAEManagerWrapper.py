import os, logging, time
from BrainConfig import config
from typing import Dict
from ILifecycle import ILifecycle, IService
from ServiceContainer import ServiceContainer
from VAEManager import VAEManager

logger = logging.getLogger("VAEManagerWrapper")

class VAEManagerWrapper(ILifecycle, IService):
    def __init__(self):
        self.manager = None
        self._container = None
    
    def initialize(self, container: ServiceContainer) -> None:
        self._container = container
        try:
            self.manager = VAEManager(
                local_model_path=config.YOUR_VAE_MODEL_PATH,
                device="cpu"
            )
            logger.info("✅ VAE管理器初始化完成")
        except Exception as e:
            logger.warning(f"⚠️  VAE管理器初始化跳过: {e}")
            self.manager = None
    
    def start(self) -> None:
        pass
    
    def stop(self) -> None:
        pass
    
    def save(self, storage_dir: str) -> None:
        # VAE模型通常不需要运行时保存
        pass
    
    def load(self, storage_dir: str) -> None:
        # VAE模型在初始化时已经加载
        pass