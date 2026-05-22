import os, logging
from typing import Optional
from ILifecycle import ILifecycle, IService
from ServiceContainer import ServiceContainer
from PersistentCortex import PersistentCortex
from HippocampusRouter import HippocampusRouter

logger = logging.getLogger("PersistentCortexWrapper")

class PersistentCortexWrapper(ILifecycle, IService):
    def __init__(self):
        self.cortex: Optional[PersistentCortex] = None
        self._container = None
        self._experts = None
        self._embedding_model = None
        self._llm = None
        self._kg_enabled = None
    
    def initialize(self, container: 'ServiceContainer') -> None:
        """只保存依赖引用，不创建实际实例"""
        self._container = container
        self._experts = container["experts"]
        self._embedding_model = container.embedding_model
        self._llm = container.llm
        self._kg_enabled = container.kg_enabled
        
        logger.info("✅ 持久化皮层包装器初始化完成")
    
    def start(self) -> None:
        """执行每日记忆衰减"""
        if self.cortex:
            # ✅ 保留原始代码中的每日记忆衰减逻辑
            self.cortex.decay_all_memories()
            logger.info("✅ 每日记忆衰减已执行")
    
    def stop(self) -> None:
        pass
    
    def save(self, storage_dir: str) -> None:
        """保存皮层状态"""
        if self.cortex:
            try:
                # ✅ 检查并调用实际存在的保存方法
                if hasattr(self.cortex, 'save_all'):
                    self.cortex.save_all()
                elif hasattr(self.cortex, 'save'):
                    self.cortex.save()
                elif hasattr(self.cortex, 'save_memories'):
                    self.cortex.save_memories()
                logger.info("✅ 持久化皮层状态保存完成")
            except Exception as e:
                logger.warning(f"⚠️  持久化皮层状态保存失败: {e}")
    
    def load(self, storage_dir: str) -> None:
        """创建PersistentCortex实例（自动加载数据）"""
        # ✅ 与原始代码完全一致：PersistentCortex在__init__中自动加载数据
        self.cortex = PersistentCortex(
            storage_dir=storage_dir,
            experts=self._experts,
            embedding_model=self._embedding_model,
            llm=self._llm
            # kg_enabled=self._kg_enabled
        )
        
        # ✅ 保留原始代码中的符号核心绑定逻辑
        if "symbolic_core" in self._container._services:
            self.cortex.symbolic_core = self._container["symbolic_core"].symbolic_core
        
        logger.info(f"✅ 持久化皮层初始化完成 | 存储目录: {storage_dir}")