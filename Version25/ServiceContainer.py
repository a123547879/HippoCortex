import os
import logging
from typing import Dict, Annotated, Optional, Any
from brain_core import BrainCore
from ILifecycle import ILifecycle
from event_system import EventBus

logger = logging.getLogger("ServiceContainer")

class ServiceContainer:
    """服务容器：管理所有组件的生命周期和依赖关系"""
    
    def __init__(self, embedding_model, llm, kg_enabled: bool = True):
        self._services: Dict[str, Any] = {}
        self._initialized: bool = False
        self._storage_dir: Optional[str] = None
        
        # 基础依赖
        self.embedding_model = embedding_model
        self.llm = llm
        self.kg_enabled = kg_enabled
        
        # 注册核心服务（EventBus是单例，不需要注册）
        self.register("brain_core", BrainCore())
    
    def register(self, name: str, service: Any):
        """注册服务"""
        if self._initialized:
            raise RuntimeError("不能在容器初始化后注册服务")
        self._services[name] = service
    
    def get(self, name: str) -> Any:
        """获取服务"""
        if name not in self._services:
            raise KeyError(f"服务未注册: {name}")
        return self._services[name]
    
    def __getitem__(self, name: str) -> Any:
        return self.get(name)
    
    def initialize_all(self, storage_dir: str):
        """初始化所有服务"""
        if self._initialized:
            logger.warning("服务容器已经初始化过了")
            return
        
        self._storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        logger.info("🧠 初始化认知系统服务容器...")
        
        # 启动事件总线处理线程
        EventBus().start_processing()
        logger.info("✅ 事件总线处理线程已启动")
        
        # 按依赖顺序初始化服务
        init_order = [
            "experts", "sdr_encoders", "cortex", "hippocampus_router",
            "symbolic_core", "vae_manager", "thalamus",
            "dopamine_system", "metacognition", "curiosity",
            "cross_modal_bridge", "think_engine",  # ✅ 确保think_engine在perception_loop之前
            "intention_service", "book_reading_service", "mind_wandering_service",
            "perception_loop", "learning_loop", "dreaming_loop", "consolidation_loop"
        ]
        
        for service_name in init_order:
            if service_name in self._services:
                service = self._services[service_name]
                if isinstance(service, ILifecycle):
                    logger.info(f"初始化服务: {service_name}")
                    service.initialize(self)
                    service.load(storage_dir)
        
        self._initialized = True
        logger.info("✅ 所有服务初始化完成")
    
    def start_all(self):
        """启动所有服务"""
        for service_name, service in self._services.items():
            if isinstance(service, ILifecycle):
                logger.info(f"启动服务: {service_name}")
                service.start()
        logger.info("✅ 所有服务启动完成")
    
    def stop_all(self):
        """停止所有服务"""
        # 先停止事件总线，确保所有事件处理完成
        EventBus().stop_processing()
        logger.info("✅ 事件总线处理线程已停止")
        
        # 按逆序停止服务
        for service_name, service in reversed(list(self._services.items())):
            if isinstance(service, ILifecycle):
                logger.info(f"停止服务: {service_name}")
                service.stop()
        logger.info("✅ 所有服务停止完成")
    
    def save_all(self):
        """保存所有服务状态"""
        if not self._storage_dir:
            raise RuntimeError("服务容器未初始化，无法保存状态")
        
        for service_name, service in self._services.items():
            if isinstance(service, ILifecycle):
                logger.info(f"保存服务状态: {service_name}")
                service.save(self._storage_dir)
        logger.info("✅ 所有服务状态保存完成")