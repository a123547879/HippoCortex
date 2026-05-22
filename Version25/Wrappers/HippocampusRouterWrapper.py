import os, logging
from ILifecycle import ILifecycle, IService
from ServiceContainer import ServiceContainer
from PersistentCortex import PersistentCortex
from HippocampusRouter import HippocampusRouter

logger = logging.getLogger("HippocampusRouterWrapper")

class HippocampusRouterWrapper(ILifecycle, IService):
    def __init__(self):
        self.router = None
        self._container = None
    
    def initialize(self, container: ServiceContainer) -> None:
        self._container = container
        self.router = HippocampusRouter(
            input_dim=container["brain_core"].config.dim,
            expert_names=list(container["experts"].keys()),
            experts=container["experts"]
        )
    
    def start(self) -> None:
        pass
    
    def stop(self) -> None:
        pass
    
    def save(self, storage_dir: str) -> None:
        router_path = os.path.join(storage_dir, "hippocampus_router.pt")
        self.router.save(router_path)
    
    def load(self, storage_dir: str) -> None:
        router_path = os.path.join(storage_dir, "hippocampus_router.pt")
        if os.path.exists(router_path):
            try:
                self.router.load(router_path)
                logger.info("✅ 海马体路由加载完成")
            except Exception as e:
                logger.warning(f"⚠️  海马体路由加载失败: {e}")
        
        # 检查是否需要初始化原型
        if not hasattr(self.router, '_prototypes_initialized') or not self.router._prototypes_initialized:
            logger.info("🧭 首次运行，初始化全专家原型...")
            self.router._initialize_prototypes_with_entities(self._container.embedding_model)
            self.save(storage_dir)