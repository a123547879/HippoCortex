# ILifecycle.py - 完全独立的接口定义
from abc import ABC, abstractmethod
from typing import Any, Optional

# 移除对ServiceContainer的导入
# 使用字符串前向引用解决类型注解问题

class ILifecycle(ABC):
    """组件生命周期接口"""
    
    @abstractmethod
    def initialize(self, container: 'ServiceContainer') -> None:
        """初始化组件，从容器获取依赖"""
        pass
    
    @abstractmethod
    def start(self) -> None:
        """启动组件"""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """停止组件"""
        pass
    
    @abstractmethod
    def save(self, storage_dir: str) -> None:
        """保存组件状态"""
        pass
    
    @abstractmethod
    def load(self, storage_dir: str) -> None:
        """加载组件状态"""
        pass

class IService(ABC):
    """服务标记接口"""
    pass