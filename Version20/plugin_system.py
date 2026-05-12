from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from brain_core import BrainCore

class Plugin(ABC):
    def __init__(self):
        self.core = BrainCore()
        self._initialized = False
    
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        pass
    
    def initialize(self):
        self._initialized = True
    
    def shutdown(self):
        self._initialized = False
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized

class PerceptionPlugin(Plugin):
    @abstractmethod
    def get_modality(self) -> str:
        pass
    
    @abstractmethod
    def perceive(self) -> Optional[Any]:
        pass

class ActionPlugin(Plugin):
    @abstractmethod
    def get_action_type(self) -> str:
        pass
    
    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> Any:
        pass