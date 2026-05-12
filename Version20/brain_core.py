import threading
import os
import datetime
from typing import Dict, Any
from BrainConfig import config
from CognitiveEnergyField import CognitiveEnergyField

class BrainCore:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        self.config = config
        self.energy_field = CognitiveEnergyField()
        self._running = False
        self._state: Dict[str, Any] = {}
        
        # 全局状态（✅ 全部正确初始化）
        self.fatigue_level = 0.0
        self.is_mind_wandering = False
        self.needs_sleep_request = False
        self.fatigue_sleep_threshold = 0.85  # ✅ 添加缺失的属性
        self.last_interaction_time = datetime.datetime.now()  # ✅ 初始化为datetime对象
        
        # 保留原来的Event对象，改个名字避免冲突
        self.interaction_event = threading.Event()
        self.interaction_event.set()
        
    
    def start(self, storage_dir: str):
        self._running = True
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.energy_field.initialize()
        self.update_interaction_time()  # ✅ 启动时更新交互时间
    
    def stop(self):
        self._running = False
        self.energy_field.shutdown()
    
    def update_interaction_time(self):
        """✅ 统一的更新交互时间方法"""
        self.last_interaction_time = datetime.datetime.now()
        self.interaction_event.set()
    
    def get_state(self, key: str, default=None):
        return self._state.get(key, default)
    
    def set_state(self, key: str, value: Any):
        self._state[key] = value
    
    @property
    def is_running(self) -> bool:
        return self._running