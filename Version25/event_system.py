import threading
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from enum import Enum, auto

class EventType(Enum):
    INPUT_RECEIVED = auto()
    MEMORY_STORED = auto()
    MEMORY_RETRIEVED = auto()
    COGNITIVE_STATE_CHANGED = auto()
    SLEEP_STARTED = auto()
    SLEEP_FINISHED = auto()
    RESPONSE_GENERATED = auto()
    INTENTION_GENERATED = auto()
    DREAM_GENERATED = auto()
    SLEEP_PROGRESS_UPDATED = "sleep_progress_updated"  # 睡眠进度更新
    SLEEP_COMPLETED = "sleep_completed"                # 睡眠完成通知

    # 新增事件
    TEXT_PERCEIVED = auto()
    THOUGHT_GENERATED = auto()
    MIND_WANDERED = auto()
    MIND_WANDER_STOPPED = auto()
    KNOWLEDGE_LEARNED = auto()
    BATCH_LEARNING_COMPLETED = auto()
    SYNAPSE_CREATED = auto()
    MEMORY_BOUND = auto()
    CONSOLIDATION_STARTED = auto()
    INTERACTION_UPDATED = auto()
    STATUS_CHANGED = auto()
    
    # ===================== 🔥 补充：缺失的关键事件类型 =====================
    # 走神开始事件（导致您错误的直接原因）
    MIND_WANDER_STARTED = auto()
    
    # 跨模态相关
    BRIDGE_TRAINED = auto()
    IMAGE_PROCESSED = auto()
    
    # 书籍阅读相关
    BOOK_READ_FINISHED = auto()
    IMAGINATION_GENERATED = auto()
    
    # 意图执行相关
    INTENTION_EXECUTED = auto()
    # ======================================================================

@dataclass
class Event:
    event_type: EventType
    data: Any = None
    timestamp: float = None

class EventBus:
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
        self._listeners: Dict[EventType, List[Callable]] = {}
        self._event_queue = []
        self._processing = False
        self._condition = threading.Condition()
    
    def subscribe(self, event_type: EventType, listener: Callable):
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(listener)
    
    def unsubscribe(self, event_type: EventType, listener: Callable):
        if event_type in self._listeners:
            self._listeners[event_type].remove(listener)
    
    def emit(self, event: Event):
        with self._condition:
            self._event_queue.append(event)
            self._condition.notify()
    
    def start_processing(self):
        self._processing = True
        threading.Thread(target=self._process_events, daemon=True).start()
    
    def stop_processing(self):
        self._processing = False
        with self._condition:
            self._condition.notify()
    
    def _process_events(self):
        while self._processing:
            with self._condition:
                while not self._event_queue and self._processing:
                    self._condition.wait()
                if not self._processing:
                    break
                event = self._event_queue.pop(0)
            
            if event.event_type in self._listeners:
                for listener in self._listeners[event.event_type]:
                    try:
                        listener(event)
                    except Exception as e:
                        print(f"Error in event listener: {e}")

def on_event(event_type: EventType):
    def decorator(func: Callable):
        EventBus().subscribe(event_type, func)
        return func
    return decorator

def safe_get_event_data(event: Event, *keys: str, default=None):
    """
    安全获取事件数据，支持多个备选键名
    例如：safe_get_event_data(event, 'mem_id', 'id') 会先找'mem_id'，再找'id'
    """
    data = event.data
    
    # 如果是Pydantic模型，先转换为字典
    if hasattr(data, 'model_dump'):
        data = data.model_dump()
    elif hasattr(data, '__dict__'):
        data = data.__dict__
    
    # 尝试所有备选键名
    for key in keys:
        if key in data:
            return data[key]
    
    return default