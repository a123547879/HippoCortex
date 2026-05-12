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