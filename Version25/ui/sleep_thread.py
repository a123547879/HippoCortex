from PyQt5.QtCore import QThread, pyqtSignal
import logging
from event_system import EventType
from Data_models import SleepReport

logger = logging.getLogger("SleepThread")


class SleepThread(QThread):
    progress_signal = pyqtSignal(int, str)
    finish_signal = pyqtSignal(SleepReport)  # 明确发射SleepReport对象
    
    def __init__(self, brain_interface, is_manual: bool = False):
        super().__init__()
        self.brain_interface = brain_interface
        self.is_manual = is_manual
        
    def run(self):
        try:
            def on_sleep_progress(event):
                self.progress_signal.emit(event.data["progress"], event.data["message"])
            
            self.brain_interface.event_bus.subscribe(EventType.SLEEP_PROGRESS_UPDATED, on_sleep_progress)
            
            sleep_report = self.brain_interface.trigger_sleep(is_manual=self.is_manual)
            
            self.finish_signal.emit(sleep_report)
            
        except Exception as e:
            logger.error(f"❌ 睡眠失败：{str(e)}", exc_info=True)
            self.finish_signal.emit(SleepReport(error=str(e)))
        finally:
            self.brain_interface.event_bus.unsubscribe(EventType.SLEEP_PROGRESS_UPDATED, on_sleep_progress)