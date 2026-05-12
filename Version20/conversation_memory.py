import json
import time
import math
import logging
from typing import List, Dict, Optional
import os

logger = logging.getLogger("ConversationMemory")

class ConversationMemory:
    """
    类脑时间衰减对话记忆系统（单对话窗口专用）
    核心特性：
    1. 遵循艾宾浩斯遗忘曲线：对话激活值随时间指数衰减
    2. 差异化衰减：重要对话衰减慢5倍，普通对话衰减快
    3. 动态上下文：按当前激活值排序取最"记得住"的对话
    4. 自然遗忘：激活值低于阈值自动从上下文中消失
    5. 自动清理：定期清理完全遗忘的对话，避免内存泄漏
    """
    
    def __init__(self, 
                 storage_path: str = None,
                 normal_decay_lambda: float = 0.1,    # 普通对话衰减系数（半衰期≈7小时）
                 important_decay_lambda: float = 0.02, # 重要对话衰减系数（半衰期≈35小时）
                 forget_threshold: float = 0.1,        # 遗忘阈值（低于此值不再出现在上下文）
                 max_context_turns: int = 8,           # 最多返回多少轮上下文
                 auto_cleanup_interval: int = 10):     # 每添加多少轮对话自动清理一次
        self.storage_path = storage_path
        
        # 衰减参数
        self.normal_decay_lambda = normal_decay_lambda
        self.important_decay_lambda = important_decay_lambda
        self.forget_threshold = forget_threshold
        self.max_context_turns = max_context_turns
        self.auto_cleanup_interval = auto_cleanup_interval
        
        # 对话存储（不再用deque，改用列表支持动态衰减）
        self.all_turns: List[Dict] = []
        
        # 待巩固的重要对话
        self.pending_consolidation: List[Dict] = []
        
        # 统计
        self.turn_count_since_last_cleanup = 0
        
        # 加载历史对话
        if storage_path:
            self._load()
    
    def add_turn(self, user_input: str, ai_response: str, metadata: Dict = None) -> str:
        """添加一轮对话，自动计算初始激活值"""
        metadata = metadata or {}
        turn_id = f"conv_{int(time.time() * 1000)}"
        
        # 自动判断重要性（可扩展更复杂的规则）
        is_important = metadata.get("is_important", False) or any(
            keyword in user_input.lower() for keyword in 
            ["记住", "重要", "别忘了", "一定要记得", "我的", "你要", "永远"]
        )
        
        turn = {
            "id": turn_id,
            "user_input": user_input,
            "ai_response": ai_response,
            "timestamp": time.time(),
            "initial_activation": 1.0,
            "is_important": is_important,
            "metadata": metadata
        }
        
        self.all_turns.append(turn)
        self.turn_count_since_last_cleanup += 1
        
        # 重要对话加入待巩固队列
        if turn["is_important"]:
            self.pending_consolidation.append(turn)
            logger.info(f"📝 标记重要对话（衰减减慢5倍）: {user_input[:30]}...")
        
        # 自动清理完全遗忘的对话
        if self.turn_count_since_last_cleanup >= self.auto_cleanup_interval:
            self._cleanup_forgotten_turns()
            self.turn_count_since_last_cleanup = 0
        
        logger.debug(f"添加对话轮次 | ID:{turn_id} | 重要:{is_important} | 当前总对话数:{len(self.all_turns)}")
        return turn_id
    
    def get_active_context(self) -> List[Dict]:
        """
        获取当前"记得住"的对话上下文
        按当前激活值从高到低排序，返回最相关的max_context_turns轮
        """
        now = time.time()
        active_turns = []
        
        for turn in self.all_turns:
            # 计算当前激活值
            current_activation = self._calculate_current_activation(turn, now)
            
            # 只保留激活值高于遗忘阈值的对话
            if current_activation >= self.forget_threshold:
                turn_with_activation = turn.copy()
                turn_with_activation["current_activation"] = current_activation
                active_turns.append(turn_with_activation)
        
        # 按激活值从高到低排序（最新+最重要的排在前面）
        active_turns.sort(key=lambda x: (-x["current_activation"], -x["timestamp"]))
        
        # 取前max_context_turns轮
        return active_turns[:self.max_context_turns]
    
    def get_recent_turns(self, n: int = 5) -> List[Dict]:
        """获取最近n轮对话（不考虑激活值，用于特殊场景）"""
        return list(self.all_turns)[-n:]
    
    def mark_important(self, turn_id: str, importance: float = 0.9):
        """手动标记对话为重要，重置激活值并减慢衰减"""
        for turn in self.all_turns:
            if turn["id"] == turn_id and not turn["is_important"]:
                turn["is_important"] = True
                turn["initial_activation"] = 1.0  # 重置激活值
                turn["timestamp"] = time.time()   # 重置时间戳，相当于重新记忆
                if turn not in self.pending_consolidation:
                    self.pending_consolidation.append(turn)
                logger.info(f"📝 手动标记对话为重要: {turn['user_input'][:30]}...")
                return True
        return False
    
    def get_pending_consolidation(self) -> List[Dict]:
        """获取待巩固的重要对话，并清空队列"""
        pending = self.pending_consolidation.copy()
        self.pending_consolidation.clear()
        return pending
    
    def clear(self):
        """清空所有对话记忆"""
        self.all_turns.clear()
        self.pending_consolidation.clear()
        self.turn_count_since_last_cleanup = 0
        logger.info("已清空所有对话记忆")
    
    def _calculate_current_activation(self, turn: Dict, now: float = None) -> float:
        """
        计算对话的当前激活值（艾宾浩斯指数衰减）
        公式：activation = initial_activation * exp(-lambda * time_hours)
        """
        if now is None:
            now = time.time()
        
        time_hours = (now - turn["timestamp"]) / 3600
        decay_lambda = self.important_decay_lambda if turn["is_important"] else self.normal_decay_lambda
        
        return turn["initial_activation"] * math.exp(-decay_lambda * time_hours)
    
    def _cleanup_forgotten_turns(self):
        """清理激活值极低的对话（低于遗忘阈值的1/10），释放内存"""
        now = time.time()
        original_count = len(self.all_turns)
        
        self.all_turns = [
            turn for turn in self.all_turns
            if self._calculate_current_activation(turn, now) >= self.forget_threshold / 10
        ]
        
        cleaned_count = original_count - len(self.all_turns)
        if cleaned_count > 0:
            logger.debug(f"🧹 自动清理了 {cleaned_count} 条完全遗忘的对话")
    
    def _load(self):
        """从文件加载历史对话"""
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
        
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.all_turns = data.get("all_turns", [])
            self.pending_consolidation = data.get("pending_consolidation", [])
            
            # 加载后立即清理一次完全遗忘的对话
            self._cleanup_forgotten_turns()
            
            logger.info(f"✅ 加载历史对话 | 有效:{len(self.all_turns)}轮 | 待巩固:{len(self.pending_consolidation)}条")
        except Exception as e:
            logger.error(f"❌ 加载对话历史失败: {e}")
    
    def save(self):
        """保存对话历史到文件"""
        if not self.storage_path:
            return
        
        try:
            data = {
                "all_turns": self.all_turns,
                "pending_consolidation": self.pending_consolidation,
                "last_saved": time.time()
            }
            
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"💾 对话历史已保存 | 有效:{len(self.all_turns)}轮")
        except Exception as e:
            logger.error(f"❌ 保存对话历史失败: {e}")
    
    def get_memory_status(self) -> Dict:
        """获取记忆状态统计（用于调试和监控）"""
        now = time.time()
        active_count = 0
        important_count = 0
        total_activation = 0.0
        
        for turn in self.all_turns:
            activation = self._calculate_current_activation(turn, now)
            if activation >= self.forget_threshold:
                active_count += 1
            if turn["is_important"]:
                important_count += 1
            total_activation += activation
        
        return {
            "total_turns": len(self.all_turns),
            "active_turns": active_count,
            "important_turns": important_count,
            "average_activation": total_activation / len(self.all_turns) if self.all_turns else 0.0,
            "pending_consolidation": len(self.pending_consolidation)
        }