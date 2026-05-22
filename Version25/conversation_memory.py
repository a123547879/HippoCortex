import json
import time
import math
import logging
from typing import List, Dict, Optional, Any
import os

# 导入实体中心统一数据契约
from Data_models import ConversationTurn, Entity, Evidence
from BrainConfig import config

logger = logging.getLogger("ConversationMemory")

class ConversationMemory:
    """
    🔥 实体中心式类脑对话记忆系统
    核心升级：对话不再是孤立文本，而是实体网络的一部分
    保留所有原有特性 + 实体关联增强 + 双向激活传播
    """
    
    def __init__(self, 
                 entity_extractor: Any,
                 cortex: Any,
                 storage_path: str = None,
                 # 从全局配置读取默认值，支持本地覆盖
                 normal_decay_lambda: float = config.CONVERSATION_MEMORY_CONFIG["normal_decay_lambda"],
                 important_decay_lambda: float = config.CONVERSATION_MEMORY_CONFIG["important_decay_lambda"],
                 forget_threshold: float = config.CONVERSATION_MEMORY_CONFIG["forget_threshold"],
                 max_context_turns: int = config.CONVERSATION_MEMORY_CONFIG["max_context_turns"],
                 auto_cleanup_interval: int = config.CONVERSATION_MEMORY_CONFIG["auto_cleanup_interval"]):
        self.entity_extractor = entity_extractor
        self.cortex = cortex
        self.storage_path = storage_path
        
        # 衰减参数（优先使用全局配置）
        self.normal_decay_lambda = normal_decay_lambda
        self.important_decay_lambda = important_decay_lambda
        self.forget_threshold = forget_threshold
        self.max_context_turns = max_context_turns
        self.auto_cleanup_interval = auto_cleanup_interval
        
        # 对话存储（使用标准ConversationTurn对象）
        self.all_turns: List[ConversationTurn] = []
        
        # 待巩固的重要对话
        self.pending_consolidation: List[ConversationTurn] = []
        
        # 统计
        self.turn_count_since_last_cleanup = 0
        
        # 加载历史对话
        if storage_path:
            self._load()
    
    def add_turn(self, user_input: str, ai_response: str, metadata: Dict = None) -> str:
        """
        添加一轮对话并自动关联实体
        ✅ 自动提取实体 ✅ 建立双向关联 ✅ 更新实体状态 ✅ 自动判断重要性
        """
        metadata = metadata or {}
        
        # 1. 提取对话中的实体（核心：对话→实体关联）
        extracted_entities: List[Entity] = []
        try:
            full_text = f"{user_input} {ai_response}"
            extracted_entities = self.entity_extractor.extract(full_text)
            logger.debug(f"🔍 对话提取到 {len(extracted_entities)} 个实体: {[e.name for e in extracted_entities]}")
        except Exception as e:
            logger.warning(f"⚠️ 对话实体提取失败: {e}，将作为普通对话处理")
        
        # 2. 自动判断重要性（结合实体重要性）
        is_important = metadata.get("is_important", False)
        # 规则1：包含重要性关键词
        is_important |= any(
            keyword in user_input.lower() for keyword in 
            ["记住", "重要", "别忘了", "一定要记得", "我的", "你要", "永远"]
        )
        # 规则2：关联了高重要性实体（>0.8）
        is_important |= any(e.importance > 0.8 for e in extracted_entities)
        # 规则3：关联了永久实体
        is_important |= any(e.is_permanent for e in extracted_entities)
        
        # 3. 创建标准对话轮次对象
        turn = ConversationTurn(
            user_input=user_input,
            ai_response=ai_response,
            timestamp=time.time(),
            initial_activation=1.0,
            is_important=is_important,
            is_consolidated=False,
            extracted_entity_ids=[e.entity_id for e in extracted_entities],
            activated_entity_ids=[e.entity_id for e in extracted_entities],
            metadata=metadata
        )
        
        # 4. 建立双向关联：对话→实体 和 实体→对话
        for entity in extracted_entities:
            # 更新实体访问状态
            entity.update_access()
            # 对话关联提升实体重要性
            entity.importance = min(0.95, entity.importance + 0.03)
            # 实体记录关联的对话
            if "conversation_turns" not in entity.metadata:
                entity.metadata["conversation_turns"] = []
            entity.metadata["conversation_turns"].append(turn.turn_id)
            # 保存实体更新
            self.cortex.update_entity(entity)
        
        # 5. 加入存储
        self.all_turns.append(turn)
        self.turn_count_since_last_cleanup += 1
        
        # 6. 重要对话加入待巩固队列
        if turn.is_important:
            self.pending_consolidation.append(turn)
            logger.info(f"📝 标记重要对话（衰减减慢5倍）: {user_input[:30]}... | 关联实体: {len(extracted_entities)}个")
        
        # 7. 自动清理完全遗忘的对话
        if self.turn_count_since_last_cleanup >= self.auto_cleanup_interval:
            self._cleanup_forgotten_turns()
            self.turn_count_since_last_cleanup = 0
        
        logger.debug(f"添加对话轮次 | ID:{turn.turn_id[:8]} | 重要:{is_important} | 当前总对话数:{len(self.all_turns)}")
        return turn.turn_id
    
    def get_active_context(self) -> List[ConversationTurn]:
        """
        获取当前"记得住"的对话上下文
        ✅ 艾宾浩斯时间衰减 ✅ 实体激活增强 ✅ 智能排序
        """
        now = time.time()
        active_turns = []
        
        for turn in self.all_turns:
            # 1. 基础时间衰减激活值
            base_activation = self._calculate_current_activation(turn, now)
            
            # 2. 实体激活增强（核心升级）
            # 如果对话关联的实体最近被访问过，提升对话激活值
            entity_boost = 0.0
            for entity_id in turn.extracted_entity_ids:
                entity = self.cortex.get_entity(entity_id)
                if entity:
                    time_since_access = (now - entity.last_accessed) / 3600  # 小时
                    if time_since_access < 1.0:  # 1小时内访问过
                        # 实体越重要，boost越大
                        entity_boost += 0.2 * entity.importance
            
            # 3. 最终激活值（不超过1.0）
            final_activation = min(1.0, base_activation + entity_boost)
            
            # 只保留激活值高于遗忘阈值的对话
            if final_activation >= self.forget_threshold:
                # 创建副本避免修改原对象
                turn_copy = turn.model_copy()
                turn_copy.initial_activation = final_activation  # 临时存储最终激活值
                active_turns.append(turn_copy)
        
        # 智能排序：先按激活值降序，再按时间降序
        active_turns.sort(key=lambda x: (-x.initial_activation, -x.timestamp))
        
        # 取前max_context_turns轮
        result = active_turns[:self.max_context_turns]
        
        # 更新激活实体ID列表（用于后续检索增强）
        for turn in result:
            self.cortex.activate_entities(turn.extracted_entity_ids, boost=0.1)
        
        logger.debug(f"获取激活上下文 | 共 {len(result)} 轮 | 最高激活值: {result[0].initial_activation:.2f}" if result else "无激活上下文")
        return result
    
    def get_recent_turns(self, n: int = 5) -> List[ConversationTurn]:
        """获取最近n轮对话（不考虑激活值，用于特殊场景）"""
        return list(self.all_turns)[-n:]
    
    def get_turns_by_entity(self, entity_id: str) -> List[ConversationTurn]:
        """获取提到指定实体的所有对话轮次（新增：实体驱动检索）"""
        return [
            turn for turn in self.all_turns 
            if entity_id in turn.extracted_entity_ids
        ]
    
    def mark_important(self, turn_id: str, importance: float = 0.9) -> bool:
        """
        手动标记对话为重要
        ✅ 重置对话激活值 ✅ 提升关联实体重要性 ✅ 加入待巩固队列
        """
        for turn in self.all_turns:
            if turn.turn_id == turn_id and not turn.is_important:
                # 标记对话为重要
                turn.is_important = True
                turn.initial_activation = 1.0  # 重置激活值
                turn.timestamp = time.time()   # 重置时间戳，相当于重新记忆
                
                # 提升所有关联实体的重要性
                for entity_id in turn.extracted_entity_ids:
                    entity = self.cortex.get_entity(entity_id)
                    if entity:
                        entity.importance = min(0.95, entity.importance + 0.1)
                        entity.is_permanent |= importance > 0.9  # 极高重要性标记为永久
                        self.cortex.update_entity(entity)
                
                # 加入待巩固队列
                if turn not in self.pending_consolidation:
                    self.pending_consolidation.append(turn)
                
                logger.info(f"📝 手动标记对话为重要: {turn.user_input[:30]}... | 关联实体: {len(turn.extracted_entity_ids)}个")
                return True
        return False
    
    def get_pending_consolidation(self) -> List[ConversationTurn]:
        """获取待巩固的重要对话，并清空队列（与ConsolidationLoop对接）"""
        pending = self.pending_consolidation.copy()
        self.pending_consolidation.clear()
        
        # 标记为已提交巩固
        for turn in pending:
            turn.is_consolidated = True
        
        logger.debug(f"返回待巩固对话 | 共 {len(pending)} 条")
        return pending
    
    def clear(self) -> None:
        """清空所有对话记忆（保留实体，仅删除对话记录）"""
        self.all_turns.clear()
        self.pending_consolidation.clear()
        self.turn_count_since_last_cleanup = 0
        logger.info("已清空所有对话记忆（实体保留）")
    
    def _calculate_current_activation(self, turn: ConversationTurn, now: float = None) -> float:
        """
        计算对话的当前激活值（艾宾浩斯指数衰减）
        公式：activation = initial_activation * exp(-lambda * time_hours)
        """
        if now is None:
            now = time.time()
        
        time_hours = (now - turn.timestamp) / 3600
        decay_lambda = self.important_decay_lambda if turn.is_important else self.normal_decay_lambda
        
        return turn.initial_activation * math.exp(-decay_lambda * time_hours)
    
    def _cleanup_forgotten_turns(self) -> None:
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
    
    def _load(self) -> None:
        """从文件加载历史对话（自动迁移旧格式）"""
        if not self.storage_path or not os.path.exists(self.storage_path):
            return
        
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 自动迁移旧格式（字典→ConversationTurn对象）
            self.all_turns = []
            for turn_data in data.get("all_turns", []):
                if isinstance(turn_data, dict) and "turn_id" not in turn_data:
                    # 旧格式字典转换
                    turn = ConversationTurn(
                        user_input=turn_data["user_input"],
                        ai_response=turn_data["ai_response"],
                        timestamp=turn_data["timestamp"],
                        initial_activation=turn_data["initial_activation"],
                        is_important=turn_data["is_important"],
                        metadata=turn_data.get("metadata", {})
                    )
                    self.all_turns.append(turn)
                else:
                    # 新格式直接反序列化
                    self.all_turns.append(ConversationTurn.from_dict(turn_data))
            
            # 加载待巩固队列
            self.pending_consolidation = []
            for turn_data in data.get("pending_consolidation", []):
                if isinstance(turn_data, dict):
                    self.pending_consolidation.append(ConversationTurn.from_dict(turn_data))
            
            # 加载后立即清理一次完全遗忘的对话
            self._cleanup_forgotten_turns()
            
            logger.info(f"✅ 加载历史对话 | 有效:{len(self.all_turns)}轮 | 待巩固:{len(self.pending_consolidation)}条")
        except Exception as e:
            logger.error(f"❌ 加载对话历史失败: {e}", exc_info=True)
    
    def save(self) -> None:
        """保存对话历史到文件（标准序列化）"""
        if not self.storage_path:
            return
        
        try:
            data = {
                "all_turns": [turn.to_dict() for turn in self.all_turns],
                "pending_consolidation": [turn.to_dict() for turn in self.pending_consolidation],
                "last_saved": time.time(),
                "version": "2.0"  # 实体中心版版本号
            }
            
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"💾 对话历史已保存 | 有效:{len(self.all_turns)}轮 | 版本:2.0")
        except Exception as e:
            logger.error(f"❌ 保存对话历史失败: {e}", exc_info=True)
    
    def get_memory_status(self) -> Dict[str, Any]:
        """获取记忆状态统计（用于调试和监控）"""
        now = time.time()
        active_count = 0
        important_count = 0
        total_activation = 0.0
        total_entities = 0
        
        for turn in self.all_turns:
            activation = self._calculate_current_activation(turn, now)
            if activation >= self.forget_threshold:
                active_count += 1
            if turn.is_important:
                important_count += 1
            total_activation += activation
            total_entities += len(turn.extracted_entity_ids)
        
        return {
            "total_turns": len(self.all_turns),
            "active_turns": active_count,
            "important_turns": important_count,
            "average_activation": round(total_activation / len(self.all_turns), 3) if self.all_turns else 0.0,
            "average_entities_per_turn": round(total_entities / len(self.all_turns), 1) if self.all_turns else 0.0,
            "pending_consolidation": len(self.pending_consolidation)
        }