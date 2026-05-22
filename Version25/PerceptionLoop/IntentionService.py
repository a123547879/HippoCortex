import datetime
import time
import random
import logging
from typing import List, Dict, Optional, Any

from ILifecycle import ILifecycle, IService
from ServiceContainer import ServiceContainer
from event_system import EventBus, Event, EventType
# ✅ 替换为实体中心统一数据契约
from Data_models import Intention, Entity, Evidence

logger = logging.getLogger("IntentionService")

class IntentionService(ILifecycle, IService):
    def __init__(self):
        self.intention_queue: List[Intention] = []
        self.max_intention_queue_size: int = 3
        self.last_intention_execution_time: datetime.datetime = datetime.datetime.now()
        self.min_intention_interval: int = 60
        self.pending_social_intention: Optional[Intention] = None
        self.pending_intention_created_at: Optional[datetime.datetime] = None
        self.PENDING_INTENTION_EXPIRE_SECONDS: int = 600
        
        # 意图类型权重（完全保留原有配置）
        self.intention_weights: Dict[str, float] = {
            "physiological": 1.5,
            "cognitive": 1.0,
            "social": 0.8,
            "exploration": 0.6
        }
        
        self._container = None
    
    def initialize(self, container: 'ServiceContainer') -> None:
        self._container = container
        
        EventBus().subscribe(EventType.COGNITIVE_STATE_CHANGED, self._on_cognitive_state_changed)
        EventBus().subscribe(EventType.INTERACTION_UPDATED, self._on_interaction_updated)
    
    def start(self) -> None:
        pass
    
    def stop(self) -> None:
        pass
    
    def save(self, storage_dir: str) -> None:
        pass
    
    def load(self, storage_dir: str) -> None:
        pass
    
    def _on_cognitive_state_changed(self, event: Event):
        """认知状态变化事件处理（完全保留原有逻辑）"""
        assessment = event.data.get("assessment", {})
        if assessment.get("confidence", 0.5) < 0.3:
            self.intention_queue.append(Intention(
                type="cognitive",
                priority=0.7,
                content="我不太确定这个问题的答案，你能再解释一下吗？",
                action="ask_clarification"
            ))
    
    def _on_interaction_updated(self, event: Event):
        """交互更新事件处理（完全保留原有逻辑）"""
        self._clear_expired_pending_intention()
    
    # ===================== 🔴 核心：实体中心式意图生成 =====================
    def generate_intentions(self) -> None:
        candidate_intentions: List[Intention] = []
        brain_core = self._container["brain_core"]
        
        # 生理意图（睡眠/疲劳）优先级最高
        if brain_core.fatigue_level > 0.5:
            if brain_core.fatigue_level >= brain_core.fatigue_sleep_threshold:
                safe_priority = max(0.0, min(2.0, 
                    brain_core.fatigue_level * self.intention_weights["physiological"] * 1.5
                ))
                candidate_intentions.append(Intention(
                    type="physiological",
                    priority=safe_priority,
                    content="我有点困了，先睡一会儿哦~ 睡醒了会更聪明的！",
                    action="express_tiredness",
                    need_sleep=True
                ))
            else:
                safe_priority = max(0.0, min(2.0, 
                    brain_core.fatigue_level * self.intention_weights["physiological"]
                ))
                candidate_intentions.append(Intention(
                    type="physiological",
                    priority=safe_priority,
                    content=f"有点累了呢，不过还能再陪你玩一会儿~",
                    action="express_tiredness",
                    need_sleep=False
                ))
        
        # 睡眠意图直接清空队列，优先执行
        if any(i.need_sleep for i in candidate_intentions):
            self.intention_queue = candidate_intentions[:1]
            logger.debug(f"🧠 已生成睡眠意图，跳过其他意图生成 | 队列长度：{len(self.intention_queue)}")
            return

        # 认知意图：主动分享重要实体
        if random.random() < 0.3:
            important_entities = self._get_important_entities(limit=3)
            if important_entities:
                entity = random.choice(important_entities)
                safe_priority = max(0.0, min(2.0, 
                    entity.importance * self.intention_weights["cognitive"]
                ))
                # 优先使用最新证据内容，兜底用实体名称
                content = entity.latest_evidence.content if entity.latest_evidence else entity.name
                candidate_intentions.append(Intention(
                    type="cognitive",
                    priority=safe_priority,
                    content=f"我想起了一件重要的事：{content[:40]}...",
                    action="review_entity",
                    context={"entity": entity}
                ))
        
        # 社交意图：分享最近聊过的实体
        if random.random() < 0.35:
            recent_entities = self._get_recent_entities(limit=15)
            if recent_entities:
                entity = random.choice(recent_entities)
                safe_priority = max(0.0, min(2.0, 
                    0.5 * self.intention_weights["social"]
                ))
                content = entity.latest_evidence.content if entity.latest_evidence else entity.name
                candidate_intentions.append(Intention(
                    type="social",
                    priority=safe_priority,
                    content=f"对了，我想起来我们之前聊过：{content[:40]}...",
                    action="share_entity",
                    context={"entity": entity}
                ))
        
        # 社交意图：主动提问知识缺口
        if random.random() < 0.2:
            knowledge_gaps = self._find_knowledge_gaps()
            if knowledge_gaps:
                gap = random.choice(knowledge_gaps)
                safe_priority = max(0.0, min(2.0, 
                    0.4 * self.intention_weights["social"]
                ))
                candidate_intentions.append(Intention(
                    type="social",
                    priority=safe_priority,
                    content=f"我一直很好奇，{gap}是什么呀？你能给我讲讲吗？",
                    action="ask_question",
                    context={"question": gap}
                ))
        
        # 探索意图：发现实体之间的关联
        if random.random() < 0.15:
            associations = self._get_random_entity_associations(limit=2)
            if len(associations) >= 2:
                safe_priority = max(0.0, min(2.0, 
                    0.3 * self.intention_weights["exploration"]
                ))
                candidate_intentions.append(Intention(
                    type="exploration",
                    priority=safe_priority,
                    content=f"我发现{associations[0][:15]}和{associations[1][:15]}之间好像有某种联系",
                    action="explore_association",
                    context={"associations": associations}
                ))
        
        # 去重并按优先级排序（同一类型只保留最高优先级）
        all_intentions = self.intention_queue + candidate_intentions
        seen_types = set()
        unique_intentions = []
        
        for intention in sorted(all_intentions, key=lambda x: -x.priority):
            if intention.type not in seen_types:
                seen_types.add(intention.type)
                unique_intentions.append(intention)
        
        self.intention_queue = unique_intentions[:self.max_intention_queue_size]
        
        logger.debug(
            f"🧠 生成了{len(candidate_intentions)}个候选意图 | "
            f"去重后保留{len(self.intention_queue)}个 | "
            f"队列类型：{[i.type for i in self.intention_queue]} | "
            f"最高优先级：{(self.intention_queue[0].priority if self.intention_queue else 0):.2f}"
        )
        
        EventBus().emit(Event(
            event_type=EventType.INTENTION_GENERATED,
            data={"count": len(self.intention_queue)},
            timestamp=time.time()
        ))
    
    def execute_highest_priority_intention(self) -> Optional[Intention]:
        """执行最高优先级意图（完全保留原有逻辑，仅适配实体上下文）"""
        if not self.intention_queue:
            return None
        
        time_since_last = (datetime.datetime.now() - self.last_intention_execution_time).total_seconds()
        if time_since_last < self.min_intention_interval:
            return None
        
        highest_intention = max(self.intention_queue, key=lambda x: x.priority)
        self.intention_queue.remove(highest_intention)
        
        logger.info(f"🎯 执行意图：{highest_intention.content} (优先级：{highest_intention.priority:.2f})")
        
        result = None
        if highest_intention.action in [
            "express_tiredness", "share_entity", "ask_question", 
            "explore_association", "review_entity"
        ]:
            result = highest_intention.content
        
        highest_intention.executed = True
        highest_intention.result = result
        self.last_intention_execution_time = datetime.datetime.now()
        
        # 社交意图标记为待处理，等待用户回复
        if highest_intention.action in ["share_entity", "ask_question", "explore_association"]:
            self.pending_social_intention = highest_intention
            self.pending_intention_created_at = datetime.datetime.now()
        
        EventBus().emit(Event(
            event_type=EventType.INTENTION_EXECUTED,
            data={"intention": highest_intention},
            timestamp=time.time()
        ))
        
        return highest_intention
    
    def _clear_expired_pending_intention(self) -> None:
        """清理过期待处理意图（完全保留原有逻辑）"""
        if (self.pending_social_intention is not None 
            and self.pending_intention_created_at is not None):
            elapsed = (datetime.datetime.now() - self.pending_intention_created_at).total_seconds()
            if elapsed > self.PENDING_INTENTION_EXPIRE_SECONDS:
                logger.info(f"🧹 清理过期的待处理意图：{self.pending_social_intention.content[:30]}...")
                self.pending_social_intention = None
                self.pending_intention_created_at = None
    
    # ===================== 🔴 实体工具方法（替代原记忆方法） =====================
    def _get_important_entities(self, limit: int = 5) -> List[Entity]:
        """获取重要性最高的实体（替代原_get_important_memories）"""
        cortex = self._container["cortex"].cortex
        important_entities: List[Entity] = []
        
        try:
            for entity in cortex.index.entities.values():
                if entity and entity.importance > 0.7:
                    important_entities.append(entity)
            
            important_entities.sort(key=lambda x: x.importance, reverse=True)
            return important_entities[:limit]
        except Exception as e:
            logger.debug(f"获取重要实体失败: {e}")
            return []
    
    def _get_recent_entities(self, limit: int = 10) -> List[Entity]:
        """获取最近访问的实体（替代原_get_recent_memories）"""
        cortex = self._container["cortex"].cortex
        recent_entities: List[Entity] = []
        
        try:
            for entity in cortex.index.entities.values():
                if entity:
                    recent_entities.append(entity)
            
            recent_entities.sort(key=lambda x: x.last_accessed, reverse=True)
            return recent_entities[:limit]
        except Exception as e:
            logger.debug(f"获取最近实体失败: {e}")
            return []
    
    def _find_knowledge_gaps(self) -> List[str]:
        """基于最近实体发现知识缺口（逻辑不变，仅适配实体内容）"""
        gaps = []
        try:
            recent_entities = self._get_recent_entities(limit=10)
            keywords = set()
            
            for entity in recent_entities:
                # 从实体名称和最新证据中提取关键词
                content = f"{entity.name} {entity.latest_evidence.content if entity.latest_evidence else ''}"
                # 简单的二元分词（中文）
                words = [content[i:i+2] for i in range(len(content)-1)]
                keywords.update([w for w in words if len(w) == 2 and not w.isspace()])
            
            if keywords:
                sample_keywords = list(keywords)[:3]
                for kw in sample_keywords:
                    gaps.append(f"和{kw}相关的知识")
            
            gaps = list(set(gaps))
            random.shuffle(gaps)
            return gaps[:5]
        except Exception as e:
            logger.debug(f"知识缺口发现失败: {e}")
            return ["一些有趣的知识"]
    
    def _get_random_entity_associations(self, limit: int = 3) -> List[str]:
        """获取随机实体用于关联探索（替代原_get_random_associations）"""
        cortex = self._container["cortex"].cortex
        associations = []
        
        try:
            entities = list(cortex.index.entities.values())
            if len(entities) > 0:
                random_entities = random.sample(entities, min(limit, len(entities)))
                for entity in random_entities:
                    # 优先使用实体名称，更简洁
                    associations.append(entity.name)
        except Exception as e:
            logger.debug(f"获取随机实体关联失败: {e}")
        
        return associations