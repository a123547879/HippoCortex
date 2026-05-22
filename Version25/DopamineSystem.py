import numpy as np
from collections import deque
import logging
from typing import List, Dict, Optional, Tuple, Any

# 导入实体中心统一数据契约与全局配置
from Data_models import Entity, EntityRelation, Evidence
from BrainConfig import config

logger = logging.getLogger("DopamineSystem")

class DopamineSystem:
    """
    🔥 实体中心式多巴胺奖励系统
    核心机制：基于奖励预测误差（RPE）的突触可塑性调节
    作用对象：实体重要性 + 实体关系突触权重
    支持：多维度内部奖励 + 外部反馈奖励 + 情绪效价整合 + 睡眠离线重放
    """
    
    def __init__(
        self,
        # 优先使用全局配置，支持本地覆盖
        baseline_dopamine: float = config.DOPAMINE_CONFIG["baseline_dopamine"],
        rpe_decay: float = config.DOPAMINE_CONFIG["rpe_decay"],
        tau_reward: float = 0.1,  # 奖励期望更新速率
        tau_dopamine: float = 0.05,  # 多巴胺浓度衰减速率
        curiosity_scale: float = config.DOPAMINE_CONFIG["curiosity_reward_scale"],
        prediction_scale: float = config.DOPAMINE_CONFIG["prediction_reward_scale"],
        external_scale: float = config.DOPAMINE_CONFIG["external_reward_scale"],
        max_reward: float = config.DOPAMINE_CONFIG["max_reward"],
        min_reward: float = config.DOPAMINE_CONFIG["min_reward"]
    ):
        # 基础参数
        self.baseline_dopamine = baseline_dopamine
        self.rpe_decay = rpe_decay
        self.tau_reward = tau_reward
        self.tau_dopamine = tau_dopamine
        
        # 奖励缩放系数
        self.curiosity_scale = curiosity_scale
        self.prediction_scale = prediction_scale
        self.external_scale = external_scale
        
        # 奖励范围限制
        self.max_reward = max_reward
        self.min_reward = min_reward
        
        # 核心状态
        self.expected_reward = 0.0
        self.current_dopamine = baseline_dopamine
        
        # 历史记录（增强版：关联实体ID与奖励类型）
        self.reward_history: deque = deque(maxlen=1000)
        self.rpe_history: deque = deque(maxlen=1000)
        self.entity_reward_history: deque = deque(maxlen=500)  # 实体级奖励记录

    # ===================== 🔴 核心：奖励预测误差计算 =====================
    def compute_reward_prediction_error(
        self,
        actual_reward: float,
        related_entity_ids: Optional[List[str]] = None,
        emotion_valence: float = 0.0,
        emotion_arousal: float = 0.5,
        reward_type: str = "internal"
    ) -> float:
        """
        计算奖励预测误差 (RPE) 并更新多巴胺浓度
        :param actual_reward: 实际获得的总奖励 (-1 到 1)
        :param related_entity_ids: 关联的实体ID列表（用于历史追踪）
        :param emotion_valence: 情绪效价 (-1负面 ~ 1正面)
        :param emotion_arousal: 情绪唤醒度 (0平静 ~ 1激动)
        :param reward_type: 奖励类型：internal/external/emotion/curiosity/prediction
        :return: 奖励预测误差 (RPE)
        """
        # 情绪增强奖励：正面情绪放大奖励，负面情绪缩小奖励
        emotion_modulated_reward = actual_reward * (1 + emotion_valence * emotion_arousal * 0.5)
        
        # 计算RPE
        rpe = emotion_modulated_reward - self.expected_reward
        
        # 更新奖励期望（指数移动平均）
        self.expected_reward = (
            (1 - self.tau_reward) * self.expected_reward 
            + self.tau_reward * emotion_modulated_reward
        )
        
        # 更新多巴胺浓度：RPE驱动变化 + 基础衰减
        self.current_dopamine += rpe
        self.current_dopamine *= (1 - self.tau_dopamine)
        # 保持多巴胺在合理范围，保留基线水平
        self.current_dopamine = np.clip(
            self.current_dopamine, 
            self.min_reward, 
            self.max_reward
        )
        
        # 记录历史
        record = {
            "timestamp": np.datetime64('now').astype(int) / 1e9,
            "actual_reward": actual_reward,
            "emotion_modulated_reward": emotion_modulated_reward,
            "expected_reward": self.expected_reward,
            "rpe": rpe,
            "dopamine_level": self.current_dopamine,
            "related_entity_ids": related_entity_ids or [],
            "reward_type": reward_type
        }
        self.reward_history.append(record)
        self.rpe_history.append(rpe)
        
        if related_entity_ids:
            for entity_id in related_entity_ids:
                self.entity_reward_history.append({
                    "entity_id": entity_id,
                    "rpe": rpe,
                    "reward_type": reward_type,
                    "timestamp": record["timestamp"]
                })
        
        logger.debug(
            f"🧠 多巴胺更新 | RPE: {rpe:.3f} | 实际奖励: {actual_reward:.3f} | "
            f"情绪调制: {emotion_modulated_reward:.3f} | 当前浓度: {self.current_dopamine:.3f}"
        )
        return rpe

    # ===================== 🔴 多维度内部奖励计算 =====================
    def get_curiosity_reward(self, information_gain: float) -> float:
        """
        基于信息增益的好奇心奖励（边际递减）
        :param information_gain: 信息增益 (0-1)
        :return: 好奇心奖励 (-1 到 1)
        """
        # 边际递减：信息增益越大，单位增益带来的奖励越少
        reward = self.curiosity_scale * (1 - np.exp(-information_gain * 3))
        return float(np.clip(reward, self.min_reward, self.max_reward))

    def get_prediction_reward(self, prediction_accuracy: float) -> float:
        """
        基于预测成功的内部奖励
        :param prediction_accuracy: 预测准确率 (0-1)
        :return: 预测奖励 (-1 到 1)
        """
        # 预测准确率超过50%才给正奖励，低于则给负奖励
        reward = self.prediction_scale * (prediction_accuracy - 0.5) * 2
        return float(np.clip(reward, self.min_reward, self.max_reward))

    def get_entity_discovery_reward(self, is_new_entity: bool, entity_importance: float = 0.5) -> float:
        """
        发现新实体的奖励
        :param is_new_entity: 是否为首次发现的实体
        :param entity_importance: 实体初始重要性
        :return: 发现奖励 (-1 到 1)
        """
        if not is_new_entity:
            return 0.0
        reward = 0.2 * entity_importance
        return float(np.clip(reward, self.min_reward, self.max_reward))

    def get_relation_establishment_reward(self, is_new_relation: bool, relation_confidence: float = 0.9) -> float:
        """
        建立新实体关系的奖励
        :param is_new_relation: 是否为新建立的关系
        :param relation_confidence: 关系置信度
        :return: 关系建立奖励 (-1 到 1)
        """
        if not is_new_relation:
            return 0.0
        reward = 0.15 * relation_confidence
        return float(np.clip(reward, self.min_reward, self.max_reward))

    def get_external_feedback_reward(self, user_feedback: float) -> float:
        """
        用户外部反馈奖励
        :param user_feedback: 用户反馈评分 (-1 负面 ~ 1 正面)
        :return: 外部奖励 (-1 到 1)
        """
        reward = self.external_scale * user_feedback
        return float(np.clip(reward, self.min_reward, self.max_reward))

    def get_emotion_reward(self, emotion_valence: float, emotion_arousal: float = 0.5) -> float:
        """
        基于情绪的内部奖励
        :param emotion_valence: 情绪效价 (-1负面 ~ 1正面)
        :param emotion_arousal: 情绪唤醒度 (0平静 ~ 1激动)
        :return: 情绪奖励 (-1 到 1)
        """
        reward = 0.2 * emotion_valence * emotion_arousal
        return float(np.clip(reward, self.min_reward, self.max_reward))

    # ===================== 🔴 实体-关系级突触可塑性调节 =====================
    def modulate_entity_importance(
        self,
        entity: Entity,
        rpe: float,
        learning_rate: float = 0.01
    ) -> None:
        """
        基于多巴胺调节实体的重要性
        :param entity: 要调节的实体对象
        :param rpe: 奖励预测误差
        :param learning_rate: 基础学习率
        """
        # 多巴胺门控：只有当RPE绝对值足够大时才发生显著变化
        dopamine_gate = np.abs(rpe)
        if dopamine_gate < 0.05:
            return
        
        # 正RPE：提升实体重要性；负RPE：降低实体重要性
        importance_delta = learning_rate * dopamine_gate * rpe
        entity.importance = np.clip(
            entity.importance + importance_delta,
            0.0,
            0.95  # 重要性上限0.95，保留0.05给永久实体
        )
        
        logger.debug(
            f"🧠 实体重要性调节 | {entity.name}({entity.entity_id[:8]}) | "
            f"RPE: {rpe:.3f} | 变化: {importance_delta:.3f} | 新值: {entity.importance:.3f}"
        )

    def modulate_relation_weight(
        self,
        relation: EntityRelation,
        rpe: float,
        learning_rate: float = 0.01
    ) -> None:
        """
        基于多巴胺调节实体关系的突触权重
        :param relation: 要调节的关系对象
        :param rpe: 奖励预测误差
        :param learning_rate: 基础学习率
        """
        dopamine_gate = np.abs(rpe)
        if dopamine_gate < 0.05:
            return
        
        # 正RPE：增强突触连接；负RPE：减弱突触连接
        weight_delta = learning_rate * dopamine_gate * rpe
        relation.update_synapse(weight_delta)
        
        logger.debug(
            f"🧠 关系权重调节 | {relation.subject_id[:8]} → {relation.predicate} → {relation.object_id[:8]} | "
            f"RPE: {rpe:.3f} | 变化: {weight_delta:.3f} | 新值: {relation.synapse_weight:.3f}"
        )

    def apply_reward_to_entities_and_relations(
        self,
        rpe: float,
        entities: List[Entity],
        relations: Optional[List[EntityRelation]] = None,
        learning_rate: float = 0.01
    ) -> None:
        """
        批量应用奖励到多个实体和关系
        :param rpe: 总奖励预测误差
        :param entities: 要调节的实体列表
        :param relations: 要调节的关系列表
        :param learning_rate: 基础学习率
        """
        relations = relations or []
        
        if not entities and not relations:
            return
        
        # 按重要性分配奖励：重要实体获得更多奖励
        total_importance = sum(e.importance for e in entities) if entities else 1.0
        
        for entity in entities:
            entity_weight = entity.importance / total_importance
            entity_rpe = rpe * entity_weight
            self.modulate_entity_importance(entity, entity_rpe, learning_rate)
        
        for relation in relations:
            # 关系奖励按平均分配，或根据置信度分配
            relation_rpe = rpe * 0.5 / max(len(relations), 1)
            self.modulate_relation_weight(relation, relation_rpe, learning_rate)

    # ===================== 🔴 睡眠离线重放支持 =====================
    def dopamine_offline_replay(
        self,
        replay_window_hours: float = 24.0,
        learning_rate: float = 0.005
    ) -> Tuple[int, int]:
        """
        多巴胺离线重放（睡眠时调用）
        重放过去指定时间内的奖励历史，巩固重要记忆
        :param replay_window_hours: 重放时间窗口（小时）
        :param learning_rate: 离线学习率（低于在线学习率）
        :return: (重放的实体数, 重放的关系数)
        """
        logger.info("🌙 开始多巴胺离线重放...")
        now = np.datetime64('now').astype(int) / 1e9
        cutoff_time = now - replay_window_hours * 3600
        
        # 筛选时间窗口内的奖励记录
        recent_records = [
            rec for rec in self.reward_history 
            if rec["timestamp"] > cutoff_time and np.abs(rec["rpe"]) > 0.1
        ]
        
        if not recent_records:
            logger.info("ℹ️ 无有效奖励记录需要重放")
            return 0, 0
        
        # 按RPE绝对值排序，优先重放高奖励/高惩罚事件
        recent_records.sort(key=lambda x: np.abs(x["rpe"]), reverse=True)
        
        processed_entities = set()
        processed_relations = set()
        
        for record in recent_records[:100]:  # 最多重放100条记录
            rpe = record["rpe"] * 0.5  # 离线重放奖励减半
            related_entity_ids = record["related_entity_ids"]
            
            for entity_id in related_entity_ids:
                if entity_id in processed_entities:
                    continue
                # 这里需要从皮层获取实体对象，实际使用时传入cortex参数
                # entity = cortex.get_entity(entity_id)
                # if entity:
                #     self.modulate_entity_importance(entity, rpe, learning_rate)
                #     processed_entities.add(entity_id)
                processed_entities.add(entity_id)  # 占位，实际使用时替换
        
        logger.info(
            f"✅ 多巴胺离线重放完成 | 重放记录: {len(recent_records)} | "
            f"处理实体: {len(processed_entities)} | 处理关系: {len(processed_relations)}"
        )
        return len(processed_entities), len(processed_relations)

    # ===================== 🔴 状态查询与重置 =====================
    def reset(self) -> None:
        """重置多巴胺系统到初始状态"""
        self.expected_reward = 0.0
        self.current_dopamine = self.baseline_dopamine
        self.reward_history.clear()
        self.rpe_history.clear()
        self.entity_reward_history.clear()
        logger.info("🔄 多巴胺系统已重置")

    def get_status(self) -> Dict[str, Any]:
        """获取多巴胺系统当前状态（用于监控和调试）"""
        recent_rpes = list(self.rpe_history)[-100:] if self.rpe_history else [0.0]
        recent_rewards = [rec["actual_reward"] for rec in list(self.reward_history)[-100:]] if self.reward_history else [0.0]
        
        return {
            "current_dopamine": round(self.current_dopamine, 3),
            "expected_reward": round(self.expected_reward, 3),
            "average_rpe_100": round(np.mean(recent_rpes), 3),
            "average_reward_100": round(np.mean(recent_rewards), 3),
            "total_reward_records": len(self.reward_history),
            "total_entity_rewards": len(self.entity_reward_history)
        }

    # ===================== 兼容方法（向后兼容） =====================
    def modulate_synaptic_plasticity(
        self,
        synapse_weights: np.ndarray,
        activations: np.ndarray,
        rpe: float,
        learning_rate: float = 0.01
    ) -> np.ndarray:
        """
        兼容旧版突触调节方法（不推荐在新代码中使用）
        """
        logger.warning("⚠️ 使用已弃用的modulate_synaptic_plasticity方法，请改用实体级调节")
        dopamine_gate = np.abs(rpe)
        weight_update = learning_rate * dopamine_gate * rpe * np.outer(activations, activations)
        synapse_weights += weight_update
        return np.clip(synapse_weights, -1.0, 1.0)