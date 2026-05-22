import time
import logging
import re
from typing import List, Dict, Optional, Any, Tuple

import torch
from brain_core import BrainCore
from event_system import EventBus, Event, EventType
from Thalamus import Thalamus
from HippocampusRouter import HippocampusRouter
from SymbolicCore import SymbolicCore
from BrainConfig import config
from DopamineSystem import DopamineSystem
from Metacognition import Metacognition
# ✅ 替换为实体中心统一数据契约
from Data_models import SleepReport, SleepStageReport, Entity, Evidence, DreamResult

logger = logging.getLogger("ConsolidationLoop")

class ConsolidationLoop:
    def __init__(self, core: BrainCore, event_bus: EventBus, llm):
        self.core: BrainCore = core
        self.event_bus: EventBus = event_bus
        self.llm: Any = llm
        
        # 组件引用
        self.thalamus: Optional[Thalamus] = None
        self.hippocampus_router: Optional[HippocampusRouter] = None
        self.symbolic_core: Optional[SymbolicCore] = None
        self.experts: Dict[str, Any] = {}
        self.cortex: Optional[Any] = None
        self.dopamine: Optional[DopamineSystem] = None
        self.metacognition: Optional[Metacognition] = None
        self.dreaming_loop: Optional[Any] = None
        self.learning_loop: Optional[Any] = None
        self.perception_loop: Optional[Any] = None
        
        # 睡眠巩固配置开关（完全保留原有配置）
        self.use_predictive_consolidation: bool = True
        self.use_cross_modal_consolidation: bool = True
        self.use_dream_consolidation: bool = True
        self.manual_sleep_epochs_multiplier: float = 1.5

    def bind_components(self, thalamus: Thalamus, hippocampus_router: HippocampusRouter, 
                       symbolic_core: SymbolicCore, experts: Dict[str, Any], cortex: Any,
                       dopamine: DopamineSystem, metacognition: Metacognition,
                       dreaming_loop: Any, learning_loop: Any,
                       perception_loop: Any = None) -> None:
        """绑定其他组件引用"""
        self.thalamus = thalamus
        self.hippocampus_router = hippocampus_router
        self.symbolic_core = symbolic_core
        self.experts = experts
        self.cortex = cortex
        self.dopamine = dopamine
        self.metacognition = metacognition
        self.dreaming_loop = dreaming_loop
        self.learning_loop = learning_loop
        self.perception_loop = perception_loop
        
        logger.info(f"✅ ConsolidationLoop 组件绑定完成 | 共 {len(self.experts)} 个专家")
        if self.use_cross_modal_consolidation and self.perception_loop:
            logger.info(f"🧠 跨模态关联巩固已启用")

    # ===================== 🔴 核心：实体中心式全脑睡眠巩固 =====================
    def sleep_consolidate_all(self, epochs: int = 3, is_manual: bool = False) -> SleepReport:
        logger.info("\n" + "="*50)
        logger.info("🌙 大脑开始睡眠巩固...")
        logger.info(f"📌 睡眠类型：{'手动触发（全量巩固）' if is_manual else '自动触发（选择性巩固）'}")
        logger.info("="*50)
        
        self.event_bus.emit(Event(EventType.SLEEP_STARTED))
        
        sleep_start_time = time.time()
        total_energy_before, _ = self.core.energy_field.total_energy([], [], [], False, 0.0, False, 0.0)

        # 手动睡眠增加epochs，保证全量巩固
        actual_epochs = int(epochs * self.manual_sleep_epochs_multiplier) if is_manual else epochs
        if is_manual:
            logger.info(f"📌 手动睡眠模式：epochs从{epochs}增加到{actual_epochs}")

        # ===================== 阶段1：浅睡 =====================
        logger.info("\n🌙 【阶段1/3】浅睡中... 整理重要对话与记忆碎片")
        self._update_sleep_progress(20, "浅睡中：整理重要对话")
        
        important_turns = []
        if self.cortex and hasattr(self.cortex, 'get_pending_conversation_consolidation'):
            important_turns = self.cortex.get_pending_conversation_consolidation() or []
            
        if important_turns:
            logger.info(f"📝 开始巩固 {len(important_turns)} 条重要对话...")
            for turn in important_turns:
                try:
                    memory_text = f"[对话记录] 用户说：{turn.user_input}，我回答：{turn.ai_response}"
                    if self.learning_loop:
                        self.learning_loop.learn(memory_text, force_expert="抽象", external_reward=0.5)
                except Exception as e:
                    logger.debug(f"跳过对话巩固: {e}")
        logger.info("✅ 浅睡完成")

        # ===================== 阶段2：深睡 =====================
        logger.info("\n🌙 【阶段2/3】深睡中... 核心记忆巩固与突触重塑")
        self._update_sleep_progress(70, "深睡中：核心记忆巩固与突触重塑")
        
        # 🔴 元认知评估实体优先级（替代原记忆优先级）
        high_priority_entities: List[Entity] = self._evaluate_entity_priorities(is_manual)
        
        thalamus_consolidated = 0
        thalamus_forgotten = 0
        if self.thalamus and hasattr(self.thalamus, 'coordinate_consolidation'):
            try:
                thalamus_consolidated, thalamus_forgotten = self.thalamus.coordinate_consolidation(
                    epochs=actual_epochs, 
                    priority_entities=high_priority_entities if high_priority_entities else None
                )
            except Exception as e:
                logger.error(f"丘脑协调巩固失败: {e}", exc_info=True)
        
        self._consolidate_hippocampal_buffer()
        self._dopamine_offline_replay(high_priority_entities)
        
        # 跨模态关联巩固（适配实体体系）
        cross_modal_consolidated = 0
        if self.use_cross_modal_consolidation and self.perception_loop:
            cross_modal_consolidated = self._cross_modal_consolidation(high_priority_entities)
        
        expert_consolidated, total_pruned, total_created = self._consolidate_experts(actual_epochs, high_priority_entities)

        total_consolidated = thalamus_consolidated + expert_consolidated + cross_modal_consolidated
        total_forgotten = thalamus_forgotten
        logger.info(f"✅ 深睡完成")
        logger.info(f"   - 丘脑短期巩固: {thalamus_consolidated}个实体")
        logger.info(f"   - 专家长期巩固: {expert_consolidated}个实体")
        logger.info(f"   - 跨模态关联巩固: {cross_modal_consolidated}对关联")
        logger.info(f"   - 总计巩固: {total_consolidated}个实体")
        logger.info(f"   - 主动遗忘: {total_forgotten}个实体")
        logger.info(f"   - 突触修剪: {total_pruned}个")
        logger.info(f"   - 突触新生: {total_created}个")

        # ===================== 阶段3：REM =====================
        logger.info("\n🌙 【阶段3/3】快速眼动睡眠中... 记忆整合与梦境生成")
        self._update_sleep_progress(95, "快速眼动睡眠中：生成梦境与整合记忆")
        
        dream_result: Optional[DreamResult] = None
        dream_consolidated = 0
        if self.dreaming_loop:
            try:
                self.dreaming_loop.expert_dreams = {}
                self.dreaming_loop.last_dream = None
                self.dreaming_loop.collect_expert_dreams()
                # 🔴 传入高优先级实体（替代原高优先级记忆）
                dream_result = self.dreaming_loop.generate_global_dream(high_priority_entities)
                
                if self.use_dream_consolidation and dream_result and dream_result.success:
                    dream_consolidated = len(dream_result.fragments) if hasattr(dream_result, 'fragments') else 0
                    logger.info(f"✅ 梦境巩固完成 | 强化了 {dream_consolidated} 条实体连接")
            except Exception as e:
                logger.error(f"梦境生成失败: {e}", exc_info=True)
        logger.info("✅ REM睡眠完成")

        # ===================== 睡眠收尾：元认知更新 =====================
        logger.info("\n🧠 元认知：更新知识置信度...")
        updated_count = 0
        if self.metacognition and high_priority_entities:
            for entity in high_priority_entities:
                try:
                    # 🔴 用实体名称作为概念键（更准确）
                    concept_key = entity.name
                    # 巩固后置信度提升10-20%
                    confidence_increase = 0.1 + 0.1 * entity.importance
                    self.metacognition.update_knowledge_confidence(concept_key, confidence_increase)
                    updated_count += 1
                except Exception as e:
                    logger.debug(f"置信度更新失败: {e}")
        logger.info(f"✅ 元认知更新完成 | 共更新 {updated_count} 个实体的置信度")

        # ===================== 唤醒 =====================
        if self.thalamus:
            self.thalamus.update_brain_state("awake")
        self.core.fatigue_level = 0.0
        self.core.is_mind_wandering = False
        self.core.needs_sleep_request = False
        
        sleep_end_time = time.time()
        total_energy_after, _ = self.core.energy_field.total_energy([], [], [], False, 0.0, False, 0.0)
        sleep_duration = round(sleep_end_time - sleep_start_time, 2)
        energy_consumed = round(max(0.0, total_energy_before - total_energy_after), 2)
        
        try:
            total_entities = len(self.cortex.index.entities) if self.cortex and hasattr(self.cortex, 'index') else 0
        except Exception as e:
            logger.debug(f"获取总实体数失败: {e}")
            total_entities = 0

        # ===================== 创建阶段报告 =====================
        light_sleep = SleepStageReport(
            important_conversations_consolidated=len(important_turns)
        )
        deep_sleep = SleepStageReport(
            high_priority_count=len(high_priority_entities),
            thalamus_consolidated=thalamus_consolidated,
            expert_consolidated=expert_consolidated,
            cross_modal_consolidated=cross_modal_consolidated,
            synapses_pruned=total_pruned,
            synapses_created=total_created,
            consolidated=total_consolidated,
            forgotten=total_forgotten
        )
        rem_sleep = SleepStageReport(
            dream_generated=dream_result is not None and dream_result.success,
            dream_content=dream_result.content if (dream_result and dream_result.success) else None,
            dream_consolidated=dream_consolidated
        )

        sleep_report = SleepReport(
            is_manual=is_manual,
            stages={"light_sleep": light_sleep, "deep_sleep": deep_sleep, "rem_sleep": rem_sleep},
            consolidated_count=total_consolidated,
            forgotten_count=total_forgotten,
            sleep_duration=sleep_duration,
            energy_consumed=energy_consumed,
            total_memories=total_entities,  # 兼容字段，实际为实体总数
            total_entities=total_entities,  # 新增实体中心字段
            dream_content=dream_result.content if (dream_result and dream_result.success) else "",
            quality_score=0,
            quality_rating=""
        )

        # 计算睡眠质量
        sleep_report.quality_score = self._calculate_sleep_quality(sleep_report)
        if sleep_report.quality_score >= 90:
            sleep_report.quality_rating = "极佳 🎉"
        elif sleep_report.quality_score >= 75:
            sleep_report.quality_rating = "良好 ✅"
        elif sleep_report.quality_score >= 60:
            sleep_report.quality_rating = "一般 ⚠️"
        else:
            sleep_report.quality_rating = "较差 ❌"

        # 打印完整报告
        logger.info("\n" + "="*50)
        logger.info("✅ 全脑睡眠巩固完成！睡眠质量报告")
        logger.info(f"📊 总实体数：{sleep_report.total_entities}")
        logger.info(f"🧠 巩固：{sleep_report.consolidated_count}个 | 遗忘：{sleep_report.forgotten_count}个")
        logger.info(f"🔗 跨模态关联：{cross_modal_consolidated}对")
        logger.info(f"⚡ 突触变化：修剪{total_pruned}个 | 新生{total_created}个")
        logger.info(f"⏱️  睡眠时长：{sleep_report.sleep_duration}秒 | 能量消耗：{sleep_report.energy_consumed}")
        logger.info(f"🏆 睡眠质量：{sleep_report.quality_rating} ({sleep_report.quality_score}分)")
        if sleep_report.dream_content:
            logger.info(f"💭 梦境内容：{sleep_report.dream_content[:80]}...")
        logger.info("="*50 + "\n")

        self.event_bus.emit(Event(EventType.SLEEP_FINISHED, sleep_report))
        
        return sleep_report

    def _update_sleep_progress(self, progress: int, message: str) -> None:
        """发送睡眠进度更新事件（完全保留原有逻辑）"""
        safe_progress = max(0, min(100, progress))
        self.event_bus.emit(Event(EventType.SLEEP_PROGRESS_UPDATED, {
            "progress": safe_progress,
            "message": message
        }))
        logger.debug(f"睡眠进度：{safe_progress}% - {message}")

    # ===================== 🔴 元认知实体优先级评估 =====================
    def _evaluate_entity_priorities(self, is_manual: bool = False) -> List[Entity]:
        """元认知驱动的实体优先级评估（替代原记忆优先级评估）"""
        high_priority_entities: List[Entity] = []
        if not self.metacognition:
            return high_priority_entities
            
        try:
            logger.info("🧠 元认知：评估实体优先级...")
            
            all_entities: List[Entity] = []
            if hasattr(self.cortex, 'index') and hasattr(self.cortex.index, 'entities'):
                all_entities = list(self.cortex.index.entities.values())
            
            if not all_entities:
                logger.info("ℹ️ 没有可巩固的实体")
                return high_priority_entities
            
            entity_priorities = []
            for entity in all_entities:
                try:
                    # 🔴 用实体名称作为概念键
                    concept_key = entity.name
                    confidence = self.metacognition.assess_knowledge_confidence(concept_key)
                    confidence_priority = 1.0 - confidence
                    
                    # 实体固有重要性
                    importance = entity.importance
                    
                    # 新鲜度：基于最后访问时间
                    recency = 1.0 - min(1.0, (time.time() - entity.last_accessed) / (7 * 24 * 3600))
                    
                    # 跨模态实体优先级提升20%
                    cross_modal_bonus = 0.2 if "multimodal_id" in entity.metadata else 0.0
                    
                    # 永久实体优先级提升30%
                    permanent_bonus = 0.3 if entity.is_permanent else 0.0
                    
                    total_priority = (
                        0.4 * confidence_priority +
                        0.35 * importance +
                        0.25 * recency +
                        cross_modal_bonus +
                        permanent_bonus
                    )
                    
                    entity_priorities.append((total_priority, entity))
                except Exception as e:
                    entity_id = getattr(entity, 'entity_id', '未知')
                    logger.debug(f"跳过无效实体 ID={entity_id}: {e}")
            
            entity_priorities.sort(key=lambda x: -x[0])
            
            if is_manual:
                high_priority_entities = [entity for (p, entity) in entity_priorities]
                logger.info(f"📌 手动睡眠模式：全量巩固 {len(high_priority_entities)} 个实体")
            else:
                top_count = max(3, int(len(entity_priorities) * 0.2))
                high_priority_entities = [entity for (p, entity) in entity_priorities[:top_count]]
                logger.info(f"🧠 元认知：筛选出 {len(high_priority_entities)} 个高优先级实体")
            
            for i, (p, entity) in enumerate(entity_priorities[:3]):
                cross_modal_marker = "🔗" if "multimodal_id" in entity.metadata else ""
                permanent_marker = "🔒" if entity.is_permanent else ""
                logger.info(f"   优先级 {i+1}: {cross_modal_marker}{permanent_marker} {entity.name} (P={p:.2f})")
                
        except Exception as e:
            logger.error(f"元认知优先级评估失败: {e}", exc_info=True)
        
        return high_priority_entities

    # ===================== 🔴 海马体缓存固化（适配实体） =====================
    def _consolidate_hippocampal_buffer(self) -> None:
        """固化海马体临时实体到对应专家"""
        if not self.hippocampus_router or not hasattr(self.hippocampus_router, "hippocampal_buffer"):
            return
            
        buffer = self.hippocampus_router.hippocampal_buffer
        if not buffer:
            return
            
        logger.info(f"🧠 固化海马体缓存：{len(buffer)} 个实体写入对应专家")
        success_count = 0
        
        for entity in buffer:
            try:
                expert_name = entity.expert
                target_expert = self.experts.get(expert_name)
                if target_expert:
                    target_expert.add_entity(entity)
                    success_count += 1
            except Exception as e:
                entity_id = getattr(entity, 'entity_id', '未知')
                logger.debug(f"实体 ID={entity_id} 固化失败: {e}")
        
        if success_count == len(buffer):
            buffer.clear()
            logger.info(f"✅ 海马体缓存全部固化完成")
        else:
            logger.warning(f"⚠️ 海马体缓存部分固化失败 | 成功:{success_count}/{len(buffer)}")

    # ===================== 🔴 多巴胺离线重放（适配实体） =====================
    def _dopamine_offline_replay(self, high_priority_entities: List[Entity]) -> None:
        """多巴胺离线重放高优先级实体（升级为预测性STDP）"""
        if not self.dopamine or not high_priority_entities:
            return
            
        try:
            logger.info("🧪 多巴胺系统：离线重放高优先级实体...")
            
            for entity in high_priority_entities:
                try:
                    importance = entity.importance
                    simulated_reward = 0.3 + 0.5 * importance
                    rpe = self.dopamine.compute_reward_prediction_error(simulated_reward)
                    
                    expert_name = entity.expert
                    expert = self.experts.get(expert_name)
                    
                    if expert and entity.sdr is not None:
                        if self.use_predictive_consolidation and hasattr(expert, 'predictive_std_update'):
                            # 预测性STDP巩固
                            predicted_sdr = expert.predict_next_sdr(entity.sdr)
                            expert.predictive_std_update(
                                pre_sdr=entity.sdr,
                                post_sdr=predicted_sdr,
                                prediction_error=torch.tensor(rpe, device=entity.sdr.device)
                            )
                            logger.debug(f"   预测性STDP重放: {entity.name} | RPE={rpe:.2f}")
                        else:
                            # 降级为经典STDP
                            original_lr = expert.stdp_learning_rate
                            expert.stdp_learning_rate = original_lr * (1.0 + rpe * 0.5)
                            expert.stdp_update(entity.sdr, entity.sdr, delta_t=10.0)
                            expert.stdp_learning_rate = original_lr
                            logger.debug(f"   经典STDP重放: {entity.name} | RPE={rpe:.2f}")
                    
                except Exception as e:
                    logger.debug(f"单个实体重放失败: {e}")
            
            logger.info(f"🧪 多巴胺离线重放完成，共重放 {len(high_priority_entities)} 个实体")
        except Exception as e:
            logger.debug(f"多巴胺离线重放跳过: {e}")

    # ===================== 🔴 跨模态关联巩固（适配实体） =====================
    def _cross_modal_consolidation(self, high_priority_entities: List[Entity]) -> int:
        """专门巩固文本-视觉实体之间的跨模态关联"""
        if not self.perception_loop or not hasattr(self.perception_loop, 'cross_modal_learning_step'):
            return 0
            
        consolidated_count = 0
        multimodal_pairs = []
        
        try:
            logger.info("🔗 开始跨模态关联巩固...")
            
            # 收集所有带multimodal_id的实体对
            multimodal_map = {}
            for entity in high_priority_entities:
                if "multimodal_id" in entity.metadata:
                    mid = entity.metadata["multimodal_id"]
                    if mid not in multimodal_map:
                        multimodal_map[mid] = {"text": None, "vision": None}
                    if entity.entity_type == "visual" or entity.expert == "视觉":
                        multimodal_map[mid]["vision"] = entity
                    else:
                        multimodal_map[mid]["text"] = entity
            
            # 筛选出完整的文本-视觉对
            for mid, pair in multimodal_map.items():
                if pair["text"] and pair["vision"] and pair["text"].sdr is not None and pair["vision"].sdr is not None:
                    multimodal_pairs.append(pair)
            
            logger.info(f"🔍 找到 {len(multimodal_pairs)} 个完整的跨模态实体对")
            
            # 对每个配对执行跨模态学习
            for pair in multimodal_pairs:
                try:
                    loss = self.perception_loop.cross_modal_learning_step(
                        text_features=pair["text"].sdr,
                        vision_features=pair["vision"].sdr,
                        target_expert="视觉"
                    )
                    consolidated_count += 1
                    logger.debug(f"   跨模态巩固: {pair['text'].name} ↔ {pair['vision'].name} | 损失={loss:.4f}")
                except Exception as e:
                    logger.debug(f"   跨模态巩固失败: {e}")
            
            logger.info(f"✅ 跨模态关联巩固完成 | 共巩固 {consolidated_count} 对关联")
            
        except Exception as e:
            logger.error(f"跨模态关联巩固失败: {e}", exc_info=True)
        
        return consolidated_count

    # ===================== 🔴 专家睡眠巩固（适配实体） =====================
    def _consolidate_experts(self, epochs: int, high_priority_entities: List[Entity]) -> Tuple[int, int, int]:
        """执行所有专家的睡眠巩固，返回(巩固数量, 修剪数量, 新生数量)"""
        total_expert_consolidated = 0
        total_pruned = 0
        total_created = 0
        
        for name, expert in self.experts.items():
            if not expert:
                continue
                
            try:
                # 符号-神经绑定巩固（完全保留原有逻辑）
                if name == "概念" and self.symbolic_core and hasattr(self.symbolic_core, 'entities'):
                    try:
                        entities_count = len(self.symbolic_core.entities)
                        logger.info(f"🧠 巩固符号-神经绑定：{entities_count} 个实体")
                        for ent_name, ent_info in self.symbolic_core.entities.items():
                            if isinstance(ent_info, dict) and "neurons" in ent_info:
                                ent_sdr = torch.zeros(expert.dim)
                                ent_sdr[ent_info["neurons"]] = 1.0
                                expert.hebbian_update(ent_sdr, ent_sdr, is_fact=True)
                    except Exception as e:
                        logger.debug(f"符号-神经绑定巩固跳过: {e}")
                
                # 筛选属于该专家的高优先级实体
                expert_priority_entity_ids = []
                if high_priority_entities:
                    expert_priority_entity_ids = [
                        entity.entity_id for entity in high_priority_entities
                        if entity.expert == name
                    ]
                
                # 调用专家的睡眠巩固
                if hasattr(expert, 'sleep_consolidate'):
                    if expert_priority_entity_ids:
                        logger.info(f"🌙 [{name}] 专家优先巩固 {len(expert_priority_entity_ids)} 个高优先级实体")
                        expert.sleep_consolidate(epochs=epochs, priority_entity_ids=expert_priority_entity_ids)
                        total_expert_consolidated += len(expert_priority_entity_ids)
                    else:
                        expert.sleep_consolidate(epochs=epochs)
                    
                    # 收集突触修剪和新生统计
                    if hasattr(expert, 'total_synapses_pruned'):
                        total_pruned += expert.total_synapses_pruned
                    if hasattr(expert, 'total_synapses_created'):
                        total_created += expert.total_synapses_created
                        
            except Exception as e:
                logger.error(f"[{name}] 专家巩固失败: {e}", exc_info=True)
        
        return total_expert_consolidated, total_pruned, total_created

    def _calculate_sleep_quality(self, report: SleepReport) -> float:
        """计算睡眠质量评分（0-100分，完全保留原有逻辑）"""
        if report.consolidated_count == 0 and report.forgotten_count == 0:
            return 85.0
        
        total_processed = report.consolidated_count + report.forgotten_count
        consolidation_rate = report.consolidated_count / total_processed if total_processed > 0 else 0
        consolidation_score = consolidation_rate * 35
        
        # 适度遗忘是健康的，最佳遗忘率10-20%
        forget_rate = report.forgotten_count / total_processed if total_processed > 0 else 0
        optimal_forget_rate = 0.15
        forget_penalty = abs(forget_rate - optimal_forget_rate) * 100
        forget_score = max(0.0, 25 - forget_penalty)
        
        # 跨模态巩固加分
        cross_modal_score = 0.0
        if hasattr(report.stages["deep_sleep"], 'cross_modal_consolidated'):
            cross_modal_count = report.stages["deep_sleep"].cross_modal_consolidated
            cross_modal_score = min(10.0, cross_modal_count * 0.5)
        
        # 突触重塑加分（健康的突触变化）
        synapse_score = 0.0
        if hasattr(report.stages["deep_sleep"], 'synapses_pruned') and hasattr(report.stages["deep_sleep"], 'synapses_created'):
            pruned = report.stages["deep_sleep"].synapses_pruned
            created = report.stages["deep_sleep"].synapses_created
            if pruned > 0 and created > 0:
                synapse_score = 10.0
        
        # 能量消耗评分（最佳消耗5-15）
        optimal_energy = 10.0
        energy_penalty = abs(report.energy_consumed - optimal_energy) * 2
        energy_score = max(0.0, 20 - energy_penalty)
        
        # 睡眠阶段完整性评分
        stage_score = 10.0
        if not report.stages["rem_sleep"].dream_generated:
            stage_score -= 3.0
        if report.stages["deep_sleep"].expert_consolidated > 0:
            stage_score += 2.0
        
        total_score = consolidation_score + forget_score + cross_modal_score + synapse_score + energy_score + stage_score
        return round(min(100.0, max(0.0, total_score)), 1)