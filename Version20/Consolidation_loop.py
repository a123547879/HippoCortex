import time
import logging
import re
from typing import List, Dict

import torch
from brain_core import BrainCore
from event_system import EventBus, Event, EventType
from Thalamus import Thalamus
from HippocampusRouterV10 import HippocampusRouter
from SymbolicCore import SymbolicCore
from BrainConfig import config
from DopamineSystem import DopamineSystem
from Metacognition import Metacognition
from Data_models import SleepReport, SleepStageReport, MemoryPacket, DreamResult

logger = logging.getLogger("ConsolidationLoop")

class ConsolidationLoop:
    def __init__(self, core: BrainCore, event_bus: EventBus, llm):
        self.core = core
        self.event_bus = event_bus
        self.llm = llm
        
        # 组件引用（由CognitiveSystem注入）
        self.thalamus = None
        self.hippocampus_router = None
        self.symbolic_core = None
        self.experts = {}
        self.cortex = None
        self.dopamine = None
        self.metacognition = None
        self.dreaming_loop = None
        self.learning_loop = None

    def bind_components(self, thalamus: Thalamus, hippocampus_router: HippocampusRouter, 
                       symbolic_core: SymbolicCore, experts: Dict, cortex,
                       dopamine: DopamineSystem, metacognition: Metacognition,
                       dreaming_loop, learning_loop):
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

    def sleep_consolidate_all(self, epochs=3, is_manual: bool = False) -> SleepReport:
        logger.info("\n" + "="*50)
        logger.info("🌙 大脑开始睡眠巩固...")
        logger.info(f"📌 睡眠类型：{'手动触发（全量巩固）' if is_manual else '自动触发（选择性巩固）'}")
        logger.info("="*50)
        
        self.event_bus.emit(Event(EventType.SLEEP_STARTED))
        
        sleep_start_time = time.time()
        total_energy_before = self.core.energy_field.total_energy([], [], [], False, 0.0, False, 0.0)[0]
        
        # 🔥 初始化强类型睡眠报告（替代原来的裸字典）
        sleep_report = SleepReport(
            is_manual=is_manual,
            stages={
                "light_sleep": SleepStageReport(),
                "deep_sleep": SleepStageReport(),
                "rem_sleep": SleepStageReport()
            }
        )

        # ===================== 阶段1：浅睡 =====================
        logger.info("\n🌙 【阶段1/3】浅睡中... 整理重要对话与记忆碎片")
        self._update_sleep_progress(20, "浅睡中：整理重要对话")
        
        important_turns = self.cortex.get_pending_conversation_consolidation()
        if important_turns:
            logger.info(f"📝 开始巩固 {len(important_turns)} 条重要对话...")
            for turn in important_turns:
                memory_text = f"[对话记录] 用户说：{turn['user_input']}，我回答：{turn['ai_response']}"
                self.learning_loop.learn(memory_text, force_expert="抽象", external_reward=0.5)
        
        # 🔥 赋值给强类型字段
        sleep_report.stages["light_sleep"].important_conversations_consolidated = len(important_turns)
        logger.info("✅ 浅睡完成")

        # ===================== 阶段2：深睡 =====================
        logger.info("\n🌙 【阶段2/3】深睡中... 核心记忆巩固与突触强化")
        self._update_sleep_progress(70, "深睡中：核心记忆巩固")
        
        # 🔥 元认知返回的是MemoryPacket列表
        high_priority_memories: List[MemoryPacket] = self._evaluate_memory_priorities(is_manual)
        
        thalamus_consolidated, thalamus_forgotten = self.thalamus.coordinate_consolidation(
            epochs=epochs, 
            priority_memories=high_priority_memories if high_priority_memories else None
        )
        
        self._consolidate_hippocampal_buffer()
        self._dopamine_offline_replay(high_priority_memories)
        expert_consolidated = self._consolidate_experts(epochs, high_priority_memories)

        total_consolidated = thalamus_consolidated + expert_consolidated
        total_forgotten = thalamus_forgotten

        # 🔥 赋值给强类型字段
        sleep_report.stages["deep_sleep"] = SleepStageReport(
            high_priority_count=len(high_priority_memories),
            thalamus_consolidated=thalamus_consolidated,
            expert_consolidated=expert_consolidated,
            consolidated=total_consolidated,
            forgotten=total_forgotten
        )
        sleep_report.consolidated_count = total_consolidated
        sleep_report.forgotten_count = total_forgotten
        
        logger.info(f"✅ 深睡完成 | 丘脑短期巩固:{thalamus_consolidated}条 | 专家长期巩固:{expert_consolidated}条 | 总计:{total_consolidated}条 | 主动遗忘:{thalamus_forgotten}条")

        # ===================== 阶段3：REM =====================
        logger.info("\n🌙 【阶段3/3】快速眼动睡眠中... 记忆整合与梦境生成")
        self._update_sleep_progress(95, "快速眼动睡眠中：生成梦境")
        
        self.dreaming_loop.expert_dreams = {}
        self.dreaming_loop.last_dream = None
        
        self.dreaming_loop.collect_expert_dreams()
        # 🔥 DreamingLoop现在返回DreamResult强类型对象
        dream_result: DreamResult = self.dreaming_loop.generate_global_dream(high_priority_memories)
        
        if dream_result.success:
            sleep_report.dream_content = dream_result.content
            logger.info(f"💭 生成梦境：{dream_result.content}")

        sleep_report.stages["rem_sleep"] = SleepStageReport(
            dream_generated=dream_result.success,
            dream_content=dream_result.content
        )
        logger.info("✅ REM睡眠完成")

        # ===================== 睡眠收尾 =====================
        self.thalamus.update_brain_state("awake")
        self.core.fatigue_level = 0.0
        self.core.is_mind_wandering = False
        self.core.needs_sleep_request = False
        
        sleep_end_time = time.time()
        total_energy_after = self.core.energy_field.total_energy([], [], [], False, 0.0, False, 0.0)[0]
        sleep_report.sleep_duration = round(sleep_end_time - sleep_start_time, 2)
        sleep_report.energy_consumed = round(total_energy_before - total_energy_after, 2)
        
        try:
            sleep_report.total_memories = len(self.cortex.index.memories)
        except Exception as e:
            logger.debug(f"获取总记忆数失败: {e}")
            sleep_report.total_memories = 0
        
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

        # 打印报告
        logger.info("\n" + "="*50)
        logger.info("✅ 全脑睡眠巩固完成！睡眠质量报告")
        logger.info(f"📊 总记忆数：{sleep_report.total_memories} | 巩固：{sleep_report.consolidated_count}条 | 遗忘：{sleep_report.forgotten_count}条")
        logger.info(f"⏱️  睡眠时长：{sleep_report.sleep_duration}秒 | 能量消耗：{sleep_report.energy_consumed}")
        logger.info(f"🏆 睡眠质量：{sleep_report.quality_rating} ({sleep_report.quality_score}分)")
        if sleep_report.dream_content:
            logger.info(f"💭 梦境内容：{sleep_report.dream_content[:50]}...")
        logger.info("="*50 + "\n")

        self.event_bus.emit(Event(EventType.SLEEP_FINISHED, sleep_report))
        
        # 🔥 返回强类型对象，不再是裸字典
        return sleep_report

    def _update_sleep_progress(self, progress: int, message: str):
        """发送睡眠进度更新事件（UI可监听）"""
        self.event_bus.emit(Event(EventType.SLEEP_PROGRESS_UPDATED, {
            "progress": progress,
            "message": message
        }))
        logger.debug(f"睡眠进度：{progress}% - {message}")

    def _evaluate_memory_priorities(self, is_manual: bool = False) -> List[MemoryPacket]:
        """元认知驱动的记忆优先级评估（支持手动全量巩固）"""
        high_priority_memories: List[MemoryPacket] = []
        if not self.metacognition:
            return high_priority_memories
            
        try:
            logger.info("🧠 元认知：评估记忆优先级...")
            
            # ✅ 修复1：直接获取已有的 MemoryPacket 对象（不是字典！）
            all_memories: List[MemoryPacket] = []
            if hasattr(self.cortex, 'index') and hasattr(self.cortex.index, 'memories'):
                all_memories = list(self.cortex.index.memories.values())
            
            if not all_memories:
                return high_priority_memories
            
            # 为每个记忆计算复习优先级
            memory_priorities = []
            for mem in all_memories:
                try:
                    concept_key = self._extract_concept_key(mem.content)
                    
                    # 1. 元认知置信度：越不熟悉越优先
                    confidence = self.metacognition.assess_knowledge_confidence(concept_key)
                    confidence_priority = 1.0 - confidence
                    
                    # 2. 记忆重要性：使用MemoryPacket内置属性
                    importance = mem.importance
                    
                    # 3. 记忆新近度：越新越优先
                    timestamp = mem.metadata.get("timestamp", time.time())
                    recency = 1.0 - min(1.0, (time.time() - timestamp) / (7 * 24 * 3600))  # 一周内
                    
                    # 综合优先级
                    total_priority = (
                        0.4 * confidence_priority +
                        0.35 * importance +
                        0.25 * recency
                    )
                    
                    memory_priorities.append((total_priority, mem))
                except Exception as e:
                    # ✅ 修复2：用 getattr 安全获取对象属性（替代字典get）
                    mem_id = getattr(mem, 'mem_id', '未知')
                    logger.debug(f"跳过无效记忆 ID={mem_id}: {e}")
            
            # 按优先级排序
            memory_priorities.sort(key=lambda x: -x[0])
            
            # 手动睡眠：返回所有记忆
            if is_manual:
                high_priority_memories = [mem for (p, mem) in memory_priorities]
                logger.info(f"📌 手动睡眠模式：返回所有 {len(high_priority_memories)} 条记忆进行全量巩固")
            else:
                # 自动睡眠：取Top 20%
                top_count = max(3, int(len(memory_priorities) * 0.2))
                high_priority_memories = [mem for (p, mem) in memory_priorities[:top_count]]
                logger.info(f"🧠 元认知：筛选出 {len(high_priority_memories)} 条高优先级记忆进行优先巩固")
            
            for i, (p, mem) in enumerate(memory_priorities[:5]):
                logger.debug(f"   优先级 {i+1}: {mem.content[:30]}... (P={p:.2f})")
                
        except Exception as e:
            logger.error(f"元认知优先级评估失败: {e}", exc_info=True)
        
        return high_priority_memories

    def _consolidate_hippocampal_buffer(self):
        """固化海马体临时记忆到对应专家（适配MemoryPacket对象）"""
        if hasattr(self.hippocampus_router, "hippocampal_buffer"):
            buffer = self.hippocampus_router.hippocampal_buffer
            if buffer:
                logger.info(f"🧠 固化海马体缓存：{len(buffer)} 条记忆写入对应专家")
                for mem in buffer:
                    # 对象属性访问（无错误）
                    expert_name = mem.expert
                    target_expert = self.experts.get(expert_name)
                    if target_expert:
                        target_expert.add_memory(
                            sdr=mem.sdr,
                            content=mem.content,
                            mem_id=mem.mem_id,
                            metadata=mem.metadata
                        )
                buffer.clear()

    def _dopamine_offline_replay(self, high_priority_memories: List[MemoryPacket]):
        """多巴胺离线重放（强化重要突触）"""
        if not self.dopamine or not high_priority_memories:
            return
            
        try:
            logger.info("🧪 多巴胺系统：离线重放高优先级记忆...")
            
            for mem in high_priority_memories:
                importance = mem.importance
                simulated_reward = 0.3 + 0.5 * importance
                
                rpe = self.dopamine.compute_reward_prediction_error(simulated_reward)
                logger.debug(f"   多巴胺重放: {mem.content[:25]}... | RPE={rpe:.2f}")
            
            logger.info(f"🧪 多巴胺离线重放完成，共重放 {len(high_priority_memories)} 条记忆")
        except Exception as e:
            logger.debug(f"多巴胺离线重放跳过: {e}")

    def _consolidate_experts(self, epochs: int, high_priority_memories: List[MemoryPacket]) -> int:
        """执行所有专家的睡眠巩固，返回总巩固数量"""
        total_expert_consolidated = 0
        
        for name, expert in self.experts.items():
            # 符号-神经绑定巩固
            if name == "概念" and self.symbolic_core:
                try:
                    logger.info(f"🧠 巩固符号-神经绑定：{len(self.symbolic_core.entities)} 个实体")
                    for ent_name, ent_info in self.symbolic_core.entities.items():
                        ent_sdr = torch.zeros(expert.dim)
                        ent_sdr[ent_info["neurons"]] = 1.0
                        expert.hebbian_update(ent_sdr, ent_sdr, is_fact=True)
                except Exception as e:
                    logger.debug(f"符号-神经绑定巩固跳过: {e}")
            
            # 筛选属于该专家的高优先级记忆
            expert_priority_mem_ids = []
            if high_priority_memories:
                expert_priority_mem_ids = [
                    mem.mem_id for mem in high_priority_memories
                    if mem.expert == name
                ]
            
            # 调用专家的睡眠巩固
            if hasattr(expert, 'sleep_consolidate'):
                if expert_priority_mem_ids:
                    logger.info(f"🌙 [{name}] 专家优先巩固 {len(expert_priority_mem_ids)} 条高优先级记忆")
                    expert.sleep_consolidate(epochs=epochs, priority_mem_ids=expert_priority_mem_ids)
                    total_expert_consolidated += len(expert_priority_mem_ids)
                else:
                    expert.sleep_consolidate(epochs=epochs)
            
        return total_expert_consolidated

    def _extract_concept_key(self, text: str) -> str:
        """从文本中提取核心概念关键词"""
        clean_text = re.sub(r'[^\w\s]', '', text)
        words = clean_text.split()
        if words:
            return max(words, key=len)
        return text[:10]

    def _calculate_sleep_quality(self, report: SleepReport) -> float:
        """计算睡眠质量评分（0-100分）"""
        if report.consolidated_count == 0 and report.forgotten_count == 0:
            return 85.0
        
        total_processed = report.consolidated_count + report.forgotten_count
        consolidation_rate = report.consolidated_count / total_processed if total_processed > 0 else 0
        consolidation_score = consolidation_rate * 40
        
        forget_rate = report.forgotten_count / total_processed if total_processed > 0 else 0
        forget_score = (1 - forget_rate) * 30
        
        energy_score = max(0.0, 20 - report.energy_consumed * 0.5)
        
        stage_score = 10.0
        if not report.stages["rem_sleep"].dream_generated:
            stage_score -= 3.0
        
        if report.stages["deep_sleep"].expert_consolidated and report.stages["deep_sleep"].expert_consolidated > 0:
            stage_score += 2.0
        
        total_score = consolidation_score + forget_score + energy_score + stage_score
        return round(min(100.0, max(0.0, total_score)), 1)