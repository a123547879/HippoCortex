import torch
import random
import logging
import re
from typing import Dict, List, Any, Optional

from brain_core import BrainCore
from event_system import EventBus
from BrainConfig import config
# ✅ 替换为实体中心统一数据契约
from Data_models import DreamResult, DreamFragment, Entity, Evidence

logger = logging.getLogger("DreamingLoop")

class DreamingLoop:
    def __init__(self, core: BrainCore, event_bus: EventBus, llm):
        self.core = core
        self.event_bus = event_bus
        self.llm = llm
        
        # 组件引用
        self.experts: Dict[str, Any] = {}
        self.learning_loop: Optional[Any] = None
        self.perception_loop: Optional[Any] = None
        self.vae_manager: Optional[Any] = None
        
        # 梦境存储
        self.expert_dreams: Dict[str, str] = {}
        self.last_dream: Optional[DreamResult] = None
        
        # 技术标签清理正则
        self.tech_tag_pattern = re.compile(r'\[(视觉记忆|梦境|概念|身份|空间|抽象|阅读想象)-.*?\]|绑定ID:[a-f0-9-]+|【.*?】')
        
        # 梦境配置开关（完全保留原有配置）
        self.use_cross_modal_dream: bool = True
        self.use_dream_consolidation: bool = True
        self.dream_surrealism: float = 0.3
        self.max_dream_depth: int = 3

    def bind_components(self, experts: Dict[str, Any], learning_loop: Any, 
                       perception_loop: Any = None, vae_manager: Any = None) -> None:
        """绑定其他组件引用"""
        self.experts = experts
        self.learning_loop = learning_loop
        self.perception_loop = perception_loop
        self.vae_manager = vae_manager
        
        logger.info(f"✅ DreamingLoop 组件绑定完成 | 共 {len(self.experts)} 个专家")
        if self.use_cross_modal_dream and self.perception_loop and self.vae_manager:
            logger.info(f"🧠 跨模态梦境已启用")

    # ===================== 🔴 核心：实体中心式梦境生成 =====================
    def generate_dream(self, dream_length: int = 3) -> DreamResult:
        """
        实体中心式梦境生成
        流程：神经随机激活 → 实体联想传播 → 提取梦境片段 → LLM重构 → 跨模态视觉生成 → 记忆巩固
        """
        logger.info("🌙 大脑进入快速眼动睡眠，开始生成梦境...")
        
        if not self.experts:
            logger.warning("⚠️ 没有可用的专家，无法生成梦境")
            return DreamResult(success=False, content="我做了一个空白的梦，什么都不记得了")
            
        active_experts = random.sample(list(self.experts.keys()), k=random.randint(1, 2))
        logger.info(f"🧠 梦境主导脑区: {active_experts}")
        
        all_activated_fragments = []
        visual_fragments = []
        
        # 神经激活模拟（完全保留原有神经计算逻辑）
        for expert_name in active_experts:
            expert = self.experts.get(expert_name)
            if not expert or not hasattr(expert, 'entities') or len(expert.entities) == 0:
                logger.debug(f"[{expert_name}] 专家无实体，跳过")
                continue
            
            try:
                # 混合初始激活：50%随机神经元放电 + 50%随机实体种子
                random_neurons = torch.randperm(expert.dim)[:int(expert.dim * 0.03)]
                initial_activation = torch.zeros(expert.dim)
                initial_activation[random_neurons] = 1.0
                
                if len(expert.entities) > 0:
                    random_entity = random.choice(expert.entities)
                    initial_activation = initial_activation * 0.5 + random_entity.sdr * 0.5
                
                # 动态传播深度（根据荒诞度调整）
                current_activation = initial_activation
                dream_depth = int(self.max_dream_depth * (1.0 + self.dream_surrealism * 0.5))
                
                for step in range(dream_depth):
                    with torch.no_grad():
                        current_activation = expert.forward(current_activation.unsqueeze(0), steps=1, top_k=50).squeeze(0)
                        # 动态噪声：荒诞度越高，噪声越大
                        noise_level = 0.08 + self.dream_surrealism * 0.1
                        current_activation += torch.randn_like(current_activation) * noise_level
                        current_activation = torch.clamp(current_activation, 0, 1)
                    
                    # 🔴 适配实体检索返回格式
                    results = expert.retrieve(current_activation, top_k=3)
                    for result in results:
                        if len(result) >= 3:
                            entity_id, score, entity_detail = result
                            if score > 0.25:
                                # 用实体最新证据作为梦境内容
                                content = entity_detail.get("latest_evidence", entity_detail.get("name", ""))
                                all_activated_fragments.append((score, content, expert_name, entity_id))
                                
                                # 收集视觉实体片段
                                if (self.use_cross_modal_dream and expert_name == "视觉" 
                                    and self.perception_loop and "multimodal_id" in entity_detail.get("metadata", {})):
                                    visual_fragments.append({
                                        "entity_id": entity_id,
                                        "content": content,
                                        "multimodal_id": entity_detail["metadata"]["multimodal_id"],
                                        "score": score
                                    })
                                
            except Exception as e:
                logger.error(f"❌ [{expert_name}] 专家神经激活失败: {e}", exc_info=True)
                continue
        
        # 提取最活跃的梦境片段
        all_activated_fragments.sort(key=lambda x: -x[0])
        top_fragments = all_activated_fragments[:dream_length * 2]
        
        if not top_fragments:
            logger.info("😴 没有激活任何实体，做了一个空白的梦")
            self.last_dream = DreamResult(
                success=False,
                content="我做了一个空白的梦，什么都不记得了",
                dominant_experts=active_experts
            )
            return self.last_dream
        
        # 去重并构建标准DreamFragment对象
        seen_contents = set()
        dream_fragments: List[DreamFragment] = []
        for score, content, expert, entity_id in top_fragments:
            clean_content = self._clean_tech_tags(content)
            content_key = clean_content.strip()[:50]
            
            if content_key not in seen_contents and len(dream_fragments) < dream_length:
                seen_contents.add(content_key)
                dream_fragments.append(DreamFragment(
                    content=clean_content,
                    source_entity_id=entity_id,  # 🔴 替换原mem_id为source_entity_id
                    activation_score=float(score),
                    expert=expert
                ))
        
        logger.info(f"🔍 提取到 {len(dream_fragments)} 个梦境片段")
        for frag in dream_fragments:
            logger.info(f"   - [{frag.expert}] {frag.content[:30]}... (激活度: {frag.activation_score:.2f})")
        
        # 跨模态视觉梦境生成（适配实体体系）
        dream_visual = None
        if self.use_cross_modal_dream and visual_fragments and self.perception_loop:
            try:
                visual_fragments.sort(key=lambda x: -x["score"])
                main_visual = visual_fragments[0]
                
                if len(visual_fragments) >= 2:
                    bridge = self.perception_loop.pons["视觉"]
                    mixed_sdr = torch.zeros(2048)
                    
                    for i, vf in enumerate(visual_fragments[:3]):
                        # 🔴 从专家中获取实体对象
                        v_entity = next((e for e in self.experts["视觉"].entities if e.entity_id == vf["entity_id"]), None)
                        if v_entity and v_entity.sdr is not None:
                            weight = 1.0 / (i + 1)
                            mixed_sdr += v_entity.sdr * weight
                    
                    # 加入梦境特有的荒诞噪声
                    mixed_sdr += torch.randn_like(mixed_sdr) * self.dream_surrealism * 0.2
                    mixed_sdr = torch.clamp(mixed_sdr, 0, 1)
                    
                    # 生成VAE潜在向量
                    dream_latent = self.vae_manager.sdr_to_latent(mixed_sdr)
                    
                    # 量化存储
                    latent_min = float(dream_latent.min())
                    latent_max = float(dream_latent.max())
                    latent_normalized = (dream_latent - latent_min) / (latent_max - latent_min + 1e-8)
                    latent_quantized = (latent_normalized * 255).to(torch.uint8).cpu().numpy()
                    
                    dream_visual = {
                        "latent": latent_quantized.tolist(),
                        "min": latent_min,
                        "max": latent_max,
                        "shape": list(dream_latent.shape),
                        "source_entities": [vf["content"] for vf in visual_fragments[:3]]
                    }
                    
                    logger.info(f"🎨 生成跨模态梦境视觉 | 融合了 {len(visual_fragments[:3])} 个视觉实体")
                
            except Exception as e:
                logger.debug(f"跨模态梦境视觉生成失败: {e}")
        
        # LLM生成式重构为连贯梦境
        try:
            fragments_text = "\n".join([f"- {frag.content}" for frag in dream_fragments])
            
            prompt = f"""
            你是一个梦境生成器，需要把下面这些碎片化的记忆重组成一个连贯、有情节、略带荒诞感的梦境故事。
            要求：
            1. 把所有片段自然地融合到一个故事里
            2. 梦境要有开头、发展和结尾
            3. 荒诞度：{int(self.dream_surrealism * 10)}分（满分10分）
            4. 语言要优美、有画面感
            5. 用第一人称"我"来叙述
            6. 不要超过200字
            7. 不要出现任何技术术语或标签

            碎片化记忆：
            {fragments_text}

            梦境故事：
            """
            
            response = self.llm.invoke(prompt)
            dream_content = response.content.strip()
            
            logger.info(f"✨ 梦境生成完成:\n{dream_content}")
            
            # 🔴 存储梦境为新实体
            dream_entity_ids = []
            if self.learning_loop:
                dream_metadata = {"is_dream": True, "dream_surrealism": self.dream_surrealism}
                if dream_visual:
                    dream_metadata["dream_visual"] = dream_visual
                
                dream_entity_ids = self.learning_loop.learn(
                    f"[梦境] {dream_content}", 
                    force_expert="抽象",
                    external_reward=0.1
                )
                if dream_entity_ids:
                    logger.info(f"💾 梦境已存入记忆 | 主实体ID: {dream_entity_ids[0]}")
            
            # 梦境记忆巩固（适配实体体系）
            if self.use_dream_consolidation and self.learning_loop:
                logger.info("🧠 开始梦境记忆巩固...")
                consolidated_count = 0
                
                for frag in dream_fragments:
                    if frag.source_entity_id and frag.expert in self.experts:
                        try:
                            expert = self.experts[frag.expert]
                            # 🔴 从皮层获取实体对象
                            entity = self.learning_loop.cortex.index.get_entity(frag.source_entity_id)
                            
                            if entity and entity.sdr is not None:
                                predicted_sdr = expert.predict_next_sdr(entity.sdr)
                                # 梦境给予轻微正奖励，强化重要记忆连接
                                rpe = torch.tensor(0.1 * frag.activation_score, device=entity.sdr.device)
                                expert.predictive_std_update(entity.sdr, predicted_sdr, rpe)
                                consolidated_count += 1
                        except Exception as e:
                            logger.debug(f"记忆巩固失败: {e}")
                
                logger.info(f"✅ 梦境记忆巩固完成 | 共强化 {consolidated_count} 条实体记忆")
            
            self.last_dream = DreamResult(
                success=True,
                content=dream_content,
                fragments=dream_fragments,
                dominant_experts=active_experts,
                main_entity_id=dream_entity_ids[0] if dream_entity_ids else None,
                visual=dream_visual
            )
            
        except Exception as e:
            logger.error(f"❌ 梦境重构失败: {e}", exc_info=True)
            fallback_dream = "我梦见了：" + "、".join([frag.content for frag in dream_fragments])
            self.last_dream = DreamResult(
                success=False,
                content=fallback_dream,
                fragments=dream_fragments,
                dominant_experts=active_experts,
                error=str(e)
            )
        
        return self.last_dream

    def collect_expert_dreams(self) -> Dict[str, str]:
        """收集所有专家生成的梦境（完全保留原有逻辑）"""
        self.expert_dreams = {}
        for name, expert in self.experts.items():
            if hasattr(expert, 'last_dream_text') and expert.last_dream_text:
                clean_dream = self._clean_tech_tags(expert.last_dream_text)
                self.expert_dreams[name] = clean_dream
                logger.info(f"🌙 收集到 [{name}] 的梦境: {clean_dream[:50]}...")
        return self.expert_dreams

    # ===================== 🔴 全局梦境融合（适配实体体系） =====================
    def generate_global_dream(self, high_priority_entities: List[Entity] = None) -> DreamResult:
        """
        生成全局梦境（融合专家梦境和高优先级实体）
        :param high_priority_entities: 高优先级实体列表（替代原high_priority_memories）
        """
        self.collect_expert_dreams()
        
        if not self.expert_dreams and not high_priority_entities:
            logger.info("😴 没有收集到任何梦境素材，做了一个空白的梦")
            self.last_dream = DreamResult(
                success=False,
                content="我做了一个空白的梦，什么都不记得了",
                expert_dreams={}
            )
            return self.last_dream
        
        try:
            logger.info("🌙 开始全局梦境重构（融合高优先级实体）...")
            
            all_dream_texts = []
            
            if self.expert_dreams:
                all_dream_texts.append("[不同脑区的活动]")
                for expert_name, dream_text in self.expert_dreams.items():
                    all_dream_texts.append(f"- {dream_text}")
            
            if high_priority_entities:
                all_dream_texts.append("\n[印象最深刻的事]")
                for entity in high_priority_entities[:5]:
                    # 🔴 用实体最新证据作为内容
                    clean_content = self._clean_tech_tags(
                        entity.latest_evidence.content if entity.latest_evidence else entity.name
                    )
                    # 高优先级实体标记
                    importance_marker = "⭐" * int(entity.importance * 3)
                    all_dream_texts.append(f"- {importance_marker} {clean_content}")
            
            fragments_text = "\n".join(all_dream_texts)
            
            prompt = f"""
                你是一个梦境整合师，需要把下面这些不同脑区的活动和印象最深刻的事，重组成一个连贯、有情节、略带荒诞感的完整梦境故事。
                要求：
                1. 把所有片段自然地融合到一个故事里
                2. 梦境要有开头、发展和结尾
                3. 荒诞度：{int(self.dream_surrealism * 10)}分（满分10分）
                4. 语言要优美、有画面感
                5. 用第一人称"我"来叙述
                6. 不要超过250字
                7. **重点突出带有⭐标记的内容**，让它们成为梦境的主线
                8. 不要出现任何技术术语或标签

                梦境素材：
                {fragments_text}

                完整梦境故事：
                """
            
            response = self.llm.invoke(prompt)
            dream_content = response.content.strip()
            
            logger.info(f"✨ 全局梦境重构完成:\n{dream_content}")
            
            # 存储全局梦境为新实体
            dream_entity_ids = []
            if self.learning_loop:
                dream_entity_ids = self.learning_loop.learn(
                    f"[梦境] {dream_content}", 
                    force_expert="抽象",
                    external_reward=0.2
                )
            
            self.last_dream = DreamResult(
                success=True,
                content=dream_content,
                expert_dreams=self.expert_dreams,
                used_high_priority_entities=[
                    self._clean_tech_tags(
                        e.latest_evidence.content[:30] if e.latest_evidence else e.name
                    ) 
                    for e in high_priority_entities[:3]
                ] if high_priority_entities else [],
                main_entity_id=dream_entity_ids[0] if dream_entity_ids else None
            )
            
        except Exception as e:
            logger.error(f"❌ 全局梦境重构失败: {e}", exc_info=True)
            fallback_parts = []
            if self.expert_dreams:
                fallback_parts.extend(self.expert_dreams.values())
            if high_priority_entities:
                fallback_parts.extend([
                    self._clean_tech_tags(e.latest_evidence.content if e.latest_evidence else e.name) 
                    for e in high_priority_entities[:3]
                ])
                
            fallback_dream = "我做了一个梦：" + "；".join(fallback_parts)
            self.last_dream = DreamResult(
                success=False,
                content=fallback_dream,
                expert_dreams=self.expert_dreams,
                error=str(e)
            )
        
        return self.last_dream

    def _clean_tech_tags(self, text: str) -> str:
        """预处理：彻底清除所有技术标签（完全保留原有逻辑）"""
        if not text:
            return ""
        return self.tech_tag_pattern.sub('', text).strip()