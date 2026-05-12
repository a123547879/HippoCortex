import torch
import random
import logging
from typing import Dict, List, Any, Optional

from brain_core import BrainCore
from event_system import EventBus
from BrainConfig import config
from Data_models import DreamResult, DreamFragment, MemoryPacket  # 🔥 导入新增的DreamFragment

logger = logging.getLogger("DreamingLoop")

class DreamingLoop:
    def __init__(self, core: BrainCore, event_bus: EventBus, llm):
        self.core = core
        self.event_bus = event_bus
        self.llm = llm
        
        # 组件引用（由CognitiveSystem注入）
        self.experts: Dict[str, Any] = {}  # 🔥 完善类型注解
        self.learning_loop: Any = None
        
        # 梦境存储（统一为DreamResult类型）
        self.expert_dreams: Dict[str, str] = {}  # 存储每个专家的梦境文本
        self.last_dream: Optional[DreamResult] = None  # 🔥 明确类型为Optional[DreamResult]

    def bind_components(self, experts: Dict[str, Any], learning_loop: Any) -> None:
        """绑定其他组件引用"""
        self.experts = experts
        self.learning_loop = learning_loop

    def generate_dream(self, dream_length: int = 3) -> DreamResult:
        """
        生成一个真正的梦境：神经激活传播 → 碎片化记忆 → LLM生成式重构
        :param dream_length: 梦境包含的记忆片段数量
        :return: 强类型梦境结果DreamResult
        """
        logger.info("🌙 大脑进入快速眼动睡眠，开始生成梦境...")
        
        # 1. 随机选择一个或多个专家作为梦境的主导脑区
        active_experts = random.sample(list(self.experts.keys()), k=random.randint(1, 2))
        logger.info(f"🧠 梦境主导脑区: {active_experts}")
        
        all_activated_sdrs = []
        
        # 2. 对每个主导专家进行神经激活模拟
        for expert_name in active_experts:
            expert = self.experts.get(expert_name)
            if not expert or not hasattr(expert, 'sdr_list') or len(expert.sdr_list) == 0:
                continue
            
            # 2.1 生成随机初始激活（模拟睡眠时的神经元自发放电）
            random_neurons = torch.randperm(expert.dim)[:int(expert.dim * 0.05)]  # 5%的神经元随机激活
            initial_activation = torch.zeros(expert.dim)
            initial_activation[random_neurons] = 1.0
            
            # 2.2 激活在突触连接中传播（模拟梦境中的联想过程）
            current_activation = initial_activation
            
            for step in range(2):  # 传播2步，模拟梦境的联想链条
                with torch.no_grad():
                    # 通过突触矩阵传播激活
                    current_activation = expert.forward(current_activation.unsqueeze(0), steps=1, top_k=50).squeeze(0)
                    # 加入噪声，模拟梦境的随机性和荒诞性
                    current_activation += torch.randn_like(current_activation) * 0.1
                    current_activation = torch.clamp(current_activation, 0, 1)
                
                # 检索激活对应的记忆
                results = expert.retrieve(current_activation, top_k=2)
                for result in results:
                    if len(result) >= 2:
                        score, content = result[0], result[1]
                        if score > 0.3:  # 只保留有一定相似度的片段
                            all_activated_sdrs.append((score, content, expert_name))
        
        # 3. 提取最活跃的记忆片段
        all_activated_sdrs.sort(key=lambda x: -x[0])
        top_fragments = all_activated_sdrs[:dream_length * 2]
        
        if not top_fragments:
            logger.info("😴 没有激活任何记忆，做了一个空白的梦")
            self.last_dream = DreamResult(
                success=False,
                content="我做了一个空白的梦，什么都不记得了",
                dominant_experts=active_experts
            )
            return self.last_dream
        
        # 去重并提取片段内容（🔥 现在使用强类型DreamFragment）
        seen_contents = set()
        dream_fragments: List[DreamFragment] = []
        for score, content, expert in top_fragments:
            if content not in seen_contents and len(dream_fragments) < dream_length:
                seen_contents.add(content)
                dream_fragments.append(DreamFragment(
                    content=content,
                    expert=expert,
                    activation_score=float(score)
                ))
        
        logger.info(f"🔍 提取到 {len(dream_fragments)} 个梦境片段")
        for frag in dream_fragments:
            logger.info(f"   - [{frag.expert}] {frag.content[:30]}... (激活度: {frag.activation_score:.2f})")
        
        # 4. 核心：LLM生成式重构为连贯梦境
        try:
            fragments_text = "\n".join([f"- {frag.content}" for frag in dream_fragments])
            
            prompt = f"""
            你是一个梦境生成器，需要把下面这些碎片化的记忆重组成一个连贯、有情节、略带荒诞感的梦境故事。
            要求：
            1. 把所有片段自然地融合到一个故事里
            2. 梦境要有开头、发展和结尾
            3. 加入一些超现实、荒诞的元素，符合梦境的特点
            4. 语言要优美、有画面感
            5. 自动去除所有技术标签，如"[视觉记忆-XX]"、"绑定ID:XX"、"[概念的梦]"等
            6. 不要出现任何代码或技术术语
            7. 不要超过200字

            碎片化记忆：
            {fragments_text}

            梦境故事：
            """
            
            response = self.llm.invoke(prompt)
            dream_content = response.content.strip()
            
            logger.info(f"✨ 梦境生成完成:\n{dream_content}")
            
            # 5. 把梦境作为新记忆存入大脑
            dream_mem_id = self.learning_loop.learn(f"[梦境] {dream_content}", force_expert="抽象")
            
            self.last_dream = DreamResult(
                success=True,
                content=dream_content,
                fragments=dream_fragments,
                dominant_experts=active_experts,
                mem_id=dream_mem_id
            )
            
        except Exception as e:
            logger.error(f"❌ 梦境重构失败: {e}", exc_info=True)
            # 降级方案：直接拼接片段
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
        """收集所有专家生成的梦境"""
        self.expert_dreams = {}
        for name, expert in self.experts.items():
            if hasattr(expert, 'last_dream_text') and expert.last_dream_text:
                self.expert_dreams[name] = expert.last_dream_text
                logger.info(f"🌙 收集到 [{name}] 的梦境: {expert.last_dream_text[:50]}...")
        return self.expert_dreams

    def generate_global_dream(self, high_priority_memories: List[MemoryPacket] = None) -> DreamResult:
        """
        生成全局梦境（融合专家梦境和高优先级记忆）
        :param high_priority_memories: 高优先级记忆列表（强类型MemoryPacket）
        :return: 强类型梦境结果DreamResult
        """
        if not self.expert_dreams and not high_priority_memories:
            logger.info("😴 没有收集到任何梦境素材，做了一个空白的梦")
            self.last_dream = DreamResult(
                success=False,
                content="我做了一个空白的梦，什么都不记得了",
                expert_dreams={}
            )
            return self.last_dream
        
        try:
            logger.info("🌙 开始全局梦境重构（融合高优先级记忆）...")
            
            # 1. 准备梦境素材：专家梦境 + 高优先级记忆内容
            all_dream_texts = []
            
            # 加入专家梦境
            for expert_name, dream_text in self.expert_dreams.items():
                all_dream_texts.append(f"[{expert_name}的梦] {dream_text}")
            
            # 加入高优先级记忆作为梦境素材
            if high_priority_memories:
                all_dream_texts.append("\n[重要记忆片段]")
                for mem in high_priority_memories[:5]:  # 取Top5
                    all_dream_texts.append(f"- {mem.content}")
            
            fragments_text = "\n".join(all_dream_texts)
            
            # 2. 用LLM重组成一个连贯的全局梦境
            prompt = f"""
                你是一个梦境整合师，需要把下面这些不同脑区的活动和重要记忆，重组成一个连贯、有情节、略带荒诞感的完整梦境故事。
                要求：
                1. 把所有片段自然地融合到一个故事里
                2. 梦境要有开头、发展和结尾
                3. 加入一些超现实、梦幻的元素
                4. 语言要优美、有画面感
                5. 不要超过200字
                6. 用第一人称"我"来叙述
                7. 重点突出[重要记忆片段]里的内容
                8. 自动去除所有技术标签，如"[视觉记忆-XX]"、"绑定ID:XX"、"[概念的梦]"等

                梦境素材：
                {fragments_text}

                完整梦境故事：
                """
            
            response = self.llm.invoke(prompt)
            dream_content = response.content.strip()
            
            logger.info(f"✨ 全局梦境重构完成:\n{dream_content}")
            
            # 3. 把梦境作为新记忆存入大脑（标记为梦境，低重要性）
            dream_mem_id = self.learning_loop.learn(f"[梦境] {dream_content}", force_expert="抽象")
            
            self.last_dream = DreamResult(
                success=True,
                content=dream_content,
                expert_dreams=self.expert_dreams,
                used_high_priority_memories=[m.content[:30] for m in high_priority_memories[:3]] if high_priority_memories else [],
                mem_id=dream_mem_id
            )
            
        except Exception as e:
            logger.error(f"❌ 全局梦境重构失败: {e}", exc_info=True)
            # 降级方案：直接拼接
            fallback_dream = "我做了一个梦：" + "；".join([v for v in self.expert_dreams.values()])
            self.last_dream = DreamResult(
                success=False,
                content=fallback_dream,
                expert_dreams=self.expert_dreams,
                error=str(e)
            )
        
        return self.last_dream