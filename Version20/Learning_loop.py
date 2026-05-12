import torch
import time
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple

from brain_core import BrainCore
from event_system import EventBus, Event, EventType
from Thalamus import Thalamus
from HippocampusRouterV10 import HippocampusRouter
from SymbolicCore import SymbolicCore
from BrainConfig import config
from DopamineSystem import DopamineSystem
from Metacognition import Metacognition
from Curiosity import Curiosity
from Data_models import MemoryPacket  # 新增：导入数据契约

logger = logging.getLogger("LearningLoop")

class LearningLoop:
    def __init__(self, core: BrainCore, event_bus: EventBus, embedding_model, llm):
        self.core: BrainCore = core
        self.event_bus: EventBus = event_bus
        self.embedding_model: Any = embedding_model
        self.llm: Any = llm
        
        # 组件引用（由CognitiveSystem注入）
        self.thalamus: Optional[Thalamus] = None
        self.hippocampus_router: Optional[HippocampusRouter] = None
        self.symbolic_core: Optional[SymbolicCore] = None
        self.experts: Dict[str, Any] = {}
        self.sdr_encoders: Dict[str, Any] = {}
        self.cortex: Optional[Any] = None
        
        # 类人学习核心模块
        self.dopamine: Optional[DopamineSystem] = None
        self.metacognition: Optional[Metacognition] = None
        self.curiosity: Optional[Curiosity] = None
        
        # 突触连接：键为(from_mem_id, to_mem_id)元组，值为权重
        self.synapses: Dict[Tuple[int, int], float] = {}  # 🔥 统一mem_id为int类型
        self.synapse_save_path: str = ""

    def bind_components(self, thalamus: Thalamus, hippocampus_router: HippocampusRouter, 
                       symbolic_core: SymbolicCore, experts: Dict[str, Any], sdr_encoders: Dict[str, Any], cortex: Any,
                       dopamine: DopamineSystem, metacognition: Metacognition, curiosity: Curiosity) -> None:
        """绑定其他组件引用"""
        self.thalamus = thalamus
        self.hippocampus_router = hippocampus_router
        self.symbolic_core = symbolic_core
        self.experts = experts
        self.sdr_encoders = sdr_encoders
        self.cortex = cortex
        self.dopamine = dopamine
        self.metacognition = metacognition
        self.curiosity = curiosity

    def set_synapse_save_path(self, path: str) -> None:
        """设置突触保存路径"""
        self.synapse_save_path = path
        self._load_synapses()

    def learn(self, text: str, force_expert: Optional[str] = None, external_reward: float = 0.0) -> Optional[int]:
        """
        学习新记忆 (集成多巴胺强化学习 + 元认知)
        :param text: 学习内容（陈述句）
        :param force_expert: 强制指定专家
        :param external_reward: 外部奖励 (-1 到 1，用户明确反馈时传入)
        :return: 记忆ID（int），如果是疑问句或被过滤则返回None
        """
        def is_declarative_sentence(q: str) -> bool:
            """判断是否为陈述句（应该被学习）"""
            question_words = ["？", "?", "哪里", "谁", "怎么", "吗", "呢"]
            
            # 特殊处理"什么"：只有当它不在否定词后面时才视为疑问词
            if "什么" in q:
                import re
                # 匹配"不是什么"、"没什么"、"没有什么"等模式
                if re.search(r'(不|没|没有)什么', q):
                    # 否定句中的"什么"不视为疑问词
                    pass
                else:
                    # 疑问句中的"什么"视为疑问词
                    return False
            
            return not any(word in q for word in question_words)
        
        # 🔥 修复：逻辑反转bug！疑问句跳过学习，陈述句才学习
        if not is_declarative_sentence(text):
            logger.info(f"🚫 输入是疑问句，跳过学习: {text[:30]}...")
            return None
        
        clip_vec = self._encode_text(text)
        clip_vec = torch.nn.functional.normalize(clip_vec.detach().squeeze(), p=2, dim=-1)
        
        passed, info_packet = self.thalamus.filter_and_relay(
            input_vec=clip_vec,
            input_text=text,
            metadata={"force_expert": force_expert}
        )
        
        if not passed:
            logger.info(f"🚫 信息被丘脑过滤: {text[:30]}...")
            return None
        
        clip_vec = info_packet["vec"]
        saliency = info_packet["saliency"]
        
        if force_expert is None:
            target_expert = self.hippocampus_router.route(clip_vec, text, True)
            self.hippocampus_router.online_learn(clip_vec, target_expert)
        else:
            target_expert = force_expert
        
        sdr_encoder = self.sdr_encoders.get(target_expert, self.sdr_encoders["概念"])
        sdr = sdr_encoder.encode(clip_vec.unsqueeze(0))
        
        # 元认知预评估
        concept_key = self._extract_concept_key(text)
        prior_confidence = 0.0
        current_time = time.time()
        
        if self.metacognition:
            try:
                prior_confidence = self.metacognition.assess_knowledge_confidence(concept_key, current_time)
                logger.debug(f"🧠 元认知：学习前对 '{concept_key}' 的置信度 = {prior_confidence:.2f}")
            except Exception as e:
                logger.debug(f"元认知预评估跳过: {e}")
        
        mem_id = self.hippocampus_router.encode(
            clip_vec=clip_vec,
            sdr=sdr,
            content=text,
            metadata={"expert": target_expert, "saliency": saliency},
            expert=target_expert
        )
        
        logger.info(f"✅ 记忆已存入海马体 | ID:{mem_id} | 专家:{target_expert}")
        
        # 发射记忆存储事件
        self.event_bus.emit(Event(EventType.MEMORY_STORED, {
            "mem_id": mem_id, 
            "content": text,
            "expert": target_expert
        }))
        
        # 符号学习
        if self.symbolic_core:
            try:
                self.symbolic_core.learn_from_dialogue("用户", text)
            except Exception as e:
                logger.debug(f"符号学习跳过: {e}")
        
        # 视觉接地
        if target_expert != "视觉" and mem_id:
            try:
                visual_results = self.cortex.search_memories(
                    clip_vec, sdr, expert_name="视觉", top_k=1, min_similarity=0.12, query_text=text
                )
                if visual_results:
                    visual_mem_id, visual_sim, visual_content, _ = visual_results[0]
                    self.create_synapse(from_mem_id=mem_id, to_mem_id=visual_mem_id, weight=0.3)
            except Exception as e:
                logger.debug(f"概念视觉接地跳过: {e}")
        
        # 多巴胺强化学习核心逻辑
        if self.dopamine and mem_id:
            try:
                # 1. 计算内部奖励
                # 1.1 好奇心奖励：基于信息增益 (与先验置信度成反比)
                information_gain = max(0.0, 1.0 - prior_confidence)
                curiosity_reward = self.dopamine.get_curiosity_reward(information_gain)
                
                # 1.2 预测奖励：基于显著性
                prediction_accuracy = saliency
                prediction_reward = self.dopamine.get_prediction_reward(prediction_accuracy)
                
                # 2. 总奖励 = 外部奖励 + 内部奖励
                total_reward = float(external_reward) + curiosity_reward + prediction_reward
                total_reward = float(np.clip(total_reward, -1.0, 1.0))
                
                # 3. 计算奖励预测误差 (RPE)
                rpe = self.dopamine.compute_reward_prediction_error(total_reward)
                
                # 4. 用多巴胺调节突触可塑性 (调节与该记忆相关的所有突触)
                if mem_id in self.cortex.index.memories:
                    # 找到所有与mem_id相连的突触
                    related_synapses = []
                    for (from_id, to_id), weight in list(self.synapses.items()):
                        if from_id == mem_id or to_id == mem_id:
                            related_synapses.append({
                                "from": from_id, 
                                "to": to_id, 
                                "weight": weight
                            })
                    
                    if related_synapses:
                        for syn in related_synapses:
                            original_weight = syn["weight"]
                            # 多巴胺调节：正RPE增强，负RPE减弱
                            weight_update = 0.01 * rpe * saliency
                            new_weight = original_weight + weight_update
                            new_weight = float(np.clip(new_weight, -1.0, 1.0))
                            
                            # 更新突触
                            key = (syn["from"], syn["to"])
                            self.synapses[key] = new_weight
                            logger.debug(f"    突触 {key} 权重: {original_weight:.2f} → {new_weight:.2f}")
                        
                        self._save_synapses()
                
                logger.info(f"🧪 多巴胺系统 | RPE={rpe:.2f} | 总奖励={total_reward:.2f} | "
                           f"(好奇={curiosity_reward:.2f}, 预测={prediction_reward:.2f}, 外部={external_reward:.2f})")
                
            except Exception as e:
                logger.debug(f"多巴胺学习跳过: {e}")
        
        # 元认知后评估与好奇心触发
        if self.metacognition and mem_id:
            try:
                # 更新元认知记录
                post_confidence = self.metacognition.assess_knowledge_confidence(concept_key, current_time)
                
                # 检查是否应该触发好奇心提问
                if self.curiosity:
                    if self.curiosity.should_ask_question(concept_key):
                        questions = self.curiosity.generate_questions(concept_key)
                        if questions:
                            logger.info(f"❓ 好奇心驱动：生成问题 - {questions[0]}")
                            # 通过事件总线发射好奇心事件
                            self.event_bus.emit(Event("CURIOSITY_TRIGGERED", {
                                "concept": concept_key,
                                "questions": questions
                            }))
                
                logger.debug(f"🧠 元认知：学习后置信度 = {post_confidence:.2f} | "
                            f"变化 = {post_confidence - prior_confidence:.2f}")
                
            except Exception as e:
                logger.debug(f"元认知后评估跳过: {e}")
        
        return mem_id
    
    def batch_init_direct_to_cortex(self, texts: List[str]) -> List[int]:
        """
        🔥 初始批量导入：直接写入皮层长期记忆
        完全跳过：丘脑、海马体、路由、多巴胺、元认知、事件、视觉接地
        只做：编码向量 → 批量直接存入皮层长期记忆
        """
        if not self.cortex:
            logger.error("❌ 皮层组件未绑定，无法批量导入")
            return []

        expert_names = []
        sdrs = []
        clip_vecs = []
        contents = []
        metadatas = []

        logger.info(f"🚀 初始批量直接导入皮层 | 共{len(texts)}条")

        for text in texts:
            try:
                # 1. 编码文本向量
                clip_vec = self._encode_text(text)
                clip_vec = torch.nn.functional.normalize(clip_vec.detach().squeeze(), p=2, dim=-1)

                target_expert = self.hippocampus_router.route(clip_vec, text, True)
                self.hippocampus_router.online_learn(clip_vec, target_expert)

                # 2. SDR编码
                sdr_encoder = self.sdr_encoders.get(target_expert, self.sdr_encoders["概念"])
                sdr = sdr_encoder.encode(clip_vec.unsqueeze(0))

                # 3. 构造基础元数据
                meta = {
                    "is_fact": True,
                    "importance": 0.7,
                    "source": "初始批量导入",
                    "tags": []
                }

                # 收集批量参数
                expert_names.append(target_expert)
                sdrs.append(sdr.detach())
                clip_vecs.append(clip_vec.detach())
                contents.append(text.strip())
                metadatas.append(meta)

            except Exception as e:
                logger.warning(f"⚠️ 文本编码失败跳过: {text[:30]} | {e}")
                continue

        # 4. 调用皮层批量接口：直接写入长期记忆，不走海马体
        mem_ids = self.cortex.batch_store_detailed_memories(
            expert_names=expert_names,
            sdrs=sdrs,
            clip_vecs=clip_vecs,
            contents=contents,
            metadatas=metadatas
        )

        logger.info(f"✅ 批量导入完成 | 成功写入皮层 {len(mem_ids)} 条记忆")
        return mem_ids

    def bind_related_memories(self, new_mem_id: int, new_mem_vec: torch.Tensor, 
                         new_mem_text: str, target_expert: str, user_input: str) -> None:
        """绑定视觉记忆与相关概念记忆（🔥 适配MemoryPacket对象）"""
        try:
            logger.info(f"🔗 开始绑定视觉记忆 | ID:{new_mem_id}")
            
            keywords = set()
            if user_input and user_input.strip():
                keywords.update(user_input.strip().split())
            
            if "[视觉记忆-" in new_mem_text:
                try:
                    tag_part = new_mem_text.split("[视觉记忆-")[1].split("]")[0]
                    keywords.add(tag_part)
                except:
                    pass
            
            if not keywords:
                keywords = {"记忆", "关联"}
            
            related_mem_ids = []
            
            for exp_name, expert in self.experts.items():
                if exp_name == target_expert:
                    continue
                
                try:
                    if hasattr(expert, 'retrieve'):
                        vec_results = expert.retrieve(new_mem_vec.unsqueeze(0), top_k=2)
                        for item in vec_results:
                            if len(item) == 5:
                                score, content, meta, _, mem_id = item
                            elif len(item) == 4:
                                mem_id, score, content, meta = item
                            else:
                                continue
                            
                            if score > 0.55:
                                related_mem_ids.append(mem_id)
                                logger.info(f"🔗 向量绑定 | 专家:{exp_name} | 记忆ID:{mem_id} | 相似度:{score:.2f}")
                except Exception as e:
                    logger.debug(f"[{exp_name}] 向量绑定跳过: {e}")
            
            if related_mem_ids:
                related_mem_ids = list(set(related_mem_ids))
                
                for rel_mem_id in related_mem_ids:
                    self.create_synapse(
                        from_mem_id=new_mem_id,
                        to_mem_id=rel_mem_id,
                        weight=0.35
                    )
                
                for rel_mem_id in related_mem_ids:
                    try:
                        # ✅ 修复：对象属性访问，替代字典语法
                        if rel_mem_id in self.cortex.index.memories:
                            rel_mem: MemoryPacket = self.cortex.index.memories[rel_mem_id]
                            if "related_mem_ids" not in rel_mem.metadata:
                                rel_mem.metadata["related_mem_ids"] = []
                            if new_mem_id not in rel_mem.metadata["related_mem_ids"]:
                                rel_mem.metadata["related_mem_ids"].append(new_mem_id)
                    except Exception as e:
                        logger.debug(f"反向绑定失败: {e}")
                
                try:
                    # ✅ 修复：对象属性访问，替代字典语法
                    if new_mem_id in self.cortex.index.memories:
                        new_mem: MemoryPacket = self.cortex.index.memories[new_mem_id]
                        new_mem.metadata["related_mem_ids"] = related_mem_ids
                except Exception as e:
                    logger.debug(f"新记忆元数据更新失败: {e}")
                
                logger.info(f"✅ 视觉记忆绑定完成 | 共绑定 {len(related_mem_ids)} 条相关记忆")
            else:
                logger.info(f"ℹ️ 未找到相关记忆，跳过绑定")
        
        except Exception as e:
            logger.error(f"❌ 记忆绑定失败: {e}", exc_info=True)

    def create_synapse(self, from_mem_id: int, to_mem_id: int, weight: float = 0.3) -> None:
        """创建双向突触连接（统一使用int类型的mem_id）"""
        key = (from_mem_id, to_mem_id)
        self.synapses[key] = weight
        reverse_key = (to_mem_id, from_mem_id)
        self.synapses[reverse_key] = weight * 0.8
        self._save_synapses()
        logger.info(f"🔗 突触建立: {from_mem_id} ↔ {to_mem_id}")

    def _load_synapses(self) -> None:
        """加载突触连接（兼容旧版字符串ID）"""
        try:
            import os
            import json
            if os.path.exists(self.synapse_save_path):
                with open(self.synapse_save_path, "r", encoding="utf-8") as f:
                    synapses_str = json.load(f)
                    # 🔥 兼容旧版：将字符串ID转换为int
                    self.synapses = {
                        (int(k.split("|")[0]), int(k.split("|")[1])): float(v)
                        for k, v in synapses_str.items()
                    }
                logger.info(f"🔗 已加载 {len(self.synapses)} 条突触连接")
        except Exception as e:
            logger.warning(f"⚠️ 加载突触连接失败: {e}")
            self.synapses = {}

    def _save_synapses(self) -> None:
        """保存突触连接"""
        try:
            import json
            synapses_str = {
                f"{k[0]}|{k[1]}": float(v)
                for k, v in self.synapses.items()
            }
            with open(self.synapse_save_path, "w", encoding="utf-8") as f:
                json.dump(synapses_str, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ 保存突触连接失败: {e}")

    def _encode_text(self, text: str) -> torch.Tensor:
        """编码文本为向量"""
        try:
            embedding = self.embedding_model.embed_query(text)
            clip_vec = torch.as_tensor(embedding, dtype=torch.float32)
            return clip_vec
        except Exception as e:
            logger.error(f"❌ 文本编码失败: {e}")
            raise

    def _extract_concept_key(self, text: str) -> str:
        """从文本中提取核心概念关键词"""
        import re
        # 移除标点符号
        clean_text = re.sub(r'[^\w\s]', '', text)
        # 按空格分割
        words = clean_text.split()
        # 返回最长的词作为核心概念
        if words:
            return max(words, key=len)
        return text[:10]