import torch
import torch.nn.functional as F
import datetime
import random
import time
import threading
import logging
from typing import List, Dict, Optional, Any, Tuple
import re
import json

from brain_core import BrainCore
from event_system import EventBus, Event, EventType
from Thalamus import Thalamus
from HippocampusRouterV10 import HippocampusRouter
from SymbolicCore import SymbolicCore
from BrainConfig import config
from Data_models import ThoughtResult, MemoryPacket, Intention, ConversationTurn

logger = logging.getLogger("PerceptionLoop")

class PerceptionLoop:
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
        
        # 走神状态
        self.mind_wandering_enabled: bool = True
        self.mind_wandering_idle_threshold: int = 30
        self.last_interaction_time: datetime.datetime = datetime.datetime.now()
        self.mind_wandering_recall_prob: float = 0.5
        self.mind_wandering_assoc_prob: float = 0.3
        self._mind_wandering_running: bool = False
        self.mind_wandering_thread: Optional[threading.Thread] = None
        
        # 意图系统（🔥 强类型化）
        self.intention_queue: List[Intention] = []
        self.max_intention_queue_size: int = 10
        self.last_intention_execution_time: datetime.datetime = datetime.datetime.now()
        self.min_intention_interval: int = 60
        self.pending_social_intention: Optional[Intention] = None  # 🔥 明确类型
        
        # 意图权重
        self.intention_weights: Dict[str, float] = {
            "physiological": 1.5,
            "cognitive": 1.0,
            "social": 0.8,
            "exploration": 0.6
        }
        
        # 符号接地缓存
        self.symbol_grounding_cache: Dict[str, Dict[str, str]] = {}
        
        # 当前查询上下文
        self.current_query_text: str = ""

    def bind_components(self, thalamus: Thalamus, hippocampus_router: HippocampusRouter, 
                       symbolic_core: SymbolicCore, experts: Dict[str, Any], sdr_encoders: Dict[str, Any], cortex: Any) -> None:
        """绑定其他组件引用"""
        self.thalamus = thalamus
        self.hippocampus_router = hippocampus_router
        self.symbolic_core = symbolic_core
        self.experts = experts
        self.sdr_encoders = sdr_encoders
        self.cortex = cortex

    def think(self, text: str, steps: int = 2, topk: int = 10, expert_last: Optional[str] = None) -> ThoughtResult:
        """主思考函数，返回强类型ThoughtResult"""
        self._update_interaction_time()
        self.current_query_text = text
        
        self.event_bus.emit(Event(EventType.INPUT_RECEIVED, {"text": text}))
        
        try:
            # 获取时间衰减对话上下文（🔥 强类型化）
            active_context: List[ConversationTurn] = []
            context_text: str = ""
            try:
                if hasattr(self.cortex, 'get_active_conversation_context'):
                    active_context = self.cortex.get_active_conversation_context()
                    if active_context:
                        context_text = self._build_context_prompt(active_context)
                        logger.debug(f"🧠 注入对话上下文 | 轮数:{len(active_context)}")
            except Exception as e:
                logger.error(f"❌ 获取对话上下文失败，跳过上下文注入: {e}")
            
            # 将上下文融入查询
            enhanced_query = f"{context_text}\n用户当前问题：{text}" if context_text else text
            
            # 初始化和路由
            clip_vec, final_expert, expert_scores, energy_detail, total_energy, \
            sdr_encoder, query_sdr, symbolic_context = self._initialize_and_route(enhanced_query, expert_last)

            # 检索记忆
            raw_results = self._retrieve_memories(clip_vec, query_sdr, text, expert_scores, total_energy)
            global_memory_pool, hippo_count = self._build_global_memory_pool(raw_results)
            
            # 融合和过滤记忆
            activated_memories, predicted_memory, prediction_error, propagated, similarity_trace = \
                self._fuse_and_filter_memories(final_expert, global_memory_pool, clip_vec, steps, topk, total_energy, text)
            final_activated_memories = self._link_visual_and_strengthen(activated_memories)
            
            # 构建思考链
            thought_chain = self._build_coherent_thought_chain(final_activated_memories, similarity_trace, 0.25)
            core_ideas = self._extract_core_ideas(final_activated_memories)
            activation_strength = propagated.norm().item() if propagated is not None else 0.0
            
            # 构建强类型思考结果
            result = ThoughtResult(
                thought_chain=thought_chain,
                core_ideas=core_ideas,
                activated_memories=[
                    m.content if isinstance(m, MemoryPacket) else m["content"] 
                    for m in final_activated_memories
                ],
                expert=final_expert,
                activation_strength=activation_strength,
                predicted_memory=predicted_memory,
                prediction_error=prediction_error,
                symbolic_context=symbolic_context,
                energy_detail=energy_detail
            )
            
            self.event_bus.emit(Event(EventType.RESPONSE_GENERATED, result))
            return result
            
        except Exception as e:
            logger.error(f"❌ 思考过程出错: {e}", exc_info=True)
            return ThoughtResult(
                thought_chain="思考失败",
                core_ideas=[],
                activated_memories=[],
                expert="概念",
                activation_strength=0.0,
                error=str(e)
            )

    # 修改后的代码（字典兼容版本）
    def _build_context_prompt(self, context_turns: List[Dict]) -> str:
        """将对话历史转换为LLM可理解的上下文提示（字典兼容版本）"""
        if not context_turns:
            return ""
        
        context_parts = ["【最近对话历史】"]
        for turn in context_turns:
            context_parts.append(f"用户：{turn['user_input']}")
            context_parts.append(f"小白：{turn['ai_response']}")
        
        return "\n".join(context_parts)

    def _initialize_and_route(self, text: str, expert_last: Optional[str]) -> Tuple[torch.Tensor, str, Dict[str, float], Dict[str, float], float, Any, torch.Tensor, str]:
        """初始化并路由查询到对应专家"""
        clip_vec = self.encode_text(text)
        clip_vec = F.normalize(clip_vec, p=2, dim=-1)
        
        target_expert = self.hippocampus_router.route(clip_vec, text)
        expert_scores = self.hippocampus_router.last_scores
        
        if expert_last is None:
            expert_last = self._get_query_expert_local(text)
        
        force_rule_keywords = {
            "视觉": ["长什么样", "样子", "长相", "外貌", "模样", "长得", "图片", "照片", "看看你"],
            "身份": ["你是谁", "我是谁", "名字", "叫什么", "主人", "你的主人", "我的名字", "你的名字", "身份"]
        }
        final_expert = target_expert
        query_lower = text.lower()
        for rule_expert, keywords in force_rule_keywords.items():
            if any(w in query_lower for w in keywords):
                final_expert = rule_expert
                break
        
        routing_probs = list(expert_scores.values()) if expert_scores else []
        triple_scores = []
        if self.symbolic_core:
            triplets = self.symbolic_core.get_all_triplets()
            triple_scores = [1.0 for _ in triplets]
        rule_match = self._get_query_expert_local(text) == final_expert
        synapse_change = self.get_synapse_change()
        
        total_energy, energy_detail = self.core.energy_field.total_energy(
            routing_probs=routing_probs, triple_scores=triple_scores, sim_scores=[],
            rule_match=rule_match, synapse_change=synapse_change,
            is_wandering=self.core.is_mind_wandering, fatigue_level=self.core.fatigue_level
        )
        
        sdr_encoder = self.sdr_encoders.get(final_expert, self.sdr_encoders["概念"])
        query_sdr = sdr_encoder.encode(clip_vec)
        
        symbolic_context = ""
        if self.symbolic_core:
            try:
                parsed = self.symbolic_core.parse_question(text)
                symbolic_results = self.symbolic_core.symbolic_retrieve(parsed)
                if symbolic_results:
                    symbolic_context = "【精准记忆】\n" + "\n".join([f"- {res['object']}" for res in symbolic_results])
            except Exception as e:
                logger.debug(f"符号检索跳过: {e}")
        
        return clip_vec, final_expert, expert_scores, energy_detail, total_energy, sdr_encoder, query_sdr, symbolic_context

    def _retrieve_memories(self, clip_vec: torch.Tensor, query_sdr: torch.Tensor, text: str, 
                          expert_scores: Dict[str, float], total_energy: float) -> List[Tuple[int, float, str, Dict]]:
        """根据能量动态调整检索阈值"""
        if total_energy < 2:
            dynamic_min_sim = 0.05
        elif total_energy < 5:
            dynamic_min_sim = 0.1
        else:
            dynamic_min_sim = 0.25
        
        raw_results = self.thalamus.schedule_retrieval(
            query_vec=clip_vec, query_sdr=query_sdr, query_text=text,
            expert_scores=expert_scores, min_similarity=dynamic_min_sim
        )
        return raw_results

    def _build_global_memory_pool(self, raw_results: List[Tuple[int, float, str, Dict]]) -> Tuple[Dict[int, Dict], int]:
        """构建全局记忆池，去重并标记来源（🔥 适配MemoryPacket）"""
        global_memory_pool: Dict[int, Dict] = {}
        seen_content = set()
        
        for mem_id, sim, content, meta in raw_results:
            content_key = content.strip()[:50]
            if content_key in seen_content:
                continue
            seen_content.add(content_key)
            
            is_hippocampus = meta.get("is_hippocampus", False)
            if is_hippocampus:
                # 🔥 核心修复：从海马体缓冲区找回原始 SDR（对象属性访问）
                original_sdr = None
                if self.hippocampus_router:
                    for mem in self.hippocampus_router.hippocampal_buffer:
                        # 废弃字典语法，改用对象属性
                        if mem.mem_id == mem_id:
                            original_sdr = mem.sdr
                            break
                
                mem_to_use = {
                    "id": mem_id, 
                    "content": content, 
                    "metadata": meta, 
                    "sdr": original_sdr
                }
            else:
                full_mem: Optional[MemoryPacket] = self.cortex.index.get_memory(mem_id)
                if full_mem and not full_mem.metadata.get("is_obsolete", False):
                    mem_to_use = {
                        "id": full_mem.mem_id,
                        "content": full_mem.content,
                        "metadata": full_mem.metadata,
                        "sdr": full_mem.sdr
                    }
                else:
                    continue
            
            global_memory_pool[mem_id] = {
                "mem": mem_to_use, "global_score": sim, "expert_score": 0.0,
                "source": "global", "cross_validated": False, "is_hippocampus": is_hippocampus
            }
        
        hippo_count = sum(1 for d in global_memory_pool.values() if d['is_hippocampus'])
        return global_memory_pool, hippo_count

    def _fuse_and_filter_memories(self, final_expert: str, global_memory_pool: Dict[int, Dict], 
                                 clip_vec: torch.Tensor, steps: int, topk: int, total_energy: float, text: str) -> Tuple[List[Dict], Optional[str], float, Optional[torch.Tensor], List[Tuple]]:
        """融合专家和全局记忆，过滤并排序"""
        expert = self.experts.get(final_expert)
        predicted_memory = None
        prediction_error = 0.0
        propagated = None
        similarity_trace = []
        activated_memories = []
        
        if expert:
            global_sdrs = []
            for mem_id, data in list(global_memory_pool.items())[:10]:
                if data["mem"]["sdr"] is not None:
                    global_sdrs.append(data["mem"]["sdr"].to(clip_vec.device))
            initial_sdr = torch.stack(global_sdrs).mean(dim=0) if global_sdrs else torch.zeros_like(clip_vec)
            
            propagated = expert.forward(initial_sdr.unsqueeze(0), steps=steps, top_k=60)
            pred_sdr = expert.predict_next_sdr(propagated.detach())
            prediction_error = expert.update_prediction(pred_sdr, propagated.detach())
            pred_results = expert.retrieve(pred_sdr, top_k=1)
            if pred_results:
                predicted_memory = pred_results[0][1]
            
            associate_results = expert.retrieve(propagated, top_k=topk*2, steps=2)
            
            for score, content, meta, idx, mem_id in associate_results:
                if mem_id in global_memory_pool:
                    global_memory_pool[mem_id]["expert_score"] = score
                    global_memory_pool[mem_id]["cross_validated"] = True
                    global_memory_pool[mem_id]["source"] = "both"
                elif not meta.get("is_obsolete", False):
                    mem_sdr = expert.mem_id_to_sdr.get(mem_id, initial_sdr)
                    global_memory_pool[mem_id] = {
                        "mem": {"id": mem_id, "content": content, "metadata": meta, "sdr": mem_sdr},
                        "global_score": 0.0, "expert_score": score, "source": "expert",
                        "cross_validated": False, "is_hippocampus": False
                    }
            
            dynamic_sim_threshold = 0.2 + min(total_energy / 30, 0.2)
            fused_results = []
            query_lower = text.lower()
            
            for mem_id, data in global_memory_pool.items():
                mem, g_score, e_score, cross, source, is_hippo = data["mem"], data["global_score"], data["expert_score"], data["cross_validated"], data["source"], data["is_hippocampus"]
                fusion_weight = 1.0
                
                if is_hippo: fusion_weight *= 1.5
                content, mem_expert = mem["content"], mem["metadata"].get("expert", "")
                if "[视觉文本]" in content or "绑定ID:" in content:
                    fusion_weight *= 1.5
                elif mem_expert == final_expert:
                    fusion_weight *= 2.0
                else:
                    fusion_weight *= 0.5
                if cross: fusion_weight *= 1.2
                
                fused_score = min(max(g_score, e_score), 1.0) * fusion_weight
                fused_results.append((fused_score, g_score, e_score, mem, cross, source, mem_expert, is_hippo))
            
            fused_results.sort(key=lambda x: -x[0])
            
            for fs, gs, es, mem, cv, s, me, ih in fused_results:
                sim = max(gs, es)
                th = dynamic_sim_threshold * 0.5 if ih else dynamic_sim_threshold
                if sim < th: continue
                activated_memories.append(mem)
                if len(activated_memories) >= topk: break
            
            if not activated_memories and fused_results:
                fallback = [r for r in fused_results if r[7]] or fused_results
                for i, (_, _, _, mem, _, _, _, _) in enumerate(fallback[:3]):
                    activated_memories.append(mem)
        
        return activated_memories, predicted_memory, prediction_error, propagated, similarity_trace

    def _link_visual_and_strengthen(self, activated_memories: List[Dict]) -> List[Dict]:
        """链接视觉记忆并增强激活计数（🔥 类型兼容版本）"""
        expanded_mem = []
        seen_ids = set()
        visual_expert = self.experts.get("视觉")
        
        for m in activated_memories:
            # 兼容MemoryPacket和字典两种格式
            mem_id = m.mem_id if isinstance(m, MemoryPacket) else m["id"]
            if mem_id in seen_ids: continue
            seen_ids.add(mem_id)
            expanded_mem.append(m)
            
            if not visual_expert:
                continue
            
            meta = m.metadata if isinstance(m, MemoryPacket) else m.get("metadata", {})
            bind_id = meta.get("multimodal_id", "")
            content = m.content if isinstance(m, MemoryPacket) else m.get("content", "")
            
            if not bind_id and "绑定ID:" in content:
                bind_id = content.split("绑定ID:")[-1].strip()
            if not bind_id: continue
            
            for idx_v, meta_v in enumerate(visual_expert.metadata_list):
                v_bind = meta_v.get("multimodal_id", "")
                if not v_bind and idx_v < len(visual_expert.content_list):
                    v_c = visual_expert.content_list[idx_v]
                    if "绑定ID:" in v_c: v_bind = v_c.split("绑定ID:")[-1].strip()
                if v_bind == bind_id:
                    vis_mem = {
                        "id": f"vis_{idx_v}",
                        "content": visual_expert.content_list[idx_v] if idx_v < len(visual_expert.content_list) else "[视觉记忆]",
                        "metadata": meta_v, "type": "visual"
                    }
                    if vis_mem["id"] not in seen_ids:
                        expanded_mem.append(vis_mem)
                        seen_ids.add(vis_mem["id"])
        
        for mem in expanded_mem:
            if isinstance(mem, MemoryPacket):
                mem.metadata["activate_count"] = mem.metadata.get("activate_count", 0) + 1
            elif isinstance(mem, dict) and "metadata" in mem:
                mem["metadata"]["activate_count"] = mem["metadata"].get("activate_count", 0) + 1
        
        return expanded_mem

    def _build_coherent_thought_chain(self, memories: List[Dict], similarity_trace: List[tuple], threshold: float) -> str:
        """构建连贯的思考链（🔥 类型兼容版本）"""
        if not memories:
            return "无思考内容"
        
        query_logic = self._extract_query_logic(self.current_query_text)
        entity = query_logic["entity"]
        logic_memories = self._build_symbolic_logic_chain(memories, query_logic)
        
        seen_contents = set()
        unique_memories = []
        for mem in logic_memories:
            content = mem.content if isinstance(mem, MemoryPacket) else mem["content"].strip()
            if content not in seen_contents:
                seen_contents.add(content)
                unique_memories.append(mem)
        
        if not unique_memories:
            return "无有效逻辑记忆"
        
        thought_parts = [f"🧠 逻辑起点：【{entity}】"]
        for idx, mem in enumerate(unique_memories):
            content = mem.content if isinstance(mem, MemoryPacket) else mem["content"].strip()
            if idx == 0:
                thought_parts.append(f"核心：{content}")
            else:
                thought_parts.append(f"→ 推导：{content}")
        
        full_chain = " | ".join(thought_parts)
        return full_chain[:800] + "..." if len(full_chain) > 800 else full_chain

    def _extract_query_logic(self, query_text: str) -> Dict[str, str]:
        """提取查询的逻辑结构"""
        if not query_text:
            return {"entity": "", "predicate": "", "type": "normal"}
        symbols = self.auto_grounding(query_text)
        return {
            "entity": symbols["entity"],
            "predicate": symbols["behavior"],
            "type": "grounded",
            "query": query_text
        }

    def _build_symbolic_logic_chain(self, memories: List[Dict], query_logic: Dict[str, str]) -> List[Dict]:
        """构建符号逻辑链（🔥 类型兼容版本）"""
        entity = query_logic["entity"]
        if not entity or not memories:
            return memories[:3]
        
        mem_logic_list = []
        for mem in memories:
            meta = mem.metadata if isinstance(mem, MemoryPacket) else mem.get("metadata", {})
            subj = meta.get("subject", "").strip()
            pred = meta.get("predicate", "").strip()
            obj = meta.get("object", "").strip()
            
            if not subj and hasattr(self.cortex, '_auto_extract_triplet'):
                try:
                    content = mem.content if isinstance(mem, MemoryPacket) else mem["content"]
                    triplet = self.cortex._auto_extract_triplet(content)
                    if triplet and len(triplet) >= 3:
                        subj, pred, obj = triplet
                        subj = str(subj).strip()
                        pred = str(pred).strip()
                        obj = str(obj).strip()
                except:
                    pass
            
            mem_logic_list.append({
                "mem": mem,
                "subj": subj,
                "pred": pred,
                "obj": obj
            })
        
        direct_logic = []
        indirect_logic = []
        irrelevant = []
        
        for item in mem_logic_list:
            s, p, o = item["subj"], item["pred"], item["obj"]
            if entity == s or entity == o:
                direct_logic.append(item)
            elif entity in s or entity in o:
                indirect_logic.append(item)
            else:
                irrelevant.append(item)
        
        ordered_items = direct_logic + indirect_logic
        
        if not ordered_items:
            return memories
        
        def get_sim(item):
            try:
                mem = item["mem"]
                content = mem.content if isinstance(mem, MemoryPacket) else mem["content"]
                mem_vec = self.encode_text(content)
                query_vec = self.encode_text(entity)
                return F.cosine_similarity(mem_vec.unsqueeze(0), query_vec.unsqueeze(0)).item()
            except:
                return 0.5
        
        ordered_items.sort(key=get_sim, reverse=True)
        return [item["mem"] for item in ordered_items]

    def _extract_core_ideas(self, memories: List[Dict]) -> List[str]:
        """提取核心思想（🔥 类型兼容版本）"""
        ideas = []
        for mem in memories:
            content = mem.content if isinstance(mem, MemoryPacket) else mem["content"]
            if "：" in content:
                ideas.append(content.split("：")[1][:15])
            else:
                ideas.append(content[:15])
        return list(set(ideas))

    def _get_query_expert_local(self, query: str) -> str:
        """本地规则判断查询所属专家"""
        symbols = self.auto_grounding(query)
        prop = symbols["property"]
        
        if "视觉" in prop or "图像" in prop:
            return "视觉"
        elif "自然风景" in prop or "建筑" in prop:
            return "空间"
        elif "人物身份" in prop or "宠物动物" in prop:
            return "身份"
        elif "知识概念" in prop or "抽象" in prop:
            return "抽象"
        else:
            return "概念"

    def auto_grounding(self, text: str) -> Dict[str, str]:
        """自动符号接地"""
        if text in self.symbol_grounding_cache:
            return self.symbol_grounding_cache[text]
        
        try:
            prompt = """
    你是实体语义提取器，严格遵守规则：
    1. 只输出一行标准JSON，不要解释、不要多余文字
    2. entity：提取句子核心实体名词
    3. behavior：精简为单个标准动词
    4. property：归纳为大类属性，只能从下面选一个：
    宠物动物 / 水果植被 / 自然风景 / 建筑居所 / 生活用品 / 人物身份 / 知识概念 / 视觉图像

    句子：%s
    输出严格格式：{"entity":"","behavior":"","property":""}
            """ % text.strip()
            
            response = self.llm.invoke(prompt)
            json_text = re.findall(r"\{.*?\}", response.content, re.S)[0]
            result = json.loads(json_text)
            
            self.symbol_grounding_cache[text] = {
                "entity": result["entity"],
                "behavior": result["behavior"],
                "property": result["property"]
            }
            return self.symbol_grounding_cache[text]
        
        except Exception as e:
            return {"entity": "未知", "behavior": "静态", "property": "通用物品"}

    def get_synapse_change(self) -> float:
        """获取全皮层突触变化量"""
        if not hasattr(self.cortex, 'experts'):
            return 0.0
        changes = []
        for exp in self.cortex.experts.values():
            if hasattr(exp, 'get_synapse_change'):
                changes.append(exp.get_synapse_change())
        return sum(changes) / len(changes) if changes else 0.0

    def encode_text(self, text: str) -> torch.Tensor:
        """编码文本为向量"""
        try:
            embedding = self.embedding_model.embed_query(text)
            clip_vec = torch.as_tensor(embedding, dtype=torch.float32)
            return clip_vec
        except Exception as e:
            logger.error(f"❌ 文本编码失败: {e}")
            raise

    def _update_interaction_time(self) -> None:
        """更新最后交互时间（有对话时调用）"""
        self.last_interaction_time = datetime.datetime.now()
        self.core.update_interaction_time()
        if self.core.is_mind_wandering:
            self._stop_mind_wandering()

    def _check_mind_wandering_trigger(self) -> None:
        """检查是否应该触发走神（定时调用）"""
        if (not self.mind_wandering_enabled 
            or self.core.is_mind_wandering 
            or self.core.fatigue_level >= self.core.fatigue_sleep_threshold):
            return
        
        idle_seconds = (datetime.datetime.now() - self.last_interaction_time).total_seconds()
        if idle_seconds >= self.mind_wandering_idle_threshold:
            self._start_mind_wandering()

    def _start_mind_wandering(self) -> None:
        """开始走神：启动后台思考线程"""
        if self.core.is_mind_wandering:
            return
            
        logger.info("🌙 大脑进入走神状态...")
        self.core.is_mind_wandering = True
        self._mind_wandering_running = True
        
        self.mind_wandering_thread = threading.Thread(target=self._mind_wandering_loop, daemon=True)
        self.mind_wandering_thread.start()

    def _stop_mind_wandering(self) -> None:
        """停止走神：瞬间回神"""
        if not self.core.is_mind_wandering:
            return
            
        logger.info("⚡ 大脑瞬间回神！")
        self.core.is_mind_wandering = False
        self._mind_wandering_running = False
        
        if self.mind_wandering_thread and self.mind_wandering_thread.is_alive():
            self.mind_wandering_thread.join(timeout=2.0)
            self.mind_wandering_thread = None

    def _mind_wandering_loop(self) -> None:
        """走神主循环（修复失控+正反馈问题）"""
        wander_start_time = datetime.datetime.now()
        # ✅ 新增：走神最大持续时间（2分钟），超时自动停止，避免无限运行
        MAX_WANDER_DURATION = 120  
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 5

        while self._mind_wandering_running:
            try:
                wander_elapsed = (datetime.datetime.now() - wander_start_time).total_seconds()
                
                # ✅ 新增1：超时自动停止走神
                if wander_elapsed > MAX_WANDER_DURATION:
                    logger.info("⏰ 走神已达最大时长，自动回神")
                    self._stop_mind_wandering()
                    break

                # ✅ 新增2：疲劳超过阈值时，立刻停止所有活动，只等待睡眠
                if self.core.fatigue_level >= self.core.fatigue_sleep_threshold:
                    logger.info("😴 疲劳已达睡眠阈值，停止所有走神活动")
                    self.core.needs_sleep_request = True
                    self._stop_mind_wandering()
                    break

                # 完整的空值安全能量计算
                if hasattr(self.hippocampus_router, 'last_scores') and self.hippocampus_router.last_scores is not None:
                    routing_probs = list(self.hippocampus_router.last_scores.values())
                else:
                    routing_probs = [0.2, 0.2, 0.2, 0.2, 0.2]

                triple_scores = []
                if self.symbolic_core and hasattr(self.symbolic_core, 'get_all_triplets'):
                    try:
                        triplets = self.symbolic_core.get_all_triplets()
                        triple_scores = [1.0 for _ in triplets]
                    except:
                        triple_scores = []

                try:
                    synapse_change = self.get_synapse_change()
                except:
                    synapse_change = 0.0

                rule_match = False

                total_energy, energy_detail = self.core.energy_field.total_energy(
                    routing_probs=routing_probs,
                    triple_scores=triple_scores,
                    sim_scores=[],
                    rule_match=rule_match,
                    # ✅ 新增3：走神时降低突触变化对能量的影响，切断正反馈
                    synapse_change=synapse_change * 0.3,  
                    is_wandering=self.core.is_mind_wandering,
                    fatigue_level=self.core.fatigue_level
                )
                
                # ✅ 新增4：限制能量最大值，避免无限上升
                total_energy = min(total_energy, 25.0)

                # 疲劳积累（降低基础疲劳，避免过快积累）
                base_fatigue = 0.001  # ✅ 从0.002降到0.001
                energy_fatigue_multiplier = 1.0 + max(0, (total_energy - 15) / 20)  # ✅ 降低乘数
                self.core.fatigue_level = min(1.0, self.core.fatigue_level + base_fatigue * energy_fatigue_multiplier)
                logger.debug(f"🧠 走神中 | 能量:{total_energy:.1f} | 疲劳:{self.core.fatigue_level:.2f} | 已运行:{wander_elapsed:.0f}秒")

                # 检查是否需要睡眠（提前触发，留缓冲）
                if not self.core.needs_sleep_request:
                    if self.core.fatigue_level >= self.core.fatigue_sleep_threshold * 0.95:
                        logger.info(f"😴 疲劳即将超限({self.core.fatigue_level:.2f})，请求睡眠...")
                        self.core.needs_sleep_request = True
                        self._stop_mind_wandering()
                        break

                # 动态调整走神概率（降低高能量时的放大倍数）
                base_recall = self.mind_wandering_recall_prob
                base_assoc = self.mind_wandering_assoc_prob
                
                if total_energy > 20:
                    dynamic_recall = min(0.7, base_recall * 1.3)  # ✅ 从1.6降到1.3
                    dynamic_assoc = min(0.6, base_assoc * 1.3)  # ✅ 从1.5降到1.3
                elif total_energy < 10:
                    dynamic_recall = max(0.2, base_recall * 0.6)
                    dynamic_assoc = max(0.1, base_assoc * 0.6)
                else:
                    dynamic_recall = base_recall
                    dynamic_assoc = base_assoc

                # 记忆闪回
                if random.random() < dynamic_recall:
                    self._mind_wandering_memory_recall()

                # 联想想象
                if random.random() < dynamic_assoc:
                    self._mind_wandering_association()

                # ✅ 新增5：疲劳超过80%时，只生成生理意图，禁止其他意图
                if self.core.fatigue_level < 0.8:
                    # 生成意图（降低生成概率，从0.3降到0.2）
                    if random.random() < 0.2:
                        self._generate_intentions()

                    # 执行意图（提高执行概率，从0.25升到0.35）
                    if random.random() < 0.35 and not self.pending_social_intention:
                        intention = self._execute_highest_priority_intention()
                        if intention and intention.action in ["express_tiredness", "share_memory", "ask_question", "explore_association", "review_memory"]:
                            self.pending_social_intention = intention
                else:
                    # 高疲劳时，只生成睡眠相关意图
                    if not any(i.type == "physiological" for i in self.intention_queue):
                        safe_priority = max(0.0, min(2.0, 
                            self.core.fatigue_level * self.intention_weights["physiological"] * 1.5
                        ))
                        self.intention_queue.append(Intention(
                            type="physiological",
                            priority=safe_priority,
                            content="我有点困了，先睡一会儿哦~ 睡醒了会更聪明的！",
                            action="express_tiredness",
                            need_sleep=True
                        ))

                consecutive_errors = 0  # 重置错误计数
                time.sleep(3)  # ✅ 从2秒延长到3秒，降低循环频率

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"❌ 走神过程出错({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {e}", exc_info=True)
                
                # ✅ 新增6：连续错误自动停止走神，避免崩溃
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.error("💥 走神连续出错，强制停止")
                    self._stop_mind_wandering()
                    break
                    
                time.sleep(3)
    
    def _mind_wandering_memory_recall(self) -> None:
        """走神时的记忆闪回（🔥 适配MemoryPacket）"""
        try:
            all_mem_ids = list(self.cortex.index.memories.keys())
            if not all_mem_ids:
                return
                
            weighted_mem_ids = []
            for mem_id in all_mem_ids:
                mem: Optional[MemoryPacket] = self.cortex.index.get_memory(mem_id)
                if not mem:
                    continue
                weight = mem.importance * 2 + mem.metadata.get("recency", 0.5)
                weighted_mem_ids.extend([mem_id] * int(weight * 10))
            
            if not weighted_mem_ids:
                return
                
            target_mem_id = random.choice(weighted_mem_ids)
            mem = self.cortex.index.get_memory(target_mem_id)
            if mem:
                self.cortex.increment_access_count(target_mem_id)
                logger.info(f"💭 记忆闪回: {mem.content[:40]}...")
                
        except Exception as e:
            logger.debug(f"记忆闪回失败: {e}")

    def _mind_wandering_association(self) -> None:
        """走神时的联想想象"""
        try:
            expert_name = random.choice(list(self.experts.keys()))
            expert = self.experts.get(expert_name)
            if not expert or not hasattr(expert, 'sdr_list') or not expert.sdr_list:
                return
                
            random_idx = random.randint(0, len(expert.sdr_list) - 1)
            start_sdr = expert.sdr_list[random_idx]
            
            with torch.no_grad():
                if hasattr(expert, 'forward'):
                    activated = expert.forward(start_sdr, steps=1, top_k=30)
                else:
                    activated = start_sdr
                
                if hasattr(expert, 'retrieve'):
                    assoc_results = expert.retrieve(activated, top_k=2)
                    if assoc_results:
                        assoc_content = assoc_results[0][1] if len(assoc_results[0]) > 1 else str(assoc_results[0])
                        logger.info(f"🤔 联想想象: → {assoc_content[:40]}...")
                    
        except Exception as e:
            logger.debug(f"联想想象失败: {e}")

    def _generate_intentions(self) -> None:
        """在走神时生成候选意图（强类型+去重+队列限制优化版）"""
        candidate_intentions: List[Intention] = []
        
        # 1. 生理意图：分级疲劳提示（优先级最高）
        if self.core.fatigue_level > 0.5:
            if self.core.fatigue_level >= self.core.fatigue_sleep_threshold:
                safe_priority = max(0.0, min(2.0, 
                    self.core.fatigue_level * self.intention_weights["physiological"] * 1.5
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
                    self.core.fatigue_level * self.intention_weights["physiological"]
                ))
                candidate_intentions.append(Intention(
                    type="physiological",
                    priority=safe_priority,
                    content=f"有点累了呢，不过还能再陪你玩一会儿~",
                    action="express_tiredness",
                    need_sleep=False
                ))
        
        # ✅ 新增：如果已经有睡眠意图，直接返回，不再生成其他意图
        if any(i.need_sleep for i in candidate_intentions):
            self.intention_queue = candidate_intentions[:1]  # 只保留最高优先级的睡眠意图
            logger.debug(f"🧠 已生成睡眠意图，跳过其他意图生成 | 队列长度：{len(self.intention_queue)}")
            return

        # 2. 认知意图：复习重要记忆
        if random.random() < 0.3:
            important_memories: List[MemoryPacket] = self._get_important_memories(limit=3)
            if important_memories:
                mem = random.choice(important_memories)
                safe_priority = max(0.0, min(2.0, 
                    mem.importance * self.intention_weights["cognitive"]
                ))
                candidate_intentions.append(Intention(
                    type="cognitive",
                    priority=safe_priority,
                    content=f"我想起了一件重要的事：{mem.content[:40]}...",
                    action="review_memory",
                    context={"memory": mem}
                ))
        
        # 3. 社交意图：主动分享近期记忆
        if random.random() < 0.35:
            recent_memories: List[MemoryPacket] = self._get_recent_memories(limit=15)
            if recent_memories:
                mem = random.choice(recent_memories)
                safe_priority = max(0.0, min(2.0, 
                    0.5 * self.intention_weights["social"]
                ))
                candidate_intentions.append(Intention(
                    type="social",
                    priority=safe_priority,
                    content=f"对了，我想起来我们之前聊过：{mem.content[:40]}...",
                    action="share_memory",
                    context={"memory": mem}
                ))
        
        # 4. 社交意图：主动提问探索知识
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
        
        # 5. 探索意图：发现记忆间的关联
        if random.random() < 0.15:
            associations = self._get_random_associations(limit=2)
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
        
        # ✅ 核心优化1：合并新旧意图 + 去重（避免重复生成相同类型的意图）
        all_intentions = self.intention_queue + candidate_intentions
        seen_types = set()
        unique_intentions = []
        
        # 按优先级从高到低去重，同一类型只保留最高优先级的1个
        for intention in sorted(all_intentions, key=lambda x: -x.priority):
            if intention.type not in seen_types:
                seen_types.add(intention.type)
                unique_intentions.append(intention)
        
        # ✅ 核心优化2：硬限制队列最大长度为3（执行间隔60秒，3个足够）
        # 原max_intention_queue_size=10太大，会导致严重堆积
        self.intention_queue = unique_intentions[:3]
        
        # ✅ 核心优化3：清空旧的替换逻辑，直接用新的排序后队列
        # 彻底删除原来逐个添加、替换最低优先级的复杂逻辑
        
        logger.debug(
            f"🧠 生成了{len(candidate_intentions)}个候选意图 | "
            f"去重后保留{len(self.intention_queue)}个 | "
            f"队列类型：{[i.type for i in self.intention_queue]} | "
            f"最高优先级：{(self.intention_queue[0].priority if self.intention_queue else 0):.2f}"
        )
        
    def _get_important_memories(self, limit: int = 5) -> List[MemoryPacket]:
        """获取重要记忆（🔥 强类型版本）"""
        important_mems: List[MemoryPacket] = []
        for mem_id in self.cortex.index.memories.keys():
            mem = self.cortex.index.get_memory(mem_id)
            if mem and mem.importance > 0.7:
                important_mems.append(mem)
        
        important_mems.sort(key=lambda x: x.importance, reverse=True)
        return important_mems[:limit]

    def _get_recent_memories(self, limit: int = 10) -> List[MemoryPacket]:
        """获取近期记忆（🔥 强类型版本）"""
        recent_mems: List[MemoryPacket] = []
        for mem_id in self.cortex.index.memories.keys():
            mem = self.cortex.index.get_memory(mem_id)
            if mem:
                recent_mems.append(mem)
        
        recent_mems.sort(key=lambda x: x.metadata.get("timestamp", 0), reverse=True)
        return recent_mems[:limit]

    def _find_knowledge_gaps(self) -> List[str]:
        """发现知识缺口（简化版）"""
        gaps = []
        try:
            recent_memories = self._get_recent_memories(limit=10)
            keywords = set()
            for mem in recent_memories:
                content = mem.content
                words = [content[i:i+2] for i in range(len(content)-1)]
                keywords.update([w for w in words if len(w) == 2])
            
            if keywords:
                sample_keywords = list(keywords)[:3]
                for kw in sample_keywords:
                    gaps.append(f"和{kw}相关的知识")
            
            gaps = list(set(gaps))
            random.shuffle(gaps)
            gaps = gaps[:5]
        except Exception as e:
            logger.debug(f"知识缺口发现失败: {e}")
            gaps = ["一些有趣的知识"]
        
        return gaps

    def _get_random_associations(self, limit: int = 3) -> List[str]:
        """获取随机关联记忆（🔥 强类型版本）"""
        associations = []
        if len(self.cortex.index.memories) > 0:
            random_mem_ids = random.sample(list(self.cortex.index.memories.keys()), min(limit, len(self.cortex.index.memories)))
            for mem_id in random_mem_ids:
                mem = self.cortex.index.get_memory(mem_id)
                if mem:
                    associations.append(mem.content)
        return associations

    def _execute_highest_priority_intention(self) -> Optional[Intention]:
        """执行最高优先级的意图（🔥 强类型版本）"""
        if not self.intention_queue:
            return None
        
        time_since_last = (datetime.datetime.now() - self.last_intention_execution_time).total_seconds()
        if time_since_last < self.min_intention_interval:
            return None
        
        # 🔥 强类型属性访问
        highest_intention = max(self.intention_queue, key=lambda x: x.priority)
        self.intention_queue.remove(highest_intention)
        
        logger.info(f"🎯 执行意图：{highest_intention.content} (优先级：{highest_intention.priority:.2f})")
        
        result = None
        if highest_intention.action in ["express_tiredness", "share_memory", "ask_question", "explore_association", "review_memory"]:
            result = highest_intention.content
        
        highest_intention.executed = True
        highest_intention.result = result
        self.last_intention_execution_time = datetime.datetime.now()
        
        return highest_intention

    def get_brain_status(self) -> Dict[str, Any]:
        """获取大脑状态信息（🔥 适配MemoryPacket）"""
        from collections import defaultdict
        import numpy as np
        
        # 安全获取总记忆数
        try:
            total_memories = len(self.cortex.index.memories)
        except:
            total_memories = 0
        
        expert_counts = defaultdict(int)
        expert_access = defaultdict(list)
        expert_sparsity = {}
        
        # 安全统计专家数据
        try:
            for mem_id in self.cortex.index.memories.keys():
                mem = self.cortex.index.get_memory(mem_id)
                if mem:
                    expert = mem.expert
                    expert_counts[expert] += 1
                    expert_access[expert].append(mem.access_count)
        except:
            pass
        
        # 安全获取专家稀疏度
        try:
            for name in self.experts.keys():
                if hasattr(self.experts[name], 'get_sparsity'):
                    expert_sparsity[name] = self.experts[name].get_sparsity()
                else:
                    expert_sparsity[name] = 0.0
        except:
            pass
        
        # 构建状态字典
        status = {
            "total_memories": total_memories,
            "ollama_model": "bge-m3",
            "embedding_dim": getattr(self.core.config, 'dim', 1024),
            "expert_distribution": {},
            "experts": {},
            "kg_enabled": getattr(self.cortex, 'kg_enabled', True),
            "is_mind_wandering": getattr(self.core, 'is_mind_wandering', False),
            "fatigue_level": getattr(self.core, 'fatigue_level', 0.0),
            "intention_queue_size": len(self.intention_queue)
        }
        
        # 安全填充专家详情
        try:
            for name in self.experts.keys():
                count = expert_counts.get(name, 0)
                access_list = expert_access.get(name, [0])
                avg_access = np.mean(access_list) if access_list else 0
                sparsity = expert_sparsity.get(name, 0.0)
                
                status["expert_distribution"][name] = count
                status["experts"][name] = {
                    "神经元": getattr(self.experts[name], 'dim', 2048),
                    "记忆数": count,
                    "平均访问": round(avg_access, 2),
                    "突触稀疏度": round(sparsity, 4)
                }
        except:
            pass
        
        return status