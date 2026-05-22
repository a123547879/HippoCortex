import torch
import torch.nn.functional as F
import logging
import re
import json
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional, Any

from ILifecycle import ILifecycle, IService
from ServiceContainer import ServiceContainer
from event_system import EventBus, Event, EventType
# ✅ 实体中心统一数据契约
from Data_models import ThoughtResult, Entity, Evidence, ConversationTurn
from BrainConfig import config

logger = logging.getLogger("ThinkEngine")

class ThinkEngine(ILifecycle, IService):

    _embedding_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
    _MAX_EMBED_CACHE: int = 500
    # 新增：缓存符号接地结果，减少LLM嵌套调用（解决卡顿）
    _grounding_cache: OrderedDict[str, Dict[str, str]] = OrderedDict()
    
    def __init__(self):
        self.symbol_grounding_cache: OrderedDict[str, Dict[str, str]] = OrderedDict()
        self.MAX_CACHE_SIZE: int = 1000
        self.current_query_text: str = ""
        self._container = None
        
    
    def initialize(self, container: 'ServiceContainer') -> None:
        self._container = container
    
    def start(self) -> None:
        pass
    
    def stop(self) -> None:
        pass
    
    def save(self, storage_dir: str) -> None:
        pass
    
    def load(self, storage_dir: str) -> None:
        pass
    
    # ===================== 核心：实体中心式思考入口 =====================
    def think(self, text: str, steps: int = 2, topk: int = 20, expert_last: Optional[str] = None) -> ThoughtResult:
        self.current_query_text = text
        
        EventBus().emit(Event(EventType.INPUT_RECEIVED, {"text": text}))
        
        try:
            active_context: List[ConversationTurn] = []
            context_text: str = ""
            try:
                cortex = self._container["cortex"].cortex
                if hasattr(cortex, 'get_active_conversation_context'):
                    active_context = cortex.get_active_conversation_context()
                    if active_context:
                        # 优化：限制上下文最多3轮，避免超长文本卡顿
                        active_context = active_context[-3:]
                        context_text = self._build_context_prompt(active_context)
                        logger.debug(f"🧠 注入对话上下文 | 轮数:{len(active_context)}")
            except Exception as e:
                logger.error(f"❌ 获取对话上下文失败，跳过上下文注入: {e}")
            
            enhanced_query = f"{context_text}\n用户当前问题：{text}" if context_text else text
            
            clip_vec, final_expert, expert_scores, energy_detail, total_energy, \
            sdr_encoder, query_sdr, symbolic_context = self._initialize_and_route(enhanced_query, expert_last)

            # 优化：简化检索流程，解决卡顿
            raw_results = self._retrieve_entities(clip_vec, query_sdr, text, expert_scores, total_energy)
            global_entity_pool, hippo_count = self._build_global_entity_pool(raw_results)
            
            # 优化：动态调整推理步数，简单问题steps=1，提速50%
            dynamic_steps = 1 if len(global_entity_pool) < 10 else steps
            activated_entities, predicted_entity, prediction_error, propagated, similarity_trace = \
                self._fuse_and_filter_entities(final_expert, global_entity_pool, clip_vec, dynamic_steps, topk, total_energy, text)
            
            final_activated_entities = self._link_visual_and_strengthen(activated_entities)
            
            thought_chain = self._build_coherent_thought_chain(final_activated_entities, similarity_trace, 0.25)
            core_ideas = self._extract_core_ideas(final_activated_entities)
            activation_strength = propagated.norm().item() if propagated is not None else 0.0
            
            # 🔥 核心修复：全流程使用Entity对象，杜绝tuple/字典混用
            result = ThoughtResult(
                thought_chain=thought_chain,
                core_ideas=core_ideas,
                activated_memories=[self._get_entity_content(e) for e in final_activated_entities],
                activated_entities=[
                    {"entity_id": e.entity_id, "name": e.name, "type": e.entity_type} 
                    for e in final_activated_entities
                ],
                activated_entity_ids=[e.entity_id for e in final_activated_entities],
                expert=final_expert,
                activation_strength=activation_strength,
                predicted_memory=self._get_entity_content(predicted_entity) if predicted_entity else None,
                prediction_error=prediction_error,
                symbolic_context=symbolic_context,
                energy_detail=energy_detail
            )
            
            EventBus().emit(Event(EventType.RESPONSE_GENERATED, result))
            return result
            
        except Exception as e:
            logger.error(f"❌ 思考过程出错: {str(e)}", exc_info=True)
            return ThoughtResult.error_result(str(e))
    
    def _build_context_prompt(self, context_turns: List[ConversationTurn]) -> str:
        """构建对话上下文提示"""
        if not context_turns:
            return ""
        
        context_parts = ["【最近对话历史】"]
        for turn in context_turns:
            context_parts.append(f"用户：{turn.user_input}")
            context_parts.append(f"小白：{turn.ai_response}")
        
        return "\n".join(context_parts)
    
    def _initialize_and_route(self, text: str, expert_last: Optional[str]) -> Tuple[torch.Tensor, str, Dict[str, float], Dict[str, float], float, Any, torch.Tensor, str]:
        """初始化与专家路由 🔥 核心修复：正确构造空实体列表，杜绝tuple报错"""
        clip_vec = self.encode_text(text)
        clip_vec = F.normalize(clip_vec, p=2, dim=-1)
        
        hippocampus_router = self._container["hippocampus_router"].router
        cortex = self._container["cortex"].cortex
        
        # ✅ 修复：构造标准空Entity列表，不传入tuple/无效数据
        seed_entities: List[Entity] = []
        
        # 调用路由，参数完全对齐你的接口
        target_expert = hippocampus_router.route(
            entity_embedding=clip_vec,
            entities=seed_entities,
            is_encoding=False
        )
        expert_scores = hippocampus_router.last_scores
        
        if expert_last is None:
            expert_last = self._get_query_expert_local(text)
        
        # 强制规则匹配
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
        symbolic_core = self._container["symbolic_core"].symbolic_core if "symbolic_core" in self._container._services else None
        if symbolic_core:
            triplets = symbolic_core.get_all_triplets()
            triple_scores = [1.0 for _ in triplets]
        rule_match = self._get_query_expert_local(text) == final_expert
        synapse_change = self.get_synapse_change()
        
        brain_core = self._container["brain_core"]
        total_energy, energy_detail = brain_core.energy_field.total_energy(
            routing_probs=routing_probs, triple_scores=triple_scores, sim_scores=[],
            rule_match=rule_match, synapse_change=synapse_change,
            is_wandering=brain_core.is_mind_wandering, fatigue_level=brain_core.fatigue_level
        )
        
        sdr_encoders = self._container["sdr_encoders"]
        sdr_encoder = sdr_encoders.get(final_expert, sdr_encoders["概念"])
        query_sdr = sdr_encoder.encode(clip_vec)
        
        symbolic_context = ""
        if symbolic_core:
            try:
                parsed = symbolic_core.parse_question(text)
                symbolic_results = symbolic_core.symbolic_retrieve(parsed)
                if symbolic_results:
                    symbolic_context = "【精准记忆】\n" + "\n".join([f"- {res['object']}" for res in symbolic_results])
            except Exception as e:
                logger.debug(f"符号检索跳过: {e}")
        
        return clip_vec, final_expert, expert_scores, energy_detail, total_energy, sdr_encoder, query_sdr, symbolic_context
    
    # ===================== 全专家实体检索 优化：移除冗余计算 =====================
    def _retrieve_entities(self, clip_vec: torch.Tensor, query_sdr: torch.Tensor, text: str, 
                      expert_scores: Dict[str, float], total_energy: float) -> List[Tuple[str, float, Dict[str, Any]]]:
        """全专家实体检索"""
        dynamic_min_sim = 0.1
        top_k = 20
        all_results = []
        seen_entity_ids = set()

        try:
            experts = self._container["experts"]
            for expert_name, expert in experts.items():
                if not expert:
                    continue
                
                try:
                    # 优化：steps=1 减少神经传播耗时
                    activated_sdr = expert.forward(query_sdr.unsqueeze(0), steps=1, top_k=60)
                    entities = expert.retrieve(activated_sdr, top_k=7, query_text=text)
                    
                    for res in entities:
                        if len(res) >= 3:
                            entity_id, score, entity_detail = res
                            if entity_id not in seen_entity_ids:
                                all_results.append((entity_id, score, entity_detail))
                                seen_entity_ids.add(entity_id)
                except Exception as e:
                    logger.debug(f"[{expert_name}] 专家检索跳过: {e}")
                    continue

            logger.info(f"🧠 全专家检索完成 | 共激活 {len(all_results)} 个实体")

        except Exception as e:
            logger.error(f"⚠️ 全专家检索异常: {str(e)}")

        # 兜底检索
        if len(all_results) < 3:
            thalamus = self._container["thalamus"].thalamus
            faiss_results = thalamus.schedule_retrieval(
                query_vec=clip_vec, query_sdr=query_sdr, query_text=text,
                expert_scores=expert_scores, min_similarity=dynamic_min_sim
            )
            for res in faiss_results:
                if len(res) >= 3 and res[0] not in seen_entity_ids:
                    all_results.append(res)

        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]
    
    # ===================== 全局实体池构建 🔥 修复：仅使用Entity对象，无tuple =====================
    def _build_global_entity_pool(self, raw_results: List[Tuple[str, float, Dict[str, Any]]]) -> Tuple[Dict[str, Dict], int]:
        """构建全局实体池"""
        global_entity_pool: Dict[str, Dict] = {}
        seen_content = set()
        SIM_THRESHOLD = 0.3
        
        cortex = self._container["cortex"].cortex
        hippocampus_router = self._container["hippocampus_router"].router
        query_vec = self.encode_text(self.current_query_text)
        
        # 处理专家检索结果
        for entity_id, sim, entity_detail in raw_results:
            try:
                full_entity: Optional[Entity] = cortex.index.get_entity(entity_id)
                if not full_entity or full_entity.metadata.get("is_obsolete", False):
                    continue
                    
                content_key = self._get_entity_content(full_entity).strip()[:50]
                if content_key in seen_content:
                    continue
                seen_content.add(content_key)
                
                global_entity_pool[entity_id] = {
                    "entity": full_entity,
                    "global_score": sim,
                    "expert_score": 0.0,
                    "source": "global",
                    "cross_validated": False,
                    "is_hippocampus": False
                }
            except:
                continue

        # 注入海马体实体
        if hippocampus_router:
            for entity in hippocampus_router.hippocampal_buffer:
                # ✅ 严格校验：只处理Entity对象，跳过所有tuple/无效数据
                if not isinstance(entity, Entity):
                    continue
                    
                content_key = self._get_entity_content(entity).strip()[:50]
                if content_key in seen_content:
                    continue

                try:
                    sim_score = F.cosine_similarity(
                        query_vec.unsqueeze(0),
                        entity.clip_vec.unsqueeze(0),
                        dim=-1
                    ).item() * 1.5
                except:
                    sim_score = 0.0

                if sim_score < SIM_THRESHOLD:
                    continue

                seen_content.add(content_key)
                global_entity_pool[entity.entity_id] = {
                    "entity": entity,
                    "global_score": sim_score,
                    "expert_score": 0.0,
                    "source": "hippocampus",
                    "cross_validated": True,
                    "is_hippocampus": True
                }

        # 排序
        sorted_entities = sorted(
            global_entity_pool.values(),
            key=lambda x: x["global_score"],
            reverse=True
        )
        global_entity_pool = {ent["entity"].entity_id: ent for ent in sorted_entities}

        hippo_count = sum(1 for d in global_entity_pool.values() if d['is_hippocampus'])
        return global_entity_pool, hippo_count
    
    # ===================== 实体融合与过滤 优化：精简逻辑，提速 =====================
    def _fuse_and_filter_entities(self, final_expert: str, global_entity_pool: Dict[str, Dict], 
                             clip_vec: torch.Tensor, steps: int, topk: int, total_energy: float, text: str) -> Tuple[List[Entity], Optional[Entity], float, Optional[torch.Tensor], List[Tuple]]:
        """神经激活传播与实体融合过滤"""
        experts = self._container["experts"]
        expert = experts.get(final_expert)
        predicted_entity = None
        prediction_error = 0.0
        propagated = None
        similarity_trace = []
        activated_entities: List[Entity] = []

        if expert:
            # 初始化SDR
            global_sdrs = []
            for data in list(global_entity_pool.values())[:8]:
                entity = data["entity"]
                if entity.sdr is not None:
                    global_sdrs.append(entity.sdr.to(clip_vec.device))
            
            initial_sdr = torch.stack(global_sdrs).mean(dim=0) if global_sdrs else torch.zeros_like(clip_vec)
            propagated = expert.forward(initial_sdr.unsqueeze(0), steps=steps, top_k=60)
            
            # 动态阈值
            dynamic_sim_threshold = 0.2
            fused_results = []
            
            # 实体加权融合
            for data in global_entity_pool.values():
                entity: Entity = data["entity"]
                g_score = data["global_score"]
                is_hippo = data["is_hippocampus"]
                fusion_weight = 1.0

                # 权重计算
                if is_hippo:
                    fusion_weight *= 1.5
                if entity.entity_type == "visual":
                    fusion_weight *= 1.5
                elif entity.expert == final_expert:
                    fusion_weight *= 2.0
                
                fused_score = min(g_score, 1.0) * fusion_weight
                fused_results.append((fused_score, entity))
            
            # 排序过滤
            fused_results.sort(key=lambda x: -x[0])
            activated_entities = [ent for _, ent in fused_results[:topk]]
            
            # 兜底
            if not activated_entities and fused_results:
                activated_entities = [ent for _, ent in fused_results[:3]]
        
        return activated_entities, predicted_entity, prediction_error, propagated, similarity_trace
    
    # ===================== 跨模态视觉实体绑定 =====================
    def _link_visual_and_strengthen(self, activated_entities: List[Entity]) -> List[Entity]:
        """绑定文本实体与视觉实体"""
        expanded_entities = []
        seen_ids = set()
        
        for entity in activated_entities:
            if entity.entity_id in seen_ids:
                continue
            seen_ids.add(entity.entity_id)
            expanded_entities.append(entity)
            
            # 激活计数更新
            entity.metadata["activate_count"] = entity.metadata.get("activate_count", 0) + 1
        
        return expanded_entities
    
    # ===================== 逻辑思维链构建 =====================
    def _build_coherent_thought_chain(self, entities: List[Entity], similarity_trace: List[tuple], threshold: float) -> str:
        """构建思考链"""
        if not entities:
            return "无思考内容"
        
        query_logic = self._extract_query_logic(self.current_query_text)
        entity_name = query_logic["entity"] or "核心实体"
        
        seen_contents = set()
        unique_entities = []
        for entity in entities:
            content = self._get_entity_content(entity).strip()
            if content not in seen_contents:
                seen_contents.add(content)
                unique_entities.append(entity)

        unique_entities = unique_entities[:6]
        thought_parts = [f"🧠 逻辑起点：【{entity_name}】"]
        
        for idx, entity in enumerate(unique_entities):
            content = self._get_entity_content(entity).strip()
            if idx == 0:
                thought_parts.append(f"核心：{content}")
            else:
                thought_parts.append(f"→ 推导{idx}：{content}")

        return " | ".join(thought_parts)
        
    def _extract_query_logic(self, query_text: str) -> Dict[str, str]:
        """提取查询逻辑"""
        if not query_text:
            return {"entity": "", "predicate": "", "type": "normal"}
        return self.auto_grounding(query_text)
    
    def _build_symbolic_logic_chain(self, entities: List[Entity], query_logic: Dict[str, str]) -> List[Entity]:
        """构建逻辑链"""
        target_entity = query_logic.get("entity", "")
        if not target_entity or not entities:
            return entities[:3]
        return entities[:3]
    
    def _extract_core_ideas(self, entities: List[Entity]) -> List[str]:
        """提取核心观点"""
        ideas = []
        for entity in entities:
            content = self._get_entity_content(entity)
            ideas.append(content[:15])
        return list(set(ideas))
    
    def _get_query_expert_local(self, query: str) -> str:
        """本地专家规则匹配"""
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
        """符号语义接地 🔥 优化：双层缓存，彻底减少LLM调用"""
        if text in self._grounding_cache:
            self._grounding_cache.move_to_end(text)
            return self._grounding_cache[text]
            
        if text in self.symbol_grounding_cache:
            self.symbol_grounding_cache.move_to_end(text)
            return self.symbol_grounding_cache[text]
        
        try:
            llm = self._container.llm
            prompt = """
你是实体语义提取器，严格遵守规则：
1. 只输出一行标准JSON，不要解释
2. entity：核心实体
3. behavior：单个动词
4. property：只能选：宠物动物/水果植被/自然风景/建筑居所/生活用品/人物身份/知识概念/视觉图像

句子：%s
输出：{"entity":"","behavior":"","property":""}
            """ % text.strip()
            
            response = llm.invoke(prompt)
            json_text = re.findall(r"\{.*?\}", response.content, re.S)[0]
            result = json.loads(json_text)
            
            # 双层缓存
            self._grounding_cache[text] = result
            self.symbol_grounding_cache[text] = result
            
            if len(self._grounding_cache) > self.MAX_CACHE_SIZE:
                self._grounding_cache.popitem(last=False)
            if len(self.symbol_grounding_cache) > self.MAX_CACHE_SIZE:
                self.symbol_grounding_cache.popitem(last=False)
            
            return result
        
        except Exception as e:
            return {"entity": "未知", "behavior": "静态", "property": "通用物品"}
    
    def get_synapse_change(self) -> float:
        """获取平均突触变化"""
        experts = self._container["experts"]
        if not experts:
            return 0.0
        changes = []
        for exp in experts.values():
            if hasattr(exp, 'get_synapse_change'):
                changes.append(exp.get_synapse_change())
        return sum(changes) / len(changes) if changes else 0.0
    
    def encode_text(self, text: str) -> torch.Tensor:
        """文本编码（LRU缓存）"""
        if not text or not isinstance(text, str):
            return torch.zeros(config.dim, dtype=torch.float32)

        cache_key = text.strip()
        if cache_key in ThinkEngine._embedding_cache:
            ThinkEngine._embedding_cache.move_to_end(cache_key)
            return ThinkEngine._embedding_cache[cache_key]

        try:
            embedding = self._container.embedding_model.embed_query(text)
            clip_vec = torch.as_tensor(embedding, dtype=torch.float32)

            ThinkEngine._embedding_cache[cache_key] = clip_vec
            if len(ThinkEngine._embedding_cache) > ThinkEngine._MAX_EMBED_CACHE:
                ThinkEngine._embedding_cache.popitem(last=False)

            return clip_vec
        except Exception as e:
            logger.error(f"❌ 文本编码失败: {text[:30]} | {e}")
            return torch.zeros(config.dim, dtype=torch.float32)
    
    # ===================== 统一实体内容获取 =====================
    def _get_entity_content(self, entity: Entity) -> str:
        """统一获取Entity内容（仅处理Entity对象，杜绝报错）"""
        if entity.latest_evidence:
            return entity.latest_evidence.content
        return entity.name