import torch
import time
import re
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple

# ✅ 仅保留必要的 LangChain 依赖（不再导入 ChatOllama/OllamaEmbeddings）
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException

from brain_core import BrainCore
from event_system import EventBus, Event, EventType
from Thalamus import Thalamus
from HippocampusRouter import HippocampusRouter
from SymbolicCore import SymbolicCore
from BrainConfig import config
from DopamineSystem import DopamineSystem
from Metacognition import Metacognition
from Curiosity import Curiosity
# ✅ 替换为实体中心统一数据契约
from Data_models import Entity, EntityRelation, Evidence, MemoryFactory

logger = logging.getLogger("LearningLoop")

# ===================== 🔴 全局配置：实体提取器参数 =====================
MAX_RETRIES = 2                              # 提取失败重试次数
# 注意：LLM 和 Embedding 模型现在由外部传入，不再在这里配置
# ======================================================================


# ✅ 重构后的 LangChain 实体提取器（接收外部传入的模型）
class LangChainEntityExtractor:
    def __init__(
        self,
        llm: Any,                # ✅ 直接接收外部传入的 ChatOllama 实例
        embeddings: Any,         # ✅ 直接接收外部传入的 OllamaEmbeddings 实例
        temperature: float = 0.1
    ):
        # 直接复用外部传入的模型，不再内部创建
        self.llm = llm
        self.embeddings = embeddings
        
        # 初始化 JSON 输出解析器
        self.json_parser = JsonOutputParser()
        
        # 构建提示词模板（适配你的专家体系）
        self.prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            请从以下文本中提取所有实体和实体之间的关系，严格按照指定的JSON格式输出。
            不要添加任何解释、多余文字或markdown格式，只输出纯JSON。

            【实体类型限制】只能使用以下类型（严格对应）：
            person(人物)、place(地点)、event(事件)、concept(概念)、object(物品)、organization(组织)、skill(技能)、system(系统)、emotion(情绪)、identity(身份)、visual(视觉)

            【关系要求】
            1. 谓词简洁准确，如：是、属于、位于、发生在、创建、拥有、参与、创作、包含、叫、取名为
            2. 置信度范围0.0-1.0，根据信息明确程度打分

            【输出格式】
            {{
                "entities": [
                    {{
                        "name": "实体名称",
                        "type": "实体类型",
                        "description": "实体的1句话简短描述"
                    }}
                ],
                "relations": [
                    {{
                        "subject": "主体实体名",
                        "predicate": "关系谓词",
                        "object": "客体实体名",
                        "confidence": 0.95
                    }}
                ]
            }}

            待处理文本：{text}
            """
        )
        
        # 构建处理链
        self.chain = self.prompt | self.llm

    def generate_embedding(self, text: str) -> List[float]:
        """生成文本的 BGE-M3 语义向量（复用外部嵌入模型）"""
        return self.embeddings.embed_query(text)

    def extract_entities_and_relations(self, text: str) -> Dict[str, Any]:
        """
        核心方法：提取实体+关系+嵌入向量
        返回格式与原有系统100%兼容
        """
        for attempt in range(MAX_RETRIES + 1):
            try:
                # 调用外部传入的 LLM 提取
                response = self.chain.invoke({"text": text})
                raw_output = response.content.strip()
                
                # 提取并解析 JSON
                json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
                if not json_match:
                    raise OutputParserException("未找到JSON格式输出")
                
                result = self.json_parser.parse(json_match.group())
                
                # 补全默认字段
                result.setdefault("entities", [])
                result.setdefault("relations", [])
                result["raw_text"] = text
                # ✅ 新增：兼容原有系统的 emotion 字段
                result["emotion"] = {"valence": 0.0, "arousal": 0.5}
                
                return result

            except Exception as e:
                if attempt < MAX_RETRIES:
                    logger.debug(f"⚠️  第 {attempt+1} 次提取失败，重试中... 错误: {str(e)[:50]}")
                    continue
                else:
                    # 最终兜底：回退到原有规则提取
                    logger.warning(f"❌ LLM提取失败，回退到规则提取")
                    return self._rule_based_fallback(text)

    def _rule_based_fallback(self, text: str) -> Dict[str, Any]:
        """终极兜底：原有规则提取逻辑（保证系统不崩溃）"""
        entities = []
        relations = []
        seen_words = set()

        # 正则提取中文关键词
        keywords = re.findall(r"[\u4e00-\u9fa5]{2,8}", text)
        
        # 实体类型匹配（对齐你的专家体系）
        TYPE_TEMPLATES = {
            "person": "人物 身份 我 你 小白 主人",
            "place": "地点 杭州 浙江 学校 城市",
            "concept": "知识 学习 科研 理论 方法 技术",
            "emotion": "真诚 友善 温柔 善良 乐观",
            "object": "宠物 小狗 伙伴 物品"
        }

        # 生成实体
        for word in keywords[:6]:
            if word in seen_words:
                continue
            seen_words.add(word)
            entities.append({
                "name": str(word),
                "type": "concept",
                "description": "规则提取的通用实体",
                "attributes": {}
            })

        # 兜底实体
        if not entities:
            entities.append({"name": "通用知识", "type": "concept", "description": "未提取到明确实体的通用内容"})

        # 生成关系
        if len(entities) >= 2:
            relations.append({
                "subject": entities[0]["name"],
                "predicate": "相关",
                "object": entities[1]["name"],
                "confidence": 0.8
            })

        return {
            "entities": entities,
            "relations": relations,
            "raw_text": text,
            "emotion": {"valence": 0.0, "arousal": 0.5}
        }


class LearningLoop:
    def __init__(self, core: BrainCore, event_bus: EventBus, embedding_model: Any, llm: Any):
        self.core: BrainCore = core
        self.event_bus: EventBus = event_bus
        self.embedding_model: Any = embedding_model  # ✅ 你传入的 OllamaEmbeddings 实例
        self.llm: Any = llm                          # ✅ 你传入的 ChatOllama 实例
        
        # 组件引用
        self.thalamus: Optional[Thalamus] = None
        self.hippocampus_router: Optional[HippocampusRouter] = None
        self.symbolic_core: Optional[SymbolicCore] = None
        self.experts: Dict[str, Any] = {}
        self.sdr_encoders: Dict[str, Any] = {}
        self.cortex: Optional[Any] = None
        self.perception_loop: Optional[Any] = None
        
        # 类人学习核心模块
        self.dopamine: Optional[DopamineSystem] = None
        self.metacognition: Optional[Metacognition] = None
        self.curiosity: Optional[Curiosity] = None
        
        # ===================== 🔴 核心修改：使用传入的模型初始化实体提取器 =====================
        self.entity_extractor = LangChainEntityExtractor(
            llm=self.llm,                # ✅ 直接传入你初始化好的 LLM
            embeddings=self.embedding_model  # ✅ 直接传入你初始化好的 Embedding 模型
        )
        # ==================================================================================
        
        # ===================== 🔴 核心替换：全局实体突触连接 =====================
        # 键：(主体实体ID, 客体实体ID)，值：全局突触权重
        self.global_entity_synapses: Dict[Tuple[str, str], float] = {}
        self.synapse_save_path: str = ""
        
        # 学习功能开关（完全保留原有配置）
        self.use_predictive_stdp: bool = True
        self.use_cross_modal_learning: bool = True
        self.sync_global_synapses: bool = True

    def bind_components(self, thalamus: Thalamus, hippocampus_router: HippocampusRouter, 
                       symbolic_core: SymbolicCore, experts: Dict[str, Any], sdr_encoders: Dict[str, Any], cortex: Any,
                       dopamine: DopamineSystem, metacognition: Metacognition, curiosity: Curiosity,
                       perception_loop: Any = None) -> None:
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
        self.perception_loop = perception_loop
        
        logger.info(f"✅ LearningLoop组件绑定完成")
        if self.perception_loop and self.use_cross_modal_learning:
            logger.info(f"🧠 跨模态脑桥学习已启用")

    def set_synapse_save_path(self, path: str) -> None:
        """设置突触保存路径"""
        self.synapse_save_path = path
        self._load_global_synapses()

    # ===================== 🔴 核心：实体中心式学习入口 =====================
    def learn(self, text: str, force_expert: Optional[str] = None, external_reward: float = 0.0) -> List[str]:
        """
        🔥 实体中心式学习新记忆（流程重构版）
        正确流程：提取实体 → 实体驱动路由 → 生成结构化记忆 → 多巴胺强化
        :param text: 学习内容（陈述句）
        :param force_expert: 强制指定专家
        :param external_reward: 外部奖励 (-1 到 1)
        :return: 本次学习创建/更新的实体ID列表
        """
        def is_declarative_sentence(q: str) -> bool:
            """判断是否为陈述句（应该被学习）"""
            question_words = ["？", "?", "哪里", "谁", "怎么", "吗", "呢"]
            if "什么" in q:
                import re
                if not re.search(r'(不|没|没有)什么', q):
                    return False
            return not any(word in q for word in question_words)
        
        # 疑问句跳过学习（完全保留原有逻辑）
        if not is_declarative_sentence(text):
            logger.info(f"🚫 输入是疑问句，跳过学习: {text[:30]}...")
            return []
        
        # ===================== 🔴 核心流程重构：先提取实体，再路由 =====================
        # 步骤1：提取实体与关系（原步骤3提前，所有后续流程基于实体）
        # ✅ 复用传入模型的实体提取器
        extraction_result = self.entity_extractor.extract_entities_and_relations(text)
        
        if not extraction_result["entities"]:
            logger.info(f"📝 未提取到实体，将文本存储到'通用知识'实体")
            extraction_result["entities"].append({
                "name": "通用知识",
                "type": "concept",
                "attributes": {}
            })
        
        # 提取情绪信息（用于多巴胺奖励）
        emotion = extraction_result.get("emotion", {"valence": 0.0, "arousal": 0.5})
        
        # 步骤2：生成文本全局向量与实体聚合向量
        global_clip_vec = self._encode_text(text)
        global_clip_vec = torch.nn.functional.normalize(global_clip_vec.detach().squeeze(), p=2, dim=-1)
        
        # 计算实体聚合向量（用于路由，权重为实体初始重要性）
        entity_vectors = []
        entity_weights = []
        for entity_data in extraction_result["entities"]:
            # 为每个实体生成独立向量（复用外部嵌入模型）
            entity_vec = self._encode_text(entity_data["name"])
            entity_vec = torch.nn.functional.normalize(entity_vec.detach().squeeze(), p=2, dim=-1)
            entity_vectors.append(entity_vec)
            entity_weights.append(entity_data.get("importance", 0.5))
        
        # 加权平均得到聚合实体向量
        if entity_vectors:
            aggregated_clip_vec = torch.stack(entity_vectors).mean(dim=0)
            aggregated_clip_vec = torch.nn.functional.normalize(aggregated_clip_vec, p=2, dim=-1)
        else:
            aggregated_clip_vec = global_clip_vec
        
        # 步骤3：丘脑过滤（传入实体信息，增强过滤准确性）
        passed, info_packet = self.thalamus.filter_and_relay(
            input_vec=aggregated_clip_vec,
            input_text=text,
            metadata={
                "force_expert": force_expert,
                "entity_count": len(extraction_result["entities"]),
                "entity_types": [e["type"] for e in extraction_result["entities"]]
            }
        )
        
        if not passed:
            logger.info(f"🚫 信息被丘脑过滤: {text[:30]}...")
            return []
        
        aggregated_clip_vec = info_packet["vec"]
        saliency = info_packet["saliency"]
        
        # 步骤4：纯实体驱动路由（不再传入text，完全基于实体）
        if force_expert is None:
            # 构建临时实体列表用于路由（仅包含路由必需的字段）
            routing_entities = []
            for entity_data in extraction_result["entities"]:
                temp_entity = MemoryFactory.create_entity(
                    name=entity_data["name"],
                    entity_type=entity_data["type"],
                    sdr=torch.zeros(config.sdr_dim),  # 路由不需要SDR
                    clip_vec=self._encode_text(entity_data["name"]).squeeze(),
                    importance=entity_data.get("importance", 0.5)
                )
                routing_entities.append(temp_entity)
            
            # 调用实体驱动路由
            target_expert = self.hippocampus_router.route(
                entity_embedding=aggregated_clip_vec,
                entities=routing_entities,
                is_encoding=True
            )
            # 在线学习路由网络
            self.hippocampus_router.online_learn(aggregated_clip_vec, target_expert)
        else:
            target_expert = force_expert
        
        # 步骤5：为每个实体生成专属SDR
        sdr_encoder = self.sdr_encoders.get(target_expert, self.sdr_encoders["概念"])
        for entity_data in extraction_result["entities"]:
            entity_vec = self._encode_text(entity_data["name"]).squeeze()
            entity_data["sdr"] = sdr_encoder.encode(entity_vec.unsqueeze(0))
            entity_data["clip_vec"] = entity_vec
            # ✅ 新增：将提取器返回的 description 存入 attributes
            entity_data.setdefault("attributes", {})["description"] = entity_data.get("description", "")
        
        # 步骤6：元认知预评估（基于实体对象）
        prior_confidences = {}
        current_time = time.time()
        for entity_data in extraction_result["entities"]:
            entity_name = entity_data["name"]
            if self.metacognition:
                try:
                    prior_confidences[entity_name] = self.metacognition.assess_knowledge_confidence(
                        entity_name, current_time
                    )
                    logger.debug(f"🧠 元认知：学习前对 '{entity_name}' 的置信度 = {prior_confidences[entity_name]:.2f}")
                except Exception as e:
                    prior_confidences[entity_name] = 0.0
                    logger.debug(f"元认知预评估跳过: {e}")
        
        # ===================== 🔴 核心：结构化记忆存储 =====================
        # 步骤7：创建标准Entity+Evidence+EntityRelation对象
        created_entities: List[Entity] = []
        created_relations: List[EntityRelation] = []
        
        # 7.1 创建全文本证据
        evidence = MemoryFactory.create_evidence(
            content=text,
            source="对话",
            sdr=sdr_encoder.encode(global_clip_vec.unsqueeze(0)),
            clip_vec=global_clip_vec,
            confidence=0.95,
            emotion_valence=emotion["valence"],
            emotion_arousal=emotion["arousal"],
            metadata={
                "saliency": saliency,
                "expert": target_expert,
                "is_fact": True
            }
        )
        
        # 7.2 创建实体并关联证据
        entity_map: Dict[str, Entity] = {}  # 实体名称→实体对象映射（用于关系创建）
        for entity_data in extraction_result["entities"]:
            entity = MemoryFactory.create_entity(
                name=entity_data["name"],
                entity_type=entity_data["type"],
                sdr=entity_data["sdr"],
                clip_vec=entity_data["clip_vec"],
                importance=entity_data.get("importance", 0.5),
                expert=target_expert,
                attributes=entity_data.get("attributes", {})
            )
            # 关联全文本证据
            entity.add_evidence(evidence)
            # 添加别名
            if "aliases" in entity_data:
                entity.aliases.update(entity_data["aliases"])
            # 添加标签
            if "tags" in entity_data:
                entity.tags.update(entity_data["tags"])
            
            created_entities.append(entity)
            entity_map[entity.name] = entity
        
        # 7.3 创建实体关系
        for rel_data in extraction_result["relations"]:
            subj_name = rel_data["subject"]
            pred = rel_data["predicate"]
            obj_name = rel_data["object"]
            
            if subj_name in entity_map and obj_name in entity_map:
                relation = MemoryFactory.create_relation(
                    subject=entity_map[subj_name],
                    predicate=pred,
                    object=entity_map[obj_name],
                    confidence=rel_data.get("confidence", 0.9),
                    initial_weight=0.1,
                    evidence=evidence
                )
                created_relations.append(relation)
        
        # 步骤8：存储到皮层（批量存储，原子操作）
        stored_entity_ids = []
        if created_entities:
            stored_entity_ids = self.cortex.store_entities(created_entities)
            logger.info(f"✅ 实体已存入皮层 | 创建/更新: {len(stored_entity_ids)} 个 | 主专家: {target_expert}")
        
        if created_relations and stored_entity_ids:
            stored_relation_ids = self.cortex.store_relations(created_relations)
            logger.info(f"✅ 关系已存入皮层 | 创建: {len(stored_relation_ids)} 个")
        
        if not stored_entity_ids:
            logger.error(f"❌ 皮层存储失败: {text[:30]}...")
            return []
        
        # 步骤9：同步到神经专家网络（完全保留原有逻辑）
        for entity in created_entities:
            if entity.entity_id in stored_entity_ids and entity.expert in self.experts:
                expert = self.experts[entity.expert]
                expert.add_entity(entity)
                logger.debug(f"🧠 同步写入神经专家网络 | {entity.expert} | 实体: {entity.name} ({entity.entity_id[:8]})")
        
        # 步骤10：发射实体存储事件（增强版，包含实体和关系信息）
        main_entity = created_entities[0] if created_entities else None
        self.event_bus.emit(Event(EventType.MEMORY_STORED, {
            "entity_ids": stored_entity_ids,
            "relation_ids": [r.relation_id for r in created_relations],
            "main_entity_id": main_entity.entity_id if main_entity else "",
            "main_entity_name": main_entity.name if main_entity else "",
            "text": text,
            "vec": aggregated_clip_vec.cpu().numpy().tolist(),
            "expert_name": target_expert,
            "saliency": saliency,
            "is_fact": True,
            "entity_count": len(stored_entity_ids),
            "relation_count": len(created_relations),
            "timestamp": time.time()
        }))
        
        # 步骤11：符号核心学习（基于结构化关系，完全保留原有逻辑）
        if self.symbolic_core and created_relations:
            try:
                for relation in created_relations:
                    self.symbolic_core.add_triplet(
                        subj=entity_map[relation.subject_id].name,
                        pred=relation.predicate,
                        obj=entity_map[relation.object_id].name,
                        entity_ids=[relation.subject_id, relation.object_id]
                    )
                logger.debug(f"✅ 符号核心学习完成 | 提取关系数: {len(created_relations)}")
            except Exception as e:
                logger.debug(f"符号学习跳过: {e}")
        
        # 步骤12：跨模态实体绑定与学习（适配实体对象）
        if target_expert != "视觉" and created_entities:
            self._bind_cross_modal_entities(
                entities=created_entities,
                global_clip_vec=aggregated_clip_vec,
                text=text,
                target_expert=target_expert
            )
        
        # ===================== 🔴 多巴胺强化学习（适配新系统） =====================
        # 步骤13：多维度奖励计算与突触调节
        if self.dopamine and stored_entity_ids:
            # 13.1 计算各维度奖励
            total_reward = 0.0
            
            # 好奇心奖励（基于信息增益=置信度提升）
            information_gain = 0.0
            for entity in created_entities:
                prior = prior_confidences.get(entity.name, 0.0)
                posterior = min(1.0, prior + 0.2)  # 学习后置信度提升
                information_gain += (posterior - prior)
            information_gain = min(1.0, information_gain / max(len(created_entities), 1))
            curiosity_reward = self.dopamine.get_curiosity_reward(information_gain)
            total_reward += curiosity_reward
            
            # 实体发现奖励
            for entity in created_entities:
                is_new = self.cortex.is_new_entity(entity.name)
                discovery_reward = self.dopamine.get_entity_discovery_reward(is_new, entity.importance)
                total_reward += discovery_reward
            
            # 关系建立奖励
            for relation in created_relations:
                is_new = self.cortex.is_new_relation(relation.subject_id, relation.predicate, relation.object_id)
                relation_reward = self.dopamine.get_relation_establishment_reward(is_new, relation.confidence)
                total_reward += relation_reward
            
            # 情绪奖励
            emotion_reward = self.dopamine.get_emotion_reward(emotion["valence"], emotion["arousal"])
            total_reward += emotion_reward
            
            # 外部奖励
            external_reward_scaled = self.dopamine.get_external_feedback_reward(external_reward)
            total_reward += external_reward_scaled
            
            # 限制总奖励范围
            total_reward = float(np.clip(total_reward, -1.0, 1.0))
            
            # 13.2 计算RPE并更新多巴胺
            rpe = self.dopamine.compute_reward_prediction_error(
                actual_reward=total_reward,
                related_entity_ids=stored_entity_ids,
                emotion_valence=emotion["valence"],
                emotion_arousal=emotion["arousal"],
                reward_type="learning"
            )
            
            # 13.3 应用奖励到所有实体和关系
            self.dopamine.apply_reward_to_entities_and_relations(
                rpe=rpe,
                entities=created_entities,
                relations=created_relations,
                learning_rate=0.01
            )
            
            logger.info(
                f"🧠 多巴胺强化完成 | 总奖励: {total_reward:.3f} | RPE: {rpe:.3f} | "
                f"好奇心: {curiosity_reward:.3f} | 外部: {external_reward_scaled:.3f}"
            )
        
        # 步骤14：元认知后评估与好奇心触发（完全保留原有逻辑）
        if self.metacognition and stored_entity_ids:
            self._metacognition_post_assessment(
                created_entity_ids=stored_entity_ids,
                extraction_result=extraction_result,
                prior_confidences=prior_confidences,
                current_time=current_time
            )
        
        return stored_entity_ids

    # ===================== 🔴 批量实体导入 =====================
    def batch_init_direct_to_cortex(self, texts: List[str]) -> List[str]:
        """
        初始批量导入：直接写入皮层长期记忆
        :param texts: 待导入的文本列表
        :return: 所有创建的实体ID列表
        """
        if not self.cortex:
            logger.error("❌ 皮层组件未绑定，无法批量导入")
            return []

        all_created_entity_ids = []
        logger.info(f"🚀 初始批量直接导入皮层 | 共{len(texts)}条文本")

        for text in texts:
            try:
                entity_ids = self.learn(text=text, external_reward=0.5)
                all_created_entity_ids.extend(entity_ids)
            except Exception as e:
                logger.warning(f"⚠️ 文本导入失败跳过: {text[:30]} | {e}")
                continue

        logger.info(f"✅ 批量导入完成 | 共创建/更新 {len(all_created_entity_ids)} 个实体")
        return all_created_entity_ids

    # ===================== 🔴 跨模态实体绑定 =====================
    def _bind_cross_modal_entities(
        self, 
        entities: List[Entity], 
        global_clip_vec: torch.Tensor, 
        text: str, 
        target_expert: str
    ) -> int:
        """
        🔥 实体中心式跨模态实体绑定（多模态通用版）
        自动建立文本实体与对应模态实体的双向关联，并触发跨模态脑桥学习
        :param entities: 待绑定的文本实体列表
        :param global_clip_vec: 全局文本CLIP向量
        :param text: 原始文本内容
        :param target_expert: 文本所属专家
        :return: 成功创建的跨模态关系数量
        """
        if not entities:
            logger.debug("ℹ️ 无待绑定的文本实体，跳过跨模态绑定")
            return 0
        
        if not self.use_cross_modal_learning:
            logger.debug("ℹ️ 跨模态学习已禁用，跳过绑定")
            return 0
        
        try:
            logger.info(f"🔗 开始跨模态实体绑定 | 文本实体数: {len(entities)} | 主专家: {target_expert}")
            
            # ===================== 🔴 步骤1：检索所有相关模态实体 =====================
            # 支持的模态专家列表（可扩展：添加"听觉"/"触觉"等）
            modal_experts = ["视觉"]
            all_modal_entities: Dict[str, List[Dict]] = {}
            
            for modal_expert in modal_experts:
                if modal_expert not in self.experts:
                    continue
                
                # 检索对应模态专家的实体（指定expert_name，提升检索准确性）
                modal_results = self.cortex.retrieve(
                    query_text=text,
                    top_k=3,
                    # expert_name=modal_expert
                )
                
                # 过滤对应类型的实体
                modal_type = modal_expert.lower()
                filtered_entities = [
                    res for res in modal_results 
                    if res.get("type") == modal_type or res.get("entity_type") == modal_type
                ]
                
                if filtered_entities:
                    all_modal_entities[modal_expert] = filtered_entities
                    logger.info(f"   📸 检索到 [{modal_expert}] 实体: {len(filtered_entities)} 个")
            
            if not all_modal_entities:
                logger.info("ℹ️ 未找到任何相关模态实体，跳过绑定")
                return 0
            
            # ===================== 🔴 步骤2：批量建立跨模态关联 =====================
            total_relations_created = 0
            existing_relations_cache = set()  # 缓存已存在的关系，防止重复创建
            
            for text_entity in entities:
                if not text_entity or not text_entity.entity_id:
                    continue
                
                # 初始化实体的跨模态绑定记录
                if "cross_modal_bindings" not in text_entity.metadata:
                    text_entity.metadata["cross_modal_bindings"] = []
                
                for modal_expert, modal_results in all_modal_entities.items():
                    modal_type = modal_expert.lower()
                    relation_predicate = f"关联{modal_expert}"
                    
                    for modal_res in modal_results:
                        modal_entity_id = modal_res["entity_id"]
                        
                        # 检查是否已存在相同关系（去重）
                        relation_key = f"{text_entity.entity_id}_{relation_predicate}_{modal_entity_id}"
                        if relation_key in existing_relations_cache:
                            continue
                        if self.cortex.index.relation_exists(
                            subject_id=text_entity.entity_id,
                            predicate=relation_predicate,
                            object_id=modal_entity_id
                        ):
                            existing_relations_cache.add(relation_key)
                            continue
                        
                        # 获取模态实体对象
                        modal_entity = self.cortex.index.get_entity(modal_entity_id)
                        if not modal_entity:
                            continue
                        
                        # ===================== 🔴 步骤3：创建双向关联关系 =====================
                        # 1. 创建文本→模态的正向关系
                        forward_relation = MemoryFactory.create_relation(
                            subject=text_entity,
                            predicate=relation_predicate,
                            object=modal_entity,
                            confidence=modal_res["activation_score"],
                            initial_weight=0.35,
                            evidence=text_entity.latest_evidence
                        )
                        self.cortex.index.add_relation(forward_relation)
                        
                        # 2. 创建模态→文本的反向关系（双向关联，提升检索效率）
                        backward_relation = MemoryFactory.create_relation(
                            subject=modal_entity,
                            predicate=f"关联文本",
                            object=text_entity,
                            confidence=modal_res["activation_score"],
                            initial_weight=0.35,
                            evidence=text_entity.latest_evidence
                        )
                        self.cortex.index.add_relation(backward_relation)
                        
                        # 缓存关系，防止重复
                        existing_relations_cache.add(relation_key)
                        existing_relations_cache.add(f"{modal_entity_id}_关联文本_{text_entity.entity_id}")
                        
                        # ===================== 🔴 步骤4：全局突触同步 =====================
                        self.create_entity_synapse(
                            from_entity_id=text_entity.entity_id,
                            to_entity_id=modal_entity_id,
                            weight=0.35
                        )
                        self.create_entity_synapse(
                            from_entity_id=modal_entity_id,
                            to_entity_id=text_entity.entity_id,
                            weight=0.35
                        )
                        
                        # ===================== 🔴 步骤5：跨模态脑桥学习 =====================
                        if (self.perception_loop 
                            and text_entity.sdr is not None 
                            and modal_entity.sdr is not None):
                            try:
                                bridge_loss = self.perception_loop.cross_modal_learning_step(
                                    text_features=text_entity.sdr,
                                    vision_features=modal_entity.sdr,  # 兼容原有接口，未来可改为modal_features
                                    target_expert=modal_expert
                                )
                                logger.info(
                                    f"   🧠 跨模态学习完成 | 损失: {bridge_loss:.4f} | "
                                    f"{text_entity.name} ↔ {modal_entity.name} [{modal_expert}]"
                                )
                            except Exception as e:
                                logger.debug(f"   ⚠️ 跨模态学习跳过: {text_entity.name} ↔ {modal_entity.name} | {e}")
                        
                        # ===================== 🔴 步骤6：记录绑定信息 =====================
                        binding_record = {
                            "modal_expert": modal_expert,
                            "modal_entity_id": modal_entity_id,
                            "modal_entity_name": modal_entity.name,
                            "confidence": modal_res["activation_score"],
                            "bind_time": time.time()
                        }
                        text_entity.metadata["cross_modal_bindings"].append(binding_record)
                        total_relations_created += 1
            
            # 批量更新实体（减少索引操作次数）
            if total_relations_created > 0:
                for text_entity in entities:
                    self.cortex.index.update_entity(text_entity)
            
            logger.info(
                f"✅ 跨模态绑定完成 | 成功创建 {total_relations_created} 个双向关系 | "
                f"涉及 {len(entities)} 个文本实体 | {len(all_modal_entities)} 种模态"
            )
            return total_relations_created
        
        except Exception as e:
            logger.error(f"❌ 跨模态实体绑定失败: {e}", exc_info=True)
            return 0

    # ===================== 🔴 全局实体突触管理 =====================
    def create_entity_synapse(self, from_entity_id: str, to_entity_id: str, weight: float = 0.3) -> None:
        """创建双向实体突触连接（同步更新专家内部STDP突触）"""
        if from_entity_id == to_entity_id:
            logger.debug(f"⚠️ 跳过自环突触 | 实体ID:{from_entity_id}")
            return
        
        key = (from_entity_id, to_entity_id)
        self.global_entity_synapses[key] = weight
        reverse_key = (to_entity_id, from_entity_id)
        self.global_entity_synapses[reverse_key] = weight * 0.8
        self._save_global_synapses()
        
        # 同步更新专家内部突触
        if self.sync_global_synapses and self.cortex:
            try:
                from_entity = self.cortex.index.get_entity(from_entity_id)
                to_entity = self.cortex.index.get_entity(to_entity_id)
                
                if from_entity and to_entity and from_entity.expert == to_entity.expert:
                    expert = self.experts.get(from_entity.expert)
                    if expert and from_entity.sdr is not None and to_entity.sdr is not None:
                        expert.hebbian_update(from_entity.sdr, to_entity.sdr, is_fact=True)
                        logger.debug(f"🧠 专家内部突触同步更新 | {from_entity.expert} | {from_entity.name}→{to_entity.name}")
            except Exception as e:
                logger.debug(f"专家突触同步跳过: {e}")
        
        logger.info(f"🔗 实体突触建立: {from_entity_id} ↔ {to_entity_id} | 权重:{weight:.2f}")

    def _load_global_synapses(self) -> None:
        """加载全局实体突触连接"""
        try:
            import os
            import json
            if os.path.exists(self.synapse_save_path):
                with open(self.synapse_save_path, "r", encoding="utf-8") as f:
                    synapses_str = json.load(f)
                    self.global_entity_synapses = {
                        (k.split("|")[0], k.split("|")[1]): float(v)
                        for k, v in synapses_str.items()
                    }
                logger.info(f"🔗 已加载 {len(self.global_entity_synapses)} 条全局实体突触连接")
        except Exception as e:
            logger.warning(f"⚠️ 加载全局实体突触失败: {e}")
            self.global_entity_synapses = {}

    def _save_global_synapses(self) -> None:
        """保存全局实体突触连接"""
        try:
            import json
            synapses_str = {
                f"{k[0]}|{k[1]}": float(v)
                for k, v in self.global_entity_synapses.items()
            }
            with open(self.synapse_save_path, "w", encoding="utf-8") as f:
                json.dump(synapses_str, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ 保存全局实体突触失败: {e}")

    # ===================== 内部工具方法 =====================
    def _encode_text(self, text: str) -> torch.Tensor:
        """编码文本为向量（完全保留原有逻辑，复用外部嵌入模型）"""
        try:
            embedding = self.embedding_model.embed_query(text)
            clip_vec = torch.as_tensor(embedding, dtype=torch.float32)
            return clip_vec
        except Exception as e:
            logger.error(f"❌ 文本编码失败: {text[:30]} | {e}")
            return torch.zeros(config.dim, dtype=torch.float32)

    # ===================== 保留原有规则提取作为终极兜底 =====================
    def _rule_based_entity_extraction(self, text: str) -> Dict[str, Any]:
        """简单规则提取（备用）"""
        entities = []
        relations = []
        
        if "是" in text:
            parts = text.split("是", 1)
            subj = parts[0].strip()
            obj = parts[1].strip()
            if subj and obj:
                entities.append({"name": subj, "type": "concept", "attributes": {}})
                entities.append({"name": obj, "type": "concept", "attributes": {}})
                relations.append({"subject": subj, "predicate": "是", "object": obj, "confidence": 0.9})
        
        return {"entities": entities, "relations": relations}

    def _dopamine_reinforcement_learning(self, created_entity_ids: List[str], extraction_result: Dict,
                                        prior_confidences: Dict[str, float], saliency: float,
                                        external_reward: float, sdr: torch.Tensor) -> None:
        """多巴胺强化学习 + 预测性STDP（适配实体体系）"""
        try:
            # 计算平均信息增益
            total_information_gain = 0.0
            for entity_data in extraction_result["entities"]:
                entity_name = entity_data["name"]
                prior_conf = prior_confidences.get(entity_name, 0.0)
                total_information_gain += max(0.0, 1.0 - prior_conf)
            avg_information_gain = total_information_gain / max(len(extraction_result["entities"]), 1)
            
            # 计算总奖励
            curiosity_reward = self.dopamine.get_curiosity_reward(avg_information_gain)
            prediction_reward = self.dopamine.get_prediction_reward(saliency)
            total_reward = float(external_reward) + curiosity_reward + prediction_reward
            total_reward = float(np.clip(total_reward, -1.0, 1.0))
            
            # 计算奖励预测误差
            rpe = self.dopamine.compute_reward_prediction_error(total_reward)
            
            # 预测性STDP更新（对每个实体执行）
            for entity_id in created_entity_ids:
                entity = self.cortex.index.get_entity(entity_id)
                if not entity or entity.expert not in self.experts:
                    continue
                
                expert = self.experts[entity.expert]
                if self.use_predictive_stdp and hasattr(expert, 'predictive_std_update'):
                    try:
                        predicted_sdr = expert.predict_next_sdr(entity.sdr)
                        expert.predictive_std_update(
                            pre_sdr=entity.sdr,
                            post_sdr=predicted_sdr,
                            prediction_error=torch.tensor(rpe, device=entity.sdr.device)
                        )
                        logger.debug(f"🧠 预测性STDP更新完成 | 实体: {entity.name} | RPE={rpe:.2f}")
                    except Exception as e:
                        logger.debug(f"预测性STDP跳过，回退到赫布学习: {e}")
                        expert.hebbian_update(entity.sdr, entity.sdr, is_fact=True)
                else:
                    expert.hebbian_update(entity.sdr, entity.sdr, is_fact=True)
            
            # 多巴胺调节全局突触
            related_synapses = []
            for (from_id, to_id), weight in list(self.global_entity_synapses.items()):
                if from_id in created_entity_ids or to_id in created_entity_ids:
                    related_synapses.append({"from": from_id, "to": to_id, "weight": weight})
            
            if related_synapses:
                for syn in related_synapses:
                    original_weight = syn["weight"]
                    weight_update = 0.01 * rpe * saliency
                    new_weight = original_weight + weight_update
                    new_weight = float(np.clip(new_weight, -1.0, 1.0))
                    key = (syn["from"], syn["to"])
                    self.global_entity_synapses[key] = new_weight
                
                self._save_global_synapses()
            
            logger.info(f"🧪 多巴胺系统 | RPE={rpe:.2f} | 总奖励={total_reward:.2f} | "
                       f"(好奇={curiosity_reward:.2f}, 预测={prediction_reward:.2f}, 外部={external_reward:.2f})")
            
        except Exception as e:
            logger.debug(f"多巴胺学习跳过: {e}")
            # 异常时回退到基础赫布学习
            for entity_id in created_entity_ids:
                entity = self.cortex.index.get_entity(entity_id)
                if entity and entity.expert in self.experts:
                    expert = self.experts[entity.expert]
                    expert.hebbian_update(entity.sdr, entity.sdr, is_fact=True)

    def _metacognition_post_assessment(self, created_entity_ids: List[str], extraction_result: Dict,
                                       prior_confidences: Dict[str, float], current_time: float) -> None:
        """元认知后评估与好奇心触发（适配实体体系）"""
        try:
            for entity_data in extraction_result["entities"]:
                entity_name = entity_data["name"]
                prior_conf = prior_confidences.get(entity_name, 0.0)
                post_conf = self.metacognition.assess_knowledge_confidence(entity_name, current_time)
                
                logger.debug(f"🧠 元认知：实体 '{entity_name}' 学习后置信度 = {post_conf:.2f} | "
                            f"变化 = {post_conf - prior_conf:.2f}")
                
                # 好奇心触发
                if self.curiosity and self.curiosity.should_ask_question(entity_name):
                    questions = self.curiosity.generate_questions(entity_name)
                    if questions:
                        logger.info(f"❓ 好奇心驱动：生成问题 - {questions[0]}")
                        self.event_bus.emit(Event("CURIOSITY_TRIGGERED", {
                            "entity_name": entity_name,
                            "questions": questions
                        }))
            
        except Exception as e:
            logger.debug(f"元认知后评估跳过: {e}")


    def save_synapses(self, storage_dir) -> None:
        """
        ✅ 兼容旧架构统一保存接口
        新架构下所有数据（实体、关系、专家权重、突触）都由PersistentCortex统一保存
        """
        logger.info("🧠 调用新架构统一保存：实体+关系+专家突触权重")
        self.cortex.save_all()