import os
import json
import torch
import numpy as np
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional, Any
import logging
import datetime
# from KnowledgeGraphMemoryV3 import KnowledgeGraphMemory
# ✅ 替换为新的实体中心数据契约
from Data_models import Entity, EntityRelation, Evidence, MemoryFactory, ConversationTurn
from torch.nn import functional as F
import time
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PersistentCortex")

try:
    import faiss
    HAS_FAISS = True
    logger.info("✅ FAISS 向量索引已加载")
except ImportError:
    HAS_FAISS = False
    logger.warning("⚠️  未安装FAISS，使用全量遍历检索（性能较低）")

from BrainConfig import config


class EntityIndex:
    """
    🔴 替换原MemoryIndex：实体中心多维度索引
    所有索引都围绕实体、关系、证据构建
    """
    def __init__(self, dim):
        self.dim = dim
        
        # ===================== 核心主存储 =====================
        self.entities: Dict[str, Entity] = {}          # 实体主存储：entity_id → Entity
        self.relations: Dict[str, EntityRelation] = {} # 关系主存储：relation_id → EntityRelation
        self.evidences: Dict[str, Evidence] = {}       # 证据主存储：evidence_id → Evidence

        # ===================== 多维度结构化索引 =====================
        self.name_index: Dict[str, str] = {}           # 实体名称索引：name → entity_id
        self.type_index: Dict[str, List[str]] = defaultdict(list)  # 实体类型索引：type → [entity_id]
        self.tag_index: Dict[str, List[str]] = defaultdict(list)   # 标签索引：tag → [entity_id]
        self.alias_index: Dict[str, str] = {}          # 别名索引：alias → entity_id
        self.subject_relation_index: Dict[str, List[str]] = defaultdict(list)  # 主体关系索引：subject_id → [relation_id]
        self.object_relation_index: Dict[str, List[str]] = defaultdict(list)   # 客体关系索引：object_id → [relation_id]

        self._faiss_id_map: Dict[int, str] = {}  # FAISS索引ID → entity_id
        self._reverse_faiss_map: Dict[str, int] = {}  # entity_id → FAISS索引ID
        self.faiss_index = None  # FAISS索引对象

        # ===================== FAISS向量索引 =====================
        self.faiss_index = None
        if HAS_FAISS:
            self._init_faiss()

    def _init_faiss(self):
        base_index = faiss.IndexFlatIP(self.dim)
        self.faiss_index = faiss.IndexIDMap(base_index)

    def _rebuild_faiss_index(self):
        logger.warning("🔧 检测到FAISS索引损坏，正在自动重建...")
        self._init_faiss()
        
        # 用实体的semantic_vec构建索引
        for entity_id, entity in self.entities.items():
            # 将字符串entity_id转换为整数ID（FAISS只支持整数ID）
            int_id = hash(entity_id) & 0x7fffffff  # 转为32位正整数
            vec = entity.semantic_vec.detach().cpu().numpy().reshape(1, -1)
            self.faiss_index.add_with_ids(vec, np.array([int_id], dtype=np.int64))
            # 保存整数ID到entity_id的映射
            self._faiss_id_map[int_id] = entity_id
        
        logger.info(f"✅ FAISS索引重建完成，共 {len(self.entities)} 个实体向量")

    # ===================== 实体操作 =====================
    def add_entity(self, entity: Entity) -> str:
        """添加实体到索引，返回entity_id"""
        if entity.entity_id in self.entities:
            logger.info(f"🔄 实体已存在，更新：{entity.name}")
            return entity.entity_id

        self.entities[entity.entity_id] = entity
        
        # 更新所有结构化索引
        self.name_index[entity.name] = entity.entity_id
        self.type_index[entity.entity_type].append(entity.entity_id)
        for tag in entity.tags:
            self.tag_index[tag].append(entity.entity_id)
        for alias in entity.aliases:
            self.alias_index[alias] = entity.entity_id

        # 更新FAISS索引
        if HAS_FAISS:
            int_id = hash(entity.entity_id) & 0x7fffffff
            self._faiss_id_map[int_id] = entity.entity_id
            vec_np = entity.semantic_vec.detach().cpu().numpy().reshape(1, -1)
            self.faiss_index.add_with_ids(vec_np, np.array([int_id], dtype=np.int64))

        logger.debug(f"✅ 实体添加成功：{entity.name} ({entity.entity_id})")
        return entity.entity_id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """通过名称或别名查找实体"""
        entity_id = self.name_index.get(name) or self.alias_index.get(name)
        return self.entities.get(entity_id) if entity_id else None

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        return [self.entities[eid] for eid in self.type_index.get(entity_type, [])]

    def get_entities_by_tag(self, tag: str) -> List[Entity]:
        return [self.entities[eid] for eid in self.tag_index.get(tag, [])]

    def delete_entity(self, entity_id: str):
        """删除实体及其所有关联关系和证据"""
        if entity_id not in self.entities:
            return

        entity = self.entities[entity_id]
        
        # 删除关联关系
        related_relations = (
            self.subject_relation_index.pop(entity_id, []) 
            + self.object_relation_index.pop(entity_id, [])
        )
        for rel_id in related_relations:
            self.relations.pop(rel_id, None)

        # 删除关联证据
        for ev in entity.evidences:
            self.evidences.pop(ev.evidence_id, None)

        # 从索引中移除
        self.name_index.pop(entity.name, None)
        self.type_index[entity.entity_type].remove(entity_id)
        for tag in entity.tags:
            self.tag_index[tag].remove(entity_id)
        for alias in entity.aliases:
            self.alias_index.pop(alias, None)

        # 从FAISS移除
        if HAS_FAISS:
            int_id = hash(entity_id) & 0x7fffffff
            self.faiss_index.remove_ids(np.array([int_id], dtype=np.int64))
            self._faiss_id_map.pop(int_id, None)

        # 从主存储移除
        del self.entities[entity_id]
        logger.info(f"🗑️  已删除实体：{entity.name} ({entity_id})")

    # ===================== 关系操作 =====================
    def add_relation(self, relation: EntityRelation) -> str:
        """添加实体间关系"""
        if relation.relation_id in self.relations:
            return relation.relation_id

        self.relations[relation.relation_id] = relation
        self.subject_relation_index[relation.subject_id].append(relation.relation_id)
        self.object_relation_index[relation.object_id].append(relation.relation_id)
        
        logger.debug(f"✅ 关系添加成功：{relation.subject_id} --{relation.predicate}--> {relation.object_id}")
        return relation.relation_id

    def get_relation(self, relation_id: str) -> Optional[EntityRelation]:
        return self.relations.get(relation_id)

    def get_entity_relations(self, entity_id: str) -> List[EntityRelation]:
        """获取实体的所有关联关系"""
        rel_ids = (
            self.subject_relation_index.get(entity_id, []) 
            + self.object_relation_index.get(entity_id, [])
        )
        return [self.relations[rid] for rid in rel_ids if rid in self.relations]

    # ===================== 证据操作 =====================
    def add_evidence(self, evidence: Evidence) -> str:
        """添加全文本证据"""
        if evidence.evidence_id in self.evidences:
            return evidence.evidence_id

        self.evidences[evidence.evidence_id] = evidence
        logger.debug(f"✅ 证据添加成功：{evidence.content[:30]}...")
        return evidence.evidence_id

    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        return self.evidences.get(evidence_id)

    # ===================== 向量检索 =====================
    def vector_search(self, query_vec: torch.Tensor, top_k: int = 100) -> List[Tuple[str, float, Entity]]:
        """语义检索最相似的实体"""
        query_np = query_vec.detach().cpu().numpy().reshape(1, -1)
        
        if HAS_FAISS and self.faiss_index.ntotal > 0:
            scores, int_ids = self.faiss_index.search(query_np, top_k)
            results = []
            for i in range(len(int_ids[0])):
                int_id = int(int_ids[0][i])
                if int_id == -1 or int_id not in self._faiss_id_map:
                    continue
                entity_id = self._faiss_id_map[int_id]
                sim = scores[0][i]
                entity = self.entities.get(entity_id)
                if entity and not entity.is_obsolete:
                    results.append((entity_id, sim, entity))
            return results
        else:
            results = []
            for entity_id, entity in self.entities.items():
                if entity.is_obsolete:
                    continue
                sim = torch.cosine_similarity(query_vec, entity.semantic_vec, dim=-1).item()
                results.append((entity_id, sim, entity))
            results.sort(key=lambda x: -x[1])
            return results[:top_k]

    # ===================== 持久化 =====================
    def save(self, file_path: str):
        save_data = {
            'version': '3.0',  # 实体中心版本号
            'entities': {},
            'relations': {},
            'evidences': {},
            'name_index': self.name_index,
            'type_index': dict(self.type_index),
            'tag_index': dict(self.tag_index),
            'alias_index': self.alias_index,
            'subject_relation_index': dict(self.subject_relation_index),
            'object_relation_index': dict(self.object_relation_index),
            '_faiss_id_map': getattr(self, '_faiss_id_map', {})
        }

        # 序列化所有核心对象
        for entity_id, entity in self.entities.items():
            save_data['entities'][entity_id] = entity.to_dict()
        for rel_id, rel in self.relations.items():
            save_data['relations'][rel_id] = rel.to_dict()
        for ev_id, ev in self.evidences.items():
            save_data['evidences'][ev_id] = ev.to_dict()

        # 原子写入
        temp_json_file = file_path + ".tmp"
        with open(temp_json_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        # 保存FAISS索引
        if HAS_FAISS and self.faiss_index.ntotal > 0:
            temp_faiss_file = file_path + ".faiss.tmp"
            faiss.write_index(self.faiss_index, temp_faiss_file)
            if os.path.exists(file_path + ".faiss"):
                os.remove(file_path + ".faiss")
            os.rename(temp_faiss_file, file_path + ".faiss")

        # 替换原文件
        if os.path.exists(file_path):
            os.remove(file_path)
        os.rename(temp_json_file, file_path)
        logger.info(f"💾 实体索引保存完成 | 实体:{len(self.entities)} | 关系:{len(self.relations)} | 证据:{len(self.evidences)}")

    def load(self, file_path: str):
        if not os.path.exists(file_path):
            # 初始化FAISS ID映射
            self._faiss_id_map = {}
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            load_data = json.load(f)

        version = load_data.get('version', '2.0')
        if version < '3.0':
            logger.error("❌ 不支持的旧版本记忆格式，请先运行迁移工具")
            self._faiss_id_map = {}
            return

        # 加载索引映射
        self.name_index = load_data['name_index']
        self.type_index = defaultdict(list, load_data['type_index'])
        self.tag_index = defaultdict(list, load_data['tag_index'])
        self.alias_index = load_data['alias_index']
        self.subject_relation_index = defaultdict(list, load_data['subject_relation_index'])
        self.object_relation_index = defaultdict(list, load_data['object_relation_index'])
        self._faiss_id_map = load_data.get('_faiss_id_map', {})

        # 反序列化核心对象
        self.entities = {}
        for entity_id, entity_dict in load_data['entities'].items():
            self.entities[entity_id] = Entity.from_dict(entity_dict)
        
        self.relations = {}
        for rel_id, rel_dict in load_data['relations'].items():
            self.relations[rel_id] = EntityRelation.from_dict(rel_dict)
        
        self.evidences = {}
        for ev_id, ev_dict in load_data['evidences'].items():
            self.evidences[ev_id] = Evidence.from_dict(ev_dict)

        # 加载FAISS索引
        if HAS_FAISS and os.path.exists(file_path + ".faiss"):
            try:
                self.faiss_index = faiss.read_index(file_path + ".faiss")
                if self.faiss_index.ntotal != len(self.entities):
                    logger.warning("⚠️  FAISS索引与实体数量不匹配，自动重建")
                    self._rebuild_faiss_index()
            except Exception as e:
                logger.error(f"❌ FAISS索引加载失败，自动重建: {e}")
                self._rebuild_faiss_index()
        else:
            if HAS_FAISS and len(self.entities) > 0:
                self._rebuild_faiss_index()

        logger.info(f"✅ 实体索引加载完成 | 实体:{len(self.entities)} | 关系:{len(self.relations)} | 证据:{len(self.evidences)}")


class PersistentCortex:
    def __init__(self, storage_dir: str, experts, embedding_model, llm, kg_enabled: bool = True):
        self.storage_dir = storage_dir
        self.experts = experts
        self.llm = llm
        self.kg_enabled = kg_enabled
        os.makedirs(storage_dir, exist_ok=True)
        self.index_file = os.path.join(storage_dir, "cortex_entity_index.json")
        self.embedding_model = embedding_model

        # self.kg = KnowledgeGraphMemory(storage_dir, enabled=kg_enabled)
        self.important_entities_file = os.path.join(storage_dir, "important_entities.json")
        self.important_entities = self._load_important_entities()
        self.conversation_memory_file = os.path.join(storage_dir, "conversation_memory.json")
        self._init_conversation_memory()

        # 替换原MemoryIndex为EntityIndex
        self.index = EntityIndex(config.dim)
        self.permanent_entities: set = set()
        self.load_all()

    def _load_important_entities(self) -> set:
        if os.path.exists(self.important_entities_file):
            try:
                with open(self.important_entities_file, "r", encoding="utf-8") as f:
                    return set(json.load(f))
            except Exception as e:
                logger.warning(f"⚠️  重要实体列表加载失败: {e}")
        return set()

    def _save_important_entities(self):
        try:
            with open(self.important_entities_file, "w", encoding="utf-8") as f:
                json.dump(list(self.important_entities), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ 重要实体列表保存失败: {e}")

    def add_important_entity(self, entity_name: str):
        self.important_entities.add(entity_name)
        self._save_important_entities()
        logger.info(f"✅ 已添加重要实体: {entity_name}")

    def remove_important_entity(self, entity_name: str):
        if entity_name in self.important_entities:
            self.important_entities.remove(entity_name)
            self._save_important_entities()
            logger.info(f"✅ 已删除重要实体: {entity_name}")
        else:
            logger.warning(f"⚠️  未找到重要实体: {entity_name}")

    # ===================== 核心：记忆存储入口 =====================
    def store_from_text(self, text: str, source: str = "对话", metadata: Dict = None) -> List[str]:
        """
        🔴 替换原store_detailed_memory：从文本创建实体中心式记忆
        这是新系统的唯一记忆存储入口
        """
        metadata = metadata or {}
        logger.info(f"🧠 开始处理文本记忆：{text[:50]}...")

        # 步骤1：生成神经向量
        sdr = self._generate_sdr(text)
        clip_vec = self._generate_clip_vec(text)

        # 步骤2：创建全文本证据
        evidence = MemoryFactory.create_evidence(
            content=text,
            source=source,
            sdr=sdr,
            clip_vec=clip_vec,
            confidence=metadata.get('confidence', 0.95),
            emotion_valence=metadata.get('emotion_valence', 0.0),
            **metadata
        )
        self.index.add_evidence(evidence)

        # 步骤3：提取实体和关系（核心！）
        extraction_result = self._extract_entities_and_relations(text)
        if not extraction_result["entities"]:
            logger.info("📝 未提取到实体，将文本存储到'通用知识'实体")
            default_entity = self._get_or_create_default_entity(sdr, clip_vec)
            default_entity.add_evidence(evidence)
            return [default_entity.entity_id]

        # 步骤4：处理实体（创建或更新）
        created_entity_ids = []
        for entity_data in extraction_result["entities"]:
            entity = self._process_entity(entity_data, sdr, clip_vec, evidence)
            created_entity_ids.append(entity.entity_id)

        # 步骤5：处理关系
        for rel_data in extraction_result["relations"]:
            self._process_relation(rel_data, evidence)

        # 步骤6：同步到专家模块
        for entity_id in created_entity_ids:
            entity = self.index.get_entity(entity_id)
            if entity.expert in self.experts:
                expert = self.experts[entity.expert]
                expert.add_entity(entity)  # 专家模块现在存储实体
                expert.hebbian_update(entity.sdr, entity.sdr, is_fact=metadata.get('is_fact', False))

        # 步骤7：同步到知识图谱（兼容原有KG）
        # if self.kg_enabled:
        #     for entity_id in created_entity_ids:
        #         entity = self.index.get_entity(entity_id)
        #         # self.kg.add_entity(entity.name, entity.entity_type, entity_id)
        #         for rel in self.index.get_entity_relations(entity_id):
        #             subj = self.index.get_entity(rel.subject_id)
        #             obj = self.index.get_entity(rel.object_id)
        #             if subj and obj:
        #                 self.kg.add_relation(subj.name, rel.predicate, obj.name)

        logger.info(f"✅ 记忆处理完成 | 创建/更新实体:{len(created_entity_ids)} | 关系:{len(extraction_result['relations'])}")
        return created_entity_ids

   # PersistentCortex.py 实体提取方法修复
    def _extract_entities_and_relations(self, text: str) -> Dict[str, Any]:
        """
        增强版实体关系提取：带重试、超时和规则降级
        """
        if not self.llm:
            return self._rule_based_entity_extraction(text)

        # 重试配置
        max_retries = 3
        retry_delay = 1.0
        timeout = 10  # 秒

        for attempt in range(max_retries):
            try:
                prompt = f"""
                请从以下文本中提取所有实体和实体之间的关系。
                严格按照JSON格式输出，不要解释、不要多余文字。
                
                实体类型只能是：person、place、event、concept、object、skill、system、emotion、identity、visual
                
                文本：{text}
                
                输出格式：
                {{
                    "entities": [
                        {{"name": "实体名称", "type": "实体类型", "attributes": {{"属性名": "属性值"}}}}
                    ],
                    "relations": [
                        {{"subject": "主体实体名", "predicate": "关系谓词", "object": "客体实体名", "confidence": 0.95}}
                    ],
                    "emotion": {{"valence": 0.0, "arousal": 0.5}}
                }}
                """
                
                # 带超时的LLM调用
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.llm.invoke, prompt)
                    response = future.result(timeout=timeout)
                
                # 提取JSON
                import re
                json_text = re.findall(r"\{.*\}", response.content, re.S)[0]
                result = json.loads(json_text)
                
                # 验证结果
                if "entities" not in result:
                    result["entities"] = []
                if "relations" not in result:
                    result["relations"] = []
                if "emotion" not in result:
                    result["emotion"] = {"valence": 0.0, "arousal": 0.5}
                
                logger.debug(f"✅ LLM实体提取成功 | 实体数: {len(result['entities'])} | 关系数: {len(result['relations'])}")
                return result
                
            except Exception as e:
                logger.warning(f"⚠️ 实体提取尝试 {attempt+1}/{max_retries} 失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # 指数退避
                else:
                    logger.error(f"❌ LLM实体提取全部失败，回退到规则提取")
                    return self._rule_based_entity_extraction(text)

    def _rule_based_entity_extraction(self, text: str) -> Dict[str, Any]:
        """规则基实体提取（降级方案）"""
        entities = []
        relations = []
        
        # 简单的"是"字句提取
        if "是" in text:
            parts = text.split("是", 1)
            subj = parts[0].strip()
            obj = parts[1].strip()
            if subj and obj:
                entities.append({"name": subj, "type": "concept", "attributes": {}})
                entities.append({"name": obj, "type": "concept", "attributes": {}})
                relations.append({"subject": subj, "predicate": "是", "object": obj, "confidence": 0.8})
        
        # 提取引号中的内容
        import re
        quotes = re.findall(r'"(.*?)"|“(.*?)”', text)
        for quote in quotes:
            content = quote[0] or quote[1]
            if content and len(content) >= 2:
                entities.append({"name": content, "type": "concept", "attributes": {}})
        
        return {
            "entities": entities,
            "relations": relations,
            "emotion": {"valence": 0.0, "arousal": 0.5}
        }

    def _rule_based_extraction(self, text: str) -> Dict:
        """简单规则提取（备用）"""
        entities = []
        relations = []
        
        # 提取"X是Y"关系
        if "是" in text:
            parts = text.split("是", 1)
            subj = parts[0].strip()
            obj = parts[1].strip()
            if subj and obj:
                entities.append({"name": subj, "type": "concept", "attributes": {}})
                entities.append({"name": obj, "type": "concept", "attributes": {}})
                relations.append({"subject": subj, "predicate": "是", "object": obj, "confidence": 0.9})
        
        return {"entities": entities, "relations": relations}

    # ===================== 实体关系处理 =====================
    def _process_entity(self, entity_data: Dict, sdr: torch.Tensor, clip_vec: torch.Tensor, evidence: Evidence) -> Entity:
        """处理单个实体：创建或更新"""
        name = entity_data["name"]
        entity_type = entity_data["type"]
        attributes = entity_data.get("attributes", {})

        # 检查实体是否已存在
        existing_entity = self.index.get_entity_by_name(name)
        if existing_entity:
            # 更新现有实体
            for key, value in attributes.items():
                existing_entity.update_attribute(key, value)
            existing_entity.add_evidence(evidence)
            existing_entity.tags.update(entity_data.get("tags", []))
            logger.info(f"🔄 更新实体：{name}")
            return existing_entity

        # 创建新实体
        entity = MemoryFactory.create_entity(
            name=name,
            entity_type=entity_type,
            sdr=sdr,
            clip_vec= clip_vec,
            semantic_vec=clip_vec,
            importance=entity_data.get("importance", 0.7),
            attributes=attributes,
            tags=set(entity_data.get("tags", [])),
            expert=self._get_expert_for_entity_type(entity_type)
        )
        entity.add_evidence(evidence)
        self.index.add_entity(entity)

        # 自动标记重要实体
        if name in self.important_entities:
            entity.is_permanent = True
            self.permanent_entities.add(entity.entity_id)

        return entity

    def _process_relation(self, rel_data: Dict, evidence: Evidence):
        """处理单个关系：创建或更新"""
        subj_name = rel_data["subject"]
        predicate = rel_data["predicate"]
        obj_name = rel_data["object"]
        confidence = rel_data.get("confidence", 0.9)

        subj = self.index.get_entity_by_name(subj_name)
        obj = self.index.get_entity_by_name(obj_name)

        if not subj or not obj:
            logger.warning(f"⚠️  关系主体或客体不存在，跳过：{subj_name} --{predicate}--> {obj_name}")
            return

        # 检查是否已有相同关系
        existing_rels = self.index.get_entity_relations(subj.entity_id)
        for rel in existing_rels:
            if rel.predicate == predicate and rel.object_id == obj.entity_id:
                # 更新现有关系的置信度和权重
                rel.confidence = max(rel.confidence, confidence)
                rel.update_synapse(0.05)  # 加强连接
                logger.info(f"🔄 更新关系：{subj_name} --{predicate}--> {obj_name}")
                return

        # 创建新关系
        relation = MemoryFactory.create_relation(
            subject=subj,
            predicate=predicate,
            object=obj,
            confidence=confidence,
            initial_weight=confidence * 0.2,
            evidence=evidence
        )
        self.index.add_relation(relation)

    def _get_or_create_default_entity(self, sdr: torch.Tensor, clip_vec: torch.Tensor) -> Entity:
        """获取或创建默认的'通用知识'实体"""
        default_entity = self.index.get_entity_by_name("通用知识")
        if not default_entity:
            default_entity = MemoryFactory.create_entity(
                name="通用知识",
                entity_type="concept",
                sdr=sdr,
                semantic_vec=clip_vec,
                importance=0.5,
                expert="抽象"
            )
            self.index.add_entity(default_entity)
        return default_entity

    def _get_expert_for_entity_type(self, entity_type: str) -> str:
        """根据实体类型分配对应的专家"""
        expert_map = {
            "person": "身份",
            "place": "空间",
            "event": "空间",
            "concept": "概念",
            "object": "概念",
            "skill": "抽象"
        }
        return expert_map.get(entity_type, "概念")

    # ===================== 记忆检索 =====================
    def retrieve(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """
        🔴 替换原search_memories：实体中心式检索
        流程：提取查询实体 → 神经激活传播 → 排序返回
        """
        logger.info(f"🔍 实体中心检索：{query_text}")

        # 步骤1：从查询中提取种子实体
        seed_entities = self._extract_query_entities(query_text)
        if not seed_entities:
            # 没有提取到实体，用语义检索找最相似的实体
            query_vec = self._generate_clip_vec(query_text)
            similar_entities = self.index.vector_search(query_vec, top_k=3)
            seed_entities = [entity for _, _, entity in similar_entities]

        if not seed_entities:
            logger.info("🔍 未找到相关实体")
            return []

        # 步骤2：神经激活传播（核心！）
        activated_entities = self._neural_activation_propagation(seed_entities, hops=2)

        # 步骤3：组装结果
        results = []
        for entity_id, activation_score in activated_entities[:top_k]:
            entity = self.index.get_entity(entity_id)
            if not entity:
                continue

            # 获取关联关系
            related_entities = []
            for rel in self.index.get_entity_relations(entity_id)[:5]:
                other_id = rel.object_id if rel.subject_id == entity_id else rel.subject_id
                other_entity = self.index.get_entity(other_id)
                if other_entity:
                    related_entities.append({
                        "predicate": rel.predicate,
                        "entity_name": other_entity.name,
                        "weight": rel.synapse_weight
                    })

            # 获取最新证据
            latest_evidence = entity.latest_evidence.content if entity.latest_evidence else ""

            results.append({
                "entity_id": entity_id,
                "name": entity.name,
                "type": entity.entity_type,
                "attributes": entity.attributes,
                "activation_score": activation_score,
                "importance": entity.importance,
                "related_entities": related_entities,
                "latest_evidence": latest_evidence,
                "all_evidences": [ev.content for ev in entity.evidences[:3]]
            })

        logger.info(f"🔍 检索完成，找到 {len(results)} 个相关实体")
        return results

    # def _extract_query_entities(self, query_text: str) -> List[Entity]:
    #     """从查询文本中提取种子实体"""
    #     entities = []
    #     # 先匹配重要实体
    #     for name in self.important_entities:
    #         if name in query_text:
    #             entity = self.index.get_entity_by_name(name)
    #             if entity:
    #                 entities.append(entity)
    #     # 再匹配所有实体名称
    #     for name in self.index.name_index.keys():
    #         if name in query_text and name not in [e.name for e in entities]:
    #             entity = self.index.get_entity_by_name(name)
    #             if entity:
    #                 entities.append(entity)
    #     return entities

    # ===================== 替换原方法 =====================
    def _extract_query_entities(self, query_text: str) -> List[Entity]:
        """
        ✅ 修复版：智能种子实体提取，过滤通用实体，优先匹配相关实体
        """
        entities = []
        query_lower = query_text.lower()
        
        # 通用实体黑名单（这些实体永远不能作为种子）
        BLACKLIST_ENTITIES = {"通用知识", "书籍", "内容", "事情", "东西", "时间", "地方", "人物"}
        
        # 1. 先匹配重要实体（优先级最高）
        for name in self.important_entities:
            if name in query_text and name not in BLACKLIST_ENTITIES:
                entity = self.index.get_entity_by_name(name)
                if entity:
                    entities.append(entity)
        
        # 2. 再匹配所有非通用实体
        for name in self.index.name_index.keys():
            if (name in query_text 
                and name not in [e.name for e in entities] 
                and name not in BLACKLIST_ENTITIES
                and len(name) >= 2):  # 过滤单字实体
                entity = self.index.get_entity_by_name(name)
                if entity:
                    entities.append(entity)
        
        # 3. 最多保留3个种子实体（避免种子太多导致激活扩散过广）
        return entities[:3]

    # ===================== 替换原方法 =====================
    def _neural_activation_propagation(self, seed_entities: List[Entity], hops: int = 2) -> List[Tuple[str, float]]:
        """
        ✅ 修复版：带神经噪声的激活传播，打破结果固定
        新增：高斯噪声、激活阈值、衰减式传播、多样性惩罚
        """
        activation = defaultdict(float)
        
        # 初始化种子实体激活值（加入微小随机噪声）
        for entity in seed_entities:
            # 基础激活 = 重要性 × (0.95~1.05) 随机噪声
            base_activation = 1.0 * entity.importance
            noise = np.random.normal(1.0, 0.05)  # 5%高斯噪声
            activation[entity.entity_id] = base_activation * noise

        # 多跳衰减式传播
        for hop in range(hops):
            new_activation = defaultdict(float)
            # 每跳衰减系数（越远的关联激活越弱）
            hop_decay = 0.7 ** (hop + 1)
            
            for entity_id, act in activation.items():
                # 激活阈值：低于0.1的不再传播，避免噪声扩散
                if act < 0.1:
                    continue
                    
                # 传播到所有关联实体
                for rel in self.index.get_entity_relations(entity_id):
                    other_id = rel.object_id if rel.subject_id == entity_id else rel.subject_id
                    # 激活值 = 当前激活 × 突触权重 × 跳数衰减 × 随机噪声
                    rel_noise = np.random.normal(1.0, 0.08)  # 8%关系噪声
                    new_act = act * rel.synapse_weight * hop_decay * rel_noise
                    new_activation[other_id] += new_act
            
            # 合并新激活值（取最大值，避免重复激活叠加）
            for entity_id, act in new_activation.items():
                activation[entity_id] = max(activation[entity_id], act)
            
            # 稀疏化：只保留激活值最高的30个实体
            if len(activation) > 30:
                sorted_items = sorted(activation.items(), key=lambda x: -x[1])
                activation = dict(sorted_items[:30])
        
        # 最终加入全局随机噪声（打破完全排序固定）
        for entity_id in activation:
            activation[entity_id] *= np.random.normal(1.0, 0.03)  # 3%全局噪声
        
        # 排序返回
        sorted_activation = sorted(activation.items(), key=lambda x: -x[1])
        return [(eid, score) for eid, score in sorted_activation if score > 0.01]

    # ===================== 记忆生命周期管理 =====================
    def increment_access(self, entity_id: str):
        """更新实体访问状态"""
        entity = self.index.get_entity(entity_id)
        if not entity:
            return
        entity.increment_access()
        
        # 同时更新关联关系的访问计数
        for rel in self.index.get_entity_relations(entity_id):
            rel.access_count += 1
            rel.last_accessed = time.time()

        # 自动标记为永久记忆
        if entity.importance >= config.permanent_importance_threshold and entity_id not in self.permanent_entities:
            self.mark_permanent(entity_id)

    def mark_permanent(self, entity_id: str):
        """标记实体为永久记忆"""
        entity = self.index.get_entity(entity_id)
        if not entity:
            return
        entity.is_permanent = True
        self.permanent_entities.add(entity_id)
        logger.info(f"🔒 实体已标记为永久：{entity.name} ({entity_id})")

    def decay_all_memories(self):
        """执行记忆自然衰减"""
        logger.info("⏳ 执行实体记忆自然衰减...")
        now = datetime.datetime.now()
        to_delete = []

        for entity_id, entity in self.index.entities.items():
            if entity.is_permanent or entity_id in self.permanent_entities:
                continue

            # 计算衰减
            days_since_access = (now - datetime.datetime.fromtimestamp(entity.last_accessed)).days
            entity.importance = max(0.0, entity.importance - days_since_access / 365 * 0.1)

            # 标记为过时
            if days_since_access >= config.entity_forget_days and entity.importance < config.entity_forget_importance_threshold:
                entity.mark_obsolete()
                to_delete.append(entity_id)

        # 删除过时实体
        for entity_id in to_delete:
            self.index.delete_entity(entity_id)
            for expert in self.experts.values():
                expert.delete_entity(entity_id)
            logger.info(f"🗑️  遗忘低价值实体：{self.index.get_entity(entity_id).name if self.index.get_entity(entity_id) else entity_id}")

        # 衰减关系权重
        for rel_id, rel in self.index.relations.items():
            days_since_access = (now - datetime.datetime.fromtimestamp(rel.last_accessed)).days
            rel.synapse_weight = max(0.0, rel.synapse_weight - days_since_access / 365 * 0.05)

        logger.info(f"✅ 记忆衰减完成，共遗忘 {len(to_delete)} 个低价值实体")

    # ===================== 睡眠巩固 =====================
    def sleep_consolidate_all(self, epochs=3):
        logger.info("\n🌙 大脑开始睡眠巩固（实体中心版）...")
        
        # 步骤1：专家模块睡眠巩固
        for name, expert in self.experts.items():
            expert.sleep_consolidate(epochs=epochs)
        
        # 步骤2：巩固重要对话
        important_turns = self.get_pending_conversation_consolidation()
        if important_turns:
            logger.info(f"📝 开始巩固 {len(important_turns)} 条重要对话...")
            for turn in important_turns:
                memory_text = f"[对话记录] 用户说：{turn.user_input}，我回答：{turn.ai_response}"
                self.store_from_text(
                    text=memory_text,
                    source="对话巩固",
                    metadata={"importance": 0.8, "is_fact": True}
                )
        
        # 步骤3：突触重塑
        self._synaptic_remodeling()
        
        # # 步骤4：知识图谱同步
        # if self.kg_enabled:
        #     self.kg.sleep_consolidate()

        logger.info("✅ 大脑睡眠巩固完成！")

    def _synaptic_remodeling(self):
        """突触重塑：修剪弱连接，生成新连接"""
        logger.info("🧠 执行突触重塑...")
        
        # 修剪弱关系
        weak_relations = []
        for rel_id, rel in self.index.relations.items():
            if abs(rel.synapse_weight) < 0.05 and rel.access_count < 3:
                weak_relations.append(rel_id)
        
        for rel_id in weak_relations:
            del self.index.relations[rel_id]
            # 从索引中移除
            self.index.subject_relation_index = {k: [v for v in vs if v != rel_id] for k, vs in self.index.subject_relation_index.items()}
            self.index.object_relation_index = {k: [v for v in vs if v != rel_id] for k, vs in self.index.object_relation_index.items()}
        
        logger.info(f"🧹 修剪了 {len(weak_relations)} 个弱连接")

    # ===================== 持久化 =====================
    def save_all(self):
        logger.info("💾 正在安全保存皮层记忆（实体中心版）...")
        
        # 保存实体索引
        temp_index_file = self.index_file + ".tmp"
        self.index.save(temp_index_file)
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        os.rename(temp_index_file, self.index_file)
        
        # 保存全局状态
        state_data = {
            'version': '3.0',
            'permanent_entities': list(self.permanent_entities)
        }
        state_file = os.path.join(self.storage_dir, "cortex_state.json")
        temp_state_file = state_file + ".tmp"
        with open(temp_state_file, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)
        if os.path.exists(state_file):
            os.remove(state_file)
        os.rename(temp_state_file, state_file)
        
        # 保存专家权重
        for name, expert in self.experts.items():
            expert_path = os.path.join(self.storage_dir, f"expert_{name}.pt")
            expert.save_weights(expert_path)
        
        # 保存其他数据
        # self.kg.save()
        self._save_important_entities()
        self._save_conversation_memory()
        
        logger.info("✅ 皮层记忆已安全保存！")

    def load_all(self):
        if not os.path.exists(self.index_file):
            logger.info("📦 无历史实体记忆，初始化新的皮层记忆系统")
            return
        
        self.index.load(self.index_file)
        
        # 加载全局状态
        state_file = os.path.join(self.storage_dir, "cortex_state.json")
        if os.path.exists(state_file):
            with open(state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            self.permanent_entities = set(state_data.get('permanent_entities', []))
        
        # 加载专家权重
        for name, expert in self.experts.items():
            expert_path = os.path.join(self.storage_dir, f"expert_{name}.pt")
            expert.load_weights(expert_path)
        
        # 加载其他数据
        # self.kg.load()
        self.important_entities = self._load_important_entities()
        self._load_conversation_memory()

        logger.info(f"✅ 历史记忆加载完成 | 总实体数:{len(self.index.entities)} | 永久实体数:{len(self.permanent_entities)}")

    # ===================== 工具方法 =====================
    def _generate_sdr(self, text: str) -> torch.Tensor:
        """生成文本的SDR向量（替换为你实际的SDR生成逻辑）"""
        # 这里只是示例，实际使用你的SDR编码器
        return torch.randn(config.sdr_dim)

    def _generate_clip_vec(self, text: str) -> torch.Tensor:
        """生成文本的CLIP向量"""
        embedding = self.embedding_model.embed_query(text)
        vec = torch.tensor(embedding, dtype=torch.float32)
        return F.normalize(vec, p=2, dim=-1)

    # ===================== 原有对话记忆系统（完全保留） =====================
    def _init_conversation_memory(self):
        self.all_conversation_turns: List[ConversationTurn] = []
        self.pending_conversation_consolidation: List[ConversationTurn] = []
        self.turn_count_since_last_cleanup = 0
        self._load_conversation_memory()
        logger.info("✅ 时间衰减对话记忆系统初始化完成")

    def add_conversation_turn(self, user_input: str, ai_response: str, metadata: Dict = None) -> str:
        metadata = metadata or {}
        is_important = metadata.get("is_important", False) or any(
            keyword in user_input.lower() for keyword in 
            ["记住", "重要", "别忘了", "一定要记得", "我的", "你要", "永远"]
        )
        turn = ConversationTurn(
            user_input=user_input,
            ai_response=ai_response,
            timestamp=datetime.datetime.now().timestamp(),
            initial_activation=1.0,
            is_important=is_important,
            metadata=metadata
        )
        self.all_conversation_turns.append(turn)
        self.turn_count_since_last_cleanup += 1
        if turn.is_important:
            self.pending_conversation_consolidation.append(turn)
            logger.info(f"📝 标记重要对话：{user_input[:30]}...")
        if self.turn_count_since_last_cleanup >= config.CONVERSATION_MEMORY_CONFIG["auto_cleanup_interval"]:
            self._cleanup_forgotten_conversations()
            self.turn_count_since_last_cleanup = 0
        return turn.id

    def get_active_conversation_context(self) -> List[ConversationTurn]:
        now = datetime.datetime.now().timestamp()
        active_turns = []
        for turn in self.all_conversation_turns:
            current_activation = self._calculate_conversation_activation(turn, now)
            if current_activation >= config.CONVERSATION_MEMORY_CONFIG["forget_threshold"]:
                turn_with_activation = turn.model_copy()
                turn_with_activation.metadata["current_activation"] = current_activation
                active_turns.append(turn_with_activation)
        active_turns.sort(key=lambda x: (-x.metadata["current_activation"], -x.timestamp))
        return active_turns[:config.CONVERSATION_MEMORY_CONFIG["max_context_turns"]]

    def get_pending_conversation_consolidation(self) -> List[ConversationTurn]:
        pending = self.pending_conversation_consolidation.copy()
        self.pending_conversation_consolidation.clear()
        return pending

    def _calculate_conversation_activation(self, turn: ConversationTurn, now: float = None) -> float:
        if now is None:
            now = datetime.datetime.now().timestamp()
        time_hours = (now - turn.timestamp) / 3600
        decay_lambda = (
            config.CONVERSATION_MEMORY_CONFIG["important_decay_lambda"] 
            if turn.is_important 
            else config.CONVERSATION_MEMORY_CONFIG["normal_decay_lambda"]
        )
        return turn.initial_activation * math.exp(-decay_lambda * time_hours)

    def _cleanup_forgotten_conversations(self):
        now = datetime.datetime.now().timestamp()
        original_count = len(self.all_conversation_turns)
        self.all_conversation_turns = [
            turn for turn in self.all_conversation_turns
            if self._calculate_conversation_activation(turn, now) >= config.CONVERSATION_MEMORY_CONFIG["forget_threshold"] / 10
        ]
        cleaned_count = original_count - len(self.all_conversation_turns)
        if cleaned_count > 0:
            logger.debug(f"🧹 自动清理了 {cleaned_count} 条完全遗忘的对话")

    def _load_conversation_memory(self):
        if not os.path.exists(self.conversation_memory_file):
            return
        try:
            with open(self.conversation_memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.all_conversation_turns = [ConversationTurn(**td) for td in data.get("all_turns", [])]
            self.pending_conversation_consolidation = [ConversationTurn(**td) for td in data.get("pending_consolidation", [])]
            self._cleanup_forgotten_conversations()
            logger.info(f"✅ 加载历史对话 | 有效:{len(self.all_conversation_turns)}轮")
        except Exception as e:
            logger.error(f"❌ 加载对话历史失败: {e}")

    def _save_conversation_memory(self):
        try:
            data = {
                "version": "2.0",
                "all_turns": [turn.model_dump() for turn in self.all_conversation_turns],
                "pending_consolidation": [turn.model_dump() for turn in self.pending_conversation_consolidation],
                "last_saved": datetime.datetime.now().timestamp()
            }
            with open(self.conversation_memory_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ 保存对话历史失败: {e}")

    def store_entities(self, entities: List[Entity]) -> List[str]:
        """批量存入实体，返回成功ID列表"""
        success_ids = []
        for ent in entities:
            try:
                self.index.add_entity(ent)
                success_ids.append(ent.entity_id)
            except Exception as e:
                logger.warning(f"存入实体失败 {ent.name}: {e}")
        return success_ids

    def store_relations(self, relations: List[EntityRelation]) -> List[str]:
        """批量存入关系，返回成功ID列表"""
        success_ids = []
        for rel in relations:
            try:
                self.index.add_relation(rel)
                success_ids.append(rel.relation_id)
            except Exception as e:
                logger.warning(f"存入关系失败 {rel.predicate}: {e}")
        return success_ids
    
    def is_new_entity(self, entity_name: str) -> bool:
        """
        ✅ 修复缺失的方法：判断实体是否为新实体（不存在则为新）
        """
        return self.index.get_entity_by_name(entity_name) is None
    

    def is_new_relation(self, subject_id: str, predicate: str, object_id: str) -> bool:
        """
        ✅ 修复缺失的方法：判断实体间关系是否为新关系
        :param subject_id: 主体实体ID
        :param predicate: 关系谓词
        :param object_id: 客体实体ID
        :return: 不存在则返回True（新关系）
        """
        # 获取主体的所有关系
        existing_relations = self.index.get_entity_relations(subject_id)
        # 遍历检查是否已有相同关系
        for rel in existing_relations:
            if rel.predicate == predicate and rel.object_id == object_id:
                return False
        return True