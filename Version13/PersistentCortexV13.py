import os
import json
import torch
import numpy as np
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional
import logging
import datetime
from KnowledgeGraphMemoryV3 import KnowledgeGraphMemory
from torch.nn import functional as F

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PersistentCortex")

# 尝试导入FAISS，没有则回退
try:
    import faiss
    HAS_FAISS = True
    logger.info("✅ FAISS 向量索引已加载")
except ImportError:
    HAS_FAISS = False
    logger.warning("⚠️  未安装FAISS，使用全量遍历检索（性能较低）")

from BrainConfig import config

class MemoryIndex:
    def __init__(self, dim):
        self.next_id = 1
        self.memories: Dict[int, Dict] = {}
        
        # 支持正负权重的关联图（正=激活，负=抑制）
        self.association_graph: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        
        # ===================== 🔥 混合索引架构：专家路由（外层）+ 语义网络（内层） =====================
        # 外层：原专家索引（保留不变，作为大抽屉）
        self.expert_index: Dict[str, List[int]] = defaultdict(list)
        
        # 内层：仿生语义索引（抽屉内的标签）
        self.semantic_index: Dict[str, List[int]] = defaultdict(list)  # {语义标签: [mem_id...]}
        self.subject_index: Dict[str, List[int]] = defaultdict(list)   # {主体: [mem_id...]}  # 彻底解决主体混淆
        self.entity_index: Dict[str, List[int]] = defaultdict(list)    # {实体ID: [mem_id...]}  # 符号-神经绑定索引
        # ====================================================================================
        
        # 保留原有标签索引
        self.tag_index: Dict[str, List[int]] = defaultdict(list)
        
        # 使用 IndexIDMap 原生支持 ID 删除
        self.dim = dim
        self.faiss_index = None
        if HAS_FAISS:
            self._init_faiss()

    def _init_faiss(self):
        """初始化 IndexIDMap"""
        base_index = faiss.IndexFlatIP(self.dim)
        self.faiss_index = faiss.IndexIDMap(base_index)

    def _rebuild_faiss_index(self):
        """
        从所有记忆自动重建FAISS索引
        当检测到索引损坏或不匹配时自动调用
        """
        logger.warning("🔧 检测到FAISS索引损坏，正在自动重建...")
        self._init_faiss()
        
        # 按ID顺序重新添加所有向量
        for mem_id in sorted(self.memories.keys()):
            vec = self.memories[mem_id]['clip_vec'].detach().cpu().numpy().reshape(1, -1)
            self.faiss_index.add_with_ids(vec, np.array([mem_id], dtype=np.int64))
        
        logger.info(f"✅ FAISS索引重建完成，共 {len(self.memories)} 条向量")

    def get_memory(self, mem_id: int) -> Optional[Dict]:
        """根据记忆ID获取完整的记忆数据"""
        return self.memories.get(mem_id, None)

    def get_by_tag(self, tag: str) -> List[int]:
        """根据标签获取所有记忆ID"""
        return self.tag_index.get(tag, [])

    def get_by_expert(self, expert_name: str) -> List[int]:
        """根据专家分区获取所有记忆ID（保留原接口）"""
        return self.expert_index.get(expert_name, [])

    # ===================== 🔥 新增：语义/主体索引查询方法 =====================
    def get_by_semantic_tag(self, tag: str) -> List[int]:
        """根据语义标签获取所有记忆ID"""
        return self.semantic_index.get(tag, [])

    def get_by_subject(self, subject: str) -> List[int]:
        """根据主体获取所有记忆ID（从根源解决主体混淆）"""
        return self.subject_index.get(subject, [])
    # ====================================================================================

    def add_memory(self, sdr: torch.Tensor, clip_vec: torch.Tensor, content: str, metadata: Dict) -> int:
        mem_id = self.next_id
        self.memories[mem_id] = {
            'id': mem_id,
            'sdr': sdr,
            'clip_vec': clip_vec,
            'content': content,
            'metadata': metadata
        }
        self.next_id += 1

        # ===================== 🔥 同时更新所有索引（外层+内层） =====================
        # 1. 外层：更新专家索引（保留原逻辑不变）
        expert = metadata.get('expert', '未知')
        self.expert_index[expert].append(mem_id)
        
        # 2. 内层：更新仿生语义索引
        # 2.1 语义标签索引
        for tag in metadata.get('semantic_tags', []):
            self.semantic_index[tag].append(mem_id)
        
        # 2.2 主体索引（核心！从根源解决主体混淆）
        subject = metadata.get('subject', '未知')
        if subject != '未知':
            self.subject_index[subject].append(mem_id)
        
        # 2.3 实体索引（符号-神经绑定）
        for entity_id in metadata.get('entity_ids', []):
            self.entity_index[entity_id].append(mem_id)
        
        # 3. 保留原有标签索引
        for tag in metadata.get('tags', []):
            self.tag_index[tag].append(mem_id)
        # ====================================================================================

        # 使用 add_with_ids 直接添加带ID的向量
        if HAS_FAISS:
            vec_np = clip_vec.detach().cpu().numpy().reshape(1, -1)
            self.faiss_index.add_with_ids(vec_np, np.array([mem_id], dtype=np.int64))

        # ===================== 保留原有的符号学习代码 =====================
        if hasattr(self, "symbolic_core"):
            try:
                self.symbolic_core.learn_from_dialogue("用户", content)
                triplet = self._auto_extract_triplet(content)
                if triplet:
                    subj, pred, obj = triplet
                    subj = self.symbolic_core.reference_learner.resolve_reference(subj)
                    obj = self.symbolic_core.reference_learner.resolve_reference(obj)
                    self.symbolic_core.add_triplet(subj, pred, obj, mem_id=mem_id)
                    metadata["triplet"] = (subj, pred, obj)
                    self.memories[mem_id]['metadata'] = metadata
            except Exception as e:
                print(f"⚠️ 符号学习跳过: {e}")
        # ====================================================================================

        return mem_id
    
    def _auto_extract_triplet(self, content: str) -> tuple:
        """
        【真正零硬编码】全自动三元组提取器
        没有任何预设名字、角色、默认值
        完全从文本字面提取，不做任何强制归一化
        """
        content = content.strip().rstrip("。！？")
        
        # 1. 处理"是"字句
        if "是" in content:
            parts = content.split("是", 1)
            subj = parts[0].strip()
            obj = parts[1].strip()
            if subj and obj:
                return (subj, "是", obj)
        
        # 2. 处理"喜欢/爱/爱好"句
        for pred in ["喜欢", "爱", "爱好"]:
            if pred in content:
                parts = content.split(pred, 1)
                subj = parts[0].strip()
                obj = parts[1].strip()
                if subj and obj:
                    return (subj, pred, obj)
        
        # 3. 处理"住在/家在/位于"句
        for pred in ["住在", "家在", "位于"]:
            if pred in content:
                parts = content.split(pred, 1)
                subj = parts[0].strip()
                obj = parts[1].strip()
                if subj and obj:
                    return (subj, "住在", obj)
        
        # 4. 处理"叫/名叫/名字是"句
        for pred in ["叫", "名叫", "名字是"]:
            if pred in content:
                parts = content.split(pred, 1)
                subj = parts[0].strip()
                obj = parts[1].strip()
                if subj and obj:
                    return (subj, "是", obj)
        
        return None

    def vector_search(self, query_vec: torch.Tensor, top_k: int = 100) -> List[Tuple[int, float, Dict]]:
        """
        向量检索：优先用FAISS，否则全量遍历
        """
        query_np = query_vec.detach().cpu().numpy().reshape(1, -1)
        
        if HAS_FAISS and self.faiss_index.ntotal > 0:
            scores, ids = self.faiss_index.search(query_np, top_k)
            results = []
            for i in range(len(ids[0])):
                mem_id = int(ids[0][i])
                if mem_id == -1:
                    continue
                sim = scores[0][i]
                if mem_id in self.memories:
                    results.append((mem_id, sim, self.memories[mem_id]))
            return results
        else:
            results = []
            for mem_id, mem in self.memories.items():
                sim = torch.cosine_similarity(query_vec, mem['clip_vec'], dim=-1).item()
                results.append((mem_id, sim, mem))
            results.sort(key=lambda x: -x[1])
            return results[:top_k]

    def get_related_memories(self, mem_id: int, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        🔥 内化版联想激活
        不再用固定分数，用【认知内化突触权重】做联想
        """
        if mem_id not in self.association_graph:
            return []
        
        cognitive_list = []
        for related_id, _ in self.association_graph[mem_id]:
            w = self.get_cognitive_synapse_weight(mem_id, related_id)
            cognitive_list.append((related_id, w))
        
        cognitive_list.sort(key=lambda x: -x[1])
        return cognitive_list[:top_k]

    def add_association(self, mem_id1: int, mem_id2: int, strength: float = 0.5):
        """正向认知突触：基础连接"""
        if mem_id1 not in self.memories or mem_id2 not in self.memories:
            return
        self.association_graph[mem_id1].append((mem_id2, strength))
        self.association_graph[mem_id2].append((mem_id1, strength))

    def add_negative_association(self, mem_id1: int, mem_id2: int):
        """负向认知抑制突触：冲突认知永久压制"""
        if mem_id1 not in self.memories or mem_id2 not in self.memories:
            return
        base_neg = -1.2
        self.association_graph[mem_id1].append((mem_id2, base_neg))
        self.association_graph[mem_id2].append((mem_id1, base_neg))
        logger.info(f"🔴 认知抑制突触内化：{mem_id1} ↔ {mem_id2} 压制权重{base_neg}")

    def get_cognitive_synapse_weight(self, source_id: int, target_id: int) -> float:
        """
        🔥 认知内化核心
        动态计算内化权重：基础突触 + 记忆活跃度 + 时间衰减 + 重要性加成
        """
        raw_weight = 0.0
        for tid, w in self.association_graph.get(source_id, []):
            if tid == target_id:
                raw_weight = w
                break

        if target_id not in self.memories:
            return 0.0
        mem = self.memories[target_id]
        meta = mem["metadata"]

        access_boost = min(1.5, 1.0 + meta.get("access_count", 0) * 0.08)
        importance_boost = meta.get("importance", 0.5) * 0.6
        recency_factor = meta.get("recency", 0.8)
        obsolete_suppress = 0.0 if meta.get("is_obsolete", False) else 1.0

        cognitive_weight = raw_weight * access_boost * recency_factor * obsolete_suppress + importance_boost
        return round(cognitive_weight, 3)

    def delete_memory(self, mem_id: int):
        if mem_id not in self.memories:
            return
        
        if HAS_FAISS and self.faiss_index.ntotal > 0:
            try:
                self.faiss_index.remove_ids(np.array([mem_id], dtype=np.int64))
            except Exception as e:
                logger.warning(f"⚠️  FAISS删除失败，重建索引: {e}")
                self._rebuild_faiss_index()
        
        # ===================== 🔥 从所有索引中删除（外层+内层） =====================
        del self.memories[mem_id]
        self.association_graph.pop(mem_id, None)
        
        # 清理关联图
        for related_list in self.association_graph.values():
            related_list[:] = [item for item in related_list if item[0] != mem_id]
        
        # 清理外层专家索引
        for expert_list in self.expert_index.values():
            if mem_id in expert_list:
                expert_list.remove(mem_id)
        
        # 清理内层语义索引
        for tag_list in self.semantic_index.values():
            if mem_id in tag_list:
                tag_list.remove(mem_id)
        
        # 清理主体索引
        for subj_list in self.subject_index.values():
            if mem_id in subj_list:
                subj_list.remove(mem_id)
        
        # 清理实体索引
        for ent_list in self.entity_index.values():
            if mem_id in ent_list:
                ent_list.remove(mem_id)
        
        # 清理原有标签索引
        for tag_list in self.tag_index.values():
            if mem_id in tag_list:
                tag_list.remove(mem_id)
        # ====================================================================================
        
        logger.info(f"🗑️  已删除记忆 ID:{mem_id}")

    def save(self, file_path: str):
        save_data = {
            'next_id': self.next_id,
            'memories': {},
            'association_graph': dict(self.association_graph),
            'tag_index': dict(self.tag_index),
            # ===================== 🔥 保存所有索引（外层+内层） =====================
            'expert_index': dict(self.expert_index),
            'semantic_index': dict(self.semantic_index),
            'subject_index': dict(self.subject_index),
            'entity_index': dict(self.entity_index)
            # ==========================================================
        }
        for mem_id, mem in self.memories.items():
            save_data['memories'][mem_id] = {
                'id': mem['id'],
                'sdr': mem['sdr'].tolist(),
                'clip_vec': mem['clip_vec'].tolist(),
                'content': mem['content'],
                'metadata': mem['metadata']
            }
        
        temp_json_file = file_path + ".tmp"
        with open(temp_json_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        if HAS_FAISS and self.faiss_index.ntotal > 0:
            temp_faiss_file = file_path + ".faiss.tmp"
            faiss.write_index(self.faiss_index, temp_faiss_file)
            
            if os.path.exists(file_path + ".faiss"):
                os.remove(file_path + ".faiss")
            os.rename(temp_faiss_file, file_path + ".faiss")
        
        if os.path.exists(file_path):
            os.remove(file_path)
        os.rename(temp_json_file, file_path)

    def load(self, file_path: str):
        if not os.path.exists(file_path):
            return
        
        with open(file_path, 'r', encoding='utf-8') as f:
            load_data = json.load(f)
        
        self.next_id = load_data['next_id']
        self.association_graph = defaultdict(list, load_data['association_graph'])
        self.tag_index = defaultdict(list, load_data['tag_index'])
        
        # ===================== 🔥 加载所有索引（外层+内层） =====================
        self.expert_index = defaultdict(list, load_data.get('expert_index', {}))
        self.semantic_index = defaultdict(list, load_data.get('semantic_index', {}))
        self.subject_index = defaultdict(list, load_data.get('subject_index', {}))
        self.entity_index = defaultdict(list, load_data.get('entity_index', {}))
        # ==========================================================
        
        self.memories = {}
        for mem_id_str, mem_data in load_data['memories'].items():
            mem_id = int(mem_id_str)
            self.memories[mem_id] = {
                'id': mem_id,
                'sdr': torch.tensor(mem_data['sdr'], dtype=torch.float32),
                'clip_vec': torch.tensor(mem_data['clip_vec'], dtype=torch.float32),
                'content': mem_data['content'],
                'metadata': mem_data['metadata']
            }
        
        if HAS_FAISS and os.path.exists(file_path + ".faiss"):
            try:
                self.faiss_index = faiss.read_index(file_path + ".faiss")
                if self.faiss_index.ntotal != len(self.memories):
                    logger.warning("⚠️  FAISS索引与记忆数量不匹配，自动重建")
                    self._rebuild_faiss_index()
            except Exception as e:
                logger.error(f"❌ FAISS索引加载失败，自动重建: {e}")
                self._rebuild_faiss_index()
        else:
            if HAS_FAISS and len(self.memories) > 0:
                self._rebuild_faiss_index()

class PersistentCortexV13:
    def __init__(self, storage_dir: str, experts, embedding_model, llm, kg_enabled: bool = True):
        """
        初始化持久化皮层记忆系统（混合架构版）
        :param storage_dir: 存储目录
        :param experts: 专家网络字典
        :param llm: LLM实例，用于知识图谱实体提取
        :param kg_enabled: 是否启用知识图谱（默认True，可关闭以提升性能）
        """
        self.storage_dir = storage_dir
        self.experts = experts
        self.llm = llm
        self.kg_enabled = kg_enabled
        os.makedirs(storage_dir, exist_ok=True)
        self.index_file = os.path.join(storage_dir, "cortex_memory_index.json")
        self.embedding_model = embedding_model

        self.kg = KnowledgeGraphMemory(storage_dir, enabled=kg_enabled)

        self.important_entities_file = os.path.join(storage_dir, "important_entities.json")
        self.important_entities = self._load_important_entities()

        self.short_term_memory: deque = deque(maxlen=20)
        self.long_term_memory: Dict[int, Dict] = {}
        self.permanent_memory: set = set()

        self.index = MemoryIndex(config.dim)
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

    def list_important_entities(self) -> List[str]:
        return list(self.important_entities)
    
    def retrieve_global_semantic_weighted(self, query_vec: torch.Tensor, expert_scores: dict, 
                                     top_k: int = 10, min_similarity: float = 0.1,
                                     query_text: str = None):
        """
        新架构核心：全局语义检索 + 专家权重加权排序
        不再隔离专家，全库检索后用路由得分加权
        """
        # 1. 全局向量检索（全量记忆，不区分任何专家）
        base_results = self.index.vector_search(query_vec, top_k=top_k * 50)
        if not base_results:
            return []

        # 基础过滤：剔除过时记忆、低相似度记忆
        filtered = []
        for mem_id, sim, mem in base_results:
            if sim < min_similarity:
                continue
            if mem["metadata"].get("is_obsolete", False):
                continue
            filtered.append((mem_id, sim, mem))

        if not filtered:
            return []

        # 2. 🔥 专家权重加权（核心改动：不屏蔽，只加权）
        weighted_results = []
        for mem_id, sim, mem in filtered:
            meta = mem["metadata"]
            mem_expert = meta.get("expert", "概念")
            
            # 路由专家权重（得分越高，加权越多）
            expert_weight = expert_scores.get(mem_expert, 0.5)
            # 混合加权公式：基础相似度 + 专家权重 + 记忆重要性
            final_score = sim * (0.3 + 0.7 * expert_weight) + meta.get("importance", 0.5) * 0.2

            # 主体匹配额外加分（保留你原有防混淆逻辑）
            if query_text:
                query_triplet = self.index._auto_extract_triplet(query_text)
                query_subject = query_triplet[0] if query_triplet else "未知"
                if "你主人" in query_text:
                    query_subject = "主人"
                elif "你" in query_text:
                    query_subject = "我"
                    
                if meta.get("subject", "") == query_subject and query_subject != "未知":
                    final_score += 0.15

            weighted_results.append((mem_id, final_score, mem["content"], meta))

        # 3. 按最终加权分数排序
        weighted_results.sort(key=lambda x: -x[1])
        return weighted_results[:top_k]
    
    # ===================== 🔥 自动语义属性提取器 =====================
    def _auto_extract_semantic_attributes(self, content: str, expert_name: str) -> Dict:
        """从记忆内容中自动提取语义属性（零硬编码）"""
        # 1. 利用你已有的三元组提取器
        triplet = self.index._auto_extract_triplet(content)
        
        # 2. 自动生成语义标签
        semantic_tags = set()
        subject = "未知"
        predicate = "未知"
        obj = "未知"
        
        if triplet:
            subject, predicate, obj = triplet
            semantic_tags.add(subject)
            semantic_tags.add(predicate)
            semantic_tags.add(obj)
        
        # 3. 补充通用语义标签（基于关键词，零硬编码）
        content_lower = content.lower()
        if "喜欢" in content_lower or "爱" in content_lower:
            semantic_tags.add("喜好")
        if "害怕" in content_lower:
            semantic_tags.add("恐惧")
        if "会" in content_lower or "能" in content_lower:
            semantic_tags.add("技能")
        if "住在" in content_lower or "位于" in content_lower:
            semantic_tags.add("地点")
        
        # 4. 把专家名作为最高级语义标签
        semantic_tags.add(expert_name)
        
        return {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "semantic_tags": list(semantic_tags),
            "entity_ids": []
        }
    # ====================================================================================

    def _extract_generic_triple(self, content: str, expert_name: str) -> dict:
        """保留原有的三元组提取，用于冲突检测"""
        import re
        content = content.strip("。，！？；:：").strip()
        
        content = re.sub(r'^(身份|指令|知识|答案|问题)[：:]', '', content)
        content = re.sub(r'^.*?记住了', '', content)
        content = content.strip()
        
        if len(content) < 2:
            return {"subject": "", "attribute": "", "value": ""}

        if expert_name == "身份":
            subject = "我"
            attribute = "个人状态"
        else:
            subject = content[:6].strip()
            attribute = expert_name

        value = content

        return {
            "subject": subject,
            "attribute": attribute,
            "value": value
        }

    def _detect_conflict_memories(self, triple: dict) -> List[int]:
        """保留原有的冲突检测，完全兼容"""
        subject = triple["subject"]
        attribute = triple["attribute"]
        new_value = triple["value"]
        
        candidate_memories = []
        for mem_id, mem in self.long_term_memory.items():
            meta = mem["metadata"]
            
            mem_subject = meta.get("subject", "")
            mem_attribute = meta.get("attribute", "")
            
            if (mem_subject 
                and mem_attribute 
                and mem_subject == subject 
                and mem_attribute == attribute 
                and not meta.get("is_obsolete", False)):
                candidate_memories.append((mem_id, mem["content"], meta.get("value", "")))
        
        if len(candidate_memories) > 100:
            logger.warning(f"⚠️  候选记忆数异常多：{len(candidate_memories)}，可能是旧记忆的 subject/attribute 没有正确设置")
            candidate_memories = candidate_memories[-100:]
        
        if not candidate_memories:
            logger.info(f"🔍 冲突检测：主体=[{subject}] | 属性=[{attribute}] | 无候选记忆")
            return []
        
        logger.info(f"🔍 冲突检测：主体=[{subject}] | 属性=[{attribute}] | 候选记忆数={len(candidate_memories)}")
        
        conflict_ids = []
        has_embedding = hasattr(self, 'embedding_model') or hasattr(self, 'brain') and hasattr(self.brain, 'embedding_model')
        
        if has_embedding:
            embedder = self.embedding_model if hasattr(self, 'embedding_model') else self.brain.embedding_model
            new_vec = embedder.embed_query(new_value)
            
            for mem_id, content, old_value in candidate_memories:
                old_vec = embedder.embed_query(old_value)
                sim = F.cosine_similarity(
                    torch.tensor(new_vec).unsqueeze(0), 
                    torch.tensor(old_vec).unsqueeze(0)
                ).item()
                
                if sim > 0.85:
                    conflict_ids.append(mem_id)
                    logger.info(f"🚨 发现冲突记忆 ID={mem_id}：{content} | 语义相似度={sim:.2f} (高置信度)")
                    continue
                
                if 0.7 <= sim <= 0.85:
                    logger.info(f"⚡ 记忆 ID={mem_id} 语义相似度={sim:.2f} (模棱两可)，启动关键词规则保险...")
                    if self._is_keyword_conflict(new_value, old_value):
                        conflict_ids.append(mem_id)
                        logger.info(f"🚨 发现冲突记忆 ID={mem_id}：{content} | 关键词规则保险判定")
                    else:
                        logger.info(f"✅ 记忆 ID={mem_id} 关键词规则检查通过，不判定为冲突")
                    continue
                
                if sim < 0.7:
                    logger.info(f"✅ 记忆 ID={mem_id} 语义相似度={sim:.2f} (低置信度)，不判定为冲突")
                    continue
        else:
            logger.warning(f"⚠️  未找到 embedding_model，完全依赖关键词规则")
            for mem_id, content, old_value in candidate_memories:
                if self._is_keyword_conflict(new_value, old_value):
                    conflict_ids.append(mem_id)
                    logger.info(f"🚨 发现冲突记忆 ID={mem_id}：{content} | 关键词规则判定")
        
        logger.info(f"🔍 共找到 {len(conflict_ids)} 条冲突记忆")
        return conflict_ids

    def _is_keyword_conflict(self, new_text: str, old_text: str) -> bool:
        """保留原有的关键词冲突检测，完全兼容"""
        time_words = ["现在", "刚才", "刚刚", "以前", "过去", "之前", "了", "。", "！", "？"]
        for w in time_words:
            new_text = new_text.replace(w, "")
            old_text = old_text.replace(w, "")
        
        neg_prefixes = ["不", "没", "无", "非"]
        new_neg_words = set()
        old_neg_words = set()
        
        for prefix in neg_prefixes:
            for word in new_text.split():
                if word.startswith(prefix) and len(word) > 1:
                    new_neg_words.add(word[1:])
            for word in old_text.split():
                if word.startswith(prefix) and len(word) > 1:
                    old_neg_words.add(word[1:])
        
        common_core = new_neg_words & old_neg_words
        if common_core:
            logger.info(f"🔍 检测到否定冲突：核心词={common_core}")
            return True
        
        opposite_pairs = [
            ("喜欢", "讨厌"), ("爱", "恨"), ("是", "不是"), 
            ("有", "没有"), ("要", "不要"), ("想", "不想"),
            ("会", "不会"), ("能", "不能"), ("可以", "不可以")
        ]
        for (pos, neg) in opposite_pairs:
            if (pos in new_text and neg in old_text) or (neg in new_text and pos in old_text):
                logger.info(f"🔍 检测到反义词冲突：{pos} ↔ {neg}")
                return True
        
        new_clean = new_text
        old_clean = old_text
        for prefix in neg_prefixes:
            new_clean = new_clean.replace(prefix, "")
            old_clean = old_clean.replace(prefix, "")
        
        new_keywords = set(new_clean.replace("我", "").replace("的", "").replace("是", "").split())
        old_keywords = set(old_clean.replace("我", "").replace("的", "").replace("是", "").split())
        
        if len(new_keywords) > 0 and len(old_keywords) > 0:
            overlap = len(new_keywords & old_keywords) / max(len(new_keywords), len(old_keywords))
            if overlap > 0.5:
                logger.info(f"🔍 检测到核心关键词重叠：重叠度={overlap:.2f}")
                return True
        
        return False

    def _check_duplicate(self, clip_vec: torch.Tensor) -> Optional[int]:
        """保留原有的语义去重，完全兼容"""
        results = self.index.vector_search(clip_vec, top_k=1)
        if results and results[0][1] > config.duplicate_threshold:
            return results[0][0]
        return None

    def store_detailed_memory(self, expert_name: str, sdr: torch.Tensor, clip_vec: torch.Tensor, content: str, metadata: Optional[Dict] = None):
        if metadata is None:
            metadata = {}

        # ===================== 核心改动1：合并三元组与语义提取，避免重复工作 =====================
        structured_info = self._extract_structured_info(content, expert_name)
        logger.info(f"🧠 结构化提取：主体={structured_info['subject']} | 谓词={structured_info['predicate']}")
        
        # ===================== 核心改动2：先做重复检测，避免无效计算 =====================
        is_identity_memory = (structured_info['attribute'] == "个人状态" and structured_info['subject'] == "我")
        if not is_identity_memory:
            duplicate_id = self._check_duplicate(clip_vec)
            if duplicate_id:
                logger.info(f"🔄 检测到重复记忆，更新访问时间 ID:{duplicate_id}")
                self._update_memory_access(duplicate_id)
                return duplicate_id

        # ===================== 核心改动3：冲突检测与旧记忆失效 =====================
        conflict_ids = self._detect_conflict_memories(structured_info)
        for mem_id in conflict_ids:
            self._mark_memory_obsolete(mem_id)

        # ===================== 核心改动4：精简元数据，但保留核心字段 =====================
        now = datetime.datetime.now().isoformat()
        full_metadata = {
            'expert': expert_name,
            'is_fact': metadata.get('is_fact', False),
            'created_at': now,
            'last_accessed': now,
            'access_count': 0,
            'importance': metadata.get('importance', 0.7),
            'recency': 1.0,
            'is_obsolete': False,
            # 核心结构化字段
            'subject': structured_info['subject'],
            'attribute': structured_info['attribute'],
            'value': structured_info['value'],
            'semantic_tags': structured_info['semantic_tags'],
            'predicate': structured_info['predicate'],
            'object': structured_info['object']
        }

        # ===================== 核心改动5：存储记忆（核心路径） =====================
        mem_id = self.index.add_memory(sdr, clip_vec, content, full_metadata)

        # ===================== 核心改动6：建立冲突记忆的负关联 =====================
        for old_id in conflict_ids:
            self.index.add_negative_association(mem_id, old_id)

        # ===================== ✅ 恢复：知识图谱代码（完全保留你的原有逻辑） =====================
        if self.kg_enabled:
            entities = self._extract_entities_from_content(content, full_metadata['semantic_tags'], structured_info)
            for entity in entities:
                entity = entity.strip()
                if len(entity) >= 2:
                    self.add_important_entity(entity)
            self.kg.add_memory_with_entities(content, expert_name, mem_id, entities)
        # ====================================================================================

        # ===================== 🔥 核心改动7：专家网络更新（切换为STDP） =====================
        if expert_name in self.experts:
            expert = self.experts[expert_name]
            
            # 🔥 检查是否启用STDP
            if hasattr(expert, 'stdp_enabled') and expert.stdp_enabled:
                # 使用STDP学习：
                # - pre_sdr = sdr（输入激活）
                # - post_sdr = sdr（输出激活）
                # - delta_t = 10.0ms（正时间差表示因果关系：输入先激活，神经元后响应）
                expert.stdp_update(sdr, sdr, delta_t=10.0)
                logger.debug(f"🧠 STDP学习 | 专家={expert_name} | delta_t=10.0ms")
            else:
                # 回退到传统赫布学习
                expert.hebbian_update(sdr, sdr, is_fact=full_metadata.get('is_fact', False))
            
            # 添加记忆到专家（这部分保持不变）
            expert.add_memory(sdr, content, mem_id=mem_id, metadata=full_metadata)

        # ===================== 核心改动8：记忆层级管理 =====================
        self.short_term_memory.appendleft(mem_id)
        self.long_term_memory[mem_id] = self.index.get_memory(mem_id)
        self._auto_build_association(mem_id, expert_name, full_metadata['semantic_tags'])
        
        if full_metadata['importance'] >= config.permanent_importance_threshold:
            self.mark_permanent(mem_id)

        logger.info(f"✅ 记忆存储成功 | ID={mem_id} | 专家={expert_name} | 主体={structured_info['subject']}")
        return mem_id

    # ===================== 辅助函数：合并后的结构化信息提取 =====================
    def _extract_structured_info(self, content: str, expert_name: str) -> Dict:
        """
        一次性提取所有结构化信息，避免重复调用
        合并了原 _extract_generic_triple 和 _auto_extract_semantic_attributes
        """
        # 1. 基础三元组提取（保留你原有的冲突检测用三元组）
        triple = self._extract_generic_triple(content, expert_name)
        
        # 2. 语义标签生成
        semantic_tags = {expert_name}
        subject, predicate, obj = triple["subject"], triple["attribute"], triple["value"]
        
        if subject and subject != "未知":
            semantic_tags.add(subject)
        if predicate and predicate != "未知":
            semantic_tags.add(predicate)
        if obj and obj != "未知":
            semantic_tags.add(obj)
        
        # 3. 补充通用语义标签
        content_lower = content.lower()
        if any(k in content_lower for k in ["喜欢", "爱", "爱好"]):
            semantic_tags.add("喜好")
        if "害怕" in content_lower:
            semantic_tags.add("恐惧")
        if any(k in content_lower for k in ["会", "能", "可以"]):
            semantic_tags.add("技能")
        if any(k in content_lower for k in ["住在", "位于", "地点"]):
            semantic_tags.add("地点")

        return {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "attribute": predicate,  # 兼容旧代码
            "value": obj,            # 兼容旧代码
            "semantic_tags": list(semantic_tags)
        }

    # ===================== 辅助函数：更新记忆访问时间 =====================
    def _update_memory_access(self, mem_id: int):
        """封装记忆访问更新逻辑"""
        mem = self.index.get_memory(mem_id)
        if mem:
            mem['metadata']['last_accessed'] = datetime.datetime.now().isoformat()
            mem['metadata']['access_count'] += 1

    # ===================== 辅助函数：标记记忆过时 =====================
    def _mark_memory_obsolete(self, mem_id: int):
        """封装旧记忆失效逻辑"""
        mem = self.index.get_memory(mem_id)
        if mem:
            mem["metadata"]["is_obsolete"] = True
            logger.warning(f"❌ 失效旧记忆 | ID={mem_id} | 内容={mem['content'][:40]}...")
    
    def _extract_entities_from_content(self, content: str, tags: List[str], triple: dict) -> List[str]:
        """保留原有的实体提取，完全兼容"""
        entities = set()
        
        for tag in tags:
            if tag and len(tag.strip()) >= 2:
                entities.add(tag.strip())
        
        if triple.get("subject") and triple["subject"] != "我":
            entities.add(triple["subject"].strip())
        
        for entity in self.important_entities:
            if entity in content:
                entities.add(entity.strip())

        entity_list = list(entities)
        logger.info(f"🔍 极简实体提取：{entity_list}")
        return entity_list

    def batch_store_detailed_memories(self, expert_names: List[str], sdrs: List[torch.Tensor], clip_vecs: List[torch.Tensor], contents: List[str], metadatas: List[Dict] = None):
        """保留原有的批量存储，完全兼容"""
        if metadatas is None:
            metadatas = [{} for _ in contents]
        
        mem_ids = []
        batch_texts = []
        batch_experts = []
        batch_entities = []
        
        for expert_name, sdr, clip_vec, content, metadata in zip(expert_names, sdrs, clip_vecs, contents, metadatas):
            duplicate_id = self._check_duplicate(clip_vec)
            if duplicate_id:
                logger.info(f"🔄 检测到语义重复记忆，跳过: {content[:30]}...")
                mem_ids.append(duplicate_id)
                continue
            
            triple = self._extract_generic_triple(content, expert_name)
            semantic_attrs = self._auto_extract_semantic_attributes(content, expert_name)
            
            now = datetime.datetime.now().isoformat()
            full_metadata = {
                'expert': expert_name,
                'is_fact': metadata.get('is_fact', False),
                'created_at': now,
                'last_accessed': now,
                'access_count': 0,
                'importance': metadata.get('importance', 0.7 if metadata.get('is_fact', False) else 0.5),
                'recency': 1.0,
                'activation': 0.8,
                'confidence': metadata.get('confidence', 0.95 if metadata.get('is_fact', False) else 0.7),
                'source': metadata.get('source', '批量导入'),
                'hierarchy': metadata.get('hierarchy', '核心' if metadata.get('is_fact', False) else '关联'),
                'tags': metadata.get('tags', self._auto_extract_tags(content, expert_name)),
                'related_memories': [],
                'is_obsolete': False,
                'subject': triple['subject'],
                'attribute': triple['attribute'],
                'value': triple['value'],
                'semantic_tags': semantic_attrs['semantic_tags'],
                'predicate': semantic_attrs['predicate'],
                'object': semantic_attrs['object']
            }
            
            mem_id = self.index.add_memory(sdr, clip_vec, content, full_metadata)
            mem_ids.append(mem_id)
            
            if expert_name in self.experts:
                expert = self.experts[expert_name]
                expert.hebbian_update(sdr, sdr, is_fact=full_metadata.get('is_fact', False))
                expert.add_memory(sdr, content, mem_id=mem_id, metadata=full_metadata)
            
            self.short_term_memory.appendleft(mem_id)
            self.long_term_memory[mem_id] = self.index.get_memory(mem_id)
            batch_texts.append(content)
            batch_experts.append(expert_name)
            
            entities = self._extract_entities_from_content(content, full_metadata['tags'], triple)
            batch_entities.append(entities)
            
            if self.kg_enabled and entities:
                for entity in entities:
                    entity = entity.strip()
                    if len(entity) >= 2:
                        self.add_important_entity(entity)
        
        if self.kg_enabled and batch_texts:
            for content, expert_name, mem_id, entities in zip(batch_texts, batch_experts, mem_ids, batch_entities):
                if mem_id in self.long_term_memory:
                    self.kg.add_memory_with_entities(content, expert_name, mem_id, entities)
        
        for mem_id, expert_name in zip(mem_ids, expert_names):
            if mem_id in self.long_term_memory:
                mem = self.long_term_memory[mem_id]
                self._auto_build_association(mem_id, expert_name, mem['metadata'].get('tags', []))
                if mem['metadata'].get('importance', 0) >= config.permanent_importance_threshold:
                    self.mark_permanent(mem_id)
        
        logger.info(f"📝 批量存储完成 | 共 {len(mem_ids)} 条记忆 | 知识图谱关联 {len(batch_entities)} 条 | 重要实体数 {len(self.important_entities)}")
        return mem_ids
    
    def _extract_query_entities(self, query_vec: torch.Tensor, expert_name: Optional[str], query_text: str = None) -> List[str]:
        """保留原有的查询实体提取，完全兼容"""
        entities = set()
        
        try:
            from BrainConfig import config
            cfg = getattr(config, "entity_extraction_config", {})
        except:
            cfg = {}
        
        min_len = cfg.get("min_entity_length", 2)
        split_chars = cfg.get("split_chars", ["，", "。", "！", "？", "；", "：", "、", " "])

        for entity in self.important_entities:
            entity = entity.strip()
            if len(entity) >= min_len:
                entities.add(entity)

        if query_text:
            import re
            
            clean_text = query_text
            for c in split_chars:
                clean_text = clean_text.replace(c, " ")
            
            words = clean_text.split()
            for word in words:
                word = word.strip()
                if len(word) >= min_len:
                    entities.add(word)

        entity_list = list(entities)
        logger.info(f"🔍 查询实体提取：{entity_list}")
        return entity_list
    
    def auto_extract_important_entities(self, top_k: int = 20, min_access_count: int = 1):
        """保留原有的自动重要实体提取，完全兼容"""
        if not self.kg_enabled or len(self.kg.G.nodes) == 0:
            logger.warning("⚠️  知识图谱为空，无法自动提取重要实体")
            return
        
        logger.info(f"\n🔍 ========== 自动提取重要实体开始 ==========")
        
        entity_scores = []
        for node_id, attrs in self.kg.G.nodes(data=True):
            if attrs.get("type") == "memory":
                continue
            
            entity_name = attrs.get("name", "")
            if not entity_name:
                continue
            
            access_count = attrs.get("access_count", 0)
            mem_count = len(attrs.get("mem_ids", []))
            
            score = access_count * 2 + mem_count
            
            if access_count >= min_access_count:
                entity_scores.append((entity_name, score, access_count, mem_count))
        
        entity_scores.sort(key=lambda x: -x[1])
        
        top_entities = [name for name, score, acc, mem in entity_scores[:top_k]]
        
        self.important_entities = set(top_entities)
        self._save_important_entities()
        
        logger.info(f"🔍 共找到 {len(entity_scores)} 个候选实体")
        logger.info(f"🔍 提取前 {len(top_entities)} 个重要实体:")
        for i, (name, score, acc, mem) in enumerate(entity_scores[:top_k]):
            logger.info(f"🔍 排名{i+1:2d} | 实体={name:20} | 分数={score:3d} | 访问={acc:2d} | 关联记忆={mem:2d}")
        
        logger.info(f"🔍 ========== 自动提取重要实体结束 ==========\n")

    # ===================== 🔥 终极检索：专家路由（外层）+ 主体过滤（内层） =====================
    def search_memories(self, query_vec: torch.Tensor, query_sdr: torch.Tensor, 
                    expert_name: Optional[str] = None, top_k: int = config.top_k, 
                    min_similarity: float = config.min_similarity, 
                    query_text: Optional[str] = None,
                    expert_scores: Optional[dict] = None):
        """
        混合架构最终检索：
        1. 全局语义检索（无专家隔离）
        2. 路由专家得分加权排序
        3. 主体过滤防混淆
        """
        # ===================== 新架构：有专家得分 → 走全局加权检索 =====================
        if expert_scores is not None:
            logger.info(f"🌍 启用全局语义检索 + 专家加权排序")
            return self.retrieve_global_semantic_weighted(
                query_vec=query_vec,
                expert_scores=expert_scores,
                top_k=top_k,
                min_similarity=min_similarity,
                query_text=query_text
            )

        # ===================== 兼容旧逻辑：无专家得分 → 走原有检索（兜底） =====================
        logger.warning(f"⚠️ 未获取专家得分，使用兼容检索模式")
        # 1. 向量检索（全库）
        base_results = self.index.vector_search(query_vec, top_k=top_k * 50)
        # logger.info(f'base_results: {len(base_results)}')
        
        # 过滤：过时记忆 + 低相似度
        filtered = []
        for mem_id, sim, mem in base_results:
            if sim < min_similarity:
                continue
            if mem["metadata"].get("is_obsolete", False):
                continue
            filtered.append((mem_id, sim, mem))
        
        if not filtered:
            return []

        # 外层专家过滤（仅兼容保留，新架构不使用）
        if expert_name is not None:
            expert_filtered = []
            for mem_id, sim, mem in filtered:
                mem_expert = mem["metadata"].get("expert", "")
                if mem_expert == expert_name:
                    expert_filtered.append((mem_id, sim, mem))
            filtered = expert_filtered
            if not filtered:
                return []

        # 内层：主体优先过滤
        if query_text:
            query_triplet = self.index._auto_extract_triplet(query_text)
            query_subject = query_triplet[0] if query_triplet else "未知"
            if "你主人" in query_text or "我的主人" in query_text:
                query_subject = "主人"
            elif "你" in query_text and "主人" not in query_text:
                query_subject = "我"
            
            if query_subject != "未知" and query_subject in self.index.subject_index:
                logger.info(f"🎯 主体过滤：锁定【{query_subject}】相关记忆")
                subject_mem_ids = set(self.index.get_by_subject(query_subject))
                subject_filtered = []
                for mem_id, sim, mem in filtered:
                    if mem_id in subject_mem_ids:
                        subject_filtered.append((mem_id, sim, mem))
                if subject_filtered:
                    filtered = subject_filtered

        # 基础打分
        scored = []
        for mem_id, sim, mem in filtered:
            meta = mem['metadata']
            base_score = sim * 0.8 + meta.get('importance', 0.5) * 0.2
            scored.append((mem_id, base_score, mem['content'], meta))

        # 专家联想加分
        if expert_name in self.experts:
            expert = self.experts[expert_name]
            expert_ret = expert.retrieve_multi_hop(query_sdr, hops=1, top_k=top_k)
            for score, content, meta, _, mid in expert_ret:
                if mid and score > 0.15 and not meta.get("is_obsolete", False):
                    for i in range(len(scored)):
                        if scored[i][0] == mid:
                            scored[i] = (scored[i][0], scored[i][1] + score * 0.5, scored[i][2], scored[i][3])

        scored.sort(key=lambda x: -x[1])
        return scored[:50]
    
    def increment_access_count(self, mem_id: int):
        mem = self.index.get_memory(mem_id)
        if not mem:
            return
        meta = mem['metadata']

        meta['access_count'] += 1
        meta['last_accessed'] = datetime.datetime.now().isoformat()
        meta['activation'] = min(1.0, meta['activation'] + 0.3)
        meta['importance'] = min(1.0, meta['importance'] + 0.05)
        meta['recency'] = 1.0

        if meta["access_count"] >= 5:
            meta["cognitive_solid"] = True

        if meta['access_count'] >= 3 and mem_id in self.short_term_memory:
            self.short_term_memory.remove(mem_id)
            logger.info(f"🧠 记忆ID:{mem_id} 认知内化升级 → 转入长期固化记忆")

        if meta['importance'] >= config.permanent_importance_threshold and mem_id not in self.permanent_memory:
            self.mark_permanent(mem_id)

    def mark_permanent(self, mem_id: int):
        mem = self.index.get_memory(mem_id)
        if not mem:
            return
        self.permanent_memory.add(mem_id)
        mem['metadata']['hierarchy'] = '永久'
        logger.info(f"🔒 记忆ID:{mem_id} 已标记为永久记忆")

    def decay_all_memories(self):
        logger.info("⏳ 执行记忆自然衰减...")
        now = datetime.datetime.now()
        to_delete = []

        for mem_id, mem in self.index.memories.items():
            if mem_id in self.permanent_memory:
                continue

            meta = mem['metadata']
            meta['activation'] = max(0.0, meta['activation'] - 0.02)
            
            create_time = datetime.datetime.fromisoformat(meta['created_at'])
            days_since_create = (now - create_time).days
            meta['recency'] = max(0.0, 1.0 - days_since_create / 365)

            last_access = datetime.datetime.fromisoformat(meta['last_accessed'])
            days_since_access = (now - last_access).days
            if days_since_access >= config.forget_days and meta['importance'] < config.forget_importance_threshold:
                to_delete.append(mem_id)

        for mem_id in to_delete:
            self.index.delete_memory(mem_id)
            self.long_term_memory.pop(mem_id, None)
            for expert in self.experts.values():
                expert.delete_memory(mem_id)
            logger.info(f"🗑️  遗忘低价值记忆 | ID:{mem_id}")

        logger.info(f"✅ 记忆衰减完成，共遗忘 {len(to_delete)} 条低价值记忆")

    def _auto_extract_tags(self, content: str, expert_name: str) -> List[str]:
        """保留原有的动态标签提取，完全兼容"""
        tags = [expert_name]
        if expert_name == "身份":
            if "你是谁" in content or "我是谁" in content:
                tags.append("身份认知")
            if "名字" in content or "叫" in content:
                tags.append("名字")
            if "主人" in content:
                tags.append("主人")
            if "关系" in content:
                tags.append("伙伴关系")
            
            if self.kg_enabled:
                for node_id, attrs in self.kg.G.nodes(data=True):
                    entity_name = attrs.get("name", "")
                    if entity_name and entity_name in content:
                        tags.append("专属身份")
                        tags.append(entity_name)
            
            for entity in self.important_entities:
                if entity in content:
                    tags.append("专属身份")
                    tags.append(entity)
                
        elif expert_name == "概念":
            if "人物" in content or "是谁" in content:
                tags.append("人物")
        elif expert_name == "空间":
            if "事件" in content or "案件" in content or "年" in content:
                tags.append("历史")
        elif expert_name == "抽象":
            if "知识" in content or "是什么" in content:
                tags.append("知识")
            if "名言" in content:
                tags.append("名言")
        return list(set(tags))

    def _auto_build_association(self, mem_id: int, expert_name: str, tags: List[str]):
        """保留原有的自动关联建立，完全兼容"""
        for tag in tags:
            tag_mem_ids = self.index.get_by_tag(tag)
            for related_id in tag_mem_ids[-5:]:
                if related_id != mem_id:
                    self.index.add_association(mem_id, related_id, strength=0.3)
        expert_mem_ids = self.index.get_by_expert(expert_name)
        for related_id in expert_mem_ids[-3:]:
            if related_id != mem_id:
                self.index.add_association(mem_id, related_id, strength=0.1)

    def save_all(self):
        logger.info("💾 正在安全保存皮层记忆...")
        
        temp_index_file = self.index_file + ".tmp"
        self.index.save(temp_index_file)
        
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        os.rename(temp_index_file, self.index_file)
        
        state_data = {
            'short_term_memory': list(self.short_term_memory),
            'permanent_memory': list(self.permanent_memory)
        }
        state_file = os.path.join(self.storage_dir, "cortex_state.json")
        temp_state_file = state_file + ".tmp"
        
        with open(temp_state_file, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)
        
        if os.path.exists(state_file):
            os.remove(state_file)
        os.rename(temp_state_file, state_file)
        
        for name, expert in self.experts.items():
            expert_path = os.path.join(self.storage_dir, f"expert_{name}.pt")
            expert.save_weights(expert_path)
        
        self.kg.save()
        self._save_important_entities()
        
        logger.info("✅ 皮层记忆已安全保存！")

    def load_all(self):
        if not os.path.exists(self.index_file):
            logger.info("📦 无历史记忆，初始化新的皮层记忆系统")
            return
        
        self.index.load(self.index_file)
        self.long_term_memory = self.index.memories
        
        state_file = os.path.join(self.storage_dir, "cortex_state.json")
        if os.path.exists(state_file):
            with open(state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            self.short_term_memory = deque(state_data['short_term_memory'], maxlen=20)
            self.permanent_memory = set(state_data['permanent_memory'])
        
        for name, expert in self.experts.items():
            expert_path = os.path.join(self.storage_dir, f"expert_{name}.pt")
            expert.load_weights(expert_path)
        
        self.kg.load()
        self.important_entities = self._load_important_entities()
        self._compat_old_memories()
        
        logger.info(f"✅ 历史记忆加载完成 | 总记忆数:{len(self.index.memories)} | 永久记忆数:{len(self.permanent_memory)}")
        if self.important_entities:
            logger.info(f"✅ 重要实体列表加载完成: {list(self.important_entities)}")
        if not self.kg_enabled:
            logger.info("⚠️  知识图谱当前处于关闭状态")

    def _compat_old_memories(self):
        """兼容旧版记忆，补全元数据"""
        now = datetime.datetime.now().isoformat()
        for mem_id, mem in self.index.memories.items():
            meta = mem['metadata']
            meta.setdefault('last_accessed', meta.get('created_at', now))
            meta.setdefault('access_count', 0)
            meta.setdefault('importance', 0.6)
            meta.setdefault('recency', 0.8)
            meta.setdefault('activation', 0.5)
            meta.setdefault('confidence', 0.9)
            meta.setdefault('source', '历史导入')
            meta.setdefault('hierarchy', '关联')
            meta.setdefault('tags', self._auto_extract_tags(mem['content'], meta.get('expert', '未知')))
            meta.setdefault('related_memories', [])
            meta.setdefault('is_obsolete', False)
            meta.setdefault('cognitive_solid', False)
            
            # 自动为旧记忆补全语义属性
            if 'semantic_tags' not in meta:
                semantic_attrs = self._auto_extract_semantic_attributes(mem['content'], meta.get('expert', '未知'))
                meta.update(semantic_attrs)
            
            # 自动为旧记忆补全主体/属性/值
            if 'subject' not in meta or 'attribute' not in meta or 'value' not in meta:
                triple = self._extract_generic_triple(mem['content'], meta.get('expert', '未知'))
                meta['subject'] = triple['subject']
                meta['attribute'] = triple['attribute']
                meta['value'] = triple['value']

    def get_expert_stats(self, expert_name: str) -> Dict:
        """保留原有的专家统计功能，完全兼容混合架构"""
        mem_ids = self.index.get_by_expert(expert_name)
        if not mem_ids:
            return {'count': 0, 'avg_access': 0.0}
        total_access = 0
        for mem_id in mem_ids:
            mem = self.index.get_memory(mem_id)
            if mem:
                total_access += mem['metadata'].get('access_count', 0)
        return {
            'count': len(mem_ids),
            'avg_access': total_access / len(mem_ids)
        }
    
    def save_brain_state(self):
        """保留原有的大脑状态保存功能，完全兼容混合架构"""
        try:
            total_memories = len(self.index.memories)
            expert_counts = {"身份":0,"概念":0,"空间":0,"抽象":0,"视觉":0}
            for k in expert_counts.keys():
                expert_counts[k] = len(self.index.get_by_expert(k))
            
            short_term = len(self.short_term_memory)
            long_term = len(self.long_term_memory)
            permanent = len(self.permanent_memory)
            
            total_possible = total_memories * 10 if total_memories > 0 else 1
            actual_associations = sum(len(assoc) for assoc in self.index.association_graph.values())
            sparsity = max(0.7, min(0.98, 1.0 - (actual_associations / total_possible)))
            
            state = {
                "total_memories": total_memories,
                "expert_distribution": expert_counts,
                "memory_layers": {"短期记忆":short_term,"长期记忆":long_term,"永久记忆":permanent},
                "experts": {
                    "身份":{"突触稀疏度":sparsity},
                    "概念":{"突触稀疏度":sparsity},
                    "空间":{"突触稀疏度":sparsity},
                    "抽象":{"突触稀疏度":sparsity},
                    "视觉":{"突触稀疏度":1.0}
                },
                "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "brain_state": "awake"
            }
            
            with open("brain_state.json", "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
                
            logger.info("💾 大脑状态已同步到桌面显示器")
            
        except Exception as e:
            logger.error(f"❌ 保存大脑状态失败: {e}")
            with open("brain_state.json", "w", encoding="utf-8") as f:
                json.dump({
                    "error": f"后端: {str(e)[:15]}",
                    "total_memories": 0,
                    "expert_distribution": {"身份":0,"概念":0,"空间":0,"抽象":0,"视觉":0},
                    "memory_layers": {"短期记忆":0,"长期记忆":0,"永久记忆":0},
                    "experts": {"身份":{"突触稀疏度":1.0},"概念":{"突触稀疏度":1.0},"空间":{"突触稀疏度":1.0},"抽象":{"突触稀疏度":1.0},"视觉":{"突触稀疏度":1.0}},
                    "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "brain_state": "error"
                }, f, ensure_ascii=False, indent=2)

    def sleep_consolidate_all(self, epochs=3):
        """保留原有的全脑睡眠巩固功能，完全兼容混合架构"""
        logger.info("\n🌙 大脑开始睡眠巩固...")
        for name, expert in self.experts.items():
            expert.sleep_consolidate(epochs=epochs)
        if self.kg_enabled:
            self.kg.sleep_consolidate()
        logger.info("✅ 大脑睡眠巩固完成！")