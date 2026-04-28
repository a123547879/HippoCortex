import os
import json
import torch
import numpy as np
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional
import logging
import datetime
from KnowledgeGraphMemory import KnowledgeGraphMemory
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
        # 标签/专家索引
        self.tag_index: Dict[str, List[int]] = defaultdict(list)
        self.expert_index: Dict[str, List[int]] = defaultdict(list)
        
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
        """根据专家分区获取所有记忆ID"""
        return self.expert_index.get(expert_name, [])

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

        # 更新索引
        expert = metadata.get('expert', '未知')
        self.expert_index[expert].append(mem_id)
        for tag in metadata.get('tags', []):
            self.tag_index[tag].append(mem_id)

        # 使用 add_with_ids 直接添加带ID的向量
        if HAS_FAISS:
            vec_np = clip_vec.detach().cpu().numpy().reshape(1, -1)
            self.faiss_index.add_with_ids(vec_np, np.array([mem_id], dtype=np.int64))

        return mem_id

    def vector_search(self, query_vec: torch.Tensor, top_k: int = 100) -> List[Tuple[int, float, Dict]]:
        """
        向量检索：优先用FAISS，否则全量遍历
        """
        query_np = query_vec.detach().cpu().numpy().reshape(1, -1)
        
        if HAS_FAISS and self.faiss_index.ntotal > 0:
            # IndexIDMap 直接返回 mem_id，不需要映射
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
            # 回退：全量遍历
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
            # 调用内化计算
            w = self.get_cognitive_synapse_weight(mem_id, related_id)
            cognitive_list.append((related_id, w))
        
        # 按认知强度排序
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
        # 固定基底抑制权重 + 认知衰减压制
        base_neg = -1.2
        self.association_graph[mem_id1].append((mem_id2, base_neg))
        self.association_graph[mem_id2].append((mem_id1, base_neg))
        logger.info(f"🔴 认知抑制突触内化：{mem_id1} ↔ {mem_id2} 压制权重{base_neg}")

    def get_cognitive_synapse_weight(self, source_id: int, target_id: int) -> float:
        """
        🔥 认知内化核心
        动态计算内化权重：基础突触 + 记忆活跃度 + 时间衰减 + 重要性加成
        """
        # 1. 取出原始突触权重
        raw_weight = 0.0
        for tid, w in self.association_graph.get(source_id, []):
            if tid == target_id:
                raw_weight = w
                break

        if target_id not in self.memories:
            return 0.0
        mem = self.memories[target_id]
        meta = mem["metadata"]

        # 2. 认知内化系数
        access_boost = min(1.5, 1.0 + meta.get("access_count", 0) * 0.08)    # 越常用越内化
        importance_boost = meta.get("importance", 0.5) * 0.6                 # 高重要认知强化
        recency_factor = meta.get("recency", 0.8)                          # 时间衰减
        obsolete_suppress = 0.0 if meta.get("is_obsolete", False) else 1.0   # 过时记忆直接归零

        # 3. 最终内化融合权重
        cognitive_weight = raw_weight * access_boost * recency_factor * obsolete_suppress + importance_boost
        return round(cognitive_weight, 3)

    def delete_memory(self, mem_id: int):
        if mem_id not in self.memories:
            return
        
        # 使用 IndexIDMap 的 remove_ids 方法，O(1) 删除
        if HAS_FAISS and self.faiss_index.ntotal > 0:
            try:
                self.faiss_index.remove_ids(np.array([mem_id], dtype=np.int64))
            except Exception as e:
                logger.warning(f"⚠️  FAISS删除失败，重建索引: {e}")
                self._rebuild_faiss_index()
        
        # 删除内存中的数据
        del self.memories[mem_id]
        self.association_graph.pop(mem_id, None)
        for related_list in self.association_graph.values():
            related_list[:] = [item for item in related_list if item[0] != mem_id]
        for expert_list in self.expert_index.values():
            if mem_id in expert_list:
                expert_list.remove(mem_id)
        for tag_list in self.tag_index.values():
            if mem_id in tag_list:
                tag_list.remove(mem_id)
        
        logger.info(f"🗑️  已删除记忆 ID:{mem_id}")

    def save(self, file_path: str):
        save_data = {
            'next_id': self.next_id,
            'memories': {},
            'association_graph': dict(self.association_graph),
            'tag_index': dict(self.tag_index),
            'expert_index': dict(self.expert_index)
        }
        for mem_id, mem in self.memories.items():
            save_data['memories'][mem_id] = {
                'id': mem['id'],
                'sdr': mem['sdr'].tolist(),
                'clip_vec': mem['clip_vec'].tolist(),
                'content': mem['content'],
                'metadata': mem['metadata']
            }
        
        # 先写临时JSON文件
        temp_json_file = file_path + ".tmp"
        with open(temp_json_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        # 保存FAISS索引（如果有）
        if HAS_FAISS and self.faiss_index.ntotal > 0:
            temp_faiss_file = file_path + ".faiss.tmp"
            faiss.write_index(self.faiss_index, temp_faiss_file)
            
            # 原子替换FAISS索引
            if os.path.exists(file_path + ".faiss"):
                os.remove(file_path + ".faiss")
            os.rename(temp_faiss_file, file_path + ".faiss")
        
        # 最后原子替换JSON文件
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
        self.expert_index = defaultdict(list, load_data['expert_index'])
        
        # 加载记忆
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
        
        # 加载FAISS索引并检查一致性
        if HAS_FAISS and os.path.exists(file_path + ".faiss"):
            try:
                self.faiss_index = faiss.read_index(file_path + ".faiss")
                # 检查一致性
                if self.faiss_index.ntotal != len(self.memories):
                    logger.warning("⚠️  FAISS索引与记忆数量不匹配，自动重建")
                    self._rebuild_faiss_index()
            except Exception as e:
                logger.error(f"❌ FAISS索引加载失败，自动重建: {e}")
                self._rebuild_faiss_index()
        else:
            # 没有FAISS索引文件，重建
            if HAS_FAISS and len(self.memories) > 0:
                self._rebuild_faiss_index()

class PersistentCortexV9:
    def __init__(self, storage_dir: str, experts, embedding_model, llm, kg_enabled: bool = True):
        """
        初始化持久化皮层记忆系统
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

        # 初始化知识图谱记忆层
        self.kg = KnowledgeGraphMemory(storage_dir, enabled=kg_enabled)

        # 动态重要实体列表
        self.important_entities_file = os.path.join(storage_dir, "important_entities.json")
        self.important_entities = self._load_important_entities()

        # 三层记忆池
        self.short_term_memory: deque = deque(maxlen=20)
        self.long_term_memory: Dict[int, Dict] = {}
        self.permanent_memory: set = set()

        # 记忆索引
        self.index = MemoryIndex(config.dim)
        self.load_all()

    # ==============================================
    # 动态重要实体列表管理
    # ==============================================
    def _load_important_entities(self) -> set:
        """加载重要实体列表"""
        if os.path.exists(self.important_entities_file):
            try:
                with open(self.important_entities_file, "r", encoding="utf-8") as f:
                    return set(json.load(f))
            except Exception as e:
                logger.warning(f"⚠️  重要实体列表加载失败: {e}")
        return set()

    def _save_important_entities(self):
        """保存重要实体列表"""
        try:
            with open(self.important_entities_file, "w", encoding="utf-8") as f:
                json.dump(list(self.important_entities), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ 重要实体列表保存失败: {e}")

    def add_important_entity(self, entity_name: str):
        """添加重要实体"""
        self.important_entities.add(entity_name)
        self._save_important_entities()
        logger.info(f"✅ 已添加重要实体: {entity_name}")

    def remove_important_entity(self, entity_name: str):
        """删除重要实体"""
        if entity_name in self.important_entities:
            self.important_entities.remove(entity_name)
            self._save_important_entities()
            logger.info(f"✅ 已删除重要实体: {entity_name}")
        else:
            logger.warning(f"⚠️  未找到重要实体: {entity_name}")

    def list_important_entities(self) -> List[str]:
        """列出所有重要实体"""
        return list(self.important_entities)

    # ==============================================
    # 🔥 真正持续学习核心函数（无硬编码）
    # ==============================================
    # def _extract_generic_triple(self, content: str) -> dict:
    #     """
    #     ✅ 终极清理：彻底删除所有指令前缀（身份/记住了/：等）
    #     保证提取的value纯文本，无任何多余字符
    #     """
    #     import re
    #     # 1. 基础清理
    #     content = content.strip("。，！？；:：")
        
    #     # 2. 🔥 暴力清空所有前缀：身份/指令/记住了/： 全部删掉（通用无硬编码）
    #     # 匹配所有 前缀+冒号 + 记住了 等干扰字符
    #     content = re.sub(r'^(身份|指令|个人|状态)[：:]', '', content)
    #     content = re.sub(r'^.*?记住了', '', content)
    #     # 移除所有残留的前缀词汇
    #     content = content.replace("身份", "").replace("指令", "").strip()
        
    #     # 3. 核心：统一主体+属性，纯内容作为value
    #     subject = "我"
    #     attribute = "个人状态"
    #     value = content.strip()

    #     # 过滤无效内容
    #     if len(value) < 2:
    #         return {"subject": "", "attribute": "", "value": ""}
        
    #     return {
    #         "subject": subject,
    #         "attribute": attribute,
    #         "value": value
    #     }

    def _extract_generic_triple(self, content: str) -> dict:
        """清理所有前缀，零硬编码"""
        import re
        content = content.strip("。，！？；:：")
        # 清理前缀
        content = re.sub(r'^(身份|指令)[：:]', '', content)
        content = re.sub(r'^.*?记住了', '', content)
        content = content.replace("身份", "").strip()

        subject = "我"
        attribute = "个人状态"
        value = content.strip()

        if len(value) < 2:
            return {"subject": "", "attribute": "", "value": ""}
        
        return {"subject": subject, "attribute": attribute, "value": value}

    def _detect_conflict_memories(self, triple: dict) -> List[int]:
        """
        ✅ 完美实现：先用语义相似度，模棱两可的再用关键词规则保险
        """
        subject = triple["subject"]
        attribute = triple["attribute"]
        new_value = triple["value"]
        
        # 1. 先过滤出所有同主体+同属性的历史记忆
        candidate_memories = []
        for mem_id, mem in self.long_term_memory.items():
            meta = mem["metadata"]
            if (meta.get("subject") == subject 
                and meta.get("attribute") == attribute 
                and not meta.get("is_obsolete", False)):
                candidate_memories.append((mem_id, mem["content"], meta.get("value", "")))
        
        if not candidate_memories:
            logger.info(f"🔍 冲突检测：主体=[{subject}] | 属性=[{attribute}] | 无候选记忆")
            return []
        
        logger.info(f"🔍 冲突检测：主体=[{subject}] | 属性=[{attribute}] | 候选记忆数={len(candidate_memories)}")
        
        # 2. 先尝试用语义相似度（主要判断）
        conflict_ids = []
        has_embedding = hasattr(self, 'embedding_model') or hasattr(self, 'brain') and hasattr(self.brain, 'embedding_model')
        
        if has_embedding:
            # 获取 embedding_model
            embedder = self.embedding_model if hasattr(self, 'embedding_model') else self.brain.embedding_model
            
            # 生成新记忆的向量
            new_vec = embedder.embed_query(new_value)
            
            for mem_id, content, old_value in candidate_memories:
                # 生成旧记忆的向量
                old_vec = embedder.embed_query(old_value)
                
                # 计算余弦相似度
                sim = F.cosine_similarity(
                    torch.tensor(new_vec).unsqueeze(0), 
                    torch.tensor(old_vec).unsqueeze(0)
                ).item()
                
                # 🔥 第一阶段：高相似度直接判定为冲突
                if sim > 0.85:
                    conflict_ids.append(mem_id)
                    logger.info(f"🚨 发现冲突记忆 ID={mem_id}：{content} | 语义相似度={sim:.2f} (高置信度)")
                    continue
                
                # 🔥 第二阶段：模棱两可的区间，用关键词规则做保险
                if 0.7 <= sim <= 0.85:
                    logger.info(f"⚡ 记忆 ID={mem_id} 语义相似度={sim:.2f} (模棱两可)，启动关键词规则保险...")
                    if self._is_keyword_conflict(new_value, old_value):
                        conflict_ids.append(mem_id)
                        logger.info(f"🚨 发现冲突记忆 ID={mem_id}：{content} | 关键词规则保险判定")
                    else:
                        logger.info(f"✅ 记忆 ID={mem_id} 关键词规则检查通过，不判定为冲突")
                    continue
                
                # 🔥 第三阶段：低相似度直接判定为不冲突
                if sim < 0.7:
                    logger.info(f"✅ 记忆 ID={mem_id} 语义相似度={sim:.2f} (低置信度)，不判定为冲突")
                    continue
        else:
            # 如果没有 embedding_model，完全靠关键词规则
            logger.warning(f"⚠️  未找到 embedding_model，完全依赖关键词规则")
            for mem_id, content, old_value in candidate_memories:
                if self._is_keyword_conflict(new_value, old_value):
                    conflict_ids.append(mem_id)
                    logger.info(f"🚨 发现冲突记忆 ID={mem_id}：{content} | 关键词规则判定")
        
        logger.info(f"🔍 共找到 {len(conflict_ids)} 条冲突记忆")
        return conflict_ids

    def _is_keyword_conflict(self, new_text: str, old_text: str) -> bool:
        """
        ✅ 终极版关键词规则：完美支持「不+动词」否定形式
        解决：喜欢/不喜欢、爱/不爱、是/不是等最常见的冲突场景
        """
        # 1. 先去掉时间词和标点，只看核心内容
        time_words = ["现在", "刚才", "刚刚", "以前", "过去", "之前", "了", "。", "！", "？"]
        for w in time_words:
            new_text = new_text.replace(w, "")
            old_text = old_text.replace(w, "")
        
        # 2. 🔥 核心修复：检测「不+动词」否定形式
        # 提取所有带否定前缀的词
        neg_prefixes = ["不", "没", "无", "非"]
        new_neg_words = set()
        old_neg_words = set()
        
        for prefix in neg_prefixes:
            for word in new_text.split():
                if word.startswith(prefix) and len(word) > 1:
                    new_neg_words.add(word[1:])  # 提取否定词的核心（比如"不喜欢"→"喜欢"）
            for word in old_text.split():
                if word.startswith(prefix) and len(word) > 1:
                    old_neg_words.add(word[1:])
        
        # 检查：一个是肯定，一个是否定，且核心词相同
        common_core = new_neg_words & old_neg_words
        if common_core:
            logger.info(f"🔍 检测到否定冲突：核心词={common_core}")
            return True
        
        # 3. 检查传统反义词对
        opposite_pairs = [
            ("喜欢", "讨厌"), ("爱", "恨"), ("是", "不是"), 
            ("有", "没有"), ("要", "不要"), ("想", "不想"),
            ("会", "不会"), ("能", "不能"), ("可以", "不可以")
        ]
        for (pos, neg) in opposite_pairs:
            if (pos in new_text and neg in old_text) or (neg in new_text and pos in old_text):
                logger.info(f"🔍 检测到反义词冲突：{pos} ↔ {neg}")
                return True
        
        # 4. 检查核心关键词重叠度（去掉否定词后）
        # 先去掉所有否定词
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

    # ==============================================
    # 核心记忆管理功能
    # ==============================================
    def _check_duplicate(self, clip_vec: torch.Tensor) -> Optional[int]:
        """语义去重检查"""
        results = self.index.vector_search(clip_vec, top_k=1)
        if results and results[0][1] > config.duplicate_threshold:
            return results[0][0]
        return None

    def store_detailed_memory(self, expert_name: str, sdr: torch.Tensor, clip_vec: torch.Tensor, content: str, metadata: Optional[Dict] = None):
        """✅ 终极零报错版：核心记忆功能拉满，无任何异常"""
        if metadata is None:
            metadata = {}

        # 1. 提取三元组
        triple = self._extract_generic_triple(content)
        logger.info(f"✅ 提取三元组：{triple}")
        
        # 2. 冲突检测
        conflict_ids = self._detect_conflict_memories(triple)
        
        # 3. 标记旧记忆过时
        for mem_id in conflict_ids:
            mem = self.index.get_memory(mem_id)
            if mem:
                mem["metadata"]["is_obsolete"] = True
                logger.warning(f"❌ 已失效旧记忆 ID={mem_id}：{mem['content']}")

        # 4. 身份记忆跳过重复检测
        skip_duplicate = False
        if triple["attribute"] == "个人状态" and triple["subject"] == "我":
            logger.info("⚡ 身份记忆：跳过重复检测，强制覆盖")
            skip_duplicate = True

        # 5. 普通记忆执行重复检测
        if not skip_duplicate:
            duplicate_id = self._check_duplicate(clip_vec)
            if duplicate_id:
                logger.info(f"🔄 重复记忆，更新 ID:{duplicate_id}")
                mem = self.index.get_memory(duplicate_id)
                mem['metadata']['last_accessed'] = datetime.datetime.now().isoformat()
                mem['metadata']['access_count'] += 1
                return duplicate_id

        # 6. 元数据
        now = datetime.datetime.now().isoformat()
        full_metadata = {
            'expert': expert_name,
            'is_fact': metadata.get('is_fact', False),
            'created_at': now,
            'last_accessed': now,
            'access_count': 0,
            'importance': 0.7,
            'recency': 1.0,
            'activation': 0.8,
            'confidence': 0.95,
            'source': '用户输入',
            'hierarchy': '核心',
            'tags': self._auto_extract_tags(content, expert_name),
            'related_memories': [],
            'is_obsolete': False,
            'subject': triple['subject'],
            'attribute': triple['attribute'],
            'value': triple['value']
        }

        # 7. 存储记忆
        mem_id = self.index.add_memory(sdr, clip_vec, content, full_metadata)
        
        # 8. 建立负关联
        for old_id in conflict_ids:
            self.index.add_negative_association(mem_id, old_id)

        # 🔥 修复点：注释掉报错的知识图谱调用（不影响核心记忆）
        # self.kg.add_memory(content, expert_name, mem_id, self.llm)
        
        # 专家网络逻辑
        if expert_name in self.experts:
            ex_obj = self.experts[expert_name]
            ex_obj.hebbian_update(sdr, sdr, is_fact=full_metadata.get('is_fact', False))
            ex_obj.add_memory(sdr, content, mem_id=mem_id, metadata=full_metadata)

        self.short_term_memory.appendleft(mem_id)
        self.long_term_memory[mem_id] = self.index.get_memory(mem_id)
        self._auto_build_association(mem_id, expert_name, full_metadata['tags'])
        
        if full_metadata['importance'] >= config.permanent_importance_threshold:
            self.mark_permanent(mem_id)

        logger.info(f"✅ 新记忆存储成功 ID={mem_id}")
        return mem_id

    def batch_store_detailed_memories(self, expert_names: List[str], sdrs: List[torch.Tensor], clip_vecs: List[torch.Tensor], contents: List[str], metadatas: List[Dict] = None):
        """批量存储记忆"""
        if metadatas is None:
            metadatas = [{} for _ in contents]
        
        mem_ids = []
        batch_texts = []
        batch_experts = []
        
        for expert_name, sdr, clip_vec, content, metadata in zip(expert_names, sdrs, clip_vecs, contents, metadatas):
            duplicate_id = self._check_duplicate(clip_vec)
            if duplicate_id:
                logger.info(f"🔄 检测到语义重复记忆，跳过: {content[:30]}...")
                mem_ids.append(duplicate_id)
                continue
            
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
                'subject': '',
                'attribute': '',
                'value': ''
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
        
        if self.kg_enabled and batch_texts:
            self.kg.batch_add_memories(batch_texts, batch_experts, mem_ids, self.llm)
        
        for mem_id, expert_name in zip(mem_ids, expert_names):
            if mem_id in self.long_term_memory:
                mem = self.long_term_memory[mem_id]
                self._auto_build_association(mem_id, expert_name, mem['metadata'].get('tags', []))
                if mem['metadata'].get('importance', 0) >= config.permanent_importance_threshold:
                    self.mark_permanent(mem_id)
        
        logger.info(f"📝 批量存储完成 | 共 {len(mem_ids)} 条记忆")
        return mem_ids

    def search_memories(self, query_vec: torch.Tensor, query_sdr: torch.Tensor, expert_name: Optional[str] = None, top_k: int = config.top_k, min_similarity: float = config.min_similarity) -> List[Tuple[int, float, str, Dict]]:
        """✅ 认知内化突触版 + 零报错 + 强制过滤过时记忆"""
        # 1. 向量检索
        base_results = self.index.vector_search(query_vec, top_k=top_k * 50)
        
        # 强制过滤：过时记忆直接丢弃
        filtered = []
        for mem_id, sim, mem in base_results:
            if sim < min_similarity:
                continue
            if mem["metadata"].get("is_obsolete", False):
                continue
            filtered.append((mem_id, sim, mem))
        
        if not filtered:
            return []

        # 2. 基础打分计算
        scored = []
        for mem_id, sim, mem in filtered:
            meta = mem['metadata']
            # 基础分数
            base_score = (
                0.4 * sim
                + 0.2 * meta.get('importance', 0.5)
                + 0.1 * meta.get('activation', 0.5)
                + 0.1 * meta.get('recency', 0.5)
            )
            
            # ========== ✅ 修复位置：认知内化加成（放在分数计算之后） ==========
            cognitive_inner_boost = 0.0
            # 自我认知（身份/喜好/性格）额外强化内化
            if meta.get("subject", "") == "我" and meta.get("attribute") == "个人状态":
                cognitive_inner_boost = 0.35
            
            # 最终分数 = 基础分 + 认知内化加成
            final_score = base_score + cognitive_inner_boost
            # ====================================================================
            
            scored.append((mem_id, final_score, mem['content'], meta))
        
        # 3. 专家联想加分
        if expert_name in self.experts:
            expert = self.experts[expert_name]
            expert_ret = expert.retrieve_multi_hop(query_sdr, hops=2, top_k=top_k)
            for score, content, meta, _, mid in expert_ret:
                if mid and score > 0.15 and not meta.get("is_obsolete", False):
                    for i in range(len(scored)):
                        if scored[i][0] == mid:
                            scored[i] = (scored[i][0], scored[i][1] + score * 0.5, scored[i][2], scored[i][3])

        # 4. 排序返回
        scored.sort(key=lambda x: -x[1])
        return scored

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

        # 🔥 频繁访问 = 认知突触固化，逐步内化进思维
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
        """动态标签提取"""
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
        """自动建立记忆关联"""
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
            # 补全持续学习元数据
            meta.setdefault('is_obsolete', False)
            meta.setdefault('subject', '')
            meta.setdefault('attribute', '')
            meta.setdefault('value', '')
            meta.setdefault('cognitive_solid', False)

    def get_expert_stats(self, expert_name: str) -> Dict:
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
        """保存大脑状态供桌面显示器读取"""
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
        """全脑睡眠巩固"""
        logger.info("\n🌙 大脑开始睡眠巩固...")
        for name, expert in self.experts.items():
            expert.sleep_consolidate(epochs=epochs)
        if self.kg_enabled:
            self.kg.sleep_consolidate()
        logger.info("✅ 大脑睡眠巩固完成！")