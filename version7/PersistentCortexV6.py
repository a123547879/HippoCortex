import os
import json
import torch
import numpy as np
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional
import logging
import datetime

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
        
        # 关联图
        self.association_graph: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        # 标签/专家索引
        self.tag_index: Dict[str, List[int]] = defaultdict(list)
        self.expert_index: Dict[str, List[int]] = defaultdict(list)
        
        # 🔥 修复：使用 IndexIDMap 原生支持 ID 删除
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
        if mem_id not in self.association_graph:
            return []
        sorted_assoc = sorted(self.association_graph[mem_id], key=lambda x: -x[1])
        return sorted_assoc[:top_k]

    def add_association(self, mem_id1: int, mem_id2: int, strength: float = 0.5):
        if mem_id1 not in self.memories or mem_id2 not in self.memories:
            return
        self.association_graph[mem_id1].append((mem_id2, strength))
        self.association_graph[mem_id2].append((mem_id1, strength))

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

class PersistentCortexV6:
    def __init__(self, storage_dir: str, experts):
        self.storage_dir = storage_dir
        self.experts = experts  # 接收专家网络引用（包含身份专家）
        os.makedirs(storage_dir, exist_ok=True)
        self.index_file = os.path.join(storage_dir, "cortex_memory_index.json")

        # 三层记忆池
        self.short_term_memory: deque = deque(maxlen=20)
        self.long_term_memory: Dict[int, Dict] = {}
        self.permanent_memory: set = set()

        # 记忆索引
        self.index = MemoryIndex(config.dim)
        self.load_all()

    def _check_duplicate(self, clip_vec: torch.Tensor) -> Optional[int]:
        """
        语义去重检查
        返回重复的记忆ID，没有则返回None
        """
        results = self.index.vector_search(clip_vec, top_k=1)
        if results and results[0][1] > config.duplicate_threshold:
            return results[0][0]
        return None

    def store_detailed_memory(self, expert_name: str, sdr: torch.Tensor, clip_vec: torch.Tensor, content: str, metadata: Optional[Dict] = None):
        if metadata is None:
            metadata = {}

        # 第一步：去重检查
        duplicate_id = self._check_duplicate(clip_vec)
        if duplicate_id:
            logger.info(f"🔄 检测到语义重复记忆，更新原记忆 ID:{duplicate_id}")
            mem = self.index.get_memory(duplicate_id)
            mem['metadata']['last_accessed'] = datetime.datetime.now().isoformat()
            mem['metadata']['access_count'] = mem['metadata'].get('access_count', 0) + 1
            return duplicate_id

        # 补全元数据
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
            'source': metadata.get('source', '用户输入'),
            'hierarchy': metadata.get('hierarchy', '核心' if metadata.get('is_fact', False) else '关联'),
            'tags': metadata.get('tags', self._auto_extract_tags(content, expert_name)),
            'related_memories': []
        }

        # 写入索引
        mem_id = self.index.add_memory(sdr, clip_vec, content, full_metadata)

        # 传递 mem_id 给专家网络
        if expert_name in self.experts:
            expert = self.experts[expert_name]
            expert.hebbian_update(sdr, sdr, is_fact=full_metadata.get('is_fact', False))
            expert.add_memory(sdr, content, mem_id=mem_id, metadata=full_metadata)

        # 记忆分层
        self.short_term_memory.appendleft(mem_id)
        self.long_term_memory[mem_id] = self.index.get_memory(mem_id)

        # 自动建立关联
        self._auto_build_association(mem_id, expert_name, full_metadata['tags'])

        # 重要性检查
        if full_metadata['importance'] >= config.permanent_importance_threshold:
            self.mark_permanent(mem_id)

        logger.info(f"📝 记忆已存储 | ID:{mem_id} | 专家:{expert_name}")
        return mem_id

    def search_memories(self, query_vec: torch.Tensor, query_sdr: torch.Tensor, expert_name: Optional[str] = None, top_k: int = config.top_k, min_similarity: float = config.min_similarity) -> List[Tuple[int, float, str, Dict]]:
        # 1. 原有向量检索
        base_results = self.index.vector_search(query_vec, top_k=top_k * 50)
        
        # 2. 最低相似度过滤
        filtered_by_sim = []
        for mem_id, sim, mem in base_results:
            if sim < min_similarity:
                continue
            filtered_by_sim.append((mem_id, sim, mem))
        
        if not filtered_by_sim:
            return []

        # 3. 综合打分
        scored_all = []
        for mem_id, sim, mem in filtered_by_sim:
            meta = mem['metadata']
            final_score = (
                0.4 * sim
                + 0.3 * meta.get('importance', 0.5)
                + 0.2 * meta.get('activation', 0.5)
                + 0.1 * meta.get('recency', 0.5)
            )
            scored_all.append((mem_id, final_score, mem['content'], mem['metadata']))
        
        # 4. 专家网络联想检索（加分制）
        expert_boost = {}
        if expert_name and expert_name in self.experts:
            expert = self.experts[expert_name]
            expert_ret = expert.retrieve_multi_hop(query_sdr, hops=2, top_k=top_k)
            
            for score, content, meta, idx, mem_id in expert_ret:
                if mem_id is not None and score > 0.15:
                    expert_boost[mem_id] = expert_boost.get(mem_id, 0) + score * 0.5
                    logger.debug(f"🧠 专家联想加分: ID={mem_id}, +{score*0.5:.3f}")

        # 5. 融合专家分数
        final_results = []
        for mem_id, final_score, content, meta in scored_all:
            if mem_id in expert_boost:
                final_score += expert_boost[mem_id]
            final_results.append((mem_id, final_score, content, meta))
        
        final_results.sort(key=lambda x: -x[1])
        return final_results

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

        if meta['access_count'] >= 3 and mem_id in self.short_term_memory:
            self.short_term_memory.remove(mem_id)
            logger.info(f"🧠 记忆ID:{mem_id} 已从短期记忆转入长期记忆")

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
        """🔥 核心修复：新增身份专家自动标签提取"""
        tags = [expert_name]
        if expert_name == "身份":
            # 身份/人称/主人/名字相关标签
            if "你是谁" in content or "我是谁" in content:
                tags.append("身份认知")
            if "名字" in content or "叫" in content:
                tags.append("名字")
            if "主人" in content:
                tags.append("主人")
            if "邓尧" in content or "小白" in content:
                tags.append("专属身份")
            if "关系" in content:
                tags.append("伙伴关系")
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
        """身份记忆自动建立关联"""
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
        
        # 先写临时文件
        temp_index_file = self.index_file + ".tmp"
        self.index.save(temp_index_file)
        
        # 原子替换
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        os.rename(temp_index_file, self.index_file)
        
        # 保存记忆池状态
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
        
        # 保存所有专家权重（包含身份专家）
        for name, expert in self.experts.items():
            expert_path = os.path.join(self.storage_dir, f"expert_{name}.pt")
            expert.save_weights(expert_path)
        
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
        
        # 加载所有专家权重（包含身份专家）
        for name, expert in self.experts.items():
            expert_path = os.path.join(self.storage_dir, f"expert_{name}.pt")
            expert.load_weights(expert_path)
        
        self._compat_old_memories()
        logger.info(f"✅ 历史记忆加载完成 | 总记忆数:{len(self.index.memories)} | 永久记忆数:{len(self.permanent_memory)}")

    def _compat_old_memories(self):
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
        """🔥 核心修复：保存大脑状态（包含身份专家）供桌面显示器读取"""
        try:
            # 1. 总记忆数
            total_memories = len(self.index.memories)
            
            # 2. 🔥 五大脑区记忆数统计（新增身份）
            expert_counts = {
                "身份": 0,
                "概念": 0,
                "空间": 0,
                "抽象": 0,
                "视觉": 0
            }
            for expert_name in expert_counts.keys():
                expert_counts[expert_name] = len(self.index.get_by_expert(expert_name))
            
            # 3. 分层记忆
            short_term = len(self.short_term_memory)
            long_term = len(self.long_term_memory)
            permanent = len(self.permanent_memory)
            
            # 4. 计算稀疏度
            total_possible = total_memories * 10 if total_memories > 0 else 1
            actual_associations = sum(len(assoc) for assoc in self.index.association_graph.values())
            sparsity = 1.0 - (actual_associations / total_possible)
            sparsity = max(0.7, min(0.98, sparsity))
            
            # 🔥 五大脑区稀疏度
            expert_sparsity = {
                "身份": sparsity,
                "概念": sparsity,
                "空间": sparsity,
                "抽象": sparsity,
                "视觉": 1.0
            }

            state = {
                "total_memories": total_memories,
                "expert_distribution": expert_counts,
                "memory_layers": {
                    "短期记忆": short_term,
                    "长期记忆": long_term,
                    "永久记忆": permanent
                },
                "experts": {
                    "身份": {"突触稀疏度": expert_sparsity["身份"]},
                    "概念": {"突触稀疏度": expert_sparsity["概念"]},
                    "空间": {"突触稀疏度": expert_sparsity["空间"]},
                    "抽象": {"突触稀疏度": expert_sparsity["抽象"]},
                    "视觉": {"突触稀疏度": expert_sparsity["视觉"]}
                },
                "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "brain_state": "awake"
            }
            
            # 保存到项目根目录
            with open("brain_state.json", "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
                
            logger.info("💾 大脑状态已同步到桌面显示器")
            
        except Exception as e:
            logger.error(f"❌ 保存大脑状态失败: {e}")
            # 出错时保存有效文件
            with open("brain_state.json", "w", encoding="utf-8") as f:
                json.dump({
                    "error": f"后端: {str(e)[:15]}",
                    "total_memories": 0,
                    "expert_distribution": {"身份":0,"概念":0,"空间":0,"抽象":0,"视觉":0},
                    "memory_layers": {"短期记忆":0,"长期记忆":0,"永久记忆":0},
                    "experts": {
                        "身份":{"突触稀疏度":1.0},
                        "概念":{"突触稀疏度":1.0},
                        "空间":{"突触稀疏度":1.0},
                        "抽象":{"突触稀疏度":1.0},
                        "视觉":{"突触稀疏度":1.0}
                    },
                    "last_update": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "brain_state": "error"
                }, f, ensure_ascii=False, indent=2)