import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import os
import logging
import json
from typing import List, Dict

from BrainConfig import config
from DynamicExpertV5 import DynamicExpert
from PersistentCortexV8 import PersistentCortexV8
from LearnableSparseEncoder import LearnableSparseEncoder
from HippocampusRouterV6 import HippocampusRouterV6

logger = logging.getLogger("AdvancedBrainV5")

class AdvancedBrainV8:
    def __init__(self, dim=1024, storage_dir="brain_v4_demo", ollama_model="nomic-embed-text", llm=None, kg_enabled: bool = True):
        """
        :param dim: 期望的 embedding 维度
        :param storage_dir: 存储目录
        :param ollama_model: Ollama 上的 embedding 模型名 (默认: nomic-embed-text)
        :param llm: LLM实例，用于知识图谱实体提取（必传）
        :param kg_enabled: 是否启用知识图谱（默认True，可关闭以提升性能）
        """
        self.dim = dim
        self.storage_dir = storage_dir
        self.ollama_model = ollama_model
        self.llm = llm
        self.kg_enabled = kg_enabled
        os.makedirs(storage_dir, exist_ok=True)
        
        # 改用 LangChain + Ollama 调用 Embedding
        logger.info(f"🤖 正在连接 Ollama 并加载模型: {ollama_model}...")
        try:
            from langchain_ollama import OllamaEmbeddings
            self.embedding_model = OllamaEmbeddings(model=ollama_model)
            # 测试连接
            test_emb = self.embedding_model.embed_query("test")
            self.actual_dim = len(test_emb)
            logger.info(f"✅ Ollama 连接成功！模型维度: {self.actual_dim}")
            
            if self.actual_dim != self.dim:
                logger.warning(f"⚠️  模型维度与期望不匹配: 期望 {self.dim}, 实际 {self.actual_dim}")
                self.dim = self.actual_dim
        except ImportError:
            logger.error("❌ 未安装 langchain-ollama，请运行: pip install langchain-ollama")
            raise
        except Exception as e:
            logger.error(f"❌ Ollama 连接失败: {e}")
            logger.error("   请确保: 1) Ollama 已安装并运行 2) 模型已拉取 (ollama pull {ollama_model})")
            raise
        
        # 固定五大脑区（强制包含身份专家，与路由完全对齐）
        self.expert_names = ["身份", "概念", "空间", "抽象", "视觉"]
        logger.info(f"🧠 五大脑区初始化完成: {self.expert_names}")
        
        # 初始化所有专家网络
        logger.info("🧠 初始化专家网络...")
        self.experts = {}
        for name in self.expert_names:
            self.experts[name] = DynamicExpert(
                name, 
                initial_dim=config.sdr_dim, 
                max_dim=config.max_expert_dim,
                active_size=config.sdr_active_size
            )
        
        # 初始化稀疏编码器
        logger.info("🔄 初始化稀疏编码器...")
        self.sdr_encoder = LearnableSparseEncoder(
            input_dim=self.dim,
            sdr_dim=config.sdr_dim, 
            active_size=config.sdr_active_size
        )
        sdr_encoder_path = os.path.join(storage_dir, "sdr_encoder.pt")
        if os.path.exists(sdr_encoder_path):
            try:
                self.sdr_encoder.load(sdr_encoder_path)
                logger.info("✅ 历史稀疏编码器加载完成")
            except Exception as e:
                logger.warning(f"⚠️  稀疏编码器加载失败，初始化新编码器: {e}")
        else:
            logger.info("🆕 初始化新的稀疏编码器")
        
        # 初始化海马体路由
        logger.info("🧭 初始化海马体路由...")
        self.hippocampus_router = HippocampusRouterV6(
            input_dim=self.dim,
            expert_names=self.expert_names
        )
        router_path = os.path.join(storage_dir, "hippocampus_router.pt")
        
        router_needs_init = False
        if os.path.exists(router_path):
            try:
                self.hippocampus_router.load(router_path)
                logger.info("✅ 海马体路由加载完成")
                if not self.hippocampus_router._prototypes_initialized:
                    router_needs_init = True
            except Exception as e:
                logger.warning(f"⚠️  海马体路由加载失败: {e}")
                router_needs_init = True
        else:
            router_needs_init = True
        
        # 初始化专家原型
        if router_needs_init:
            logger.info("🧭 首次运行，初始化全专家原型（含身份认知）...")
            self.hippocampus_router._initialize_prototypes_with_embedding(self.embedding_model)
            self.hippocampus_router.save(router_path)
            logger.info("✅ 专家原型初始化并保存完成")
        
        self.cortex = PersistentCortexV8(storage_dir, self.experts, self.llm, kg_enabled=kg_enabled)
        
        # 执行每日记忆衰减
        self.cortex.decay_all_memories()

    # ==============================================
    # 工具方法
    # ==============================================
    def get_expert(self, expert_name: str) -> DynamicExpert:
        """获取指定名称的专家脑区"""
        return self.experts.get(expert_name)

    # ==============================================
    def think(self, text: str, steps: int = 2, topk: int = 10) -> Dict:
        try:
            # ====================== 1. 文本编码与路由 ======================
            clip_vec = self.encode_text(text)
            # 规范归一化：保持维度，避免squeeze导致批量维度丢失
            clip_vec = F.normalize(clip_vec, p=2, dim=-1)
            query_sdr = self.sdr_encoder.encode(clip_vec)
            target_expert = self.hippocampus_router.route(clip_vec, text)

            # ====================== 2. 检索种子记忆 ======================
            # 直接取top3，无需冗余切片
            raw_results = self.cortex.search_memories(
                clip_vec, query_sdr,
                expert_name=target_expert,
                top_k=3,
                min_similarity=0.3
            )

            # 无检索结果：统一返回格式
            if not raw_results:
                return {
                    "thought_chain": "无候选记忆",
                    "core_ideas": [],
                    "activated_memories": [],
                    "seed_memories": [],
                    "associated_memories": [],
                    "expert": target_expert,
                    "activation_strength": 0.0,
                    "error": None
                }

            # ====================== 3. 收集有效种子记忆（SDR） ======================
            seed_sdrs: List[torch.Tensor] = []
            seed_memories: List[dict] = []
            for mem_id, sim, content, meta in raw_results:
                mem = self.cortex.index.get_memory(mem_id)
                if mem["metadata"].get("is_obsolete", False):
                    continue
                if mem and "sdr" in mem:
                    # 保证所有SDR张量设备一致（CPU/GPU）
                    seed_sdrs.append(mem['sdr'].to(clip_vec.device))
                    seed_memories.append(mem)

            # 无有效种子SDR：直接返回
            if not seed_sdrs:
                return {
                    "thought_chain": "无有效种子记忆",
                    "core_ideas": [],
                    "activated_memories": [],
                    "seed_memories": [],
                    "associated_memories": [],
                    "expert": target_expert,
                    "activation_strength": 0.0,
                    "error": None
                }

            # ====================== 4. 获取专家网络 ======================
            expert = self.experts.get(target_expert)
            if not expert:
                logger.warning(f"未找到专家模块: {target_expert}")
                all_memories = seed_memories
                activated_memories = []
                propagated = None
            else:
                # ====================== 5. 神经激活传播（核心逻辑） ======================
                # 堆叠种子SDR并计算平均激活
                initial_activation = torch.stack(seed_sdrs).mean(dim=0, keepdim=True)
                
                # 执行神经传播
                propagated = expert.forward(initial_activation, steps=steps, top_k=60)
                
                # 检索联想记忆
                associate_results = expert.retrieve(propagated, top_k=topk, steps=1)
                
                # ====================== 6. 过滤联想记忆（去重） ======================
                seed_ids = {m["id"] for m in seed_memories}
                activated_memories = []
                for score, content, meta, idx, mem_id in associate_results:
                    if mem_id and mem_id not in seed_ids:
                        mem = self.cortex.index.get_memory(mem_id)
                        if mem:
                            activated_memories.append(mem)
                
                # 合并种子记忆 + 联想记忆
                all_memories = seed_memories + activated_memories

            # ====================== 7. 生成思考结果 ======================
            thought_chain = self._build_thought_chain(all_memories)
            core_ideas = self._extract_core_ideas(all_memories)
            
            # 计算激活强度：规范判断，避免变量未定义
            activation_strength = propagated.norm().item() if propagated is not None else 0.0

            # ====================== 8. 统一返回结果 ======================
            return {
                "thought_chain": thought_chain,
                "core_ideas": core_ideas,
                "activated_memories": [m["content"] for m in all_memories],
                "seed_memories": [m["content"] for m in seed_memories],
                "associated_memories": [m["content"] for m in activated_memories],
                "expert": target_expert,
                "activation_strength": activation_strength,
                "error": None
            }

        except Exception as e:
            # 打印错误堆栈，方便调试
            logger.error(f"❌ 思考过程出错: {e}", exc_info=True)
            return {
                "thought_chain": "思考失败",
                "core_ideas": [],
                "activated_memories": [],
                "seed_memories": [],
                "associated_memories": [],
                "expert": None,
                "activation_strength": 0.0,
                "error": f"思考失败: {str(e)}"
            }
        
    # ==============================================
    # 🔥 终极修复：思考辅助方法（无警告+稳定匹配）
    # ==============================================
    def _get_retrieved_memory_vectors(self, memories: List[str], expert_name: str) -> List[torch.Tensor]:
        """✅ 100%消除张量警告 + 正确提取SDR向量"""
        vectors = []
        for mem in self.cortex.index.memories.values():
            if mem["content"] in memories and mem["metadata"].get("expert") == expert_name:
                sdr_data = mem["sdr"]
                # 兼容所有数据类型，彻底消除PyTorch警告
                if isinstance(sdr_data, torch.Tensor):
                    sdr_tensor = sdr_data.detach().clone()
                else:
                    sdr_tensor = torch.as_tensor(sdr_data, dtype=torch.float32)
                vectors.append(sdr_tensor)
        return vectors

    def _search_activation(self, expert: DynamicExpert, activation: torch.Tensor, topk: int = 5) -> List[int]:
        """
        ✅ 正确修复：DynamicExpert没有index，改用SDR相似度匹配记忆
        从专家的sdr_list中找和激活态最相似的SDR，返回对应记忆ID
        """
        try:
            # 专家内部存储了所有记忆的SDR，直接计算相似度
            sim_scores = []
            for idx, sdr in enumerate(expert.sdr_list):
                # 稀疏编码余弦相似度
                sim = F.cosine_similarity(activation, sdr.unsqueeze(0), dim=-1).item()
                sim_scores.append((idx, sim))
            
            # 按相似度排序，取topk
            sim_scores.sort(key=lambda x: -x[1])
            top_indices = [idx for idx, sim in sim_scores[:topk]]
            return top_indices
        except Exception as e:
            logger.warning(f"⚠️ 激活搜索失败: {e}")
            return []

    def _build_thought_chain(self, memories: List[Dict]) -> str:
        """构建联想思路链"""
        if not memories:
            return "无关联记忆"
        contents = [m["content"][:35] + "..." if len(m["content"]) > 35 else m["content"] for m in memories]
        return " → ".join(contents)

    def _extract_core_ideas(self, memories: List[Dict]) -> List[str]:
        """提取思考的核心概念"""
        ideas = []
        for mem in memories:
            content = mem["content"]
            if "：" in content:
                ideas.append(content.split("：")[1][:15])
            else:
                ideas.append(content[:15])
        return list(set(ideas))

    def _get_identity_core_memory(self) -> str:
        """获取核心身份记忆"""
        id_memories = [m["content"] for m in self.cortex.index.memories.values() if m["metadata"].get("expert") == "身份"]
        return "\n".join(id_memories[:3]) if id_memories else "我是小白"
    
    # ==============================================
    # 原有功能（完全保留）
    # ==============================================
    def encode_text(self, text):
        try:
            embedding = self.embedding_model.embed_query(text)
            clip_vec = torch.as_tensor(embedding, dtype=torch.float32)
            return clip_vec
        except Exception as e:
            logger.error(f"❌ 文本编码失败: {e}")
            raise RuntimeError(f"Ollama embedding 失败: {e}") from e

    def learn(self, text, force_expert=None):
        clip_vec = self.encode_text(text)
        clip_vec = F.normalize(clip_vec.detach().squeeze(), p=2, dim=-1)
        sdr = self.sdr_encoder.encode(clip_vec.unsqueeze(0))
        
        if force_expert is None:
            target_expert = self.hippocampus_router.route(clip_vec, text)
            self.hippocampus_router.online_learn(clip_vec, target_expert)
        else:
            target_expert = force_expert
        
        self.cortex.store_detailed_memory(target_expert, sdr, clip_vec, text)
        logger.info(f"✅ 记忆已存入 【{target_expert}】 专家: {text[:30]}...")

    def batch_learn(self, texts: List[str], force_experts: List[str] = None):
        if force_experts is None:
            force_experts = [None for _ in texts]
        
        batch_clip_vecs = []
        batch_sdrs = []
        batch_experts = []
        
        for text, force_expert in zip(texts, force_experts):
            try:
                clip_vec = self.encode_text(text)
                clip_vec = F.normalize(clip_vec.detach().squeeze(), p=2, dim=-1)
                sdr = self.sdr_encoder.encode(clip_vec.unsqueeze(0))
                
                if force_expert is None:
                    target_expert = self.hippocampus_router.route(clip_vec, text)
                    self.hippocampus_router.online_learn(clip_vec, target_expert)
                else:
                    target_expert = force_expert
                
                batch_clip_vecs.append(clip_vec)
                batch_sdrs.append(sdr)
                batch_experts.append(target_expert)
            except Exception as e:
                logger.error(f"❌ 预处理失败: {text[:50]}... 错误: {e}")
        
        if batch_clip_vecs:
            self.cortex.batch_store_detailed_memories(
                batch_experts,
                batch_sdrs,
                batch_clip_vecs,
                texts
            )

    def recall_compositional(self, text, target_expert=None):
        clip_vec = self.encode_text(text)
        clip_vec = F.normalize(clip_vec.detach().squeeze(), p=2, dim=-1)
        query_sdr = self.sdr_encoder.encode(clip_vec.unsqueeze(0))
        
        if target_expert is None:
            target_expert = self.hippocampus_router.route(clip_vec, text)
        
        logger.info(f"🔍 在 【{target_expert if target_expert else '全专家'}】 检索记忆...")
        results = self.cortex.search_memories(
            clip_vec,
            query_sdr,
            expert_name=target_expert,
            top_k=config.top_k,
            min_similarity=config.min_similarity
        )
        
        if not results and target_expert is not None:
            logger.info(f"⚠️  【{target_expert}】 无结果，全专家检索...")
            results = self.cortex.search_memories(
                clip_vec,
                query_sdr,
                expert_name=None,
                top_k=config.top_k,
                min_similarity=config.min_similarity - 0.05
            )
        
        print(f"  找到 {len(results)} 条候选记忆")
        for i, (mem_id, sim, content, meta) in enumerate(results[:5]):
            print(f"    候选 {i+1}: 得分={sim:.3f}, 专家={meta.get('expert', '?')}, 内容={content[:40]}...")
        
        if not results:
            return [], None
        
        memories = [content for _, _, content, _ in results]
        best_sim = results[0][1]
        novelty_score = 1.0 - best_sim
        
        if novelty_score > 0.65:
            self.learn(text)
            return [], None
        
        for mem_id, _, _, _ in results:
            self.cortex.increment_access_count(mem_id)
        
        return memories, {'similarity': best_sim}

    def sleep_consolidate_all(self, epochs=3):
        logger.info("\n🌙 大脑开始睡眠巩固（五脑区同步+知识图谱）...")
        for name, expert in self.experts.items():
            expert.sleep_consolidate(epochs=epochs)
        self.cortex.sleep_consolidate_all(epochs=epochs)
        logger.info("✅ 全脑睡眠巩固完成！")

    def save_all(self):
        sdr_encoder_path = os.path.join(self.storage_dir, "sdr_encoder.pt")
        self.sdr_encoder.save(sdr_encoder_path)
        
        router_path = os.path.join(self.storage_dir, "hippocampus_router.pt")
        self.hippocampus_router.save(router_path)
        
        self.cortex.save_all()
        self.cortex.save_brain_state()
        
        logger.info("✅ 所有大脑数据已安全保存！")

    def get_brain_status(self):
        total_memories = len(self.cortex.index.memories)
        
        expert_counts = defaultdict(int)
        expert_access = defaultdict(list)
        expert_sparsity = {}
        
        for mem in self.cortex.index.memories.values():
            expert = mem['metadata'].get('expert', '未知')
            expert_counts[expert] += 1
            expert_access[expert].append(mem['metadata'].get('access_count', 0))
        
        for name in self.expert_names:
            expert_sparsity[name] = self.experts[name].get_sparsity()
        
        status = {
            "total_memories": total_memories,
            "ollama_model": self.ollama_model,
            "embedding_dim": self.dim,
            "expert_distribution": {},
            "experts": {},
            "kg_enabled": self.kg_enabled
        }
        
        for name in self.expert_names:
            count = expert_counts.get(name, 0)
            access_list = expert_access.get(name, [0])
            avg_access = np.mean(access_list) if access_list else 0
            sparsity = expert_sparsity.get(name, 0.0)
            
            status["expert_distribution"][name] = count
            status["experts"][name] = {
                "神经元": self.experts[name].dim,
                "记忆数": count,
                "平均访问": round(avg_access, 2),
                "突触稀疏度": round(sparsity, 4)
            }
        
        return status

    def redistribute_memories(self):
        logger.info("🔄 开始全脑记忆重新分配（修正身份记忆错分）...")
        total_redis = 0
        for mem_id, mem in list(self.cortex.index.memories.items()):
            content = mem['content']
            old_expert = mem['metadata']['expert']
            clip_vec = mem['clip_vec']
            new_expert = self.hippocampus_router.route(clip_vec, content)
            
            if new_expert != old_expert:
                mem['metadata']['expert'] = new_expert
                if old_expert in self.experts:
                    self.experts[old_expert].delete_memory(mem_id)
                if new_expert in self.experts:
                    self.experts[new_expert].add_memory(
                        mem['sdr'], content, mem_id=mem_id, metadata=mem['metadata']
                    )
                if old_expert in self.cortex.index.expert_index and mem_id in self.cortex.index.expert_index[old_expert]:
                    self.cortex.index.expert_index[old_expert].remove(mem_id)
                self.cortex.index.expert_index[new_expert].append(mem_id)
                total_redis += 1
                logger.debug(f"   记忆迁移: {old_expert} → {new_expert} | {content[:20]}...")
        logger.info(f"✅ 记忆重分配完成！共修正 {total_redis} 条错分记忆")
        return total_redis

    def add_important_entity(self, entity_name: str):
        self.cortex.add_important_entity(entity_name)

    def remove_important_entity(self, entity_name: str):
        self.cortex.remove_important_entity(entity_name)

    def list_important_entities(self) -> list:
        return self.cortex.list_important_entities()

    def enable_kg(self):
        self.kg_enabled = True
        self.cortex.kg_enabled = True
        logger.info("✅ 知识图谱已启用")

    def disable_kg(self):
        self.kg_enabled = False
        self.cortex.kg_enabled = False
        logger.info("✅ 知识图谱已禁用（性能模式）")