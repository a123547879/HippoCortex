import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import os
import logging
import json

from BrainConfig import config
from DynamicExpertV4 import DynamicExpert
from PersistentCortexV6 import PersistentCortexV6
from LearnableSparseEncoder import LearnableSparseEncoder
from HippocampusRouterV5 import HippocampusRouterV5

logger = logging.getLogger("AdvancedBrainV5")

class AdvancedBrainV6:
    def __init__(self, dim=1024, storage_dir="brain_v4_demo", ollama_model="nomic-embed-text"):
        """
        :param dim: 期望的 embedding 维度
        :param storage_dir: 存储目录
        :param ollama_model: Ollama 上的 embedding 模型名 (默认: nomic-embed-text)
        """
        self.dim = dim
        self.storage_dir = storage_dir
        self.ollama_model = ollama_model
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
        
        # 🔥 核心修复：固定五大脑区（强制包含身份专家，与路由完全对齐）
        self.expert_names = ["身份", "概念", "空间", "抽象", "视觉"]
        logger.info(f"🧠 五大脑区初始化完成: {self.expert_names}")
        
        # 初始化所有专家网络（包含身份专家）
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
        
        # 🔥 初始化海马体路由（支持身份专家）
        logger.info("🧭 初始化海马体路由...")
        self.hippocampus_router = HippocampusRouterV5(
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
        
        # 初始化专家原型（包含身份专家原型）
        if router_needs_init:
            logger.info("🧭 首次运行，初始化全专家原型（含身份认知）...")
            self.hippocampus_router._initialize_prototypes_with_embedding(self.embedding_model)
            self.hippocampus_router.save(router_path)
            logger.info("✅ 专家原型初始化并保存完成")
        
        # 初始化持久化皮层
        self.cortex = PersistentCortexV6(storage_dir, self.experts)
        
        # 执行每日记忆衰减
        self.cortex.decay_all_memories()

    def encode_text(self, text):
        """
        文本编码（失败抛出异常）
        """
        try:
            embedding = self.embedding_model.embed_query(text)
            clip_vec = torch.tensor(embedding, dtype=torch.float32)
            return clip_vec
        except Exception as e:
            logger.error(f"❌ 文本编码失败: {e}")
            raise RuntimeError(f"Ollama embedding 失败: {e}") from e

    def learn(self, text, force_expert=None):
        """学习一条知识（身份知识自动路由到身份专家）"""
        clip_vec = self.encode_text(text)
        clip_vec = F.normalize(clip_vec.detach().squeeze(), p=2, dim=-1)
        
        # 生成SDR
        sdr = self.sdr_encoder.encode(clip_vec.unsqueeze(0))
        
        # 路由决策（身份知识最高优先级）
        if force_expert is None:
            target_expert = self.hippocampus_router.route(clip_vec, text)
            self.hippocampus_router.online_learn(clip_vec, target_expert)
        else:
            target_expert = force_expert
        
        # 存储记忆
        self.cortex.store_detailed_memory(target_expert, sdr, clip_vec, text)
        logger.info(f"✅ 记忆已存入 【{target_expert}】 专家: {text[:30]}...")

    def recall_compositional(self, text, target_expert=None):
        """组合式记忆检索（优先检索身份专家）"""
        clip_vec = self.encode_text(text)
        clip_vec = F.normalize(clip_vec.detach().squeeze(), p=2, dim=-1)
        
        query_sdr = self.sdr_encoder.encode(clip_vec.unsqueeze(0))
        
        # 身份知识优先路由
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
        
        # 兜底：无结果则全专家搜索
        if not results and target_expert is not None:
            logger.info(f"⚠️  【{target_expert}】 无结果，全专家检索...")
            results = self.cortex.search_memories(
                clip_vec,
                query_sdr,
                expert_name=None,
                top_k=config.top_k,
                min_similarity=config.min_similarity - 0.05
            )
        
        # 打印检索结果
        print(f"  找到 {len(results)} 条候选记忆")
        for i, (mem_id, sim, content, meta) in enumerate(results[:5]):
            print(f"    候选 {i+1}: 得分={sim:.3f}, 专家={meta.get('expert', '?')}, 内容={content[:40]}...")
        
        if not results:
            return [], None
        
        # 新奇度检测
        memories = [content for _, _, content, _ in results]
        best_sim = results[0][1]
        novelty_score = 1.0 - best_sim
        
        if novelty_score > 0.65:
            self.learn(text)
            return [], None
        
        # 更新访问次数
        for mem_id, _, _, _ in results:
            self.cortex.increment_access_count(mem_id)
        
        return memories, {'similarity': best_sim}

    def sleep_consolidate_all(self, epochs=3):
        """全脑睡眠巩固（包含身份专家独立修剪）"""
        logger.info("\n🌙 大脑开始睡眠巩固（五脑区同步）...")
        for name, expert in self.experts.items():
            expert.sleep_consolidate(epochs=epochs)
        logger.info("✅ 全脑睡眠巩固完成！")

    def save_all(self):
        """保存所有大脑数据"""
        # 保存编码器
        sdr_encoder_path = os.path.join(self.storage_dir, "sdr_encoder.pt")
        self.sdr_encoder.save(sdr_encoder_path)
        
        # 保存路由
        router_path = os.path.join(self.storage_dir, "hippocampus_router.pt")
        self.hippocampus_router.save(router_path)
        
        # 保存皮层记忆
        self.cortex.save_all()
        self.cortex.save_brain_state()
        
        logger.info("✅ 所有大脑数据已安全保存！")

    def get_brain_status(self):
        """获取大脑状态（显示身份专家数据）"""
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
        }
        
        # 🔥 显示五大脑区完整状态（含身份专家）
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
        """
        🔥 核心功能：重新分配记忆（自动修正身份记忆错分问题）
        把所有「你/我/主人/名字」记忆迁移到身份专家
        """
        logger.info("🔄 开始全脑记忆重新分配（修正身份记忆错分）...")
        
        total_redis = 0
        for mem_id, mem in list(self.cortex.index.memories.items()):
            content = mem['content']
            old_expert = mem['metadata']['expert']
            
            # 用最新路由重新分配
            clip_vec = mem['clip_vec']
            new_expert = self.hippocampus_router.route(clip_vec, content)
            
            if new_expert != old_expert:
                # 更新元数据
                mem['metadata']['expert'] = new_expert
                
                # 迁移专家记忆
                if old_expert in self.experts:
                    self.experts[old_expert].delete_memory(mem_id)
                if new_expert in self.experts:
                    self.experts[new_expert].add_memory(
                        mem['sdr'], content, mem_id=mem_id, metadata=mem['metadata']
                    )
                
                # 更新索引
                if old_expert in self.cortex.index.expert_index and mem_id in self.cortex.index.expert_index[old_expert]:
                    self.cortex.index.expert_index[old_expert].remove(mem_id)
                self.cortex.index.expert_index[new_expert].append(mem_id)
                
                total_redis += 1
                logger.debug(f"   记忆迁移: {old_expert} → {new_expert} | {content[:20]}...")
        
        logger.info(f"✅ 记忆重分配完成！共修正 {total_redis} 条错分记忆")
        return total_redis