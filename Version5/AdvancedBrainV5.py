import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import os
import logging

from BrainConfig import config
from DynamicExpertV3 import DynamicExpert
from PersistentCortexV5 import PersistentCortexV5
from LearnableSparseEncoder import LearnableSparseEncoder
from HippocampusRouterV4 import HippocampusRouterV4

logger = logging.getLogger("AdvancedBrainV5")

class AdvancedBrainV5:
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
            # 测试一下连接
            test_emb = self.embedding_model.embed_query("test")
            self.actual_dim = len(test_emb)
            logger.info(f"✅ Ollama 连接成功！模型维度: {self.actual_dim}")
            
            if self.actual_dim != self.dim:
                logger.warning(f"⚠️  模型维度与期望不匹配: 期望 {self.dim}, 实际 {self.actual_dim}")
                logger.warning(f"   将自动使用实际维度 {self.actual_dim}")
                self.dim = self.actual_dim
        except ImportError:
            logger.error("❌ 未安装 langchain-ollama，请运行: pip install langchain-ollama")
            raise
        except Exception as e:
            logger.error(f"❌ Ollama 连接失败: {e}")
            logger.error("   请确保: 1) Ollama 已安装并运行 2) 模型已拉取 (ollama pull {ollama_model})")
            raise
        
        # 初始化专家网络
        logger.info("🧠 初始化专家网络...")
        self.expert_names = config.expert_names
        self.experts = {}
        for name in self.expert_names:
            self.experts[name] = DynamicExpert(
                name, 
                initial_dim=config.sdr_dim, 
                max_dim=config.max_expert_dim,
                active_size=config.sdr_active_size
            )
        
        # 初始化稀疏编码器 (使用实际维度)
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
                logger.warning(f"⚠️  历史稀疏编码器加载失败，初始化新编码器: {e}")
        else:
            logger.info("🆕 初始化新的稀疏编码器")
        
        # 初始化海马体路由
        logger.info("🧭 初始化海马体路由...")
        self.hippocampus_router = HippocampusRouterV4(
            input_dim=self.dim,
            expert_names=self.expert_names
        )
        router_path = os.path.join(storage_dir, "hippocampus_router.pt")
        
        # 🔥 修复：先加载路由，再检查是否需要初始化原型
        router_needs_init = False
        if os.path.exists(router_path):
            try:
                self.hippocampus_router.load(router_path)
                logger.info("✅ 海马体路由加载完成")
                # 检查原型是否已初始化
                if not self.hippocampus_router._prototypes_initialized:
                    router_needs_init = True
            except Exception as e:
                logger.warning(f"⚠️  海马体路由加载失败: {e}")
                router_needs_init = True
        else:
            router_needs_init = True
        
        # 🔥 修复：如果需要，初始化专家原型
        if router_needs_init:
            logger.info("🧭 首次运行，正在初始化专家原型...")
            self.hippocampus_router._initialize_prototypes_with_embedding(self.embedding_model)
            # 保存初始化后的路由
            self.hippocampus_router.save(router_path)
            logger.info("✅ 专家原型初始化并保存完成")
        
        # 初始化持久化皮层
        self.cortex = PersistentCortexV5(storage_dir, self.experts)
        
        # 执行每日记忆衰减
        self.cortex.decay_all_memories()

    def encode_text(self, text):
        """
        失败即抛异常，绝不返回随机向量
        """
        try:
            # 调用 Ollama 获取 embedding
            embedding = self.embedding_model.embed_query(text)
            clip_vec = torch.tensor(embedding, dtype=torch.float32)
            
            return clip_vec
        except Exception as e:
            logger.error(f"❌ 文本编码失败: {e}")
            # 直接抛异常，中断学习流程
            raise RuntimeError(f"Ollama embedding 失败: {e}") from e

    def learn(self, text, force_expert=None):
        """学习一条知识"""
        # 编码文本
        clip_vec = self.encode_text(text)
        clip_vec = F.normalize(clip_vec.detach().squeeze(), p=2, dim=-1)
        
        # 生成SDR
        sdr = self.sdr_encoder.encode(clip_vec.unsqueeze(0))
        
        # 使用海马体路由
        if force_expert is None:
            target_expert = self.hippocampus_router.route(clip_vec, text)
            # 在线学习路由
            self.hippocampus_router.online_learn(clip_vec, target_expert)
        else:
            target_expert = force_expert
        
        # 存储记忆
        self.cortex.store_detailed_memory(target_expert, sdr, clip_vec, text)

    def recall_compositional(self, text, target_expert=None):
        """组合式记忆检索"""
        # 编码文本
        clip_vec = self.encode_text(text)
        clip_vec = F.normalize(clip_vec.detach().squeeze(), p=2, dim=-1)
        
        # 生成SDR
        query_sdr = self.sdr_encoder.encode(clip_vec.unsqueeze(0))
        
        # 使用海马体路由
        if target_expert is None:
            target_expert = self.hippocampus_router.route(clip_vec, text)
        
        # 先搜指定专家，搜不到自动全专家搜索
        logger.info(f"🔍 先在 [{target_expert if target_expert else '全专家'}] 分区寻找...")
        results = self.cortex.search_memories(
            clip_vec,
            query_sdr,
            expert_name=target_expert,
            top_k=config.top_k,
            min_similarity=config.min_similarity
        )
        
        # 核心兜底：如果指定专家没找到，自动放宽到全专家
        if not results and target_expert is not None:
            logger.info(f"⚠️  [{target_expert}] 分区未找到，放宽到全专家搜索...")
            results = self.cortex.search_memories(
                clip_vec,
                query_sdr,
                expert_name=None,
                top_k=config.top_k,
                min_similarity=config.min_similarity - 0.05
            )
        
        print(f"  找到 {len(results)} 条候选记忆")
        if len(results) > 5:
            for i, (mem_id, sim, content, meta) in enumerate(results[:5]):
                print(f"    候选 {i+1}: 综合分={sim:.3f}, 专家={meta.get('expert', '?')}, 内容={content[:40]}...")
        else:
            for i, (mem_id, sim, content, meta) in enumerate(results):
                print(f"    候选 {i+1}: 综合分={sim:.3f}, 专家={meta.get('expert', '?')}, 内容={content[:40]}...")
        
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
        """所有专家进行睡眠巩固"""
        logger.info("\n🌙 大脑开始睡眠巩固...")
        for name, expert in self.experts.items():
            expert.sleep_consolidate(epochs=epochs)
        logger.info("✅ 大脑睡眠巩固完成！")

    def save_all(self):
        """保存所有大脑数据"""
        # 保存稀疏编码器
        sdr_encoder_path = os.path.join(self.storage_dir, "sdr_encoder.pt")
        self.sdr_encoder.save(sdr_encoder_path)
        
        # 保存海马体路由
        router_path = os.path.join(self.storage_dir, "hippocampus_router.pt")
        self.hippocampus_router.save(router_path)
        
        # 保存皮层记忆
        self.cortex.save_all()
        
        logger.info("✅ 所有大脑数据已安全保存！")

    def get_brain_status(self):
        """获取大脑状态"""
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
            "total_samples": total_memories,
            "ollama_model": self.ollama_model,
            "embedding_dim": self.dim,
            "expert_distribution": {},
            "experts": {},
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
        """
        🔥 新增：重新分配现有记忆到正确的专家
        解决历史数据路由错误的问题
        """
        logger.info("🔄 开始重新分配现有记忆...")
        
        total_redis = 0
        for mem_id, mem in list(self.cortex.index.memories.items()):
            content = mem['content']
            old_expert = mem['metadata']['expert']
            
            # 用当前的路由重新判断专家
            clip_vec = mem['clip_vec']
            new_expert = self.hippocampus_router.route(clip_vec, content)
            
            if new_expert != old_expert:
                # 更新元数据
                mem['metadata']['expert'] = new_expert
                
                # 从旧专家删除
                if old_expert in self.experts:
                    self.experts[old_expert].delete_memory(mem_id)
                
                # 添加到新专家
                if new_expert in self.experts:
                    self.experts[new_expert].add_memory(
                        mem['sdr'], content, mem_id=mem_id, metadata=mem['metadata']
                    )
                
                # 更新索引
                if old_expert in self.cortex.index.expert_index:
                    if mem_id in self.cortex.index.expert_index[old_expert]:
                        self.cortex.index.expert_index[old_expert].remove(mem_id)
                self.cortex.index.expert_index[new_expert].append(mem_id)
                
                total_redis += 1
                logger.debug(f"   记忆ID:{mem_id} 从 [{old_expert}] 移动到 [{new_expert}]")
        
        logger.info(f"✅ 记忆重新分配完成，共移动 {total_redis} 条记忆")
        return total_redis