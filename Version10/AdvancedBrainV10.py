import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import os
import logging
import json
from typing import List, Dict

from BrainConfig import config
from DynamicExpertV6 import DynamicExpert
from PersistentCortexV9 import PersistentCortexV9
from LearnableSparseEncoder import LearnableSparseEncoder
from HippocampusRouterV7 import HippocampusRouterV7  # ✅ 修复1：统一导入 V7
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("AdvancedBrainV5")

class AdvancedBrainV10:
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
        
        # ====================== 🔥 核心修改：每个专家专属稀疏编码器 ======================
        logger.info("🔄 初始化专家专属稀疏编码器...")
        self.sdr_encoders = {}  # 从全局一个 → 每个专家一个
        for name in self.expert_names:
            self.sdr_encoders[name] = LearnableSparseEncoder(
                input_dim=self.dim,
                sdr_dim=config.sdr_dim, 
                active_size=config.sdr_active_size,
                expert_name=name  # 🔥 关键：传入专家名称，加载差异化配置
            )
            
            # 加载每个专家的历史编码器
            encoder_path = os.path.join(storage_dir, f"sdr_encoder_{name}.pt")
            if os.path.exists(encoder_path):
                try:
                    self.sdr_encoders[name].load(encoder_path)
                    logger.info(f"✅ [{name}] 专家历史稀疏编码器加载完成")
                except Exception as e:
                    logger.warning(f"⚠️  [{name}] 专家稀疏编码器加载失败，初始化新编码器: {e}")
            else:
                logger.info(f"🆕 初始化 [{name}] 专家新的稀疏编码器")
        
        # 初始化海马体路由
        logger.info("🧭 初始化海马体路由...")
        self.hippocampus_router = HippocampusRouterV7(  # ✅ 修复2：统一使用 V7
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
        
        self.cortex = PersistentCortexV9(storage_dir, self.experts, embedding_model=self.embedding_model, llm= self.llm, kg_enabled=kg_enabled)
        
        # 执行每日记忆衰减
        self.cortex.decay_all_memories()

    # ==============================================
    # 工具方法
    # ==============================================
    def get_expert(self, expert_name: str) -> DynamicExpert:
        """获取指定名称的专家脑区"""
        return self.experts.get(expert_name)

    def think(self, text: str, steps: int = 2, topk: int = 10) -> Dict:
        try:
            # ====================== 全局常量：类脑常识边界阈值（关键！） ======================
            CONTEXT_SIM_THRESHOLD = 0.40
            SIMILARITY_DEBUG = True
            
            # ====================== 1. 文本编码与路由 ======================
            clip_vec = self.encode_text(text)
            clip_vec = F.normalize(clip_vec, p=2, dim=-1)
            target_expert = self.hippocampus_router.route(clip_vec, text)
            
            # ====================== 🔥 核心修改：调用对应专家的专属编码器 ======================
            sdr_encoder = self.sdr_encoders.get(target_expert, self.sdr_encoders["概念"])
            query_sdr = sdr_encoder.encode(clip_vec)

            # ====================== 2. 检索种子记忆 ======================
            raw_results = self.cortex.search_memories(
                clip_vec, query_sdr,
                expert_name=target_expert,
                top_k=3,
                min_similarity=0.3
            )

            # ====================== 🔥 核心3：路由正反馈（检索到记忆说明路由正确） ======================
            if raw_results:
                self.hippocampus_router.online_learn(clip_vec, target_expert)

            if not raw_results:
                return {
                    "thought_chain": "无候选记忆",
                    "core_ideas": [],
                    "activated_memories": [],
                    "seed_memories": [],
                    "associated_memories": [],
                    "expert": target_expert,
                    "activation_strength": 0.0,
                    "predicted_memory": None,
                    "prediction_error": 0.0,
                    "similarity_trace": [],
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
                    seed_sdrs.append(mem['sdr'].to(clip_vec.device))
                    seed_memories.append(mem)

            if not seed_sdrs:
                return {
                    "thought_chain": "无有效种子记忆",
                    "core_ideas": [],
                    "activated_memories": [],
                    "seed_memories": [],
                    "associated_memories": [],
                    "expert": target_expert,
                    "activation_strength": 0.0,
                    "predicted_memory": None,
                    "prediction_error": 0.0,
                    "similarity_trace": [],
                    "error": None
                }

            # ====================== 4. 获取专家网络 ======================
            expert = self.experts.get(target_expert)
            predicted_memory = None
            prediction_error = 0.0
            propagated = None
            similarity_trace = []
            
            if expert:
                # ====================== 5. 神经激活传播 + SNN脉冲时序 ======================
                initial_activation = torch.stack(seed_sdrs).mean(dim=0, keepdim=True)
                propagated = expert.forward(initial_activation, steps=steps, top_k=60)

                # ====================== 🔥 预测编码（保留原功能） ======================
                pred_sdr = expert.predict_next_sdr(propagated.detach())
                prediction_error = expert.update_prediction(pred_sdr, propagated.detach())
                pred_results = expert.retrieve(pred_sdr, top_k=1)
                if pred_results:
                    _, pred_content, _, _, pred_mem_id = pred_results[0]
                    predicted_memory = pred_content

                # ====================== 6. ✅ 第一重保险：关联记忆收集 + 实时停止 ======================
                seed_ids = {m["id"] for m in seed_memories}
                activated_memories = []
                associate_results = expert.retrieve(propagated, top_k=topk, steps=1)
                
                current_context_mem = seed_memories[-1] if seed_memories else None
                chain_broken = False
                
                for score, content, meta, idx, mem_id in associate_results:
                    if chain_broken or mem_id in seed_ids:
                        continue
                        
                    mem = self.cortex.index.get_memory(mem_id)
                    if not mem or "sdr" not in mem:
                        continue

                    sim = F.cosine_similarity(
                        current_context_mem["sdr"].unsqueeze(0),
                        mem["sdr"].unsqueeze(0)
                    ).item()
                    
                    if sim < CONTEXT_SIM_THRESHOLD:
                        logger.info(f"🧠 思考链断裂（相似度：{sim:.2f} < {CONTEXT_SIM_THRESHOLD}），自动停止联想")
                        similarity_trace.append((current_context_mem["content"], mem["content"], sim, "BROKEN"))
                        chain_broken = True
                        break

                    activated_memories.append(mem)
                    similarity_trace.append((current_context_mem["content"], mem["content"], sim, "OK"))
                    current_context_mem = mem

                all_memories = seed_memories + activated_memories

                # ====================== 7. ✅ 第二重保险：思考链生成前的连贯性二次校验 ======================
                if len(all_memories) > 1:
                    coherent_memories = [all_memories[0]]
                    for i in range(1, len(all_memories)):
                        prev_mem = coherent_memories[-1]
                        curr_mem = all_memories[i]
                        
                        sim = F.cosine_similarity(
                            prev_mem["sdr"].unsqueeze(0),
                            curr_mem["sdr"].unsqueeze(0)
                        ).item()
                        
                        if sim < CONTEXT_SIM_THRESHOLD:
                            continue
                        coherent_memories.append(curr_mem)
                    all_memories = coherent_memories

            else:
                logger.warning(f"未找到专家模块: {target_expert}")
                all_memories = seed_memories
                activated_memories = []

            # ====================== 8. 生成思考链（带相似度标记） ======================
            thought_chain = self._build_coherent_thought_chain(all_memories, similarity_trace, CONTEXT_SIM_THRESHOLD)
            core_ideas = self._extract_core_ideas(all_memories)
            activation_strength = propagated.norm().item() if propagated is not None else 0.0

            # ====================== 9. 返回结果（新增相似度轨迹） ======================
            return {
                "thought_chain": thought_chain,
                "core_ideas": core_ideas,
                "activated_memories": [m["content"] for m in all_memories],
                "seed_memories": [m["content"] for m in seed_memories],
                "associated_memories": [m["content"] for m in activated_memories],
                "expert": target_expert,
                "activation_strength": activation_strength,
                "predicted_memory": predicted_memory,
                "prediction_error": prediction_error,
                "similarity_trace": similarity_trace,
                "error": None
            }

        except Exception as e:
            logger.error(f"❌ 思考过程出错: {e}", exc_info=True)
            return {
                "thought_chain": "思考失败",
                "core_ideas": [],
                "activated_memories": [],
                "seed_memories": [],
                "associated_memories": [],
                "expert": None,
                "activation_strength": 0.0,
                "predicted_memory": None,
                "prediction_error": 0.0,
                "similarity_trace": [],
                "error": f"思考失败: {str(e)}"
            }

    # ====================== 新增辅助函数：生成带相似度标记的思考链 ======================
    def _build_coherent_thought_chain(self, memories: List[dict], similarity_trace: List[tuple], threshold: float) -> str:
        """生成带相似度标记的思考链，清晰显示断裂点"""
        if not memories:
            return "无思考内容"
        
        thought_parts = [f"🤯 大脑思考完成 | 思路：{memories[0]['content']}"]
        for i in range(1, len(memories)):
            sim_info = next((t for t in similarity_trace if t[0] == memories[i-1]["content"] and t[1] == memories[i]["content"]), None)
            if sim_info:
                prev_content, curr_content, sim, status = sim_info
                if status == "OK":
                    thought_parts.append(f"→ {curr_content} (相似度: {sim:.2f} ✅)")
                else:
                    thought_parts.append(f"→ {curr_content} (相似度: {sim:.2f} ❌ 【思考链断裂】)")
            else:
                thought_parts.append(f"→ {memories[i]['content']}")
        
        if len(memories) < len(similarity_trace)+1:
            thought_parts.append("🛑 自动停止思考：检测到不相关内容")
        
        return " | ".join(thought_parts)
        
    # ==============================================
    # 🔥 终极修复：思考辅助方法（无警告+稳定匹配）
    # ==============================================
    def _get_retrieved_memory_vectors(self, memories: List[str], expert_name: str) -> List[torch.Tensor]:
        """✅ 100%消除张量警告 + 正确提取SDR向量"""
        vectors = []
        for mem in self.cortex.index.memories.values():
            if mem["content"] in memories and mem["metadata"].get("expert") == expert_name:
                sdr_data = mem["sdr"]
                if isinstance(sdr_data, torch.Tensor):
                    sdr_tensor = sdr_data.detach().clone()
                else:
                    sdr_tensor = torch.as_tensor(sdr_data, dtype=torch.float32)
                vectors.append(sdr_tensor)
        return vectors

    def _search_activation(self, expert: DynamicExpert, activation: torch.Tensor, topk: int = 5) -> List[int]:
        """
        ✅ 正确修复：DynamicExpert没有index，改用SDR相似度匹配记忆
        """
        try:
            sim_scores = []
            for idx, sdr in enumerate(expert.sdr_list):
                sim = F.cosine_similarity(activation, sdr.unsqueeze(0), dim=-1).item()
                sim_scores.append((idx, sim))
            
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
        
        if force_expert is None:
            target_expert = self.hippocampus_router.route(clip_vec, text)
            self.hippocampus_router.online_learn(clip_vec, target_expert)
        else:
            target_expert = force_expert
        
        # ====================== 🔥 核心修改：调用对应专家的专属编码器 ======================
        sdr_encoder = self.sdr_encoders.get(target_expert, self.sdr_encoders["概念"])
        sdr = sdr_encoder.encode(clip_vec.unsqueeze(0))
        
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
                
                if force_expert is None:
                    target_expert = self.hippocampus_router.route(clip_vec, text)
                    self.hippocampus_router.online_learn(clip_vec, target_expert)
                else:
                    target_expert = force_expert
                
                # ====================== 🔥 核心修改：调用对应专家的专属编码器 ======================
                sdr_encoder = self.sdr_encoders.get(target_expert, self.sdr_encoders["概念"])
                sdr = sdr_encoder.encode(clip_vec.unsqueeze(0))
                
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
        
        if target_expert is None:
            target_expert = self.hippocampus_router.route(clip_vec, text)
        
        # ====================== 🔥 核心修改：调用对应专家的专属编码器 ======================
        sdr_encoder = self.sdr_encoders.get(target_expert, self.sdr_encoders["概念"])
        query_sdr = sdr_encoder.encode(clip_vec.unsqueeze(0))
        
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
        # ====================== 🔥 核心修改：保存每个专家的专属编码器 ======================
        for name in self.expert_names:
            encoder_path = os.path.join(self.storage_dir, f"sdr_encoder_{name}.pt")
            self.sdr_encoders[name].save(encoder_path)
            logger.info(f"💾 [{name}] 专家稀疏编码器已保存: {encoder_path}")
        
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
    
    def force_clean_all_experts(self):
        """
        🔥 终极清理：不依赖海马体路由，直接根据内容关键词强制分配专家
        """
        logger.info("🔧 开始全专家终极强制清理...")
        total_moved = 0
        
        expert_keywords = {
            "视觉": ["图片", "图像", "照片", "视觉", "看", "画", "图", "长什么样", "颜色", "形状", "大小"],
            "空间": ["事件", "历史", "年", "月", "日", "发生", "发现", "地点", "哪里", "战争", "会议"],
            "概念": ["人物", "是什么", "定义", "概念", "职业", "动物", "植物", "物体", "元谋人", "氏族", "华夏族"],
            "抽象": ["知识", "道理", "名言", "原理", "定律", "方法", "技术", "甲骨文"],
            "身份": ["我是谁", "你是谁", "我叫", "你叫", "名字", "身份", "主人", "我是", "你是", "关系", "小白", "邓尧"]
        }
        
        for mem_id, mem in list(self.cortex.index.memories.items()):
            old_expert = mem["metadata"]["expert"]
            content = mem["content"].lower()
            
            new_expert = "抽象"
            for expert, keywords in expert_keywords.items():
                if any(keyword in content for keyword in keywords):
                    new_expert = expert
                    break
            
            if new_expert != old_expert:
                mem["metadata"]["expert"] = new_expert
                
                if old_expert in self.experts:
                    self.experts[old_expert].delete_memory(mem_id)
                
                if new_expert in self.experts:
                    self.experts[new_expert].add_memory(
                        mem["sdr"], mem["content"], mem_id=mem_id, metadata=mem["metadata"]
                    )
                
                if old_expert in self.cortex.index.expert_index and mem_id in self.cortex.index.expert_index[old_expert]:
                    self.cortex.index.expert_index[old_expert].remove(mem_id)
                self.cortex.index.expert_index[new_expert].append(mem_id)
                
                total_moved += 1
                logger.debug(f"   迁移记忆: {old_expert} → {new_expert} | {mem['content'][:30]}...")
        
        logger.info(f"✅ 全专家终极清理完成！共迁移 {total_moved} 条错分记忆")
        
        for name, expert in self.experts.items():
            expert_mem_ids = self.cortex.index.get_by_expert(name)
            if expert_mem_ids:
                logger.info(f"🧠 重新训练 [{name}] 专家突触...")
                for mem_id in expert_mem_ids:
                    mem = self.cortex.index.get_memory(mem_id)
                    if mem and "sdr" in mem:
                        expert.hebbian_update(mem["sdr"], mem["sdr"], is_fact=True)
        
        return total_moved

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