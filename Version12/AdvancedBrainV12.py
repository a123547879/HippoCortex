import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import os
import logging
import json
import datetime
import random
import time
import threading
from typing import List, Dict

from BrainConfig import config
from DynamicExpertV6 import DynamicExpert
from PersistentCortexV10 import PersistentCortexV10
from LearnableSparseEncoder import LearnableSparseEncoder
from HippocampusRouterV7 import HippocampusRouterV7
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("AdvancedBrainV12")

class AdvancedBrainV12:
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
        self.hippocampus_router = HippocampusRouterV7(
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
        
        self.cortex = PersistentCortexV10(storage_dir, self.experts, embedding_model=self.embedding_model, llm= self.llm, kg_enabled=kg_enabled)
        
        # 执行每日记忆衰减
        self.cortex.decay_all_memories()

        # ====================== 🔥 走神状态系统 ======================
        self.mind_wandering_enabled = True
        self.is_mind_wandering = False
        self.mind_wandering_start_time = None
        self.fatigue_level = 0.0
        self.fatigue_sleep_threshold = 0.85
        self.mind_wandering_idle_threshold = 30
        self.last_interaction_time = datetime.datetime.now()
        self.mind_wandering_recall_prob = 0.5
        self.mind_wandering_assoc_prob = 0.3
        self.needs_sleep_request = False  # 睡眠请求标志
        self.mind_wandering_thread = None
        self._mind_wandering_running = False

        # ====================== 🔥 新增：意图驱动系统 ======================
        self.intention_queue = []  # 意图队列
        self.max_intention_queue_size = 10  # 最大意图队列长度
        self.last_intention_execution_time = datetime.datetime.now()  # 上次执行意图的时间
        self.min_intention_interval = 60  # 两次主动意图的最小间隔（秒）
        self.pending_social_intention = None  # 待执行的社交意图（供界面读取）
        
        # 意图权重配置
        self.intention_weights = {
            "physiological": 1.5,  # 生理意图权重最高
            "cognitive": 1.0,      # 认知意图
            "social": 0.8,         # 社交意图
            "exploration": 0.6     # 探索意图
        }
        # ====================================================================

    # ==============================================
    # 🔥 新增：意图驱动核心方法
    # ==============================================
    def _generate_intentions(self):
        """在走神时生成候选意图"""
        candidate_intentions = []
        
        # 1. 生理意图：睡眠提示（优先级最高，但不直接执行，只提示）
        if self.fatigue_level > 0.6:
            candidate_intentions.append({
                "type": "physiological",
                "priority": self.fatigue_level * self.intention_weights["physiological"],
                "content": f"我感觉有点累了，疲劳度已经有{int(self.fatigue_level * 100)}%了",
                "action": "express_tiredness"
            })
        
        # 2. 认知意图：复习重要记忆
        if random.random() < 0.3:
            important_memories = self._get_important_memories(limit=3)
            if important_memories:
                mem = random.choice(important_memories)
                candidate_intentions.append({
                    "type": "cognitive",
                    "priority": mem["metadata"]["importance"] * self.intention_weights["cognitive"],
                    "content": f"我想起了一件重要的事：{mem['content'][:40]}...",
                    "action": "review_memory",
                    "context": {"memory": mem}
                })
        
        # 3. 社交意图：主动分享
        if random.random() < 0.35:
            recent_memories = self._get_recent_memories(limit=15)
            if recent_memories:
                mem = random.choice(recent_memories)
                candidate_intentions.append({
                    "type": "social",
                    "priority": 0.5 * self.intention_weights["social"],
                    "content": f"对了，我想起来我们之前聊过：{mem['content'][:40]}...",
                    "action": "share_memory",
                    "context": {"memory": mem}
                })
        
        # 4. 社交意图：主动提问
        if random.random() < 0.2:
            knowledge_gaps = self._find_knowledge_gaps()
            if knowledge_gaps:
                gap = random.choice(knowledge_gaps)
                candidate_intentions.append({
                    "type": "social",
                    "priority": 0.4 * self.intention_weights["social"],
                    "content": f"我一直很好奇，{gap}是什么呀？你能给我讲讲吗？",
                    "action": "ask_question",
                    "context": {"question": gap}
                })
        
        # 5. 探索意图：关联记忆探索
        if random.random() < 0.15:
            associations = self._get_random_associations(limit=2)
            if len(associations) >= 2:
                candidate_intentions.append({
                    "type": "exploration",
                    "priority": 0.3 * self.intention_weights["exploration"],
                    "content": f"我发现{associations[0][:15]}和{associations[1][:15]}之间好像有某种联系",
                    "action": "explore_association",
                    "context": {"associations": associations}
                })
        
        # 按优先级排序
        candidate_intentions.sort(key=lambda x: x["priority"], reverse=True)
        
        # 添加到意图队列
        for intention in candidate_intentions:
            if len(self.intention_queue) < self.max_intention_queue_size:
                self.intention_queue.append(intention)
            else:
                # 替换优先级最低的意图
                min_priority_idx = min(range(len(self.intention_queue)), 
                                     key=lambda i: self.intention_queue[i]["priority"])
                if intention["priority"] > self.intention_queue[min_priority_idx]["priority"]:
                    self.intention_queue[min_priority_idx] = intention
        
        logger.debug(f"🧠 生成了{len(candidate_intentions)}个候选意图，当前队列长度：{len(self.intention_queue)}")

    def _get_important_memories(self, limit=5):
        """获取重要记忆"""
        important_mems = [m for m in self.cortex.index.memories.values() 
                         if m["metadata"].get("importance", 0) > 0.7]
        important_mems.sort(key=lambda x: x["metadata"]["importance"], reverse=True)
        return important_mems[:limit]

    def _get_recent_memories(self, limit=10):
        """获取近期记忆"""
        recent_mems = list(self.cortex.index.memories.values())
        recent_mems.sort(key=lambda x: x["metadata"].get("timestamp", 0), reverse=True)
        return recent_mems[:limit]

    def _find_knowledge_gaps(self):
        """发现知识缺口（简化实现）"""
        return ["量子纠缠", "黑洞", "人工智能的未来", "宇宙的起源", "意识是什么"]

    def _get_random_associations(self, limit=3):
        """获取随机关联记忆"""
        associations = []
        if len(self.cortex.index.memories) > 0:
            random_mems = random.sample(list(self.cortex.index.memories.values()), min(limit, len(self.cortex.index.memories)))
            associations = [m["content"] for m in random_mems]
        return associations

    def _execute_highest_priority_intention(self):
        """执行最高优先级的意图"""
        if not self.intention_queue:
            return None
        
        # 检查最小间隔
        time_since_last = (datetime.datetime.now() - self.last_intention_execution_time).total_seconds()
        if time_since_last < self.min_intention_interval:
            return None
        
        # 取出最高优先级意图
        highest_intention = max(self.intention_queue, key=lambda x: x["priority"])
        self.intention_queue.remove(highest_intention)
        
        logger.info(f"🎯 执行意图：{highest_intention['content']} (优先级：{highest_intention['priority']:.2f})")
        
        # 执行意图
        result = None
        if highest_intention["action"] in ["express_tiredness", "share_memory", "ask_question", "explore_association", "review_memory"]:
            # 这些是需要主动和用户说的意图
            result = highest_intention["content"]
        
        highest_intention["executed"] = True
        highest_intention["result"] = result
        self.last_intention_execution_time = datetime.datetime.now()
        
        return highest_intention

    def get_pending_social_intention(self):
        """获取并清除待执行的社交意图（确保只执行一次）"""
        if hasattr(self, 'pending_social_intention') and self.pending_social_intention:
            intention = self.pending_social_intention
            self.pending_social_intention = None  # 消费后立即清空
            logger.debug(f"🧠 取出待执行意图: {intention['content']}")
            return intention
        return None

    # ==============================================
    # 走神状态核心方法
    # ==============================================
    def _update_interaction_time(self):
        """更新最后交互时间（有对话时调用）"""
        self.last_interaction_time = datetime.datetime.now()
        if self.is_mind_wandering:
            self._stop_mind_wandering()

    def _check_mind_wandering_trigger(self):
        """检查是否应该触发走神（定时调用）"""
        if (not self.mind_wandering_enabled 
            or self.is_mind_wandering 
            or self.fatigue_level >= self.fatigue_sleep_threshold):
            return
        
        idle_seconds = (datetime.datetime.now() - self.last_interaction_time).total_seconds()
        if idle_seconds >= self.mind_wandering_idle_threshold:
            self._start_mind_wandering()

    def _start_mind_wandering(self):
        """开始走神：启动后台思考线程"""
        if self.is_mind_wandering:
            return
            
        logger.info("🌙 大脑进入走神状态...")
        self.is_mind_wandering = True
        self.mind_wandering_start_time = datetime.datetime.now()
        self._mind_wandering_running = True
        
        self.mind_wandering_thread = threading.Thread(target=self._mind_wandering_loop, daemon=True)
        self.mind_wandering_thread.start()

    def _stop_mind_wandering(self):
        """停止走神：瞬间回神"""
        if not self.is_mind_wandering:
            return
            
        logger.info("⚡ 大脑瞬间回神！")
        self.is_mind_wandering = False
        self._mind_wandering_running = False
        
        if self.mind_wandering_thread and self.mind_wandering_thread.is_alive():
            self.mind_wandering_thread.join(timeout=2.0)
            self.mind_wandering_thread = None

    # ====================== 🔥 修改：走神循环集成意图生成 ======================
    def _mind_wandering_loop(self):
        """走神主循环：记忆闪回 + 联想想象 + 疲劳积累 + 意图生成"""
        while self._mind_wandering_running:
            try:
                # 1. 疲劳积累
                self.fatigue_level = min(1.0, self.fatigue_level + 0.002)
                logger.debug(f"🧠 走神中... 疲劳度: {self.fatigue_level:.2f}")
                
                # 2. 疲劳达到阈值 → 设置睡眠请求标志
                if self.fatigue_level >= self.fatigue_sleep_threshold and not self.needs_sleep_request:
                    logger.info("😴 疲劳度达到阈值，向界面发送睡眠请求...")
                    self.needs_sleep_request = True
                
                # 3. 随机记忆闪回
                if random.random() < self.mind_wandering_recall_prob:
                    self._mind_wandering_memory_recall()
                
                # 4. 轻量联想想象
                if random.random() < self.mind_wandering_assoc_prob:
                    self._mind_wandering_association()
                
                # 🔥 5. 生成意图（30%概率）
                if random.random() < 0.3:
                    self._generate_intentions()
                
                # 🔥 6. 尝试执行意图（25%概率）
                if random.random() < 0.25 and not self.pending_social_intention:
                    intention = self._execute_highest_priority_intention()
                    if intention and intention["action"] in ["express_tiredness", "share_memory", "ask_question", "explore_association", "review_memory"]:
                        self.pending_social_intention = intention
                
                # 暂停2秒
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ 走神过程出错: {e}")
                time.sleep(2)
    
    def check_and_consume_sleep_request(self) -> bool:
        """检查是否有睡眠请求"""
        if self.needs_sleep_request:
            self.needs_sleep_request = False
            return True
        return False

    def _mind_wandering_memory_recall(self):
        """走神时的记忆闪回"""
        try:
            all_mem_ids = list(self.cortex.long_term_memory.keys())
            if not all_mem_ids:
                return
                
            weighted_mem_ids = []
            for mem_id in all_mem_ids:
                mem = self.cortex.index.get_memory(mem_id)
                if not mem:
                    continue
                meta = mem["metadata"]
                weight = meta.get("importance", 0.5) * 2 + meta.get("recency", 0.5)
                weighted_mem_ids.extend([mem_id] * int(weight * 10))
            
            if not weighted_mem_ids:
                return
                
            target_mem_id = random.choice(weighted_mem_ids)
            mem = self.cortex.index.get_memory(target_mem_id)
            if mem:
                self.cortex.increment_access_count(target_mem_id)
                logger.info(f"💭 记忆闪回: {mem['content'][:40]}...")
                
        except Exception as e:
            logger.debug(f"记忆闪回失败: {e}")

    def _mind_wandering_association(self):
        """走神时的联想想象"""
        try:
            expert_name = random.choice(self.expert_names)
            expert = self.experts[expert_name]
            
            if not expert.sdr_list:
                return
                
            random_idx = random.randint(0, len(expert.sdr_list) - 1)
            start_sdr = expert.sdr_list[random_idx]
            
            with torch.no_grad():
                activated = expert.forward(start_sdr, steps=1, top_k=30)
                assoc_results = expert.retrieve(activated, top_k=2)
                
                if assoc_results:
                    assoc_content = assoc_results[0][1]
                    logger.info(f"🤔 联想想象: → {assoc_content[:40]}...")
                    
        except Exception as e:
            logger.debug(f"联想想象失败: {e}")

    def reset_fatigue(self):
        """重置疲劳度"""
        self.fatigue_level = 0.0
        logger.info("🔋 疲劳度已重置")

    # ==============================================
    # 工具方法与原有功能
    # ==============================================
    def get_expert(self, expert_name: str) -> DynamicExpert:
        return self.experts.get(expert_name)

    def think(self, text: str, steps: int = 2, topk: int = 10) -> Dict:
        self._update_interaction_time()
        
        try:
            CONTEXT_SIM_THRESHOLD = 0.30
            
            clip_vec = self.encode_text(text)
            clip_vec = F.normalize(clip_vec, p=2, dim=-1)
            target_expert = self.hippocampus_router.route(clip_vec, text)
            
            sdr_encoder = self.sdr_encoders.get(target_expert, self.sdr_encoders["概念"])
            query_sdr = sdr_encoder.encode(clip_vec)

            raw_results = self.cortex.search_memories(
                clip_vec, query_sdr,
                expert_name=target_expert,
                top_k=3,
                min_similarity=0.3,
                query_text= text
            )

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

            expert = self.experts.get(target_expert)
            predicted_memory = None
            prediction_error = 0.0
            propagated = None
            similarity_trace = []
            
            if expert:
                initial_activation = torch.stack(seed_sdrs).mean(dim=0, keepdim=True)
                propagated = expert.forward(initial_activation, steps=steps, top_k=60)

                pred_sdr = expert.predict_next_sdr(propagated.detach())
                prediction_error = expert.update_prediction(pred_sdr, propagated.detach())
                pred_results = expert.retrieve(pred_sdr, top_k=1)
                if pred_results:
                    _, pred_content, _, _, pred_mem_id = pred_results[0]
                    predicted_memory = pred_content

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

            thought_chain = self._build_coherent_thought_chain(all_memories, similarity_trace, CONTEXT_SIM_THRESHOLD)
            core_ideas = self._extract_core_ideas(all_memories)
            activation_strength = propagated.norm().item() if propagated is not None else 0.0

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

    def _build_coherent_thought_chain(self, memories: List[dict], similarity_trace: List[tuple], threshold: float) -> str:
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
        
    def _get_retrieved_memory_vectors(self, memories: List[str], expert_name: str) -> List[torch.Tensor]:
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
        if not memories:
            return "无关联记忆"
        contents = [m["content"][:35] + "..." if len(m["content"]) > 35 else m["content"] for m in memories]
        return " → ".join(contents)

    def _extract_core_ideas(self, memories: List[Dict]) -> List[str]:
        ideas = []
        for mem in memories:
            content = mem["content"]
            if "：" in content:
                ideas.append(content.split("：")[1][:15])
            else:
                ideas.append(content[:15])
        return list(set(ideas))

    def _get_identity_core_memory(self) -> str:
        id_memories = [m["content"] for m in self.cortex.index.memories.values() if m["metadata"].get("expert") == "身份"]
        return "\n".join(id_memories[:3]) if id_memories else "我是小白"
    
    def encode_text(self, text):
        try:
            embedding = self.embedding_model.embed_query(text)
            clip_vec = torch.as_tensor(embedding, dtype=torch.float32)
            return clip_vec
        except Exception as e:
            logger.error(f"❌ 文本编码失败: {e}")
            raise RuntimeError(f"Ollama embedding 失败: {e}") from e

    def learn(self, text, force_expert=None):
        self._update_interaction_time()
        
        clip_vec = self.encode_text(text)
        clip_vec = F.normalize(clip_vec.detach().squeeze(), p=2, dim=-1)
        
        if force_expert is None:
            target_expert = self.hippocampus_router.route(clip_vec, text)
            self.hippocampus_router.online_learn(clip_vec, target_expert)
        else:
            target_expert = force_expert
        
        sdr_encoder = self.sdr_encoders.get(target_expert, self.sdr_encoders["概念"])
        sdr = sdr_encoder.encode(clip_vec.unsqueeze(0))
        
        self.cortex.store_detailed_memory(target_expert, sdr, clip_vec, text)
        logger.info(f"✅ 记忆已存入 【{target_expert}】 专家: {text[:30]}...")

    def batch_learn(self, texts: List[str], force_experts: List[str] = None):
        self._update_interaction_time()
        
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
        self._update_interaction_time()
        
        clip_vec = self.encode_text(text)
        clip_vec = F.normalize(clip_vec.detach().squeeze(), p=2, dim=-1)
        
        if target_expert is None:
            target_expert = self.hippocampus_router.route(clip_vec, text)
        
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
        
        self.reset_fatigue()
        self.is_mind_wandering = False
        self._mind_wandering_running = False
        self.needs_sleep_request = False
        self.last_interaction_time = datetime.datetime.now()
        self.intention_queue = []  # 清空意图队列
        self.pending_social_intention = None  # 清空待执行意图
        
        logger.info("✅ 全脑睡眠巩固完成！所有状态已重置")
        return None

    def save_all(self):
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
            "kg_enabled": self.kg_enabled,
            "is_mind_wandering": self.is_mind_wandering,
            "fatigue_level": self.fatigue_level,
            "intention_queue_size": len(self.intention_queue)  # 🔥 新增：意图队列长度
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
        logger.info("🔧 开始全专家终极强制清理...")
        total_moved = 0
        
        expert_keywords = {
            "视觉": ["图片", "图像", "照片", "视觉", "看", "画", "图", "长什么样", "颜色", "形状", "大小"],
            "空间": ["事件", "历史", "年", "月", "日", "发生", "发现", "地点", "哪里", "战争", "会议"],
            "概念": ["人物", "是什么", "定义", "概念", "职业", "动物", "植物", "物体", "元谋人", "氏族", "华夏族"],
            "抽象": ["知识", "道理", "名言", "原理", "定律", "方法", "技术", "甲骨文"],
            "身份": ["我是谁", "你是谁", "我叫", "你叫", "名字", "身份", "主人", "我是", "你是", "关系", "小白", "邓尧"]
        }
        
        MEM_NODE_PREFIX = "mem_"
        knowledge_graph = getattr(self, 'knowledge_graph', None)

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

                if knowledge_graph and knowledge_graph.enabled:
                    try:
                        mem_node = f"{MEM_NODE_PREFIX}{mem_id}"
                        if mem_node not in knowledge_graph.G:
                            continue
                        
                        knowledge_graph.G.nodes[mem_node]["expert"] = new_expert
                        
                        for neighbor in knowledge_graph.G.neighbors(mem_node):
                            node_attrs = knowledge_graph.G.nodes[neighbor]
                            if node_attrs.get("type") != "memory":
                                knowledge_graph.G.nodes[neighbor]["expert"] = new_expert
                        
                        knowledge_graph._clear_cache()
                        
                    except Exception as e:
                        logger.debug(f"   图谱同步忽略（记忆{mem_id}）: {str(e)[:50]}")
        
        logger.info(f"✅ 全专家终极清理完成！共迁移 {total_moved} 条错分记忆")
        
        for name, expert in self.experts.items():
            expert_mem_ids = self.cortex.index.get_by_expert(name)
            if expert_mem_ids:
                logger.info(f"🧠 重新训练 [{name}] 专家突触...")
                for mem_id in expert_mem_ids:
                    mem = self.cortex.index.get_memory(mem_id)
                    if mem and "sdr" in mem:
                        expert.hebbian_update(mem["sdr"], mem["sdr"], is_fact=True)
        
        if knowledge_graph:
            try:
                knowledge_graph.save()
                logger.info("💾 知识图谱已同步保存")
            except:
                pass
        
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