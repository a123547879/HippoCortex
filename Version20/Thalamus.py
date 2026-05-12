import torch
import time
import logging
from typing import List, Dict, Optional, Tuple
from collections import deque

logger = logging.getLogger("Thalamus")

class Thalamus:
    """
    丘脑核心模块：信息中继、注意力调控、记忆巩固协调、意识整合
    完全基于神经科学丘脑功能设计
    """
    def __init__(self, 
                 input_dim: int = 1024,
                 attention_threshold: float = 0.3,
                 consolidation_threshold: float = 0.6,
                 max_short_term_capacity: int = 50):
        super().__init__()
        self.input_dim = input_dim
        
        # ========== 丘脑核心参数 ==========
        self.attention_threshold = attention_threshold
        self.consolidation_threshold = consolidation_threshold
        self.max_short_term_capacity = max_short_term_capacity
        
        # ========== 注意力系统 ==========
        self.current_attention_focus = None
        self.attention_weights = {}
        self.saliency_map = {}
        
        # ========== 记忆筛选系统 ==========
        self.memory_importance_scorer = self._default_importance_scorer
        
        # ========== 状态调控 ==========
        self.brain_state = "awake"
        self.arousal_level = 1.0
        
        # ========== 关联模块引用 ==========
        self.hippocampus = None
        self.cortex = None
        self.energy_field = None
        self.experts = None
        
        logger.info("🧠 丘脑模块初始化完成")
    
    def bind_modules(self, hippocampus, cortex, energy_field, experts):
        """绑定关联模块"""
        self.hippocampus = hippocampus
        self.cortex = cortex
        self.energy_field = energy_field
        self.experts = experts
        logger.info("✅ 丘脑已绑定所有脑区模块")
    
    # ===================== 核心功能1：信息过滤与中继 =====================
    def filter_and_relay(self, 
                        input_vec: torch.Tensor, 
                        input_text: str = None,
                        metadata: Dict = None) -> Tuple[bool, Dict]:
        """
        过滤输入信息，只有足够显著的信息才能进入海马体
        """
        metadata = metadata or {}
        input_text = input_text or ""
        
        # 关键信息强制放行
        is_identity = input_text.startswith("身份：")
        is_important = metadata.get('importance', 0.0) >= 0.7
        is_pending_consolidate = metadata.get('pending_consolidation', False)
        
        if is_identity or is_important or is_pending_consolidate:
            saliency = 1.0
            enhanced_vec = input_vec * 1.5
            logger.info(f"✅ 丘脑【强制放行关键信息】: {input_text[:40]}...")
            return True, {
                "vec": enhanced_vec,
                "text": input_text,
                "metadata": metadata,
                "saliency": saliency,
                "timestamp": time.time()
            }

        # 普通信息过滤
        saliency = self._calculate_saliency(input_vec, input_text, metadata)
        dynamic_threshold = self._get_dynamic_attention_threshold()
        
        if saliency < dynamic_threshold:
            logger.debug(f"🚫 丘脑过滤低显著性信息: {input_text[:30]}... (显著性:{saliency:.2f} < 阈值:{dynamic_threshold:.2f})")
            return False, {}
        
        enhanced_vec = input_vec * (1.0 + saliency * 0.5)
        
        if saliency > 0.8:
            self.current_attention_focus = input_text
            logger.info(f"🎯 丘脑锁定注意力焦点: {input_text[:50]}... (显著性:{saliency:.2f})")
        
        logger.debug(f"✅ 丘脑中继信息: {input_text[:30]}... (显著性:{saliency:.2f})")

        if metadata and 'vae_latent' in metadata and metadata['vae_latent'] is not None:
            logger.debug(f"🧠 确认VAE数据经过丘脑 | 大小: ~{len(str(metadata['vae_latent']))//1024}KB")

        return True, {
            "vec": enhanced_vec,
            "text": input_text,
            "metadata": metadata,
            "saliency": saliency,
            "timestamp": time.time()
        }
        
    def _calculate_saliency(self, input_vec: torch.Tensor, input_text: str = None, metadata: Dict = None) -> float:
        """计算信息显著性"""
        saliency = 0.5
        
        if self.cortex and hasattr(self.cortex, 'index'):
            results = self.cortex.index.vector_search(input_vec, top_k=1)
            if results:
                novelty = 1.0 - results[0][1]
                saliency += novelty * 0.4
        
        if input_text and self.cortex and hasattr(self.cortex, 'important_entities'):
            for entity in self.cortex.important_entities:
                if entity in input_text:
                    saliency += 0.2
                    break
        
        if metadata and metadata.get('importance', 0.5) > 0.7:
            saliency += 0.3
        
        saliency *= self.arousal_level
        return min(1.0, saliency)
    
    def _get_dynamic_attention_threshold(self) -> float:
        """基于认知能量动态调整注意力阈值"""
        if not self.energy_field:
            return self.attention_threshold
        
        total_energy = self.energy_field.total_energy(
            routing_probs=[], triple_scores=[], sim_scores=[],
            rule_match=True, synapse_change=0.0,
            is_wandering=(self.brain_state == "wandering"),
            fatigue_level=0.0
        )[0]
        
        energy_factor = min(0.5, total_energy / 50.0)
        dynamic_threshold = self.attention_threshold + energy_factor
        
        if self.brain_state == "wandering":
            dynamic_threshold += 0.2
        
        return min(0.8, dynamic_threshold)
    
    # ===================== 核心功能2：记忆巩固协调 =====================
    def coordinate_consolidation(self, epochs: int = 3, priority_memories: Optional[List] = None) -> Tuple[int, int]:
        """协调海马体到皮层的记忆巩固过程"""
        if not self.hippocampus or not self.cortex:
            logger.error("❌ 丘脑未绑定海马体或皮层，无法执行巩固")
            return 0, 0
        
        logger.info("\n🧠 丘脑开始协调记忆巩固过程...")
        
        priority_mem_ids = set()
        if priority_memories:
            priority_mem_ids = {m["id"] for m in priority_memories if "id" in m}
            logger.info(f"🧠 元认知：收到 {len(priority_mem_ids)} 条高优先级记忆标记")
        
        short_term_memories = list(self.hippocampus.hippocampal_buffer)
        if not short_term_memories:
            logger.info("✅ 海马体无短期记忆，跳过巩固")
            return 0, 0
        
        consolidated_count = 0
        forgotten_count = 0
        priority_consolidated = 0
        
        for mem in short_term_memories:
            importance = self.memory_importance_scorer(mem)
            
            # 兼容 MemoryPacket 获取 ID
            if hasattr(mem, 'mem_id'):
                mem_id = mem.mem_id
            else:
                mem_id = mem.get("mem_id", "")
                
            is_priority = mem_id in priority_mem_ids
            
            if is_priority:
                importance += 0.25
                # 兼容获取内容
                content = mem.content if hasattr(mem, 'content') else mem.get('content', '')
                logger.debug(f"🧠 元认知优先记忆：{content[:25]}... | 加分后重要性={importance:.2f}")
            
            if importance >= self.consolidation_threshold:
                # 兼容修改元数据
                if hasattr(mem, 'metadata'):
                    mem.metadata["is_priority_consolidation"] = True
                    mem.metadata["priority_score"] = importance
                else:
                    mem["metadata"]["is_priority_consolidation"] = True
                    mem["metadata"]["priority_score"] = importance
                
                success = self._transfer_to_cortex(mem)
                if success:
                    consolidated_count += 1
                    if is_priority:
                        priority_consolidated += 1
                    content = mem.content if hasattr(mem, 'content') else mem.get('content', '')
                    logger.debug(f"✅ 记忆巩固成功: {content[:30]}... (重要性:{importance:.2f})")
            else:
                content = mem.content if hasattr(mem, 'content') else mem.get('content', '')
                logger.debug(f"🗑️  丘脑筛选遗忘: {content[:30]}... (重要性:{importance:.2f})")
        
        # 清空已巩固记忆
        self.hippocampus.hippocampal_buffer = deque(
            [m for m in self.hippocampus.hippocampal_buffer 
             if self.memory_importance_scorer(m) < self.consolidation_threshold],
            maxlen=self.max_short_term_capacity
        )
        
        if priority_memories:
            logger.info(f"🧠 元认知优先巩固：{priority_consolidated}/{len(priority_mem_ids)} 条高优先级记忆成功固化")
        
        logger.info(f"✅ 丘脑记忆巩固完成 | 成功:{consolidated_count} | 遗忘:{forgotten_count}")
        return consolidated_count, forgotten_count
    
    def _default_importance_scorer(self, memory) -> float:
        """修复：兼容 MemoryPacket 对象 + 字典"""
        score = 0.35

        # 兼容获取显著性
        if hasattr(memory, 'metadata'):
            saliency = memory.metadata.get('saliency', 0.6)
        else:
            saliency = memory.get('saliency', 0.6)
        score += saliency * 0.4

        # 回放次数
        if hasattr(memory, 'replay_count'):
            replay_count = memory.replay_count
        elif hasattr(memory, 'metadata'):
            replay_count = memory.metadata.get('replay_count', 0)
        else:
            replay_count = memory.get('replay_count', 0)
        score += min(0.3, replay_count * 0.08)

        # 新鲜度
        if hasattr(memory, 'metadata'):
            timestamp = memory.metadata.get('timestamp')
        else:
            timestamp = memory.get('timestamp')
            
        if timestamp:
            age_hours = (time.time() - timestamp) / 3600
            freshness = max(0.15, 1.0 - age_hours / 24.0)
        else:
            freshness = 0.15
        score += freshness * 0.2

        # 重要实体
        if hasattr(memory, 'content'):
            content = memory.content
        else:
            content = memory.get('content', '')
            
        if self.cortex and hasattr(self.cortex, 'important_entities'):
            for entity in self.cortex.important_entities:
                if entity in content:
                    score += 0.12
                    break

        return min(1.0, score)
    
    def _transfer_to_cortex(self, memory) -> bool:
        """将单条记忆从海马体转移到皮层（兼容MemoryPacket）"""
        try:
            # 自动适配 MemoryPacket / 字典
            if hasattr(memory, 'expert'):
                expert_name = memory.expert
                sdr = memory.sdr
                clip_vec = memory.clip_vec
                content = memory.content
                metadata = memory.metadata
                replay_count = getattr(memory, 'replay_count', 0)
            else:
                expert_name = memory['expert']
                sdr = memory['sdr']
                clip_vec = memory['clip_vec']
                content = memory['content']
                metadata = memory['metadata']
                replay_count = memory.get('replay_count', 0)
            
            metadata['consolidated_at'] = time.time()
            metadata['consolidation_epochs'] = replay_count
            
            self.cortex.store_detailed_memory(
                expert_name=expert_name,
                sdr=sdr,
                clip_vec=clip_vec,
                content=content,
                metadata=metadata
            )
            
            return True
        except Exception as e:
            logger.error(f"❌ 记忆转移失败: {e}", exc_info=True)
            return False
    
    def schedule_retrieval(self, 
                      query_vec: torch.Tensor, 
                      query_sdr: torch.Tensor,
                      query_text: str = None,
                      expert_scores: Dict = None,
                      min_similarity= None) -> List[Tuple]:
        """丘脑统一调度记忆检索"""
        all_results = []
        
        # 海马体检索
        if self.hippocampus:
            hippo_results = self._retrieve_hippocampus(query_vec, query_text)
            all_results.extend(hippo_results)
            logger.info(f"🧠 海马体检索到 {len(hippo_results)} 条短期记忆")
        
        # 皮层检索
        if self.cortex:
            cortex_results = self.cortex.search_memories(
                query_vec=query_vec,
                query_sdr=query_sdr,
                query_text=query_text,
                top_k= 20,
                min_similarity= min_similarity,
                expert_scores=expert_scores
            )
            all_results.extend(cortex_results)
            logger.info(f"🧠 皮层检索到 {len(cortex_results)} 条长期记忆")
        
        # 目标专家加权
        if expert_scores:
            target_expert = max(expert_scores.items(), key=lambda x: x[1])[0]
            weighted_results = []
            for res in all_results:
                mem_id, score, content, meta = res
                if meta.get("expert", "") == target_expert:
                    score += 0.2
                weighted_results.append((mem_id, score, content, meta))
            all_results = weighted_results
        
        # 排序+去重
        all_results.sort(key=lambda x: -x[1])
        seen_ids = set()
        seen_contents = set()
        unique_results = []
        
        for res in all_results:
            mem_id = res[0]
            content = res[2].strip()
            
            if mem_id not in seen_ids:
                is_duplicate = False
                for seen in seen_contents:
                    if len(set(content) & set(seen)) / max(len(content), len(seen)) > 0.9:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    seen_ids.add(mem_id)
                    seen_contents.add(content)
                    unique_results.append(res)

        return unique_results
    
    def _extract_query_triplet(self, query_text: str) -> tuple:
        """三元组提取逻辑"""
        query_text = query_text.strip().rstrip("。！？")
        if "是" in query_text:
            parts = query_text.split("是", 1)
            return (parts[0].strip(), "是", parts[1].strip())
        return None
        
    def _retrieve_hippocampus(self, query_vec: torch.Tensor, query_text: str = None) -> List[Tuple]:
        """检索海马体短期记忆（🔥 修复：兼容MemoryPacket）"""
        results = []
        for mem in self.hippocampus.hippocampal_buffer:
            # 兼容获取向量
            if hasattr(mem, 'clip_vec'):
                mem_vec = mem.clip_vec.to(query_vec.device)
                mem_id = mem.mem_id
                content = mem.content
                meta = mem.metadata.copy() if hasattr(mem.metadata, 'copy') else dict(mem.metadata)
            else:
                mem_vec = mem['clip_vec'].to(query_vec.device)
                mem_id = mem['mem_id']
                content = mem['content']
                meta = mem['metadata'].copy()
            
            sim = torch.nn.functional.cosine_similarity(query_vec, mem_vec, dim=-1).item()
            weighted_score = sim * 1.3
            meta['is_hippocampus'] = True
            
            results.append((
                mem_id,
                weighted_score,
                content,
                meta
            ))
        
        results.sort(key=lambda x: -x[1])
        return results
        
    # ===================== 核心功能4：状态调控 =====================
    def update_brain_state(self, new_state: str):
        """更新大脑状态"""
        self.brain_state = new_state
        
        if new_state == "awake":
            self.arousal_level = 1.0
            self.attention_threshold = 0.3
            logger.info("🧠 丘脑切换到清醒状态，注意力全开")
        elif new_state == "wandering":
            self.arousal_level = 0.6
            self.attention_threshold = 0.5
            logger.info("🧠 丘脑切换到走神状态，注意力降低")
        elif new_state == "sleep":
            self.arousal_level = 0.2
            self.attention_threshold = 0.8
            logger.info("🧠 丘脑切换到睡眠状态，开始记忆巩固")