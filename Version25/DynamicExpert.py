from typing import List, Tuple, Dict, Optional, Any, TypedDict
import torch
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import random

# ✅ 替换为新的实体中心数据契约
from Data_models import Entity, EntityRelation, Evidence, MemoryFactory

logger = logging.getLogger("DynamicExpert")

# ------------------------------
# 类型定义（适配实体体系）
# ------------------------------
class ActivatedEntitiesResult(TypedDict):
    core_entities: List[str]  # 核心实体名称
    activated_entities: List[Dict[str, Any]]  # 激活的实体详情
    thought_chain: str  # 联想思路
    activation_strength: float  # 整体激活强度

class STDPStatistics(TypedDict):
    total_updates: int
    avg_change: float
    max_change: float
    current_timestep: int
    positive_weights: int
    negative_weights: int
    zero_weights: int

# ------------------------------
# 🔥 实体中心式神经专家模块
# ------------------------------
class DynamicExpert(nn.Module):
    def __init__(
        self, 
        name: str, 
        initial_dim: int = 2048, 
        max_dim: int = 8192, 
        active_size: int = 60,
        local_bias_enabled: bool = True, 
        local_bias_strength: float = 0.4, 
        cross_partition_decay: float = 0.7,
        max_neurons: int = 10000, 
        neuron_count: int = 2048,
    ):
        super().__init__()
        self.name: str = name
        self.dim: int = initial_dim
        self.max_dim: int = max_dim
        self.active_size: int = active_size
        self.last_dream_text: str = ""
        self.name = name
        self.max_neurons = max_neurons
        self.neuron_count = neuron_count
        
        # 加载专家配置（完全保留）
        try:
            from BrainConfig import config
            self.expert_cfg: Dict[str, Any] = config.EXPERT_CONFIG.get(name, config.EXPERT_CONFIG["概念"])
        except:
            self.expert_cfg = {
                "sparsity": 0.03,
                "local_radius": 0.2,
                "core_bias": 0.5,
                "sdr_active_count": 15
            }
            logger.warning(f"[{self.name}] 未找到 config.EXPERT_CONFIG，使用默认参数")
        
        # 局部连接配置（完全保留）
        self.local_bias_enabled: bool = local_bias_enabled
        self.local_bias_strength: float = local_bias_strength
        self.cross_partition_decay: float = cross_partition_decay
        self.partition_tags: List[str] = [name] * initial_dim
        
        # 赫布学习突触权重（完全保留，神经计算核心不变）
        self.synapse: nn.Parameter = nn.Parameter(torch.zeros(initial_dim, initial_dim), requires_grad=False)
        if self.local_bias_enabled:
            self._init_local_bias()
        
        # STDP参数（完全保留）
        self.stdp_enabled: bool = True
        self.tau_plus: float = 20.0
        self.tau_minus: float = 20.0
        self.A_plus: float = 0.01
        self.A_minus: float = 0.012
        self.stdp_learning_rate: float = 0.01

        # 预测性STDP配置（完全保留）
        self.use_predictive_stdp: bool = True
        self.expected_error: float = 0.1
        self.pe_threshold: float = 0.05
        self.predictive_lr_scale: float = 0.8
        
        # 脉冲历史（完全保留）
        self.spike_history: List[Tuple[int, int, bool]] = []
        self.current_timestep: int = 0
        self._last_delta_t: float = 0.0
        
        # 突触追踪（完全保留）
        self.synapse_change_trace: List[float] = []
        
        # ===================== 🔴 核心替换：实体存储 =====================
        # ✅ 唯一真实存储：仅保留Entity列表
        self.entities: List[Entity] = []
        
        # ✅ 性能优化映射表（SDR哈希 ↔ 实体ID）
        self.sdr_to_entity_id: Dict[int, str] = {}
        self.entity_id_to_sdr: Dict[str, int] = {}
        
        # 神经元统计（完全保留）
        self.neuron_activation_counts: torch.Tensor = torch.zeros(self.dim, dtype=torch.int32)
        self.neuron_coactivation_matrix: torch.Tensor = torch.zeros(self.dim, self.dim, dtype=torch.int32)
        self.synapse_update_counts: torch.Tensor = torch.zeros_like(self.synapse, dtype=torch.int32)
        
        # 突触修剪与新生（完全保留）
        self.pruning_enabled: bool = True
        self.synaptogenesis_enabled: bool = True
        self.pruning_percentile: float = 0.10
        self.new_synapse_rate: float = 0.05
        self.new_synapse_initial_weight: float = 0.1
        self.max_synapse_density: float = 0.15
        
        self.total_synapses_pruned: int = 0
        self.total_synapses_created: int = 0

        self.entity_id_to_index: dict[str, int] = {}  # 实体ID → 专家内索引
        self.index_to_entity_id: dict[int, str] = {}  # 专家内索引 → 实体ID

        self.free_neurons = [] # 随便给个默认值，修复属性缺失

    # ------------------------------
    # ✅ 神经初始化（完全保留，无需修改）
    # ------------------------------
    def _init_local_bias(self) -> None:
        logger.info(f"🧠 [{self.name}] 专家初始化局部连接偏置（差异化模式）...")
        cfg = self.expert_cfg
        partition_size = int(self.dim * 0.2)
        local_radius_px = int(self.dim * cfg["local_radius"])
        
        self.synapse.data = torch.randn(self.dim, self.dim) * 0.05
        core_mask = torch.zeros(self.dim, self.dim)
        core_mask[:partition_size, :partition_size] = cfg["core_bias"]
        self.synapse.data += core_mask
        
        sparsity_mask = torch.rand(self.dim, self.dim) > cfg["sparsity"]
        self.synapse.data[sparsity_mask] = 0.0
        
        for i in range(self.dim):
            start = max(0, i - local_radius_px)
            end = min(self.dim, i + local_radius_px)
            self.synapse.data[i, :start] = 0.0
            self.synapse.data[i, end:] = 0.0
        
        same_partition_count = sum(1 for i in range(self.dim) for j in range(self.dim) 
                                  if i != j and self.partition_tags[i] == self.partition_tags[j])
        local_connectivity = (torch.sum(self.synapse.data > 0.1).item() / max(same_partition_count, 1)) * 100
        core_connectivity = (torch.sum(self.synapse.data[:partition_size, :partition_size] > 0.1).item() / max(partition_size*partition_size, 1)) * 100
        
        logger.info(f"✅ [{self.name}] 局部连接偏置初始化完成")
        logger.info(f"   - 全局局部连接率: {local_connectivity:.2f}%")
        logger.info(f"   - 核心功能区连接率: {core_connectivity:.2f}%")
        logger.info(f"   - 专家配置: 稀疏度={cfg['sparsity']}, 连接半径={cfg['local_radius']}, 核心偏置={cfg['core_bias']}")

    # ------------------------------
    # ✅ 脉冲记录与STDP学习（完全保留，无需修改）
    # ------------------------------
    def record_spikes(self, sdr: torch.Tensor, is_pre_synaptic: bool = True) -> None:
        active_neurons = torch.where(sdr > 0.1)[0].cpu().numpy()
        for neuron_id in active_neurons:
            self.spike_history.append((self.current_timestep, int(neuron_id), is_pre_synaptic))
        
        if self.current_timestep > 100:
            cutoff = self.current_timestep - 100
            self.spike_history = [(t, n, p) for t, n, p in self.spike_history if t >= cutoff]

    def stdp_update(self, pre_sdr: torch.Tensor, post_sdr: torch.Tensor, delta_t: float = 0.0) -> None:
        if not self.stdp_enabled:
            self.hebbian_update(pre_sdr, post_sdr)
            return
        
        if pre_sdr.dim() == 1:
            pre_sdr = pre_sdr.unsqueeze(0)
        if post_sdr.dim() == 1:
            post_sdr = post_sdr.unsqueeze(0)
        
        self.record_spikes(pre_sdr, is_pre_synaptic=True)
        self.record_spikes(post_sdr, is_pre_synaptic=False)
        
        pre_active = torch.where(pre_sdr > 0.1)[1].cpu().numpy()
        post_active = torch.where(post_sdr > 0.1)[1].cpu().numpy()
        
        delta_w = torch.zeros_like(self.synapse.data)
        
        if delta_t != 0:
            for i in pre_active:
                for j in post_active:
                    if delta_t > 0:
                        delta_w[j, i] += self.A_plus * torch.exp(-torch.tensor(delta_t) / self.tau_plus)
                    else:
                        delta_w[j, i] -= self.A_minus * torch.exp(torch.tensor(delta_t) / self.tau_minus)
        else:
            for t_pre, pre_neuron, _ in self.spike_history:
                if pre_neuron not in pre_active:
                    continue
                for t_post, post_neuron, _ in self.spike_history:
                    if post_neuron not in post_active:
                        continue
                    dt = t_post - t_pre
                    if dt > 0:
                        delta_w[post_neuron, pre_neuron] += self.A_plus * torch.exp(-torch.tensor(dt) / self.tau_plus)
                    elif dt < 0:
                        delta_w[post_neuron, pre_neuron] -= self.A_minus * torch.exp(torch.tensor(dt) / self.tau_minus)
        
        self.synapse.data += self.stdp_learning_rate * delta_w
        self.synapse.data = torch.clamp(self.synapse.data, -1.0, 1.0)
        
        with torch.no_grad():
            pre_active_torch = torch.where(pre_sdr > 0.1)[1].cpu()
            self.neuron_activation_counts[pre_active_torch] += 1
            post_active_torch = torch.where(post_sdr > 0.1)[1].cpu()
            self.neuron_activation_counts[post_active_torch] += 1
            
            for i in pre_active_torch:
                for j in post_active_torch:
                    self.neuron_coactivation_matrix[i, j] += 1
                    self.synapse_update_counts[j, i] += 1
        
        self.neuron_activation_counts = (self.neuron_activation_counts * 0.995).to(torch.int32)
        self.neuron_coactivation_matrix = (self.neuron_coactivation_matrix * 0.995).to(torch.int32)
        
        total_change = torch.sum(torch.abs(delta_w)).item()
        self.synapse_change_trace.append(total_change)
        self.current_timestep += 1
        logger.debug(f"🧠 STDP更新 | 总权重变化: {total_change:.4f} | LTP/LTD平衡: {torch.sum(delta_w > 0).item()}/{torch.sum(delta_w < 0).item()}")

    def predictive_std_update(self, 
                             pre_sdr: torch.Tensor, 
                             post_sdr: torch.Tensor,
                             prediction_error: torch.Tensor) -> None:
        if not self.stdp_enabled:
            return
        
        device = self.synapse.device
        if pre_sdr.dim() == 1:
            pre_sdr = pre_sdr.unsqueeze(0).to(device)
        if post_sdr.dim() == 1:
            post_sdr = post_sdr.unsqueeze(0).to(device)
        prediction_error = prediction_error.to(device)
        
        self.record_spikes(pre_sdr, is_pre_synaptic=True)
        self.record_spikes(post_sdr, is_pre_synaptic=False)
        
        pre_active = torch.where(pre_sdr > 0.1)[1]
        post_active = torch.where(post_sdr > 0.1)[1]
        
        if len(pre_active) == 0 or len(post_active) == 0:
            return
        
        effective_pe = torch.where(
            torch.abs(prediction_error) > self.pe_threshold,
            prediction_error,
            torch.tensor(0.0, device=device)
        )
        
        if torch.abs(effective_pe) < 1e-6:
            logger.debug(f"🧠 [{self.name}] 预测误差过小，跳过学习 | PE={prediction_error.item():.4f}")
            return
        
        delta_w = torch.zeros_like(self.synapse.data)
        
        if hasattr(self, '_last_delta_t') and self._last_delta_t != 0:
            dt = self._last_delta_t
            if dt > 0:
                delta_w[post_active[:, None], pre_active[None, :]] += self.A_plus * torch.exp(-torch.tensor(dt) / self.tau_plus)
            else:
                delta_w[post_active[:, None], pre_active[None, :]] -= self.A_minus * torch.exp(torch.tensor(dt) / self.tau_minus)
        else:
            for t_pre, pre_neuron, _ in self.spike_history:
                if pre_neuron not in pre_active:
                    continue
                for t_post, post_neuron, _ in self.spike_history:
                    if post_neuron not in post_active:
                        continue
                    dt = t_post - t_pre
                    if dt > 0:
                        delta_w[post_neuron, pre_neuron] += self.A_plus * torch.exp(-torch.tensor(dt) / self.tau_plus)
                    elif dt < 0:
                        delta_w[post_neuron, pre_neuron] -= self.A_minus * torch.exp(torch.tensor(dt) / self.tau_minus)
        
        delta_w = delta_w * effective_pe * self.stdp_learning_rate * self.predictive_lr_scale
        
        self.synapse.data += delta_w
        self.synapse.data = torch.clamp(self.synapse.data, -1.0, 1.0)
        
        with torch.no_grad():
            self.neuron_activation_counts[pre_active] += 1
            self.neuron_activation_counts[post_active] += 1
            
            for i in pre_active:
                for j in post_active:
                    self.neuron_coactivation_matrix[i, j] += 1
                    self.synapse_update_counts[j, i] += 1
        
        self.neuron_activation_counts = (self.neuron_activation_counts * 0.995).to(torch.int32)
        self.neuron_coactivation_matrix = (self.neuron_coactivation_matrix * 0.995).to(torch.int32)
        
        total_change = torch.sum(torch.abs(delta_w)).item()
        self.synapse_change_trace.append(total_change)
        self.current_timestep += 1
        
        logger.debug(f"🧠 [{self.name}] 预测性STDP更新 | PE={prediction_error.item():.4f} | 有效PE={effective_pe.item():.4f} | 总权重变化={total_change:.4f}")

    def hebbian_update(self, pre_sdr: torch.Tensor, post_sdr: torch.Tensor, is_fact: bool = False) -> None:
        if pre_sdr.dim() == 1:
            pre_sdr = pre_sdr.unsqueeze(0)
        if post_sdr.dim() == 1:
            post_sdr = post_sdr.unsqueeze(0)
        
        lr = 0.02 if is_fact else 0.01
        decay = 0.001
        delta = lr * torch.matmul(post_sdr.T, pre_sdr)
        self.synapse.data += delta
        self.synapse.data -= decay * self.synapse.data
        self.synapse.data = torch.clamp(self.synapse.data, -1.0, 1.0)

    # ------------------------------
    # ✅ 神经前向传播（完全保留，无需修改）
    # ------------------------------
    # def forward(self, sdr: torch.Tensor, steps: int = 2, top_k: int = 60) -> torch.Tensor:
    #     if sdr.dim() > 1:
    #         sdr = sdr.squeeze()
    #     activation = sdr.float().unsqueeze(0)
        
    #     for step in range(steps):
    #         activation = self.snn_pulse_decay(activation, step, steps)
    #         activation = torch.sigmoid(torch.matmul(activation, self.synapse.T))
    #         top_k_actual = min(top_k, activation.shape[-1])
    #         top_values, top_indices = torch.topk(activation, k=top_k_actual, dim=-1)
    #         new_activation = torch.zeros_like(activation)
    #         new_activation.scatter_(-1, top_indices, top_values)
    #         activation = new_activation
        
    #     return activation.squeeze(0)

    # ===================== 替换原方法 =====================
    def forward(self, sdr: torch.Tensor, steps: int = 2, top_k: int = 60, add_noise: bool = True) -> torch.Tensor:
        """
        ✅ 修复版：带动态神经噪声的前向传播，打破突触固化
        """
        if sdr.dim() > 1:
            sdr = sdr.squeeze()
        activation = sdr.float().unsqueeze(0)
        
        for step in range(steps):
            activation = self.snn_pulse_decay(activation, step, steps)
            activation = torch.sigmoid(torch.matmul(activation, self.synapse.T))
            
            # ✅ 新增：每步加入微小高斯噪声（模拟大脑神经噪声）
            if add_noise:
                noise = torch.randn_like(activation) * 0.02  # 2%噪声
                activation += noise
            
            top_k_actual = min(top_k, activation.shape[-1])
            top_values, top_indices = torch.topk(activation, k=top_k_actual, dim=-1)
            new_activation = torch.zeros_like(activation)
            new_activation.scatter_(-1, top_indices, top_values)
            activation = new_activation
        
        return activation.squeeze(0)

    # ------------------------------
    # ✅ 核心：实体神经检索
    # ------------------------------
    def retrieve(self, query_sdr: torch.Tensor, top_k: int = 30, steps: int = 3, target_expert: Optional[str] = None, query_text: str = "") -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        实体中心式神经检索
        返回格式：(entity_id, 得分, 实体详情字典)
        """
        logger.info(f"[{self.name}] 神经检索：当前总实体数 = {len(self.entities)}")
        if not self.entities:
            return []
        
        # 神经激活传播（核心逻辑完全不变）
        activated_sdr = self.forward(query_sdr, steps=steps)
        results = []
        total_entities = len(self.entities)
        
        for idx, entity in enumerate(self.entities):
            # 突触激活重叠得分（完全复用原有逻辑）
            activate_threshold = 0.1
            overlap = torch.sum((activated_sdr > activate_threshold) & (entity.sdr > activate_threshold)).item()
            total_active = torch.sum(activated_sdr > activate_threshold).item()
            recall_rate = overlap / total_active if total_active > 0 else 0.0

            # 吸引子相似度（完全复用原有逻辑）
            mem_evolved = self.forward(entity.sdr, steps=steps)
            attractor_sim = F.cosine_similarity(activated_sdr, mem_evolved, dim=-1).item()
            synaptic_score = recall_rate * 0.7 + attractor_sim * 0.3

            # 时间衰减（使用Entity的created_at字段）
            time_decay = 0.995 ** (total_entities - idx - 1)
            time_decay = max(time_decay, 0.4)
            
            # 专家权重（使用Entity的expert计算属性）
            expert_weight = 1.0
            if target_expert:
                if entity.expert == target_expert:
                    expert_weight = 1.2
            
            # 查询类型加权（完全复用原有逻辑）
            query_lower = query_text.lower()
            if any(k in query_lower for k in ["哪里", "在哪", "地址", "位置", "住", "学校"]):
                if entity.expert == "空间":
                    expert_weight *= 1.5
            if any(k in query_lower for k in ["谁", "名字", "身份", "主人", "叫什么"]):
                if entity.expert == "身份":
                    expert_weight *= 1.5

            # 使用频率加权（使用Entity的access_count字段）
            use_weight = 1.0 + min(entity.access_count * 0.1, 0.5)

            final_score = synaptic_score * time_decay * expert_weight * use_weight
            
            if final_score < 0.001:
                continue
            
            # 返回实体详情
            entity_detail = {
                "name": entity.name,
                "type": entity.entity_type,
                "attributes": entity.attributes,
                "latest_evidence": entity.latest_evidence.content if entity.latest_evidence else "",
                "metadata": entity.metadata
            }
            results.append((entity.entity_id, final_score, entity_detail))
        
        # 兜底逻辑
        if not results and len(self.entities) > 0:
            logger.info(f"[{self.name}] 神经检索：兜底返回前3条实体")
            for i in range(min(3, len(self.entities))):
                entity = self.entities[i]
                entity_detail = {
                    "name": entity.name,
                    "type": entity.entity_type,
                    "attributes": entity.attributes,
                    "latest_evidence": entity.latest_evidence.content if entity.latest_evidence else "",
                    "metadata": entity.metadata
                }
                results.append((entity.entity_id, 0.5, entity_detail))
        
        results.sort(key=lambda x: -x[1])
        logger.info(f"[{self.name}] 神经检索：返回 {len(results)} 个实体")
        return results[:top_k]

    def retrieve_multi_hop(self, query_sdr: torch.Tensor, hops: int = 3, top_k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        """多跳实体联想检索（逻辑不变，仅适配返回格式）"""
        current_sdr = query_sdr
        all_results = []
        for hop in range(hops):
            current_sdr = self.forward(current_sdr, steps=1)
            hop_results = self.retrieve(current_sdr, top_k=top_k // hops)
            all_results.extend(hop_results)
        all_results.sort(key=lambda x: -x[1])
        return all_results[:top_k]

    # ------------------------------
    # ✅ 核心：添加实体
    # ------------------------------
    # DynamicExpert.py 实体ID生成修复
    def add_entity(self, entity: Entity) -> None:
        """添加实体到专家网络（修复张量索引崩溃 + 哈希冲突）"""
        if entity.entity_id in self.entity_id_to_index:
            logger.debug(f"实体已存在，跳过添加: {entity.name} ({entity.entity_id})")
            return
        
        # 分配神经元索引
        if len(self.free_neurons) > 0:
            neuron_idx = self.free_neurons.pop()
        else:
            neuron_idx = len(self.entities)
            if neuron_idx >= self.max_neurons:
                logger.warning(f"⚠️ 专家 [{self.name}] 神经元已满，无法添加新实体")
                return
        
        # 建立索引
        self.entity_id_to_index[entity.entity_id] = neuron_idx
        self.index_to_entity_id[neuron_idx] = entity.entity_id
        self.entities.append(entity)
        
        # ===================== 核心修复：synapses → synapse（单数） =====================
        if entity.sdr is not None and isinstance(entity.sdr, torch.Tensor):
            try:
                # 获取激活的神经元（张量）
                active_neurons = torch.where(entity.sdr > 0.5)[0]
                # 关键：张量转普通整数，安全索引
                for neuron in active_neurons.cpu().numpy().tolist():
                    if 0 <= neuron < self.neuron_count:
                        # ✅ 修复：self.synapse（你定义的名字），不是 self.synapses
                        self.synapse[neuron_idx, neuron] = 0.5
            except Exception as e:
                logger.warning(f"⚠️ 专家 [{self.name}] 实体 {entity.name} 突触初始化异常: {str(e)}")
        
        logger.debug(f"✅ 实体添加到专家 [{self.name}]: {entity.name} ({entity.entity_id})")
    
    # ✅ 核心：删除实体
    # ------------------------------
    def delete_entity(self, entity_id: str) -> None:
        """删除指定实体"""
        if entity_id not in self.entity_id_to_sdr:
            logger.warning(f"[{self.name}] 未找到实体ID: {entity_id}")
            return
        
        # 清理映射表
        sdr_hash = self.entity_id_to_sdr.pop(entity_id)
        self.sdr_to_entity_id.pop(sdr_hash, None)
        
        # 清理真实存储
        self.entities = [e for e in self.entities if e.entity_id != entity_id]
        
        logger.info(f"[{self.name}] 实体 {entity_id} 已删除 | 剩余实体:{len(self.entities)}")

    # ------------------------------
    # ✅ 睡眠巩固（适配实体体系）
    # ------------------------------
    def sleep_consolidate(
        self, 
        epochs: int = 3, 
        priority_entity_ids: Optional[List[str]] = None,
        dopamine_system: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        🔥 实体中心式睡眠巩固（神经科学对齐版）
        三阶段流程：高优先级实体强化回放 → 梦境随机回放 → 全量突触重塑
        :param epochs: 全量回放轮数
        :param priority_entity_ids: 元认知标记的高优先级实体ID
        :param dopamine_system: 多巴胺系统实例（用于离线重放）
        :return: 巩固统计结果（用于生成睡眠报告）
        """
        if not self.entities:
            logger.info(f"🌙 专家 [{self.name}] 无实体，跳过睡眠巩固")
            return {
                "entities_consolidated": 0,
                "entities_skipped": 0,
                "synapses_pruned": 0,
                "synapses_created": 0,
                "dream_entities": [],
                "dream_text": ""
            }
        
        logger.info(f"\n🌙 专家 [{self.name}] 开始睡眠巩固 (epochs={epochs})")
        start_time = time.time()
        
        # ===================== 🔴 步骤1：过滤有效实体 =====================
        # 跳过已过时、已完全巩固的实体
        valid_entities = []
        for idx, entity in enumerate(self.entities):
            if entity.is_obsolete:
                logger.debug(f"   ⏭️ 跳过过时实体: {entity.name}")
                continue
            if entity.consolidation_level >= 1.0:
                logger.debug(f"   ⏭️ 跳过已完全巩固实体: {entity.name}")
                continue
            valid_entities.append((idx, entity))
        
        if not valid_entities:
            logger.info(f"🌙 专家 [{self.name}] 无需要巩固的实体，跳过")
            return {
                "entities_consolidated": 0,
                "entities_skipped": len(self.entities),
                "synapses_pruned": 0,
                "synapses_created": 0,
                "dream_entities": [],
                "dream_text": ""
            }
        
        logger.info(f"🧠 待巩固实体: {len(valid_entities)}/{len(self.entities)} 个")
        
        # ===================== 🔴 步骤2：计算回放优先级 =====================
        # 多维度优先级评分（0-1）：重要性(40%) + 新鲜度(30%) + 巩固进度(20%) + 优先级标记(10%)
        priority_scores = {}
        priority_indices = set()
        
        # 标记高优先级实体
        if priority_entity_ids:
            logger.info(f"🧠 元认知高优先级实体: {len(priority_entity_ids)} 个")
            for idx, entity in valid_entities:
                if entity.entity_id in priority_entity_ids:
                    priority_indices.add(idx)
        
        # 计算每个实体的综合优先级
        for idx, entity in valid_entities:
            # 1. 重要性得分（0-1）
            importance_score = entity.importance
            
            # 2. 新鲜度得分（0-1）：30天衰减到0
            days_since_access = (time.time() - entity.last_accessed) / (24 * 3600)
            freshness_score = max(0.0, 1.0 - days_since_access / 30)
            
            # 3. 巩固进度得分（0-1）：未巩固的优先级更高
            consolidation_score = 1.0 - entity.consolidation_level
            
            # 4. 优先级标记得分
            priority_score = 1.0 if idx in priority_indices else 0.0
            
            # 综合得分
            total_score = (
                0.4 * importance_score 
                + 0.3 * freshness_score 
                + 0.2 * consolidation_score 
                + 0.1 * priority_score
            )
            
            priority_scores[idx] = total_score
            logger.debug(
                f"   优先级 | {entity.name:20} | 重要性:{importance_score:.2f} | 新鲜度:{freshness_score:.2f} | "
                f"巩固度:{consolidation_score:.2f} | 综合:{total_score:.2f}"
            )
        
        # ===================== 🔴 步骤3：高优先级实体强化回放 =====================
        if priority_indices:
            logger.info(f"\n🧠 开始高优先级实体强化回放 (额外2轮)")
            # 按优先级排序高优先级实体
            sorted_priority = sorted(
                [idx for idx in priority_indices],
                key=lambda x: priority_scores[x],
                reverse=True
            )
            
            for extra_epoch in range(2):
                logger.debug(f"   强化回放轮次 {extra_epoch+1}/2")
                for idx in sorted_priority:
                    entity = self.entities[idx]
                    # 更新实体回放状态
                    entity.replay_count += 1
                    entity.update_access()
                    # 神经可塑性更新
                    if self.stdp_enabled:
                        self.stdp_update(entity.sdr, entity.sdr, delta_t=15.0)
                    else:
                        self.hebbian_update(
                            entity.sdr, 
                            entity.sdr, 
                            is_fact=entity.metadata.get('is_fact', False)
                        )
                    logger.debug(f"      ✅ 强化回放 | {entity.name} | 回放次数:{entity.replay_count}")
        
        # ===================== 🔴 步骤4：梦境随机回放 =====================
        logger.info(f"\n🌙 开始梦境随机回放")
        # 按优先级加权采样（优先级越高，被选中概率越大）
        all_indices = [idx for idx, _ in valid_entities]
        weights = [priority_scores[idx] for idx in all_indices]
        weights = np.array(weights) / sum(weights)  # 归一化
        
        # 采样6个实体用于梦境回放
        dream_count = min(6, len(valid_entities))
        dream_indices = np.random.choice(all_indices, size=dream_count, replace=False, p=weights)
        
        dream_fragments = []
        dream_contents = []
        
        for idx in dream_indices:
            entity = self.entities[idx]
            # 更新实体回放状态
            entity.replay_count += 1
            entity.update_access()
            # 神经激活
            active_neurons = torch.where(entity.sdr > 0.1)[0].cpu().numpy()
            neuron_count = len(active_neurons)
            # 标记是否为高优先级
            is_priority = idx in priority_indices
            priority_marker = " ⭐" if is_priority else ""
            # 获取最新证据内容
            evidence_content = entity.latest_evidence.content[:30] + "..." if entity.latest_evidence else "无证据"
            
            logger.info(
                f"     🌙 梦境回放{priority_marker} | 神经元:{neuron_count:3d} | "
                f"实体:{entity.name:15} | 证据:{evidence_content}"
            )
            
            # 结构化梦境片段（对接DreamResult）
            dream_fragments.append({
                "entity_id": entity.entity_id,
                "entity_name": entity.name,
                "content": evidence_content,
                "activation_score": priority_scores[idx],
                "is_priority": is_priority,
                "expert": self.name
            })
            dream_contents.append(f"{entity.name}（{evidence_content}）")
        
        # 生成梦境文本
        if dream_contents:
            self.last_dream_text = "我刚刚梦里梦到了：" + "，还梦到了：".join(dream_contents)
        else:
            self.last_dream_text = ""
        
        logger.info(f"✅ 梦境回放完成 | 共回放 {len(dream_fragments)} 个实体")
        
        # ===================== 🔴 步骤5：全量实体巩固回放 =====================
        logger.info(f"\n🧠 开始全量实体巩固回放 (epochs={epochs})")
        # 按优先级从高到低排序所有有效实体
        sorted_all = sorted(
            valid_entities,
            key=lambda x: priority_scores[x[0]],
            reverse=True
        )
        
        consolidated_count = 0
        for epoch in range(epochs):
            logger.debug(f"   全量回放轮次 {epoch+1}/{epochs}")
            for idx, entity in sorted_all:
                # 神经可塑性更新
                if self.stdp_enabled:
                    self.stdp_update(entity.sdr, entity.sdr, delta_t=10.0)
                else:
                    self.hebbian_update(
                        entity.sdr, 
                        entity.sdr, 
                        is_fact=entity.metadata.get('is_fact', False)
                    )
                # 提升巩固进度
                entity.consolidation_level = min(1.0, entity.consolidation_level + 0.1 / epochs)
                consolidated_count += 1
        
        # ===================== 🔴 步骤6：多巴胺离线重放 =====================
        if dopamine_system:
            logger.info(f"\n🧠 开始多巴胺离线重放")
            # 筛选过去24小时内有奖励记录的实体
            reward_entities = [
                entity for _, entity in valid_entities
                if time.time() - entity.created_at < 24 * 3600
                and entity.importance > 0.7
            ]
            if reward_entities:
                logger.info(f"   待重放高奖励实体: {len(reward_entities)} 个")
                # 这里可以调用多巴胺系统的离线重放方法
                # dopamine_system.dopamine_offline_replay_for_entities(reward_entities)
        
        # ===================== 🔴 步骤7：突触重塑 =====================
        logger.info(f"\n🔧 开始突触重塑")
        # 智能修剪和新生
        pruned = self.synaptic_pruning()
        created = self.synaptogenesis()
        
        # 基础弱连接清理
        weak_threshold = 0.01
        num_weak = torch.sum(torch.abs(self.synapse.data) < weak_threshold).item()
        total_synapses = self.synapse.data.numel()
        self.synapse.data[torch.abs(self.synapse.data) < weak_threshold] = 0.0
        
        # 计算稀疏度
        sparsity = self.get_sparsity() * 100
        
        # ===================== 🔴 步骤8：统计与返回 =====================
        duration = time.time() - start_time
        logger.info(f"\n✅ 专家 [{self.name}] 睡眠巩固完成 | 耗时: {duration:.2f}秒")
        logger.info(f"   巩固实体: {consolidated_count} 个")
        logger.info(f"   突触稀疏度: {sparsity:.2f}%")
        logger.info(f"   基础弱连接清理: {num_weak}/{total_synapses} ({num_weak/total_synapses:.2%})")
        logger.info(f"   智能修剪: {pruned} 个 | 智能新生: {created} 个")
        logger.info(f"   累计修剪: {self.total_synapses_pruned} | 累计新生: {self.total_synapses_created}")
        
        return {
            "entities_consolidated": consolidated_count,
            "entities_skipped": len(self.entities) - len(valid_entities),
            "synapses_pruned": pruned + num_weak,
            "synapses_created": created,
            "dream_entities": dream_fragments,
            "dream_text": self.last_dream_text,
            "sparsity": round(sparsity, 2),
            "duration": round(duration, 2)
        }

    def get_sparsity(self) -> float:
        if self.synapse is None:
            return 0.0
        return (torch.abs(self.synapse.data) < 0.01).float().mean().item()

    # ------------------------------
    # ✅ 保存/加载（适配实体体系）
    # ------------------------------
    def save_weights(self, path: str) -> None:
        try:
            torch.save({
                'version': '3.0',  # 实体中心版本号
                'synapse': self.synapse.data,
                'entities': self.entities,
                'sdr_to_entity_id': self.sdr_to_entity_id,
                'entity_id_to_sdr': self.entity_id_to_sdr,
                'dim': self.dim,
                'partition_tags': self.partition_tags,
                'local_bias_enabled': self.local_bias_enabled,
                'stdp_enabled': self.stdp_enabled,
                'tau_plus': self.tau_plus,
                'tau_minus': self.tau_minus,
                'A_plus': self.A_plus,
                'A_minus': self.A_minus,
                'current_timestep': self.current_timestep,
                'neuron_activation_counts': self.neuron_activation_counts,
                'neuron_coactivation_matrix': self.neuron_coactivation_matrix,
                'synapse_update_counts': self.synapse_update_counts,
                'total_synapses_pruned': self.total_synapses_pruned,
                'total_synapses_created': self.total_synapses_created,
            }, path)
            logger.info(f"💾 专家 [{self.name}] 权重已保存: {path} | 实体数:{len(self.entities)} | 版本:3.0")
        except Exception as e:
            logger.error(f"❌ 专家 [{self.name}] 权重保存失败: {e}")

    def load_weights(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning(f"[{self.name}] 权重文件不存在，初始化新权重")
            return
        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
            version = data.get('version', '2.0')
            
            # 加载神经参数（完全保留）
            self.synapse.data = data['synapse']
            self.dim = data.get('dim', self.dim)
            self.partition_tags = data.get('partition_tags', [self.name] * self.dim)
            self.local_bias_enabled = data.get('local_bias_enabled', True)
            self.stdp_enabled = data.get('stdp_enabled', True)
            self.tau_plus = data.get('tau_plus', 20.0)
            self.tau_minus = data.get('tau_minus', 20.0)
            self.A_plus = data.get('A_plus', 0.01)
            self.A_minus = data.get('A_minus', 0.012)
            self.current_timestep = data.get('current_timestep', 0)
            self.neuron_activation_counts = data.get('neuron_activation_counts', torch.zeros(self.dim, dtype=torch.int32))
            self.neuron_coactivation_matrix = data.get('neuron_coactivation_matrix', torch.zeros(self.dim, self.dim, dtype=torch.int32))
            self.synapse_update_counts = data.get('synapse_update_counts', torch.zeros_like(self.synapse, dtype=torch.int32))
            self.total_synapses_pruned = data.get('total_synapses_pruned', 0)
            self.total_synapses_created = data.get('total_synapses_created', 0)
            
            # 加载实体数据
            self.entities = data.get('entities', [])
            self.sdr_to_entity_id = data.get('sdr_to_entity_id', {})
            self.entity_id_to_sdr = data.get('entity_id_to_sdr', {})
            
            logger.info(f"✅ 专家 [{self.name}] 加载完成 | 版本:{version} | 实体数: {len(self.entities)}")
        except Exception as e:
            logger.error(f"❌ 专家 [{self.name}] 权重加载失败: {e}", exc_info=True)
            self.synapse.data = torch.zeros(self.dim, self.dim)
            if self.local_bias_enabled:
                self._init_local_bias()
            self.entities = []
            self.sdr_to_entity_id = {}
            self.entity_id_to_sdr = {}

    # ------------------------------
    # ✅ 实体激活（适配实体体系）
    # ------------------------------
    def activate_entities(self, entity_semantic_vecs: List[torch.Tensor], steps: int = 2) -> ActivatedEntitiesResult:
        if not self.entities:
            return ActivatedEntitiesResult(
                core_entities=[], 
                activated_entities=[], 
                thought_chain="无激活实体", 
                activation_strength=0.0
            )
        
        # 平均语义向量作为初始激活
        activation = torch.stack(entity_semantic_vecs).mean(dim=0)
        activation = activation / (activation.norm() + 1e-8)
        activation = self.forward(activation, steps=steps)
        
        # 获取激活的实体
        activated_indices = self._retrieve_activated_indices_safe(activation, topk=5)
        activated_entities = [self.get_entity_by_idx_safe(i) for i in activated_indices if i >= 0]
        
        # 构建联想链
        chain = self._build_thought_chain(activated_entities)
        core_entities = list(set([ent["name"] for ent in activated_entities]))
        
        return ActivatedEntitiesResult(
            core_entities=core_entities,
            activated_entities=activated_entities,
            thought_chain=chain,
            activation_strength=activation.norm().item()
        )

    def _retrieve_activated_indices_safe(self, activation: torch.Tensor, topk: int = 5) -> List[int]:
        if not self.entities:
            return []
        results = []
        for i, entity in enumerate(self.entities):
            sim = F.cosine_similarity(activation.squeeze(), entity.sdr, dim=-1).item()
            results.append((sim, i))
        results.sort(key=lambda x: -x[0])
        return [i for (sim, i) in results[:topk]]

    def get_entity_by_idx_safe(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= len(self.entities):
            return {"name": "", "type": "", "attributes": {}, "latest_evidence": ""}
        entity = self.entities[idx]
        return {
            "entity_id": entity.entity_id,
            "name": entity.name,
            "type": entity.entity_type,
            "attributes": entity.attributes,
            "latest_evidence": entity.latest_evidence.content if entity.latest_evidence else ""
        }

    def _build_thought_chain(self, entities: List[Dict[str, Any]]) -> str:
        if not entities:
            return "无关联实体"
        chain = " → ".join([f"{ent['name']}({ent['type']})" for ent in entities])
        return f"联想思路：{chain}"

    # ------------------------------
    # ✅ 预测相关方法（完全保留，无需修改）
    # ------------------------------
    def init_predictor(self) -> None:
        if not hasattr(self, "predict_head"):
            self.predict_head = nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.LayerNorm(self.dim),
                nn.GELU(),
                nn.Linear(self.dim, self.dim)
            ).to(self.synapse.device)
            self.predict_head.eval()

    def snn_pulse_decay(self, activation: torch.Tensor, step: int, total_steps: int) -> torch.Tensor:
        with torch.no_grad():
            decay_rate = torch.exp(-torch.linspace(0, 1.5, total_steps, device=activation.device)[step])
        return activation * decay_rate

    def predict_next_sdr(self, current_activation: torch.Tensor) -> torch.Tensor:
        self.init_predictor()
        with torch.no_grad():
            return self.predict_head(current_activation)

    def update_prediction(self, pred_sdr: torch.Tensor, real_sdr: torch.Tensor) -> float:
        if pred_sdr.dim() == 1:
            pred_sdr = pred_sdr.unsqueeze(0)
        if real_sdr.dim() == 1:
            real_sdr = real_sdr.unsqueeze(0)
        
        with torch.no_grad():
            prediction_error = F.mse_loss(pred_sdr, real_sdr)
        
        rpe = prediction_error - self.expected_error
        self.expected_error = 0.9 * self.expected_error + 0.1 * prediction_error
        
        try:
            if self.use_predictive_stdp:
                self.predictive_std_update(real_sdr, pred_sdr, rpe)
            else:
                delta_t = 5.0 * (1.0 - prediction_error.item())
                self._last_delta_t = delta_t
                self.stdp_update(real_sdr, pred_sdr, delta_t=delta_t)
        except Exception as e:
            logger.error(f"❌ [{self.name}] STDP更新失败，回退到赫布学习: {e}")
            self.hebbian_update(real_sdr, pred_sdr)
        
        return prediction_error.item()

    # ------------------------------
    # ✅ 突触修剪与新生（完全保留，无需修改）
    # ------------------------------
    def get_synapse_change(self) -> float:
        if not hasattr(self, 'synapse'):
            return 0.0
        return float(torch.sum(torch.abs(self.synapse)).item())

    def synaptic_pruning(self, pruning_percentile: Optional[float] = None) -> int:
        if not self.pruning_enabled:
            return 0
        pruning_percentile = pruning_percentile or self.pruning_percentile
        
        with torch.no_grad():
            weight_importance = torch.abs(self.synapse.data)
            update_importance = torch.log1p(self.synapse_update_counts.float())
            total_importance = weight_importance * (1.0 + 0.5 * update_importance)
            non_zero_mask = (self.synapse.data != 0)
            non_zero_importance = total_importance[non_zero_mask]
            
            if len(non_zero_importance) == 0:
                return 0
            
            threshold = torch.quantile(non_zero_importance, pruning_percentile)
            prune_mask = (total_importance < threshold) & non_zero_mask
            num_pruned = int(prune_mask.sum().item())
            self.synapse.data[prune_mask] = 0.0
            self.synapse_update_counts[prune_mask] = 0
            self.total_synapses_pruned += num_pruned
            current_sparsity = (self.synapse.data == 0).float().mean().item()
            logger.info(f"🧠 [{self.name}] 突触修剪 | 修剪: {num_pruned} | 累计: {self.total_synapses_pruned} | 当前稀疏度: {current_sparsity:.2%}")
            return num_pruned

    def synaptogenesis(self, new_synapse_rate: Optional[float] = None) -> int:
        if not self.synaptogenesis_enabled:
            return 0
        new_synapse_rate = new_synapse_rate or self.new_synapse_rate
        
        with torch.no_grad():
            activation_threshold = torch.quantile(self.neuron_activation_counts.float(), 0.7)
            high_activation_neurons = torch.where(self.neuron_activation_counts > activation_threshold)[0].tolist()
            
            if len(high_activation_neurons) < 2:
                high_activation_neurons = list(range(min(50, self.dim)))
            
            candidate_pairs = []
            max_candidates = 2000
            sample_size = min(len(high_activation_neurons) * 20, max_candidates)
            
            for _ in range(sample_size):
                i = random.choice(high_activation_neurons)
                j = random.choice(high_activation_neurons)
                if i != j and abs(self.synapse.data[j, i].item()) < 0.01:
                    coactivation = self.neuron_coactivation_matrix[i, j].item()
                    candidate_pairs.append((i, j, coactivation))
            
            if not candidate_pairs:
                for _ in range(50):
                    i = random.randint(0, self.dim-1)
                    j = random.randint(0, self.dim-1)
                    if i != j and abs(self.synapse.data[j, i].item()) < 0.01:
                        candidate_pairs.append((i, j, 0))
            
            candidate_pairs.sort(key=lambda x: -x[2])
            current_non_zero = (self.synapse.data != 0).sum().item()
            max_possible = int(self.dim * self.dim * self.max_synapse_density) - current_non_zero
            num_to_create = min(len(candidate_pairs), max(int(current_non_zero * new_synapse_rate), 50), max_possible)
            
            created_count = 0
            for i, j, coactivation in candidate_pairs[:num_to_create]:
                self.synapse.data[j, i] = self.new_synapse_initial_weight * (1.0 + coactivation / 10.0)
                self.synapse_update_counts[j, i] = 1
                created_count += 1
            
            self.total_synapses_created += created_count
            logger.info(f"🧠 [{self.name}] 突触新生 | 新增: {created_count} | 累计: {self.total_synapses_created}")
            return created_count

    def analyze_stdp_statistics(self) -> STDPStatistics:
        if not self.synapse_change_trace:
            return STDPStatistics(
                total_updates=0, avg_change=0.0, max_change=0.0,
                current_timestep=self.current_timestep, positive_weights=0,
                negative_weights=0, zero_weights=self.synapse.data.numel()
            )
        
        return STDPStatistics(
            total_updates=len(self.synapse_change_trace),
            avg_change=sum(self.synapse_change_trace) / len(self.synapse_change_trace),
            max_change=max(self.synapse_change_trace),
            current_timestep=self.current_timestep,
            positive_weights=torch.sum(self.synapse.data > 0).item(),
            negative_weights=torch.sum(self.synapse.data < 0).item(),
            zero_weights=torch.sum(self.synapse.data == 0).item()
        )