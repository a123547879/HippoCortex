# 在文件顶部添加导入
from typing import List, Tuple, Dict, Optional, Any, TypedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import random

logger = logging.getLogger("DynamicExpert")

# 在导入后添加
class ActivatedMemoriesResult(TypedDict):
    """activate_memories方法的返回值类型"""
    core_ideas: List[str]
    activated_memories: List[Dict[str, Any]]
    thought_chain: str
    activation_strength: float

class STDPStatistics(TypedDict):
    """analyze_stdp_statistics方法的返回值类型"""
    total_updates: int
    avg_change: float
    max_change: float
    current_timestep: int
    positive_weights: int
    negative_weights: int
    zero_weights: int

class DynamicExpert(nn.Module):
    def __init__(
        self, 
        name: str, 
        initial_dim: int = 2048, 
        max_dim: int = 8192, 
        active_size: int = 60,
        local_bias_enabled: bool = True, 
        local_bias_strength: float = 0.4, 
        cross_partition_decay: float = 0.7
    ):
        super().__init__()
        self.name: str = name  # 支持：身份/概念/空间/抽象/视觉
        self.dim: int = initial_dim
        self.max_dim: int = max_dim
        self.active_size: int = active_size
        self.last_dream_text: str = ""
        
        # 加载专家专属差异化配置
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
        
        # 局部连接偏置配置
        self.local_bias_enabled: bool = local_bias_enabled
        self.local_bias_strength: float = local_bias_strength
        self.cross_partition_decay: float = cross_partition_decay
        self.partition_tags: List[str] = [name] * initial_dim
        
        # 赫布学习突触权重矩阵
        self.synapse: nn.Parameter = nn.Parameter(torch.zeros(initial_dim, initial_dim), requires_grad=False)
        if self.local_bias_enabled:
            self._init_local_bias()
        
        # STDP核心参数
        self.stdp_enabled: bool = True  # 是否启用STDP
        self.tau_plus: float = 20.0       # LTP时间常数(ms) - 突触前先激活
        self.tau_minus: float = 20.0      # LTD时间常数(ms) - 突触后先激活
        self.A_plus: float = 0.01         # LTP幅度
        self.A_minus: float = 0.012       # LTD幅度（略大于A_plus，实现整体稳定性）
        self.stdp_learning_rate: float = 0.01  # STDP学习率
        
        # 脉冲历史记录（用于STDP时序计算）
        self.spike_history: List[Tuple[int, int, bool]] = []  # 存储 (timestep, neuron_id, is_pre)
        self.current_timestep: int = 0
        
        # 突触追踪器（用于可视化和调试）
        self.synapse_change_trace: List[float] = []
        
        # 历史SDR记忆库
        self.sdr_list: List[torch.Tensor] = []
        self.content_list: List[str] = []
        self.metadata_list: List[Dict[str, Any]] = []
        
        # SDR ↔ mem_id 双向映射（稳定版）
        self.sdr_to_mem_id: Dict[int, int] = {}  # sdr_hash -> mem_id
        self.mem_id_to_sdr: Dict[int, int] = {}  # mem_id -> sdr_hash
        
        # 突触修剪与新生核心参数
        self.neuron_activation_counts: torch.Tensor = torch.zeros(self.dim, dtype=torch.int32)
        self.neuron_coactivation_matrix: torch.Tensor = torch.zeros(self.dim, self.dim, dtype=torch.int32)
        self.synapse_update_counts: torch.Tensor = torch.zeros_like(self.synapse, dtype=torch.int32)
        
        self.pruning_enabled: bool = True          # 是否启用修剪
        self.synaptogenesis_enabled: bool = True   # 是否启用新生
        self.pruning_percentile: float = 0.10       # 每次修剪最弱的10%突触
        self.new_synapse_rate: float = 0.05         # 每次新生最多新增5%的突触
        self.new_synapse_initial_weight: float = 0.1 # 新突触的初始权重
        self.max_synapse_density: float = 0.15      # 最大突触密度（15%）
        
        self.total_synapses_pruned: int = 0
        self.total_synapses_created: int = 0
    
    def _init_local_bias(self) -> None:
        """
        终极修复版：
        1. 激活核心功能区（前20%神经元）
        2. 专家差异化连接模式
        3. 局部连接半径差异化
        """
        logger.info(f"🧠 [{self.name}] 专家初始化局部连接偏置（差异化模式）...")
        
        cfg = self.expert_cfg
        partition_size = int(self.dim * 0.2)  # 前20%为核心功能区
        local_radius_px = int(self.dim * cfg["local_radius"])  # 连接半径（像素级）
        
        # 1. 基础初始化：小噪声
        self.synapse.data = torch.randn(self.dim, self.dim) * 0.05
        
        # 核心修复1：激活核心功能区
        core_mask = torch.zeros(self.dim, self.dim)
        core_mask[:partition_size, :partition_size] = cfg["core_bias"]
        self.synapse.data += core_mask
        
        # 核心修复2：专家差异化稀疏度
        sparsity_mask = torch.rand(self.dim, self.dim) > cfg["sparsity"]
        self.synapse.data[sparsity_mask] = 0.0
        
        # 核心修复3：局部连接半径差异化
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
    
    def record_spikes(self, sdr: torch.Tensor, is_pre_synaptic: bool = True) -> None:
        """
        记录脉冲发放时间，用于STDP时序计算
        :param sdr: 稀疏分布表征
        :param is_pre_synaptic: 是否为突触前脉冲
        """
        active_neurons = torch.where(sdr > 0.1)[0].cpu().numpy()
        for neuron_id in active_neurons:
            self.spike_history.append((self.current_timestep, int(neuron_id), is_pre_synaptic))
        
        if self.current_timestep > 100:
            cutoff = self.current_timestep - 100
            self.spike_history = [(t, n, p) for t, n, p in self.spike_history if t >= cutoff]
    
    def stdp_update(self, pre_sdr: torch.Tensor, post_sdr: torch.Tensor, delta_t: float = 0.0) -> None:
        """
        真正的STDP（脉冲时序依赖可塑性）+ 激活记录
        核心原理：
        - 如果突触前脉冲在突触后脉冲之前(Δt > 0) → 突触增强(LTP)
        - 如果突触前脉冲在突触后脉冲之后(Δt < 0) → 突触抑制(LTD)
        """
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
            
            if delta_t != 0 or len(pre_active_torch) > 0:
                for i in pre_active_torch:
                    for j in post_active_torch:
                        self.synapse_update_counts[j, i] += 1
        
        self.neuron_activation_counts = (self.neuron_activation_counts * 0.995).to(torch.int32)
        self.neuron_coactivation_matrix = (self.neuron_coactivation_matrix * 0.995).to(torch.int32)
        
        total_change = torch.sum(torch.abs(delta_w)).item()
        self.synapse_change_trace.append(total_change)
        
        self.current_timestep += 1
        
        logger.debug(f"🧠 STDP更新 | 总权重变化: {total_change:.4f} | LTP/LTD平衡: {torch.sum(delta_w > 0).item()}/{torch.sum(delta_w < 0).item()}")
    
    def hebbian_update(self, pre_sdr: torch.Tensor, post_sdr: torch.Tensor, is_fact: bool = False) -> None:
        """
        保留原有的赫布学习作为备选方案
        改进的赫布学习规则（类脑核心学习机制）
        局部连接偏置作为初始优势，赫布学习会在其基础上继续强化
        """
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
    
    def forward(self, sdr: torch.Tensor, steps: int = 2, top_k: int = 60) -> torch.Tensor:
        """
        SDR在专家网络中传播激活（核心类脑推理）
        局部连接偏置 + SNN脉冲时序衰减
        """
        if sdr.dim() == 1:
            sdr = sdr.unsqueeze(0)
        
        activation = sdr.float()
        
        for step in range(steps):
            activation = self.snn_pulse_decay(activation, step, steps)
            activation = torch.sigmoid(torch.matmul(activation, self.synapse.T))
            
            top_k_actual = min(top_k, activation.shape[-1])
            top_values, top_indices = torch.topk(activation, k=top_k_actual, dim=-1)
            new_activation = torch.zeros_like(activation)
            new_activation.scatter_(-1, top_indices, top_values)
            activation = new_activation
        
        return activation.squeeze(0)
    
    def retrieve(self, query_sdr: torch.Tensor, top_k: int = 10, steps: int = 2, target_expert: Optional[str] = None) -> List[Tuple[float, str, Dict[str, Any], int, Optional[int]]]:
        if not self.sdr_list:
            return []
        
        activated_sdr = self.forward(query_sdr, steps=steps)
        
        results = []
        total_mem = len(self.sdr_list)
        for i, hist_sdr in enumerate(self.sdr_list):
            sim = F.cosine_similarity(activated_sdr, hist_sdr, dim=-1).item()
            time_decay = 0.995 ** (total_mem - i - 1)
            expert_weight = 1.0
            
            if target_expert and i < len(self.metadata_list):
                mem_expert = self.metadata_list[i].get("expert", "")
                if mem_expert == target_expert:
                    expert_weight = 1.2

            activate_count = self.metadata_list[i].get("activate_count", 0) if i < len(self.metadata_list) else 0
            use_weight = 1.0 + (activate_count * 0.1)

            final_score = sim * time_decay * expert_weight * use_weight
            
            if final_score < 0.1:
                continue
            
            meta = self.metadata_list[i] if i < len(self.metadata_list) else {}
            content = self.content_list[i] if i < len(self.content_list) else ""
            sdr_hash = hash(hist_sdr.numpy().tobytes())
            mem_id = self.sdr_to_mem_id.get(sdr_hash, None)
            
            results.append((final_score, content, meta, i, mem_id))
        
        results.sort(key=lambda x: -x[0])
        return results[:top_k]
    
    def retrieve_multi_hop(self, query_sdr: torch.Tensor, hops: int = 3, top_k: int = 10) -> List[Tuple[float, str, Dict[str, Any], int, Optional[int]]]:
        """多跳联想检索（深度类脑推理）"""
        current_sdr = query_sdr
        all_results = []
        
        for hop in range(hops):
            current_sdr = self.forward(current_sdr, steps=1)
            hop_results = self.retrieve(current_sdr, top_k=top_k // hops)
            for score, content, meta, idx, mem_id in hop_results:
                all_results.append((score * (0.8 ** hop), content, meta, idx, mem_id))
        
        all_results.sort(key=lambda x: -x[0])
        return all_results[:top_k]
    
    def add_memory(self, sdr: torch.Tensor, content: str, mem_id: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        添加记忆到专家网络（支持身份/概念等所有脑区，兼容新格式）
        """
        metadata = metadata or {}
        
        sdr_cpu = sdr.squeeze(0).detach().cpu()
        self.sdr_list.append(sdr_cpu)
        self.content_list.append(content)
        self.metadata_list.append(metadata)
        
        if mem_id is not None:
            sdr_hash = hash(sdr_cpu.numpy().tobytes())
            self.sdr_to_mem_id[sdr_hash] = mem_id
            self.mem_id_to_sdr[mem_id] = sdr_hash
    
    def delete_memory(self, mem_id: int) -> None:
        """
        安全删除记忆（仅解除映射，不破坏索引）
        """
        if mem_id not in self.mem_id_to_sdr:
            logger.warning(f"[{self.name}] 未找到记忆ID: {mem_id}")
            return
        
        sdr_hash = self.mem_id_to_sdr.pop(mem_id)
        self.sdr_to_mem_id.pop(sdr_hash, None)
        logger.info(f"[{self.name}] 记忆ID {mem_id} 已删除（映射解除）")
    
    def sleep_consolidate(self, epochs: int = 3, priority_mem_ids: Optional[List[int]] = None) -> None:
        """
        睡眠巩固：梦境记忆回放 + 重放学习 + 突触修剪与新生（真正实现做梦）
        支持元认知传入的 priority_mem_ids，优先巩固高优先级记忆
        """
        if not self.sdr_list:
            logger.info(f"🌙 专家 [{self.name}] 无记忆，跳过睡眠巩固")
            return
        
        logger.info(f"🌙 专家 [{self.name}] 开始睡眠巩固 (epochs={epochs})...")
        
        priority_indices = []
        if priority_mem_ids:
            logger.info(f"🧠 元认知：收到 {len(priority_mem_ids)} 条高优先级记忆标记，优先回放")
            for mem_id in priority_mem_ids:
                if mem_id in self.mem_id_to_sdr:
                    for idx, sdr in enumerate(self.sdr_list):
                        sdr_hash = hash(sdr.numpy().tobytes())
                        if sdr_hash == self.mem_id_to_sdr[mem_id]:
                            priority_indices.append(idx)
                            break
            
            if priority_indices:
                logger.info(f"🧠 元认知：定位到 {len(priority_indices)} 条高优先级记忆，优先回放")
                for extra_epoch in range(2):
                    for idx in priority_indices:
                        sdr = self.sdr_list[idx]
                        is_fact = self.metadata_list[idx].get('is_fact', False) if idx < len(self.metadata_list) else False
                        
                        if self.stdp_enabled:
                            self.stdp_update(sdr, sdr, delta_t=15.0)
                        else:
                            self.hebbian_update(sdr, sdr, is_fact=is_fact)
                        
                        content = self.content_list[idx] if idx < len(self.content_list) else ""
                        logger.debug(f"   🧠 优先回放 | 记忆:{content[:30]}...")
        
        logger.info(f"😴 专家 [{self.name}] 进入梦境，开始记忆回放 & 神经元激活...")
        memory_pool = []
        for idx, (sdr, content, meta) in enumerate(zip(self.sdr_list, self.content_list, self.metadata_list)):
            freshness = (len(self.sdr_list) - idx) / len(self.sdr_list)
            if idx in priority_indices:
                freshness *= 1.5
            memory_pool.append((freshness, idx, sdr, content, meta))
        
        memory_pool.sort(reverse=True, key=lambda x: x[0])
        top_candidates = memory_pool[:min(15, len(memory_pool))]
        random.shuffle(top_candidates)
        dream_memories = top_candidates[:6]
        
        dream_log = []
        dream_contents = []
        for freshness, idx, sdr, content, meta in dream_memories:
            active_neurons = torch.where(sdr > 0.1)[0].cpu().numpy()
            neuron_count = len(active_neurons)
            log_content = content[:50] + "..." if len(content) > 50 else content
            priority_marker = " [优先记忆]" if idx in priority_indices else ""
            logger.info(f"     🌙 梦境回放{priority_marker} | 神经元激活:{neuron_count} | 内容:{log_content}")
            dream_log.append({
                "memory_idx": idx,
                "active_neurons": neuron_count,
                "content": log_content,
                "is_priority": idx in priority_indices
            })
            dream_contents.append(content[:60])
        
        if dream_contents:
            self.last_dream_text = "我刚刚梦里梦到了：" + "，还梦到了：".join(dream_contents)
        else:
            self.last_dream_text = ""
        logger.info(f"✅ 专家 [{self.name}] 梦境结束，共回放 {len(dream_memories)} 段记忆")
        
        for epoch in range(epochs):
            for i in range(len(self.sdr_list)):
                sdr = self.sdr_list[i]
                is_fact = self.metadata_list[i].get('is_fact', False) if i < len(self.metadata_list) else False
                if self.stdp_enabled:
                    self.stdp_update(sdr, sdr, delta_t=10.0)
                else:
                    self.hebbian_update(sdr, sdr, is_fact=is_fact)
        
        logger.info(f"🧠 [{self.name}] 开始突触重塑（修剪+新生）...")
        pruned = self.synaptic_pruning()
        created = self.synaptogenesis()
        
        weak_threshold = 0.01
        num_weak = torch.sum(torch.abs(self.synapse.data) < weak_threshold).item()
        total = self.synapse.data.numel()
        self.synapse.data[torch.abs(self.synapse.data) < weak_threshold] = 0.0
        
        sparsity = self.get_sparsity() * 100
        logger.info(f"✅ 专家 [{self.name}] 睡眠巩固完成 | 稀疏度: {sparsity:.2f}%")
        logger.info(f"   基础修剪弱连接: {num_weak}/{total} ({num_weak/total:.2%})")
        logger.info(f"   智能修剪: {pruned} | 智能新生: {created} | 累计修剪: {self.total_synapses_pruned} | 累计新生: {self.total_synapses_created}")
    
    def get_sparsity(self) -> float:
        """计算突触稀疏度（类脑健康度指标）"""
        if self.synapse is None:
            return 0.0
        return (torch.abs(self.synapse.data) < 0.01).float().mean().item()
    
    def save_weights(self, path: str) -> None:
        """保存专家权重+记忆+映射关系（全脑区通用，完整保存新格式metadata）"""
        try:
            torch.save({
                'synapse': self.synapse.data,
                'sdr_list': self.sdr_list,
                'content_list': self.content_list,
                'metadata_list': self.metadata_list,
                'sdr_to_mem_id': self.sdr_to_mem_id,
                'mem_id_to_sdr': self.mem_id_to_sdr,
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
            logger.info(f"💾 专家 [{self.name}] 权重已保存: {path}")
        except Exception as e:
            logger.error(f"❌ 专家 [{self.name}] 权重保存失败: {e}")
    
    def load_weights(self, path: str) -> None:
        """加载专家权重（兼容旧版本+身份专家+新格式metadata）"""
        if not os.path.exists(path):
            logger.warning(f"[{self.name}] 权重文件不存在，初始化新权重")
            return
        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
            self.synapse.data = data['synapse']
            self.sdr_list = data.get('sdr_list', [])
            self.content_list = data.get('content_list', [])
            self.metadata_list = data.get('metadata_list', [])
            self.sdr_to_mem_id = data.get('sdr_to_mem_id', {})
            self.mem_id_to_sdr = data.get('mem_id_to_sdr', {})
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
            
            logger.info(f"✅ 专家 [{self.name}] 加载完成 | 记忆数: {len(self.sdr_list)} | 局部偏置: {'启用' if self.local_bias_enabled else '禁用'} | STDP: {'启用' if self.stdp_enabled else '禁用'}")
            logger.info(f"   累计修剪: {self.total_synapses_pruned} | 累计新生: {self.total_synapses_created}")
        except Exception as e:
            logger.error(f"❌ 专家 [{self.name}] 权重加载失败: {e}，重置为初始状态")
            self.synapse.data = torch.zeros(self.dim, self.dim)
            if self.local_bias_enabled:
                self._init_local_bias()
            self.sdr_list = []
            self.content_list = []
            self.metadata_list = []
    
    def activate_memories(self, memory_clip_vecs: List[torch.Tensor], steps: int = 2) -> ActivatedMemoriesResult:
        """
        核心：让检索到的记忆在专家网络中激活传播（真正思考）
        :param memory_clip_vecs: 初始激活的记忆向量
        :param steps: 突触传播步数
        :return: 完整的思考链路
        """
        if not memory_clip_vecs:
            return {"chain": [], "core_ideas": [], "activated_memories": []}
        
        activation = torch.stack(memory_clip_vecs).mean(dim=0).unsqueeze(0)
        activation = activation / (activation.norm() + 1e-8)
        
        for _ in range(steps):
            activation = self.forward(activation)
        
        activated_indices = self._retrieve_activated_indices_safe(activation, topk=5)
        activated_memories = [self.get_memory_by_idx_safe(i) for i in activated_indices if i >= 0]
        
        chain = self._build_thought_chain(activated_memories)
        core_ideas = list(set([mem["content"].split("：")[1][:20] for mem in activated_memories if "：" in mem["content"]]))
        
        return {
            "core_ideas": core_ideas,
            "activated_memories": activated_memories,
            "thought_chain": chain,
            "activation_strength": activation.norm().item()
        }
    
    def _retrieve_activated_indices_safe(self, activation: torch.Tensor, topk: int = 5) -> List[int]:
        """安全版：根据神经激活态，检索被激活的记忆索引"""
        if not self.sdr_list:
            return []
        
        results = []
        for i, hist_sdr in enumerate(self.sdr_list):
            sim = F.cosine_similarity(activation.squeeze(0), hist_sdr, dim=-1).item()
            results.append((sim, i))
        results.sort(key=lambda x: -x[0])
        return [i for (sim, i) in results[:topk]]
    
    def get_memory_by_idx_safe(self, idx: int) -> Dict[str, Any]:
        """安全版：根据索引获取记忆内容"""
        if idx < 0 or idx >= len(self.sdr_list):
            return {"content": "", "metadata": {}}
        return {
            "content": self.content_list[idx] if idx < len(self.content_list) else "",
            "metadata": self.metadata_list[idx] if idx < len(self.metadata_list) else {}
        }
    
    def _build_thought_chain(self, memories: List[Dict[str, Any]]) -> str:
        """把激活的记忆 → 结构化思路链"""
        if not memories:
            return "无关联记忆"
        chain = " → ".join([mem["content"][:30] + "..." if len(mem["content"]) > 30 else mem["content"] for mem in memories])
        return f"联想思路：{chain}"
    
    def init_predictor(self) -> None:
        """初始化预测头（懒加载，自动匹配设备）"""
        if not hasattr(self, "predict_head"):
            self.predict_head = nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.LayerNorm(self.dim),
                nn.GELU(),
                nn.Linear(self.dim, self.dim)
            ).to(self.synapse.device)
            self.predict_head.eval()
    
    def snn_pulse_decay(self, activation: torch.Tensor, step: int, total_steps: int) -> torch.Tensor:
        """轻量SNN脉冲衰减（时间维度，无梯度冲突）"""
        with torch.no_grad():
            decay_rate = torch.exp(-torch.linspace(0, 1.5, total_steps, device=activation.device)[step])
        return activation * decay_rate
    
    def predict_next_sdr(self, current_activation: torch.Tensor) -> torch.Tensor:
        """预测下一个SDR（禁用梯度，纯推理）"""
        self.init_predictor()
        with torch.no_grad():
            return self.predict_head(current_activation)
    
    def update_prediction(self, pred_sdr: torch.Tensor, real_sdr: torch.Tensor) -> float:
        """
        轻量预测编码更新（无反向传播！纯数值更新）
        兼容你现有所有逻辑，不破坏突触/记忆
        """
        if pred_sdr.dim() == 1:
            pred_sdr = pred_sdr.unsqueeze(0)
        if real_sdr.dim() == 1:
            real_sdr = real_sdr.unsqueeze(0)
        
        with torch.no_grad():
            loss = F.mse_loss(pred_sdr, real_sdr).item()
        
        try:
            if self.stdp_enabled:
                delta_t = 5.0 * (1.0 - loss)
                self.stdp_update(real_sdr, pred_sdr, delta_t=delta_t)
            else:
                self.hebbian_update(real_sdr, pred_sdr)
        except:
            pass
        
        return loss
    
    def get_synapse_change(self) -> float:
        """获取突触权重总变化量（用于认知能量场计算）"""
        if not hasattr(self, 'synapse'):
            return 0.0
        return float(torch.sum(torch.abs(self.synapse)).item())
    
    def synaptic_pruning(self, pruning_percentile: Optional[float] = None) -> int:
        """
        自动修剪弱突触：
        1. 计算每个突触的"重要性"（权重强度 + 更新频率）
        2. 删除重要性最低的一部分突触
        3. 维持大脑的稀疏性和能量效率
        """
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
        """
        修复版：突触新生（解决新生为0的问题）
        """
        if not self.synaptogenesis_enabled:
            return 0
        
        new_synapse_rate = new_synapse_rate or self.new_synapse_rate
        
        with torch.no_grad():
            activation_threshold = torch.quantile(self.neuron_activation_counts.float(), 0.7)
            high_activation_neurons = torch.where(self.neuron_activation_counts > activation_threshold)[0].tolist()
            
            if len(high_activation_neurons) < 2:
                return 0
            
            candidate_pairs = []
            max_candidates = 2000
            
            sample_size = min(len(high_activation_neurons) * 20, max_candidates)
            
            for _ in range(sample_size):
                i = random.choice(high_activation_neurons)
                j = random.choice(high_activation_neurons)
                
                if i == j:
                    continue
                
                coactivation = self.neuron_coactivation_matrix[i, j].item()
                current_weight = self.synapse.data[j, i].item()
                
                if coactivation > 2 and abs(current_weight) < 0.01:
                    candidate_pairs.append((i, j, coactivation))
            
            if not candidate_pairs:
                logger.info(f"🧠 [{self.name}] 没有足够的共激活对，随机生成新突触")
                for _ in range(20):
                    i = random.choice(high_activation_neurons)
                    j = random.choice(high_activation_neurons)
                    if i != j and abs(self.synapse.data[j, i].item()) < 0.01:
                        candidate_pairs.append((i, j, 1))
            
            candidate_pairs.sort(key=lambda x: -x[2])
            
            current_non_zero = (self.synapse.data != 0).sum().item()
            max_possible = int(self.dim * self.dim * self.max_synapse_density) - current_non_zero
            num_to_create = min(len(candidate_pairs), max(int(current_non_zero * new_synapse_rate), 50), max_possible)
            
            if num_to_create <= 0:
                return 0
            
            created_count = 0
            for i, j, coactivation in candidate_pairs[:num_to_create]:
                self.synapse.data[j, i] = self.new_synapse_initial_weight * (1.0 + coactivation / 10.0)
                self.synapse_update_counts[j, i] = 1
                created_count += 1
            
            self.total_synapses_created += created_count
            
            logger.info(f"🧠 [{self.name}] 突触新生 | 新增: {created_count} | 累计: {self.total_synapses_created} | 候选对: {len(candidate_pairs)}")
            
            return created_count
    
    def analyze_stdp_statistics(self) -> STDPStatistics:
        """分析STDP学习统计信息"""
        if not self.synapse_change_trace:
            return {
                "total_updates": 0,
                "avg_change": 0.0,
                "max_change": 0.0,
                "current_timestep": self.current_timestep,
                "positive_weights": 0,
                "negative_weights": 0,
                "zero_weights": 0
            }
        
        return {
            "total_updates": len(self.synapse_change_trace),
            "avg_change": sum(self.synapse_change_trace) / len(self.synapse_change_trace),
            "max_change": max(self.synapse_change_trace) if self.synapse_change_trace else 0.0,
            "current_timestep": self.current_timestep,
            "positive_weights": torch.sum(self.synapse.data > 0).item(),
            "negative_weights": torch.sum(self.synapse.data < 0).item(),
            "zero_weights": torch.sum(self.synapse.data == 0).item()
        }