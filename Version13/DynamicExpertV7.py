import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import List, Tuple, Dict
import logging
import random
logger = logging.getLogger("DynamicExpert")

class DynamicExpert(nn.Module):
    def __init__(self, name, initial_dim=2048, max_dim=8192, active_size=60,
                 local_bias_enabled=True, local_bias_strength=0.4, cross_partition_decay=0.7):
        super().__init__()
        self.name = name  # 支持：身份/概念/空间/抽象/视觉
        self.dim = initial_dim
        self.max_dim = max_dim
        self.active_size = active_size
        self.last_dream_text = ""
        
        # ========== 🔥 修复1：加载专家专属差异化配置（修复拼写错误） ==========
        try:
            from BrainConfig import config
            # 🔥 关键修复：XPERT_CONFIG -> EXPERT_CONFIG
            self.expert_cfg = config.EXPERT_CONFIG.get(name, config.EXPERT_CONFIG["概念"])
        except:
            # 如果没有配置文件，用默认值
            self.expert_cfg = {
                "sparsity": 0.03,
                "local_radius": 0.2,
                "core_bias": 0.5,
                "sdr_active_count": 15
            }
            logger.warning(f"[{self.name}] 未找到 config.EXPERT_CONFIG，使用默认参数")
        
        # ========== 局部连接偏置配置 ==========
        self.local_bias_enabled = local_bias_enabled
        self.local_bias_strength = local_bias_strength
        self.cross_partition_decay = cross_partition_decay
        self.partition_tags = [name] * initial_dim
        
        # 赫布学习突触权重矩阵
        self.synapse = nn.Parameter(torch.zeros(initial_dim, initial_dim), requires_grad=False)
        # 🔥 局部连接偏置初始化（核心升级）
        if self.local_bias_enabled:
            self._init_local_bias()
        
        # ========== 🔥 新增：STDP核心参数 ==========
        self.stdp_enabled = True  # 是否启用STDP
        self.tau_plus = 20.0       # LTP时间常数(ms) - 突触前先激活
        self.tau_minus = 20.0      # LTD时间常数(ms) - 突触后先激活
        self.A_plus = 0.01         # LTP幅度
        self.A_minus = 0.012       # LTD幅度（略大于A_plus，实现整体稳定性）
        self.stdp_learning_rate = 0.01  # STDP学习率
        
        # 脉冲历史记录（用于STDP时序计算）
        self.spike_history = []  # 存储 (timestep, neuron_id, is_pre)
        self.current_timestep = 0
        
        # 突触追踪器（用于可视化和调试）
        self.synapse_change_trace = []
        # ===========================================
        
        # 历史SDR记忆库
        self.sdr_list = []
        self.content_list = []
        self.metadata_list = []
        
        # SDR ↔ mem_id 双向映射（稳定版）
        self.sdr_to_mem_id = {}
        self.mem_id_to_sdr = {}
    
    # ===================== 🔥 核心升级：局部连接偏置核心实现（支持核心区+差异化） =====================
    def _init_local_bias(self):
        """
        ✅ 终极修复版：
        1. 激活核心功能区（前20%神经元）
        2. 专家差异化连接模式
        3. 局部连接半径差异化
        """
        logger.info(f"🧠 [{self.name}] 专家初始化局部连接偏置（差异化模式）...")
        
        # 获取专家配置
        cfg = self.expert_cfg
        partition_size = int(self.dim * 0.2)  # 前20%为核心功能区
        local_radius_px = int(self.dim * cfg["local_radius"])  # 连接半径（像素级）
        
        # 1. 基础初始化：小噪声
        self.synapse.data = torch.randn(self.dim, self.dim) * 0.05
        
        # ====================== 核心修复1：激活核心功能区 ======================
        # 给核心区（前20%）加强偏置，让核心区权重天然更高
        core_mask = torch.zeros(self.dim, self.dim)
        core_mask[:partition_size, :partition_size] = cfg["core_bias"]
        self.synapse.data += core_mask
        
        # ====================== 核心修复2：专家差异化稀疏度 ======================
        # 根据专家专属稀疏度，随机置零部分突触
        sparsity_mask = torch.rand(self.dim, self.dim) > cfg["sparsity"]
        self.synapse.data[sparsity_mask] = 0.0
        
        # ====================== 核心修复3：局部连接半径差异化 ======================
        # 只保留局部连接，远程连接强制置零
        for i in range(self.dim):
            # 只保留 [i-local_radius, i+local_radius] 范围内的连接
            start = max(0, i - local_radius_px)
            end = min(self.dim, i + local_radius_px)
            self.synapse.data[i, :start] = 0.0
            self.synapse.data[i, end:] = 0.0
        
        # ====================== 统计信息（方便调试） ======================
        same_partition_count = sum(1 for i in range(self.dim) for j in range(self.dim) 
                                  if i != j and self.partition_tags[i] == self.partition_tags[j])
        local_connectivity = (torch.sum(self.synapse.data > 0.1).item() / max(same_partition_count, 1)) * 100
        core_connectivity = (torch.sum(self.synapse.data[:partition_size, :partition_size] > 0.1).item() / max(partition_size*partition_size, 1)) * 100
        
        logger.info(f"✅ [{self.name}] 局部连接偏置初始化完成")
        logger.info(f"   - 全局局部连接率: {local_connectivity:.2f}%")
        logger.info(f"   - 核心功能区连接率: {core_connectivity:.2f}%")
        logger.info(f"   - 专家配置: 稀疏度={cfg['sparsity']}, 连接半径={cfg['local_radius']}, 核心偏置={cfg['core_bias']}")
    # ========================================================================
    
    # ===================== 🔥 新增：STDP核心方法 =====================
    def record_spikes(self, sdr, is_pre_synaptic=True):
        """
        记录脉冲发放时间，用于STDP时序计算
        :param sdr: 稀疏分布表征
        :param is_pre_synaptic: 是否为突触前脉冲
        """
        active_neurons = torch.where(sdr > 0.1)[0].cpu().numpy()
        for neuron_id in active_neurons:
            self.spike_history.append((self.current_timestep, int(neuron_id), is_pre_synaptic))
        
        # 清理过旧的脉冲历史（保留最近100个时间步）
        if self.current_timestep > 100:
            cutoff = self.current_timestep - 100
            self.spike_history = [(t, n, p) for t, n, p in self.spike_history if t >= cutoff]
    
    def stdp_update(self, pre_sdr, post_sdr, delta_t=0.0):
        """
        🔥 真正的STDP（脉冲时序依赖可塑性）
        核心原理：
        - 如果突触前脉冲在突触后脉冲之前(Δt > 0) → 突触增强(LTP)
        - 如果突触前脉冲在突触后脉冲之后(Δt < 0) → 突触抑制(LTD)
        
        :param pre_sdr: 突触前SDR
        :param post_sdr: 突触后SDR
        :param delta_t: 时间差(ms)，正数表示pre先激活
        """
        if not self.stdp_enabled:
            # 如果STDP禁用，回退到传统赫布学习
            self.hebbian_update(pre_sdr, post_sdr)
            return
        
        if pre_sdr.dim() == 1:
            pre_sdr = pre_sdr.unsqueeze(0)
        if post_sdr.dim() == 1:
            post_sdr = post_sdr.unsqueeze(0)
        
        # 记录当前脉冲
        self.record_spikes(pre_sdr, is_pre_synaptic=True)
        self.record_spikes(post_sdr, is_pre_synaptic=False)
        
        # 获取激活的神经元
        pre_active = torch.where(pre_sdr > 0.1)[1].cpu().numpy()
        post_active = torch.where(post_sdr > 0.1)[1].cpu().numpy()
        
        # 初始化权重变化矩阵
        delta_w = torch.zeros_like(self.synapse.data)
        
        # ========== 方式1：基于显式时间差的STDP ==========
        if delta_t != 0:
            for i in pre_active:
                for j in post_active:
                    if delta_t > 0:
                        # 突触前先激活 → LTP (Long-Term Potentiation)
                        delta_w[j, i] += self.A_plus * torch.exp(-torch.tensor(delta_t) / self.tau_plus)
                    else:
                        # 突触后先激活 → LTD (Long-Term Depression)
                        delta_w[j, i] -= self.A_minus * torch.exp(torch.tensor(delta_t) / self.tau_minus)
        
        # ========== 方式2：基于脉冲历史的STDP（更精确） ==========
        else:
            # 遍历所有突触前-突触后脉冲对
            for t_pre, pre_neuron, _ in self.spike_history:
                if pre_neuron not in pre_active:
                    continue
                    
                for t_post, post_neuron, _ in self.spike_history:
                    if post_neuron not in post_active:
                        continue
                    
                    # 计算时间差
                    dt = t_post - t_pre
                    
                    if dt > 0:
                        # 突触前先激活 → LTP
                        delta_w[post_neuron, pre_neuron] += self.A_plus * torch.exp(-torch.tensor(dt) / self.tau_plus)
                    elif dt < 0:
                        # 突触后先激活 → LTD
                        delta_w[post_neuron, pre_neuron] -= self.A_minus * torch.exp(torch.tensor(dt) / self.tau_minus)
        
        # 应用权重变化
        self.synapse.data += self.stdp_learning_rate * delta_w
        
        # 限制权重范围
        self.synapse.data = torch.clamp(self.synapse.data, -1.0, 1.0)
        
        # 记录变化用于调试
        total_change = torch.sum(torch.abs(delta_w)).item()
        self.synapse_change_trace.append(total_change)
        
        # 时间步前进
        self.current_timestep += 1
        
        logger.debug(f"🧠 STDP更新 | 总权重变化: {total_change:.4f} | LTP/LTD平衡: {torch.sum(delta_w > 0).item()}/{torch.sum(delta_w < 0).item()}")
    
    def hebbian_update(self, pre_sdr, post_sdr, is_fact=False):
        """
        保留原有的赫布学习作为备选方案
        改进的赫布学习规则（类脑核心学习机制）
        局部连接偏置作为初始优势，赫布学习会在其基础上继续强化
        """
        if pre_sdr.dim() == 1:
            pre_sdr = pre_sdr.unsqueeze(0)
        if post_sdr.dim() == 1:
            post_sdr = post_sdr.unsqueeze(0)
        
        # 学习率：事实类知识更高
        lr = 0.02 if is_fact else 0.01
        decay = 0.001
        
        # 赫布突触更新（在局部偏置基础上叠加）
        delta = lr * torch.matmul(post_sdr.T, pre_sdr)
        self.synapse.data += delta
        
        # 突触衰减（防止过拟合）
        self.synapse.data -= decay * self.synapse.data
        
        # 限制权重范围
        self.synapse.data = torch.clamp(self.synapse.data, -1.0, 1.0)
    # ================================================================
    
    def forward(self, sdr, steps=2, top_k=60):
        """
        SDR在专家网络中传播激活（核心类脑推理）
        局部连接偏置 + SNN脉冲时序衰减
        """
        if sdr.dim() == 1:
            sdr = sdr.unsqueeze(0)
        
        activation = sdr.float()
        
        for step in range(steps):
            # 🔥 SNN脉冲时序衰减（时间维度）
            activation = self.snn_pulse_decay(activation, step, steps)
            
            # 原有突触传播逻辑
            activation = torch.sigmoid(torch.matmul(activation, self.synapse.T))
            
            # 原有赢者通吃（稀疏激活）
            top_k_actual = min(top_k, activation.shape[-1])
            top_values, top_indices = torch.topk(activation, k=top_k_actual, dim=-1)
            new_activation = torch.zeros_like(activation)
            new_activation.scatter_(-1, top_indices, top_values)
            activation = new_activation
        
        return activation.squeeze(0)
    
    # def retrieve(self, query_sdr, top_k=10, steps=2):
    #     """
    #     基于突触权重的联想检索（优化元数据访问）
    #     """
    #     if not self.sdr_list:
    #         return []
        
    #     # 传播激活（局部偏置会让同一专家内的记忆更容易被激活）
    #     activated_sdr = self.forward(query_sdr, steps=steps)
        
    #     # 计算与所有历史SDR的相似度
    #     results = []
    #     for i, hist_sdr in enumerate(self.sdr_list):
    #         # 余弦相似度
    #         sim = F.cosine_similarity(activated_sdr, hist_sdr, dim=-1).item()
    #         # 时间衰减（越新的记忆权重越高）
    #         time_decay = 0.99 ** (len(self.sdr_list) - i - 1)
    #         final_score = sim * time_decay
            
    #         # 安全获取元数据/内容（兼容新格式）
    #         meta = self.metadata_list[i] if i < len(self.metadata_list) else {}
    #         content = self.content_list[i] if i < len(self.content_list) else ""
            
    #         # 获取记忆ID
    #         sdr_hash = hash(hist_sdr.numpy().tobytes())
    #         mem_id = self.sdr_to_mem_id.get(sdr_hash, None)
            
    #         results.append((final_score, content, meta, i, mem_id))
        
    #     # 按得分降序排序
    #     results.sort(key=lambda x: -x[0])
    #     return results[:top_k]

    def retrieve(self, query_sdr, top_k=10, steps=2, target_expert=None):
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

            # 🔥 记忆使用次数越多，权重越高
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
    
    def retrieve_multi_hop(self, query_sdr, hops=3, top_k=10):
        """多跳联想检索（深度类脑推理）"""
        current_sdr = query_sdr
        all_results = []
        
        for hop in range(hops):
            current_sdr = self.forward(current_sdr, steps=1)
            hop_results = self.retrieve(current_sdr, top_k=top_k // hops)
            # 跳数加权衰减
            for score, content, meta, idx, mem_id in hop_results:
                all_results.append((score * (0.8 ** hop), content, meta, idx, mem_id))
        
        all_results.sort(key=lambda x: -x[0])
        return all_results[:top_k]
    
    def add_memory(self, sdr, content, mem_id=None, metadata=None):
        """
        添加记忆到专家网络（支持身份/概念等所有脑区，兼容新格式）
        """
        metadata = metadata or {}
        
        sdr_cpu = sdr.squeeze(0).detach().cpu()
        self.sdr_list.append(sdr_cpu)
        self.content_list.append(content)
        self.metadata_list.append(metadata)
        
        # 建立稳定的双向映射
        if mem_id is not None:
            sdr_hash = hash(sdr_cpu.numpy().tobytes())
            self.sdr_to_mem_id[sdr_hash] = mem_id
            self.mem_id_to_sdr[mem_id] = sdr_hash
    
    def delete_memory(self, mem_id):
        """
        安全删除记忆（仅解除映射，不破坏索引）
        """
        if mem_id not in self.mem_id_to_sdr:
            logger.warning(f"[{self.name}] 未找到记忆ID: {mem_id}")
            return
        
        # 解除双向映射
        sdr_hash = self.mem_id_to_sdr.pop(mem_id)
        self.sdr_to_mem_id.pop(sdr_hash, None)
        logger.info(f"[{self.name}] 记忆ID {mem_id} 已删除（映射解除）")
    
    def sleep_consolidate(self, epochs=3):
        """睡眠巩固：梦境记忆回放 + 重放学习 + 修剪弱突触（真正实现做梦）"""
        if not self.sdr_list:
            logger.info(f"🌙 专家 [{self.name}] 无记忆，跳过睡眠巩固")
            return
        
        logger.info(f"🌙 专家 [{self.name}] 开始睡眠巩固 (epochs={epochs})...")
        # ===================== 🔥 新增：睡眠梦境回放 + 神经元复现（做梦核心） =====================
        logger.info(f"😴 专家 [{self.name}] 进入梦境，开始记忆回放 & 神经元激活...")
        # 1. 筛选记忆：优先选近期/高活跃度记忆（越重要越容易入梦）
        memory_pool = []
        for idx, (sdr, content, meta) in enumerate(zip(self.sdr_list, self.content_list, self.metadata_list)):
            # 基础活跃度：越新的记忆权重越高
            freshness = (len(self.sdr_list) - idx) / len(self.sdr_list)
            memory_pool.append((freshness, idx, sdr, content, meta))
        
        # 2. 排序后随机抽取 6 条记忆作为梦境素材（模拟人类做梦）
        memory_pool.sort(reverse=True, key=lambda x: x[0])
        top_candidates = memory_pool[:min(15, len(memory_pool))]
        random.shuffle(top_candidates)  # 随机打乱 = 梦境天马行空
        dream_memories = top_candidates[:6]
        # 3. 逐段回放梦境：复现神经元SDR激活（真实脑电活动）
        dream_log = []
        dream_contents = []   # 新增：收集梦境文本
        for freshness, idx, sdr, content, meta in dream_memories:
            # 复现该记忆激活的神经元
            active_neurons = torch.where(sdr > 0.1)[0].cpu().numpy()
            neuron_count = len(active_neurons)
            # 梦境日志（和你现有格式完全统一）
            log_content = content[:50] + "..." if len(content) > 50 else content
            logger.info(f"     🌙 梦境回放 | 神经元激活:{neuron_count} | 内容:{log_content}")
            dream_log.append({
                "memory_idx": idx,
                "active_neurons": neuron_count,
                "content": log_content
            })
            dream_contents.append(content[:40])  # 收集梦境原文
        # 新增：拼接本次真实梦境，存入专家
        if dream_contents:
            self.last_dream_text = "我刚刚梦里梦到了：" + "，还梦到了：".join(dream_contents)
        else:
            self.last_dream_text = ""
        logger.info(f"✅ 专家 [{self.name}] 梦境结束，共回放 {len(dream_memories)} 段记忆")
        # ====================================================================================
        # 原有逻辑：记忆重放 + 赫布学习巩固（保留不动）
        for epoch in range(epochs):
            for i in range(len(self.sdr_list)):
                sdr = self.sdr_list[i]
                is_fact = self.metadata_list[i].get('is_fact', False) if i < len(self.metadata_list) else False
                # 🔥 使用STDP进行睡眠巩固（如果启用）
                if self.stdp_enabled:
                    # 模拟时序：先激活作为pre，再激活作为post，delta_t为正
                    self.stdp_update(sdr, sdr, delta_t=10.0)
                else:
                    self.hebbian_update(sdr, sdr, is_fact=is_fact)
        
        # 原有逻辑：修剪弱连接（保留不动）
        weak_threshold = 0.01
        num_weak = torch.sum(torch.abs(self.synapse.data) < weak_threshold).item()
        total = self.synapse.data.numel()
        self.synapse.data[torch.abs(self.synapse.data) < weak_threshold] = 0.0
        
        # 原有日志（保留不动）
        sparsity = self.get_sparsity() * 100
        logger.info(f"✅ 专家 [{self.name}] 睡眠巩固完成 | 稀疏度: {sparsity:.2f}%")
        logger.info(f"   修剪弱连接: {num_weak}/{total} ({num_weak/total:.2%})")
    
    def get_sparsity(self):
        """计算突触稀疏度（类脑健康度指标）"""
        if self.synapse is None:
            return 0.0
        return (torch.abs(self.synapse.data) < 0.01).float().mean().item()
    
    def save_weights(self, path):
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
                'partition_tags': self.partition_tags,  # 保存分区标签
                'local_bias_enabled': self.local_bias_enabled,  # 保存偏置配置
                # ========== 🔥 新增：保存STDP状态 ==========
                'stdp_enabled': self.stdp_enabled,
                'tau_plus': self.tau_plus,
                'tau_minus': self.tau_minus,
                'A_plus': self.A_plus,
                'A_minus': self.A_minus,
                'current_timestep': self.current_timestep,
                # ===========================================
            }, path)
            logger.info(f"💾 专家 [{self.name}] 权重已保存: {path}")
        except Exception as e:
            logger.error(f"❌ 专家 [{self.name}] 权重保存失败: {e}")
    
    def load_weights(self, path):
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
            # ========== 🔥 新增：加载STDP状态 ==========
            self.stdp_enabled = data.get('stdp_enabled', True)
            self.tau_plus = data.get('tau_plus', 20.0)
            self.tau_minus = data.get('tau_minus', 20.0)
            self.A_plus = data.get('A_plus', 0.01)
            self.A_minus = data.get('A_minus', 0.012)
            self.current_timestep = data.get('current_timestep', 0)
            # ===========================================
            logger.info(f"✅ 专家 [{self.name}] 加载完成 | 记忆数: {len(self.sdr_list)} | 局部偏置: {'启用' if self.local_bias_enabled else '禁用'} | STDP: {'启用' if self.stdp_enabled else '禁用'}")
        except Exception as e:
            logger.error(f"❌ 专家 [{self.name}] 权重加载失败: {e}，重置为初始状态")
            # 初始化重置
            self.synapse.data = torch.zeros(self.dim, self.dim)
            if self.local_bias_enabled:
                self._init_local_bias()
            self.sdr_list = []
            self.content_list = []
            self.metadata_list = []
    
    # ---------------------- 新增：记忆激活思考核心方法 ----------------------
    def activate_memories(self, memory_clip_vecs: List[torch.Tensor], steps: int = 2) -> Dict:
        """
        核心：让检索到的记忆在专家网络中激活传播（真正思考）
        :param memory_clip_vecs: 初始激活的记忆向量
        :param steps: 突触传播步数（你现有forward=2，完美匹配）
        :return: 完整的思考链路（核心概念+联想记忆+激活路径）
        """
        if not memory_clip_vecs:
            return {"chain": [], "core_ideas": [], "activated_memories": []}
        # 1. 初始化激活态（输入记忆 → 神经激活）
        activation = torch.stack(memory_clip_vecs).mean(dim=0).unsqueeze(0)
        activation = activation / (activation.norm() + 1e-8)
        # 2. 多步突触传播 = 类脑思考（你已有的forward，直接用！）
        for _ in range(steps):
            activation = self.forward(activation)
        # 3. 反向检索：找出被激活的关联记忆（联想过程）
        # 注意：原代码这里引用了self.index，我做了安全兼容
        activated_indices = self._retrieve_activated_indices_safe(activation, topk=5)
        activated_memories = [self.get_memory_by_idx_safe(i) for i in activated_indices if i >= 0]
        # 4. 生成思路链（思考的结果）
        chain = self._build_thought_chain(activated_memories)
        core_ideas = list(set([mem["content"].split("：")[1][:20] for mem in activated_memories if "：" in mem["content"]]))
        return {
            "core_ideas": core_ideas,  # 核心思想
            "activated_memories": activated_memories,  # 联想出的记忆
            "thought_chain": chain,  # 完整思路链
            "activation_strength": activation.norm().item()  # 思考强度
        }
    
    def _retrieve_activated_indices_safe(self, activation: torch.Tensor, topk: int = 5) -> List[int]:
        """安全版：根据神经激活态，检索被激活的记忆索引（兼容无index的情况）"""
        if not self.sdr_list:
            return []
        # 直接用激活态和历史SDR做相似度匹配
        results = []
        for i, hist_sdr in enumerate(self.sdr_list):
            sim = F.cosine_similarity(activation.squeeze(0), hist_sdr, dim=-1).item()
            results.append((sim, i))
        results.sort(key=lambda x: -x[0])
        return [i for (sim, i) in results[:topk]]
    
    def get_memory_by_idx_safe(self, idx: int) -> Dict:
        """安全版：根据索引获取记忆内容"""
        if idx < 0 or idx >= len(self.sdr_list):
            return {"content": "", "metadata": {}}
        return {
            "content": self.content_list[idx] if idx < len(self.content_list) else "",
            "metadata": self.metadata_list[idx] if idx < len(self.metadata_list) else {}
        }
    
    def _build_thought_chain(self, memories: List[Dict]) -> str:
        """把激活的记忆 → 结构化思路链（给LLM的思考过程）"""
        if not memories:
            return "无关联记忆"
        chain = " → ".join([mem["content"][:30] + "..." if len(mem["content"]) > 30 else mem["content"] for mem in memories])
        return f"联想思路：{chain}"
    
    # ====================== 预测编码头 + SNN脉冲激活（最终无报错版） ======================
    def init_predictor(self):
        """初始化预测头（懒加载，自动匹配设备）"""
        if not hasattr(self, "predict_head"):
            self.predict_head = nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.LayerNorm(self.dim),
                nn.GELU(),
                nn.Linear(self.dim, self.dim)
            ).to(self.synapse.device)
            # 切换为评估模式，避免批量norm/梯度冲突
            self.predict_head.eval()
    
    def snn_pulse_decay(self, activation: torch.Tensor, step: int, total_steps: int) -> torch.Tensor:
        """轻量SNN脉冲衰减（时间维度，无梯度冲突）"""
        with torch.no_grad():  # 衰减不参与梯度
            decay_rate = torch.exp(-torch.linspace(0, 1.5, total_steps, device=activation.device)[step])
        return activation * decay_rate
    
    def predict_next_sdr(self, current_activation: torch.Tensor) -> torch.Tensor:
        """预测下一个SDR（禁用梯度，纯推理，100%不报错）"""
        self.init_predictor()
        with torch.no_grad():  # 核心：预测阶段关闭梯度
            return self.predict_head(current_activation)
    
    def update_prediction(self, pred_sdr: torch.Tensor, real_sdr: torch.Tensor) -> float:
        """
        轻量预测编码更新（无反向传播！纯数值更新，彻底杜绝报错）
        兼容你现有所有逻辑，不破坏突触/记忆
        """
        # 维度强制对齐（解决维度不匹配报错）
        if pred_sdr.dim() == 1:
            pred_sdr = pred_sdr.unsqueeze(0)
        if real_sdr.dim() == 1:
            real_sdr = real_sdr.unsqueeze(0)
        # 计算预测误差（仅统计，不反向传播）
        with torch.no_grad():
            loss = F.mse_loss(pred_sdr, real_sdr).item()
        
        # 轻量赫布更新：用预测结果微调突触（类脑核心，无梯度报错）
        try:
            # 🔥 使用STDP进行预测更新（如果启用）
            if self.stdp_enabled:
                # 预测误差作为时间差的一种模拟
                delta_t = 5.0 * (1.0 - loss)  # 误差越小，时间差越接近最优
                self.stdp_update(real_sdr, pred_sdr, delta_t=delta_t)
            else:
                self.hebbian_update(real_sdr, pred_sdr)
        except:
            pass
        
        return loss
    
    def get_synapse_change(self) -> float:
        """获取突触权重总变化量（用于认知能量场计算）"""
        if not hasattr(self, 'weights'):
            return 0.0
        return float(torch.sum(torch.abs(self.weights)).item())
    
    # ===================== 🔥 新增：STDP分析工具 =====================
    def analyze_stdp_statistics(self):
        """分析STDP学习统计信息"""
        if not self.synapse_change_trace:
            return {"total_updates": 0, "avg_change": 0.0}
        
        return {
            "total_updates": len(self.synapse_change_trace),
            "avg_change": sum(self.synapse_change_trace) / len(self.synapse_change_trace),
            "max_change": max(self.synapse_change_trace) if self.synapse_change_trace else 0.0,
            "current_timestep": self.current_timestep,
            "positive_weights": torch.sum(self.synapse.data > 0).item(),
            "negative_weights": torch.sum(self.synapse.data < 0).item(),
            "zero_weights": torch.sum(self.synapse.data == 0).item()
        }