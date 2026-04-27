import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import List, Tuple
import logging

logger = logging.getLogger("DynamicExpert")

class DynamicExpert(nn.Module):
    def __init__(self, name, initial_dim=2048, max_dim=8192, active_size=60):
        super().__init__()
        self.name = name  # 支持：身份/概念/空间/抽象/视觉
        self.dim = initial_dim
        self.max_dim = max_dim
        self.active_size = active_size
        
        # 赫布学习突触权重矩阵
        self.synapse = nn.Parameter(torch.zeros(initial_dim, initial_dim), requires_grad=False)
        # 历史SDR记忆库
        self.sdr_list = []
        self.content_list = []
        self.metadata_list = []
        
        # SDR ↔ mem_id 双向映射（稳定版）
        self.sdr_to_mem_id = {}  # sdr哈希 -> 记忆ID
        self.mem_id_to_sdr = {}  # 记忆ID -> sdr哈希

    def forward(self, sdr, steps=2, top_k=60):
        """
        SDR在专家网络中传播激活（核心类脑推理）
        """
        if sdr.dim() == 1:
            sdr = sdr.unsqueeze(0)
        
        activation = sdr.float()
        
        for _ in range(steps):
            # 突触传播
            activation = torch.sigmoid(torch.matmul(activation, self.synapse.T))
            
            # 赢者通吃（稀疏激活）
            top_k_actual = min(top_k, activation.shape[-1])
            top_values, top_indices = torch.topk(activation, k=top_k_actual, dim=-1)
            new_activation = torch.zeros_like(activation)
            new_activation.scatter_(-1, top_indices, top_values)
            activation = new_activation
        
        return activation.squeeze(0)

    def retrieve(self, query_sdr, top_k=10, steps=2):
        """
        基于突触权重的联想检索
        """
        if not self.sdr_list:
            return []
        
        # 传播激活
        activated_sdr = self.forward(query_sdr, steps=steps)
        
        # 计算与所有历史SDR的相似度
        results = []
        for i, hist_sdr in enumerate(self.sdr_list):
            # 余弦相似度
            sim = F.cosine_similarity(activated_sdr, hist_sdr, dim=-1).item()
            # 时间衰减（越新的记忆权重越高）
            time_decay = 0.99 ** (len(self.sdr_list) - i - 1)
            final_score = sim * time_decay
            
            # 安全获取元数据/内容
            meta = self.metadata_list[i] if i < len(self.metadata_list) else {}
            content = self.content_list[i] if i < len(self.content_list) else ""
            
            # 获取记忆ID
            sdr_hash = hash(hist_sdr.numpy().tobytes())
            mem_id = self.sdr_to_mem_id.get(sdr_hash, None)
            
            results.append((final_score, content, meta, i, mem_id))
        
        # 按得分降序排序
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

    def hebbian_update(self, pre_sdr, post_sdr, is_fact=False):
        """
        改进的赫布学习规则（类脑核心学习机制）
        """
        if pre_sdr.dim() == 1:
            pre_sdr = pre_sdr.unsqueeze(0)
        if post_sdr.dim() == 1:
            post_sdr = post_sdr.unsqueeze(0)
        
        # 学习率：事实类知识更高
        lr = 0.02 if is_fact else 0.01
        decay = 0.001
        
        # 赫布突触更新
        delta = lr * torch.matmul(post_sdr.T, pre_sdr)
        self.synapse.data += delta
        
        # 突触衰减（防止过拟合）
        self.synapse.data -= decay * self.synapse.data
        
        # 限制权重范围
        self.synapse.data = torch.clamp(self.synapse.data, -1.0, 1.0)

    def add_memory(self, sdr, content, mem_id=None, metadata=None):
        """
        添加记忆到专家网络（支持身份/概念等所有脑区）
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
        """睡眠巩固：重放记忆+修剪弱突触（全脑区通用）"""
        if not self.sdr_list:
            logger.info(f"🌙 专家 [{self.name}] 无记忆，跳过睡眠巩固")
            return
        
        logger.info(f"🌙 专家 [{self.name}] 开始睡眠巩固 (epochs={epochs})...")
        
        # 记忆重放（类脑睡眠巩固机制）
        for epoch in range(epochs):
            for i in range(len(self.sdr_list)):
                sdr = self.sdr_list[i]
                is_fact = self.metadata_list[i].get('is_fact', False) if i < len(self.metadata_list) else False
                self.hebbian_update(sdr, sdr, is_fact=is_fact)
        
        # 修剪弱连接（模拟人脑突触修剪）
        weak_threshold = 0.01
        num_weak = torch.sum(torch.abs(self.synapse.data) < weak_threshold).item()
        total = self.synapse.data.numel()
        self.synapse.data[torch.abs(self.synapse.data) < weak_threshold] = 0.0
        
        # 优化日志输出
        sparsity = self.get_sparsity() * 100
        logger.info(f"✅ 专家 [{self.name}] 睡眠巩固完成 | 稀疏度: {sparsity:.2f}%")
        logger.info(f"   修剪弱连接: {num_weak}/{total} ({num_weak/total:.2%})")

    def get_sparsity(self):
        """计算突触稀疏度（类脑健康度指标）"""
        if self.synapse is None:
            return 0.0
        return (torch.abs(self.synapse.data) < 0.01).float().mean().item()

    def save_weights(self, path):
        """保存专家权重+记忆+映射关系（全脑区通用）"""
        try:
            torch.save({
                'synapse': self.synapse.data,
                'sdr_list': self.sdr_list,
                'content_list': self.content_list,
                'metadata_list': self.metadata_list,
                'sdr_to_mem_id': self.sdr_to_mem_id,
                'mem_id_to_sdr': self.mem_id_to_sdr,
                'dim': self.dim
            }, path)
            logger.info(f"💾 专家 [{self.name}] 权重已保存: {path}")
        except Exception as e:
            logger.error(f"❌ 专家 [{self.name}] 权重保存失败: {e}")

    def load_weights(self, path):
        """加载专家权重（兼容旧版本+身份专家）"""
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
            logger.info(f"✅ 专家 [{self.name}] 加载完成 | 记忆数: {len(self.sdr_list)}")
        except Exception as e:
            logger.error(f"❌ 专家 [{self.name}] 权重加载失败: {e}，重置为初始状态")
            # 初始化重置
            self.synapse.data = torch.zeros(self.dim, self.dim)
            self.sdr_list = []
            self.content_list = []
            self.metadata_list = []