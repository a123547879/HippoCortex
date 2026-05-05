import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from collections import deque

class LearnableSparseEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        sdr_dim: int = 2048,
        active_size: int = 60,
        temperature: float = 0.1,
        learning_rate: float = 1e-3,
        momentum: float = 0.9,
        expert_name: str = "概念",  # 🔥 新增：专家名称
    ):
        super().__init__()
        self.input_dim = input_dim
        self.sdr_dim = sdr_dim
        self.base_active_size = active_size  # 保存基础值
        self.temperature = temperature
        self.expert_name = expert_name  # 🔥 新增：保存专家名称
        
        # ========== 🔥 新增：加载专家专属差异化配置 ==========
        self._load_expert_config()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, sdr_dim * 2),
            nn.LayerNorm(sdr_dim * 2),
            nn.GELU(),
            nn.Linear(sdr_dim * 2, sdr_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(sdr_dim, sdr_dim // 2),
            nn.LayerNorm(sdr_dim // 2),
            nn.GELU(),
            nn.Linear(sdr_dim // 2, input_dim),
        )
        
        self.lateral_inhibition = nn.Parameter(torch.eye(sdr_dim) * 0.5)
        self.register_buffer('activation_history', torch.zeros(sdr_dim))
        self.register_buffer('history_count', torch.zeros(1))
        
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, betas=(momentum, 0.999)
        )
        self.training_buffer = deque(maxlen=1000)

    # ========== 🔥 新增：加载专家配置的辅助方法 ==========
    def _load_expert_config(self):
        """加载专家专属配置，动态调整 active_size 和 core_bias"""
        try:
            from BrainConfig import config 
            self.expert_cfg = config.EXPERT_CONFIG.get(self.expert_name, config.EXPERT_CONFIG["概念"])
            # 动态调整 active_size（集群编码）
            self.active_size = self.expert_cfg.get("sdr_active_count", self.base_active_size)
            self.core_bias = self.expert_cfg.get("core_bias", 0.5)
            self.partition_size = int(self.sdr_dim * 0.2)  # 前20%为核心功能区
        except Exception as e:
            # 如果没有配置文件，用默认值
            self.expert_cfg = {}
            self.active_size = self.base_active_size
            self.core_bias = 0.5
            self.partition_size = int(self.sdr_dim * 0.2)
            import logging
            logging.warning(f"[LearnableSparseEncoder-{self.expert_name}] 未加载到专家配置，使用默认值: {e}")

    def _competitive_activation(self, pre_activations, k=None):
        """
        🔥 终极修复版：
        1. ✅ 保留：核心功能区（前20%）加偏置
        2. ✅ 保留：侧抑制机制
        3. ✅ 保留：软激活+硬激活+直通估计器
        4. 🔥 新增：支持动态激活数量 k
        """
        batch_size = pre_activations.shape[0]
        
        # ====================== 确定激活数量 k ======================
        if k is None:
            # 🔥 动态计算 k（如果没有指定）
            base_k = self.active_size  # 基础激活数（原来的固定值）
            
            # 计算输入复杂度：pre_activations 的标准差
            activation_std = torch.std(pre_activations).item()
            
            # 动态调整：复杂输入激活更多神经元
            # 标准差越大，表示输入越"丰富"，需要更多神经元来编码
            dynamic_k = base_k + int(activation_std * 30)
            
            # 限制在合理范围内：[base_k, base_k * 5]
            k = max(base_k, min(dynamic_k, base_k * 5))
        
        # ====================== 核心修复：给核心功能区加偏置（保留不变） ======================
        biased_activations = pre_activations.clone()
        if biased_activations.dim() == 2:
            biased_activations[:, :self.partition_size] += self.core_bias
        else:
            biased_activations[:self.partition_size] += self.core_bias
        
        # ====================== 原有逻辑：侧抑制（保留不变） ======================
        inhibited = biased_activations - torch.matmul(
            F.softmax(self.lateral_inhibition, dim=1),
            biased_activations.unsqueeze(-1)
        ).squeeze(-1) * 0.3
        
        # ====================== 原有逻辑：软激活+硬激活+直通估计器（仅修改k） ======================
        softmax_vals = F.softmax(inhibited / self.temperature, dim=-1)
        
        # 🔥 唯一修改：使用动态 k 替代固定的 self.active_size
        topk_vals, topk_idx = torch.topk(inhibited, k, dim=-1)
        
        hard_mask = torch.zeros_like(biased_activations).scatter_(-1, topk_idx, 1.0)
        sdr = hard_mask - softmax_vals.detach() + softmax_vals
        
        return sdr, topk_idx

    def encode(self, x, return_stats=False, k= 2):
        """
        🔥 修复版：支持传递动态 k 值
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        pre_activations = self.encoder(x)
        
        # 🔥 传递 k 给竞争激活
        sdr, topk_idx = self._competitive_activation(pre_activations, k=k)
        
        if return_stats:
            stats = {
                "activation_count": topk_idx.shape[-1],
                "k_used": k if k is not None else "dynamic",
                "pre_activation_std": torch.std(pre_activations).item()
            }
            return sdr.squeeze(0) if x.shape[0] == 1 else sdr, stats
    
        return sdr.squeeze(0) if x.shape[0] == 1 else sdr

    def decode(self, sdr):
        """
        解码：将SDR重建回原始连续向量
        """
        return self.decoder(sdr)

    def forward(self, x):
        """
        完整前向传播：编码 -> 解码 -> 计算重建损失
        :return: (sdr, reconstructed, recon_loss, stats)
        """
        original_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        sdr = self.encode(x)
        reconstructed = self.decode(sdr)
        
        # 🔥 修复：确保 reconstructed 和 x 的形状一致
        if reconstructed.shape != x.shape:
            reconstructed = reconstructed.view_as(x)
        
        recon_loss = F.mse_loss(reconstructed, x)
        
        # 恢复 sdr 的原始形状
        if len(original_shape) == 1:
            sdr = sdr.squeeze(0)
        
        return sdr, reconstructed, recon_loss, {}

    def train_step(self, x):
        """
        单步训练
        """
        self.optimizer.zero_grad()
        _, _, loss, stats = self.forward(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return stats

    def online_learn(self, x, force_train=False):
        """
        在线学习：积累数据到buffer，够32个就训练一次
        """
        self.training_buffer.append(x.detach().cpu())
        stats = {'buffer_size': len(self.training_buffer)}
        if force_train or len(self.training_buffer) >= 32:
            device = x.device
            batch = torch.stack(list(self.training_buffer)[-32:]).to(device)
            train_stats = self.train_step(batch)
            stats.update(train_stats)
            self.training_buffer.clear()
        return stats

    def compute_similarity(self, sdr1, sdr2):
        """
        计算两个SDR之间的相似度
        """
        if sdr1.dim() == 1:
            sdr1 = sdr1.unsqueeze(0)
        if sdr2.dim() == 1:
            sdr2 = sdr2.unsqueeze(0)
        dot_sim = torch.sum(sdr1 * sdr2, dim=-1)
        normalization = (sdr1.sum(dim=-1) + sdr2.sum(dim=-1)) / 2 + 1e-8
        similarity = dot_sim / normalization
        return similarity.item() if similarity.numel() == 1 else similarity

    def save(self, path):
        """
        保存模型权重和优化器状态
        """
        torch.save({
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        """
        加载模型权重和优化器状态
        """
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            self.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def __call__(self, x):
        """
        🔥 修复：让直接调用 model(x) 时也返回 sdr
        """
        return self.encode(x)