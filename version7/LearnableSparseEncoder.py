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
    ):
        super().__init__()
        self.input_dim = input_dim
        self.sdr_dim = sdr_dim
        self.active_size = active_size
        self.temperature = temperature
        
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

    def _competitive_activation(self, pre_activations):
        """
        竞争激活机制：赢者通吃 + 侧抑制
        """
        batch_size = pre_activations.shape[0]
        # 侧抑制
        inhibited = pre_activations - torch.matmul(
            F.softmax(self.lateral_inhibition, dim=1),
            pre_activations.unsqueeze(-1)
        ).squeeze(-1) * 0.3
        
        # 软激活
        softmax_vals = F.softmax(inhibited / self.temperature, dim=-1)
        # 硬激活：赢者通吃
        topk_vals, topk_idx = torch.topk(inhibited, self.active_size, dim=-1)
        hard_mask = torch.zeros_like(pre_activations).scatter_(-1, topk_idx, 1.0)
        # 直通估计器（Straight-Through Estimator）
        sdr = hard_mask - softmax_vals.detach() + softmax_vals
        return sdr, topk_idx

    def encode(self, x, return_stats=False):
        """
        编码：将连续向量转换为稀疏分布表征（SDR）
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        pre_activations = self.encoder(x)
        sdr, topk_idx = self._competitive_activation(pre_activations)
        if return_stats:
            return sdr.squeeze(0) if x.shape[0] == 1 else sdr, {}
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