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
        expert_name: str = "概念",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.sdr_dim = sdr_dim
        self.base_active_size = active_size
        self.temperature = temperature
        self.expert_name = expert_name
        
        # 加载专家专属差异化配置
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
        
        # ✅ 修复：侧抑制改为非自抑制矩阵（类脑正确逻辑）
        self.lateral_inhibition = nn.Parameter(torch.randn(sdr_dim, sdr_dim) * 0.01)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, betas=(momentum, 0.999)
        )
        self.training_buffer = deque(maxlen=1000)

    def _load_expert_config(self):
        """加载专家专属配置"""
        try:
            from BrainConfig import config 
            self.expert_cfg = config.EXPERT_CONFIG.get(self.expert_name, config.EXPERT_CONFIG["概念"])
            self.active_size = self.expert_cfg.get("sdr_active_count", self.base_active_size)
            self.core_bias = self.expert_cfg.get("core_bias", 0.5)
            self.partition_size = int(self.sdr_dim * 0.2)
        except Exception as e:
            self.expert_cfg = {}
            self.active_size = self.base_active_size
            self.core_bias = 0.5
            self.partition_size = int(self.sdr_dim * 0.2)
            import logging
            logging.warning(f"[LearnableSparseEncoder-{self.expert_name}] 未加载到专家配置，使用默认值: {e}")

    def _competitive_activation(self, pre_activations, k=None):
        """
        竞争激活（动态稀疏编码 + 核心区偏置 + 侧抑制）
        """
        batch_size = pre_activations.shape[0]
        
        # ✅ 修复：动态k计算（严格限制范围，保证稀疏性）
        if k is None:
            base_k = self.active_size
            activation_std = torch.std(pre_activations).item()
            dynamic_k = base_k + int(activation_std * 20)
            # 强制限制：最小30，最大2倍基础值（保证稀疏）
            k = int(max(30, min(dynamic_k, base_k * 2)))
        
        # 核心功能区偏置
        biased_activations = pre_activations.clone()
        if biased_activations.dim() == 2:
            biased_activations[:, :self.partition_size] += self.core_bias
        else:
            biased_activations[:self.partition_size] += self.core_bias
        
        # 侧抑制
        inhibited = biased_activations - torch.matmul(
            F.softmax(self.lateral_inhibition, dim=1),
            biased_activations.unsqueeze(-1)
        ).squeeze(-1) * 0.3
        
        # 软激活+硬激活+直通估计器
        softmax_vals = F.softmax(inhibited / self.temperature, dim=-1)
        topk_vals, topk_idx = torch.topk(inhibited, k, dim=-1)
        
        hard_mask = torch.zeros_like(biased_activations).scatter_(-1, topk_idx, 1.0)
        sdr = hard_mask - softmax_vals.detach() + softmax_vals
        
        return sdr, topk_idx

    def encode(self, x, return_stats=False, k=None):  # ✅ 修复：默认k=None
        """
        编码：向量 → SDR
        ✅ 修复：默认k=None，启用动态激活
        ✅ 修复：统一返回值形状
        """
        # 统一维度
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        pre_activations = self.encoder(x)
        sdr, topk_idx = self._competitive_activation(pre_activations, k=k)
        
        # 压缩维度（保持输出统一）
        sdr = sdr.squeeze(0)
        
        if return_stats:
            stats = {
                "activation_count": topk_idx.shape[-1],
                "k_used": k if k is not None else "dynamic",
                "pre_activation_std": torch.std(pre_activations).item()
            }
            return sdr, stats
        
        return sdr

    def decode(self, sdr):
        """SDR → 原始向量"""
        if sdr.dim() == 1:
            sdr = sdr.unsqueeze(0)
        return self.decoder(sdr).squeeze(0)

    def forward(self, x):
        """完整前向传播：编码+解码+损失"""
        original_x = x
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        sdr = self.encode(x)
        reconstructed = self.decode(sdr)
        recon_loss = F.mse_loss(reconstructed, original_x)
        
        return sdr, reconstructed, recon_loss

    def train_step(self, x):
        """单步训练"""
        self.optimizer.zero_grad()
        _, _, loss = self.forward(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return {"loss": loss.item()}

    def online_learn(self, x, force_train=False):
        """在线学习"""
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
        """SDR余弦相似度"""
        if sdr1.dim() == 1:
            sdr1 = sdr1.unsqueeze(0)
        if sdr2.dim() == 1:
            sdr2 = sdr2.unsqueeze(0)
        
        sim = F.cosine_similarity(sdr1, sdr2, dim=-1)
        return sim.item() if sim.numel() == 1 else sim

    def save(self, path):
        """保存模型"""
        torch.save({
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        """加载模型"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            self.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def __call__(self, x):
        return self.encode(x)