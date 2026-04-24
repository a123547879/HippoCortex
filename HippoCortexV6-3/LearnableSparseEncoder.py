import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# import os
import json
import datetime
from transformers import (
    CLIPTokenizer, CLIPTextModel,
    AutoTokenizer, AutoModelForCausalLM
)
from typing import List, Tuple, Optional
import numpy as np
from collections import deque, defaultdict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# ==============================
# 核心类脑记忆组件（保持原有架构）
# ==============================
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
        batch_size = pre_activations.shape[0]
        inhibited = pre_activations - torch.matmul(
            F.softmax(self.lateral_inhibition, dim=1),
            pre_activations.unsqueeze(-1)
        ).squeeze(-1) * 0.3
        
        softmax_vals = F.softmax(inhibited / self.temperature, dim=-1)
        topk_vals, topk_idx = torch.topk(inhibited, self.active_size, dim=-1)
        hard_mask = torch.zeros_like(pre_activations).scatter_(-1, topk_idx, 1.0)
        sdr = hard_mask - softmax_vals.detach() + softmax_vals
        return sdr, topk_idx

    def encode(self, x, return_stats=False):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        pre_activations = self.encoder(x)
        sdr, topk_idx = self._competitive_activation(pre_activations)
        if return_stats:
            return sdr.squeeze(0) if x.shape[0] == 1 else sdr, {}
        return sdr.squeeze(0) if x.shape[0] == 1 else sdr

    def decode(self, sdr):
        return self.decoder(sdr)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        sdr = self.encode(x)
        reconstructed = self.decode(sdr)
        recon_loss = F.mse_loss(reconstructed, x)
        return sdr, reconstructed, recon_loss, {}

    def train_step(self, x):
        self.optimizer.zero_grad()
        _, _, loss, stats = self.forward(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return stats

    def online_learn(self, x, force_train=False):
        self.training_buffer.append(x.detach().cpu())
        stats = {'buffer_size': len(self.training_buffer)}
        if force_train or len(self.training_buffer) >= 32:
            batch = torch.stack(list(self.training_buffer)[-32:]).to(x.device)
            train_stats = self.train_step(batch)
            stats.update(train_stats)
            self.training_buffer.clear()
        return stats

    def compute_similarity(self, sdr1, sdr2):
        if sdr1.dim() == 1:
            sdr1 = sdr1.unsqueeze(0)
        if sdr2.dim() == 1:
            sdr2 = sdr2.unsqueeze(0)
        dot_sim = torch.sum(sdr1 * sdr2, dim=-1)
        normalization = (sdr1.sum(dim=-1) + sdr2.sum(dim=-1)) / 2 + 1e-8
        similarity = dot_sim / normalization
        return similarity.item() if similarity.numel() == 1 else similarity

    def save(self, path):
        torch.save({'state_dict': self.state_dict(), 'optimizer': self.optimizer.state_dict()}, path)

    def load(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location='cpu', weights_only= False)
            self.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
