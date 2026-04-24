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
from LearnableSparseEncoder import LearnableSparseEncoder


class HippocampusRouterV2:
    def __init__(self, expert_names, storage_dir, input_dim=512, sdr_dim=2048, active_size=60):
        self.expert_names = expert_names
        self.storage_dir = storage_dir
        self.input_dim = input_dim
        self.sdr_dim = sdr_dim
        
        encoder_path = os.path.join(storage_dir, "sdr_encoder.pt")
        if os.path.exists(encoder_path):
            self.encoder = LearnableSparseEncoder(input_dim, sdr_dim, active_size)
            self.encoder.load(encoder_path)
        else:
            self.encoder = LearnableSparseEncoder(input_dim, sdr_dim, active_size)
        
        self.expert_prototypes = nn.Parameter(torch.randn(len(expert_names), sdr_dim) * 0.1)
        self.expert_proj = nn.Linear(sdr_dim, len(expert_names), bias=False)
        self.route_temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.route_history = deque(maxlen=1000)
        self.proj_path = os.path.join(storage_dir, "hippo_v2.pt")
        self.load_projections()

    def load_projections(self):
        if os.path.exists(self.proj_path):
            data = torch.load(self.proj_path, map_location='cpu', weights_only= False)
            self.expert_prototypes.data = data['prototypes']
            self.expert_proj.load_state_dict(data['proj'])
            self.route_temperature.data = data['temperature']

    def save_projections(self):
        torch.save({
            'prototypes': self.expert_prototypes.data,
            'proj': self.expert_proj.state_dict(),
            'temperature': self.route_temperature.data,
        }, self.proj_path)

    def encode(self, input_vec):
        return self.encoder.encode(input_vec)

    def route(self, input_vec, return_confidence=False):
        sdr = self.encode(input_vec)
        if sdr.dim() == 1:
            sdr = sdr.unsqueeze(0)
        
        prototype_sims = F.cosine_similarity(
            sdr.unsqueeze(1), self.expert_prototypes.unsqueeze(0), dim=-1
        )
        proj_logits = self.expert_proj(sdr)
        combined_scores = prototype_sims + proj_logits * 0.5
        gates = F.softmax(combined_scores / F.softplus(self.route_temperature), dim=-1)
        
        hard_idx = torch.argmax(gates, dim=-1)
        hard_gates = torch.zeros_like(gates).scatter_(-1, hard_idx.unsqueeze(-1), 1.0)
        gates = hard_gates - gates.detach() + gates
        
        self.route_history.append({
            'gates': gates.squeeze(0).detach().cpu().numpy(),
            'sdr_sparsity': (sdr > 0.5).float().mean().item(),
        })
        
        expert_idx = hard_idx.item()
        confidence = gates[0, expert_idx].item()
        
        if return_confidence:
            return self.expert_names[expert_idx], confidence, gates.squeeze(0)
        return self.expert_names[expert_idx]

    def update_expert_prototype(self, expert_name, sdr, momentum=0.9):
        idx = self.expert_names.index(expert_name)
        if sdr.dim() == 1:
            sdr = sdr.unsqueeze(0)
        current = self.expert_prototypes[idx]
        updated = momentum * current + (1 - momentum) * sdr.mean(dim=0)
        self.expert_prototypes.data[idx] = F.normalize(updated, p=2, dim=-1)

    def match_score(self, sdr1, sdr2):
        return self.encoder.compute_similarity(sdr1, sdr2)