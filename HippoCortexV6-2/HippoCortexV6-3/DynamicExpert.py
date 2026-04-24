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


class DynamicExpert:
    def __init__(self, expert_id, initial_dim=2048, max_dim=8192):
        self.expert_id = expert_id
        self.dim = initial_dim
        self.max_dim = max_dim
        self.synapse = torch.zeros((self.dim, self.dim))
        self.neuron_usage = torch.zeros(self.dim)
        self.total_grown = 0
        self.total_pruned = 0

    def hebbian_update(self, pre_sdr, post_sdr, is_fact=False):
        if pre_sdr.size(0) != self.dim:
            new_pre = torch.zeros(self.dim, dtype=pre_sdr.dtype)
            new_pre[:min(pre_sdr.size(0), self.dim)] = pre_sdr[:min(pre_sdr.size(0), self.dim)]
            pre_sdr = new_pre
        if post_sdr.size(0) != self.dim:
            new_post = torch.zeros(self.dim, dtype=post_sdr.dtype)
            new_post[:min(post_sdr.size(0), self.dim)] = post_sdr[:min(post_sdr.size(0), self.dim)]
            post_sdr = new_post
        
        active_neurons = torch.where(pre_sdr == 1)[0]
        self.neuron_usage[active_neurons] += 1.0
        self.neuron_usage *= 0.995
        
        update = torch.outer(post_sdr.float(), pre_sdr.float()) * 0.05
        self.synapse = torch.clamp(self.synapse + update, 0, 1.0)

    def save_weights(self, path):
        torch.save({
            'synapse': self.synapse, 'neuron_usage': self.neuron_usage,
            'dim': self.dim, 'total_grown': self.total_grown, 'total_pruned': self.total_pruned,
        }, path)

    def load_weights(self, path):
        if os.path.exists(path):
            data = torch.load(path, map_location='cpu', weights_only= False)
            self.synapse = data['synapse']
            self.neuron_usage = data['neuron_usage']
            self.dim = data['dim']
            self.total_grown = data.get('total_grown', 0)
            self.total_pruned = data.get('total_pruned', 0)