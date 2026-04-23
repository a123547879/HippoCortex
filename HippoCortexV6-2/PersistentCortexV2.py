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

from TopologicalMemoryIndex import TopologicalMemoryIndex


class PersistentCortexV2:
    def __init__(self, storage_dir="brain_data_v2"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.index = TopologicalMemoryIndex(sdr_dim=2048, storage_dir=storage_dir)
        self.expert_namespaces = {}

    # def store_detailed_memory(self, expert_name, global_sdr, token_seq, content, metadata=None):
    #     if global_sdr.dim() == 1:
    #         global_sdr = global_sdr.unsqueeze(0)
    #     meta = {'expert': expert_name, 'token_seq': token_seq.detach().cpu().tolist()}
    #     if metadata:
    #         meta.update(metadata)
    #     mem_id = self.index.add(global_sdr.squeeze(0), content, meta)
    #     if expert_name not in self.expert_namespaces:
    #         self.expert_namespaces[expert_name] = []
    #     self.expert_namespaces[expert_name].append(mem_id)
    #     if mem_id % 10 == 0:
    #         self.index.save()
    #     return mem_id

    def store_detailed_memory(self, expert_name, global_sdr, token_seq, content, metadata=None):
        if global_sdr.dim() == 1:
            global_sdr = global_sdr.unsqueeze(0)
        meta = {
            'expert': expert_name, 
            'sdr': global_sdr.detach().cpu(),
            'clip_vec': token_seq.detach().cpu()  # 🔥 同时存储CLIP向量
        }
        if metadata:
            meta.update(metadata)
        mem_id = self.index.add(global_sdr.squeeze(0), content, meta)
        if expert_name not in self.expert_namespaces:
            self.expert_namespaces[expert_name] = []
        self.expert_namespaces[expert_name].append(mem_id)
        if mem_id % 10 == 0:
            self.index.save()
        return mem_id

    # def search_memories(self, query_sdr, expert_name=None, top_k=5, min_similarity=0.1):
    #     if query_sdr.dim() == 1:
    #         query_sdr = query_sdr.unsqueeze(0)
        
    #     # 🔥 搜索所有记忆
    #     results = self.index.search(query_sdr.squeeze(0), top_k=top_k * 10, fast_mode=False)
        
    #     filtered = []
    #     for mem_id, sim, mem_data in results:
    #         if sim < min_similarity:
    #             continue
            
    #         # 🔥 如果指定了专家，才过滤；否则不过滤
    #         if expert_name and mem_data['metadata'].get('expert') != expert_name:
    #             continue
            
    #         filtered.append((
    #             mem_id, 
    #             sim, 
    #             mem_data['content'],
    #             mem_data['metadata']
    #         ))
        
    #     filtered.sort(key=lambda x: x[1], reverse=True)
    #     return filtered[:top_k]

    def search_memories(self, query_sdr, expert_name=None, min_similarity=0.1):
        if query_sdr.dim() == 1:
            query_sdr = query_sdr.unsqueeze(0)
        
        # 1. 先搜索所有记忆（扩大搜索池）
        all_results = self.index.search(query_sdr.squeeze(0), fast_mode=False)
        
        # 2. 按专家分组
        expert_groups = defaultdict(list)
        for mem_id, sim, mem_data in all_results:
            if sim < min_similarity:
                continue
            exp = mem_data['metadata'].get('expert', '未知')
            expert_groups[exp].append((mem_id, sim, mem_data['content'], mem_data['metadata']))
        
        # 3. 按比例取记忆
        final_results = []
        
        if expert_name:
            # ===================== 指定了专家的情况 =====================
            # 目标专家：取前 50%
            if expert_name in expert_groups:
                target_list = expert_groups[expert_name]
                take_n = max(1, len(target_list))  # 至少取1条
                final_results.extend(target_list[:take_n])
            
            # 其他专家：各取前 20%
            for exp, mem_list in expert_groups.items():
                if exp == expert_name:
                    continue
                take_n = max(1, len(mem_list) // 5)  # 1/5 ≈ 20%
                final_results.extend(mem_list[:take_n])
        
        else:
            # ===================== 没指定专家的情况 =====================
            # 所有专家：各取前 50%
            for exp, mem_list in expert_groups.items():
                take_n = max(1, len(mem_list) // 2)
                final_results.extend(mem_list[:take_n])
        
        # 4. 重新按相似度全局排序，返回 top_k
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results

    def increment_access_count(self, mem_id):
        self.index.increment_access(mem_id)

    def save(self):
        self.index.save()

    def get_expert_stats(self, expert_name):
        mem_ids = self.expert_namespaces.get(expert_name, [])
        if not mem_ids:
            return {'count': 0, 'avg_access': 0}
        accesses = []
        for mid in mem_ids:
            if mid in self.index.memories:
                accesses.append(self.index.memories[mid]['access_count'])
        return {
            'count': len(mem_ids),
            'avg_access': np.mean(accesses) if accesses else 0,
        }