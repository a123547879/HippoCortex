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

class TopologicalMemoryIndex:
    def __init__(self, sdr_dim, storage_dir):
        self.sdr_dim = sdr_dim
        self.storage_dir = storage_dir
        self.index_path = os.path.join(storage_dir, "topo_index.pt")
        self.segments = 32
        self.segment_size = sdr_dim // self.segments
        self.inverted_index = [defaultdict(list) for _ in range(self.segments)]
        self.memories = {}
        self.next_id = 0
        self.load()

    def _get_segment_signature(self, sdr, segment_idx):
        start = segment_idx * self.segment_size
        end = start + self.segment_size
        segment = sdr[start:end]
        active_bits = torch.where(segment > 0.5)[0].tolist()
        return ",".join(map(str, active_bits))

    def add(self, sdr, content, metadata=None):
        mem_id = self.next_id
        self.next_id += 1
        self.memories[mem_id] = {
            'sdr': sdr.detach().cpu(),
            'content': content,
            'metadata': metadata or {},
            'access_count': 0,
            'created_at': datetime.datetime.now().isoformat(),
        }
        for seg in range(self.segments):
            sig = self._get_segment_signature(sdr, seg)
            self.inverted_index[seg][sig].append(mem_id)
        return mem_id

    # def search(self, query_sdr, top_k=5, fast_mode=True):
    #     if fast_mode:
    #         candidate_scores = defaultdict(int)
    #         for seg in range(self.segments):
    #             sig = self._get_segment_signature(query_sdr, seg)
    #             for mem_id in self.inverted_index[seg].get(sig, []):
    #                 candidate_scores[mem_id] += 1
            
    #         # 🔥 修复：把阈值从2降到1，确保所有有重叠的记忆都能进入候选
    #         candidates = [mid for mid, score in candidate_scores.items() if score >= 1]
    #     else:
    #         candidates = list(self.memories.keys())
        
    #     if not candidates:
    #         return []
        
    #     results = []
    #     for mem_id in candidates:
    #         mem = self.memories[mem_id]
    #         # 🔥 SDR专用相似度：Jaccard相似度变种
    #         overlap = torch.sum(query_sdr * mem['sdr'])
    #         overlap = torch.sum(query_sdr * mem['sdr'])
    #         total_active = torch.sum(query_sdr) + torch.sum(mem['sdr'])
    #         sim = (2 * overlap) / total_active if total_active > 0 else 0.0

    #         # 🔥 新增：目标专家记忆加分，非目标降权
    #         # if expert_name and mem['metadata'].get('expert') == expert_name:
    #         #     sim *= 1.5  # 对应专家权重拉满
    #         # else:
    #         #     sim *= 0.5  # 其他专家直接砍半
    #         results.append((mem_id, sim.item(), mem))
        
    #     results.sort(key=lambda x: x[1], reverse=True)
    #     return results[:top_k]

    def search(self, query_sdr, fast_mode=False):
        candidates = list(self.memories.keys())
        
        if not candidates:
            return []
        
        results = []
        for mem_id in candidates:
            mem = self.memories[mem_id]
            
            # 🔥 关键修复：直接用CLIP向量计算余弦相似度
            if 'clip_vec' in mem['metadata']:
                query_clip = query_sdr  # 现在传进来的是CLIP向量
                mem_clip = mem['metadata']['clip_vec']
                dot = torch.sum(query_clip * mem_clip)
                norm = torch.norm(query_clip) * torch.norm(mem_clip) + 1e-8
                sim = (dot / norm).item()
            else:
                # 回退到SDR相似度
                overlap = torch.sum(query_sdr * mem['sdr'])
                union = torch.sum(query_sdr) + torch.sum(mem['sdr']) - overlap
                sim = (overlap / (union + 1e-8)).item()
            
            # ==============================================
            # 🔥 新增：抽象专家降权 50%
            # ==============================================
            if mem['metadata'].get('expert') == '抽象':
                sim = sim * 0.9  # 直接砍半
            
            results.append((mem_id, sim, mem))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def increment_access(self, mem_id):
        if mem_id in self.memories:
            self.memories[mem_id]['access_count'] += 1

    def save(self):
        torch.save({
            'memories': self.memories, 'next_id': self.next_id,
            'inverted_index': [dict(seg) for seg in self.inverted_index],
        }, self.index_path)

    def load(self):
        if os.path.exists(self.index_path):
            data = torch.load(self.index_path, map_location='cpu', weights_only= False)
            self.memories = data['memories']
            self.next_id = data['next_id']
            for i, seg_data in enumerate(data['inverted_index']):
                for sig, mem_ids in seg_data.items():
                    self.inverted_index[i][sig] = mem_ids
