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

# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


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

    def search(self, query_sdr, top_k=5, fast_mode=False):
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
        return results[:top_k]

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

    def search_memories(self, query_sdr, expert_name=None, top_k=5, min_similarity=0.1):
        if query_sdr.dim() == 1:
            query_sdr = query_sdr.unsqueeze(0)
        
        # 🔥 搜索所有记忆
        results = self.index.search(query_sdr.squeeze(0), top_k=top_k * 10, fast_mode=False)
        
        filtered = []
        for mem_id, sim, mem_data in results:
            if sim < min_similarity:
                continue
            
            # 🔥 如果指定了专家，才过滤；否则不过滤
            if expert_name and mem_data['metadata'].get('expert') != expert_name:
                continue
            
            filtered.append((
                mem_id, 
                sim, 
                mem_data['content'],
                mem_data['metadata']
            ))
        
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:top_k]

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


# ==============================
# 核心大脑系统
# ==============================
class AdvancedBrainV2:
    def __init__(self, dim=512, storage_dir="brain_v2"):
        self.dim = dim
        self.storage_dir = storage_dir
        self.expert_names = ["视觉", "概念", "空间", "抽象"]
        os.makedirs(storage_dir, exist_ok=True)
        
        self.cortex = PersistentCortexV2(storage_dir)
        self.hippo = HippocampusRouterV2(
            self.expert_names, storage_dir, input_dim=dim, sdr_dim=2048, active_size=60
        )
        
        self.experts = {}
        for name in self.expert_names:
            exp = DynamicExpert(name, initial_dim=2048, max_dim=8192)
            exp.load_weights(os.path.join(storage_dir, f"{name}_weights.pt"))
            self.experts[name] = exp
        
        self.tokenizer = CLIPTokenizer.from_pretrained(r"D:\2250111005\HippoCortex\clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained(r"D:\2250111005\HippoCortex\clip-vit-large-patch14")
        
        self.blur_counter = 0
        self.learn_stats = {'total_samples': 0, 'expert_distribution': {n: 0 for n in self.expert_names}}
        
        # 初始化专家语义原型
        with torch.no_grad():
            expert_init_texts = {
                "视觉": "图片 图像 视觉 颜色 形状 画面 照片 绘画 艺术",
                "概念": "人物 名字 身份 职业 关系 朋友 家人 知识 事实",
                "空间": "位置 地点 坐标 方向 地图 空间 距离 案件 故事",
                "抽象": "理论 定律 逻辑 数学 哲学 思想 名言 推理",
            }
            
            if not os.path.exists(os.path.join(storage_dir, "hippo_v2.pt")):
                init_clip_vecs = []
                for name, text in expert_init_texts.items():
                    clip_vec = self.encode_text(text)
                    clip_vec = F.normalize(clip_vec.detach(), p=2, dim=-1)
                    init_clip_vecs.append(clip_vec)
                
                for i, (name, text) in enumerate(expert_init_texts.items()):
                    clip_vec = init_clip_vecs[i]
                    sdr = self.hippo.encode(clip_vec)
                    idx = self.expert_names.index(name)
                    self.hippo.expert_prototypes.data[idx] = sdr.squeeze()

    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
        # 🔥 关键修复：用 pooler_output 而不是 last_hidden_state[:, 0, :]
        return F.normalize(outputs.pooler_output.squeeze(), p=2, dim=-1)

    def learn(self, text, is_fact=False):
        # ========== 第一层：强制前缀路由（扩展版）==========
        prefix_map = {
            "人物:": "概念", "技能:": "概念",
            "案件:": "空间", "事件:": "空间",  # 事件也是时空类
            "名言:": "抽象", "方法:": "抽象",  # 方法偏抽象逻辑
            "视觉:": "视觉", "图像:": "视觉", "图片:": "视觉",
        }
        
        # 先检查硬前缀
        for prefix, expert in prefix_map.items():
            if text.startswith(prefix):
                expert_name = expert
                break
        else:
            # ========== 第二层：关键词路由（新增）==========
            keyword_map = {
                "概念": ["人物", "职业", "身份", "关系", "朋友", "家人", "名字", "医生", "教授", "侦探"],
                "空间": ["案件", "事件", "地点", "位置", "地图", "现场", "谋杀", "盗窃", "年份", "时间"],
                "抽象": ["名言", "方法", "逻辑", "推理", "理论", "观察", "细节", "思维"],
                "视觉": ["图像", "颜色", "形状", "画面", "照片", "绘画"],
            }
            
            text_lower = text.lower()
            scores = {name: 0 for name in self.expert_names}
            for expert, keywords in keyword_map.items():
                scores[expert] = sum(1 for kw in keywords if kw in text_lower)
            
            # 如果关键词有明确 winner，用它
            max_score = max(scores.values())
            if max_score >= 2:  # 至少命中2个关键词才算
                expert_name = max(scores, key=scores.get)
            else:
                # ========== 第三层：CLIP向量路由（保底）==========
                clip_vec = self.encode_text(text)
                clip_vec = F.normalize(clip_vec.detach(), p=2, dim=-1)
                expert_name = self.hippo.route(clip_vec)
        
        # 确保 clip_vec 存在（如果走了前缀/关键词分支，需要补编码）
        if 'clip_vec' not in locals():
            clip_vec = self.encode_text(text)
            clip_vec = F.normalize(clip_vec.detach(), p=2, dim=-1)
        
        sdr = self.hippo.encode(clip_vec)
        self.hippo.encoder.online_learn(clip_vec)
        
        # 🔥 暂时禁用原型更新，防止滚雪球
        self.hippo.update_expert_prototype(expert_name, sdr)
        
        self.experts[expert_name].hebbian_update(sdr, sdr, is_fact)
        self.cortex.store_detailed_memory(expert_name, sdr, clip_vec, text, metadata={'is_fact': is_fact})
        self.experts[expert_name].save_weights(os.path.join(self.storage_dir, f"{expert_name}_weights.pt"))
        self.hippo.encoder.save(os.path.join(self.storage_dir, "sdr_encoder.pt"))
        self.hippo.save_projections()
        
        self.learn_stats['total_samples'] += 1
        self.learn_stats['expert_distribution'][expert_name] += 1
        return expert_name

    # def recall_compositional(self, text):
    #     clip_vec = self.encode_text(text)
    #     clip_vec = F.normalize(clip_vec.detach().squeeze(), p=2, dim=-1)
    #     query_sdr = self.hippo.encode(clip_vec)
        
    #     # 🔥 不限制专家！直接搜索所有记忆！
    #     print(f"🔍 遍历所有专家，寻找最佳匹配...")
    #     results = self.cortex.search_memories(
    #         query_sdr, 
    #         expert_name=None,  # 🔥 不限制专家！
    #         top_k=5, 
    #         min_similarity=0.0
    #     )
        
    #     print(f"  找到 {len(results)} 条候选记忆")
    #     for i, (mem_id, sim, content, meta) in enumerate(results):
    #         print(f"    候选 {i+1}: 相似度={sim:.3f}, 专家={meta.get('expert', '?')}, 内容={content[:30]}...")
        
    #     best_match = -1.0
    #     best_result = None
    #     if results:
    #         best_match = results[0][1]
    #         best_result = results[0]
    #         print(f"✅ 最佳匹配: 相似度={best_match:.3f}, 专家={best_result[3].get('expert', '?')}")
        
    #     novelty_score = 1.0 - best_match
    #     if novelty_score > 0.65:
    #         self.learn(text)
    #         return "🤖 我学到新知识了！", None
        
    #     if best_result:
    #         mem_id, sim, content, meta = best_result
    #         self.cortex.increment_access_count(mem_id)
    #         return content, {'similarity': sim, 'expert': meta.get('expert', best_result[3].get('expert', '?'))}
        
    #     return "❓ 没有找到相关记忆", None

    def recall_compositional(self, text):
        clip_vec = self.encode_text(text)
        clip_vec = F.normalize(clip_vec.detach().squeeze(), p=2, dim=-1)
        
        print(f"🔍 遍历所有专家，寻找最佳匹配...")
        results = self.cortex.search_memories(
            clip_vec,
            expert_name=None,
            top_k= 100,  # 🔥 只取前3条最相关的
            min_similarity=0.0
        )
        
        print(f"  找到 {len(results)} 条候选记忆")
        for i, (mem_id, sim, content, meta) in enumerate(results[:5]):
            print(f"    候选 {i+1}: 相似度={sim:.3f}, 专家={meta.get('expert', '?')}, 内容={content[:40]}...")
        
        if not results:
            return [], None
        
        # 🔥 返回所有前3条记忆，而不是只返回第一条
        memories = [content for _, _, content, _ in results]
        best_sim = results[0][1]
        
        novelty_score = 1.0 - best_sim
        if novelty_score > 0.65:
            self.learn(text)
            return [], None
        
        # 增加访问计数
        for mem_id, _, _, _ in results:
            self.cortex.increment_access_count(mem_id)
        
        return memories, {'similarity': best_sim}

    # def get_brain_status(self):
    #     status = {
    #         "total_memories": sum(self.cortex.get_expert_stats(n)['count'] for n in self.expert_names),
    #         "total_samples": self.learn_stats['total_samples'],
    #         "expert_distribution": self.learn_stats['expert_distribution'],
    #         "experts": {},
    #     }
    #     for name, exp in self.experts.items():
    #         stats = self.cortex.get_expert_stats(name)
    #         status["experts"][name] = {
    #             "神经元": exp.dim, "记忆数": stats['count'], "平均访问": round(stats['avg_access'], 2),
    #         }
    #     return status

    def get_brain_status(self):
        # 🔥 从实际存储中统计，完全抛弃统计变量
        total_memories = len(self.cortex.index.memories)
        
        # 统计各专家实际记忆数和访问量
        expert_counts = defaultdict(int)
        expert_access = defaultdict(list)
        
        for mem in self.cortex.index.memories.values():
            expert = mem['metadata'].get('expert', '未知')
            expert_counts[expert] += 1
            expert_access[expert].append(mem.get('access_count', 0))
        
        status = {
            "total_memories": total_memories,
            "total_samples": total_memories,  # 学习的样本数等于记忆数
            "expert_distribution": {},
            "experts": {},
        }
        
        for name in self.expert_names:
            count = expert_counts.get(name, 0)
            access_list = expert_access.get(name, [0])
            avg_access = np.mean(access_list) if access_list else 0
            
            status["expert_distribution"][name] = count
            status["experts"][name] = {
                "神经元": self.experts[name].dim,
                "记忆数": count,
                "平均访问": round(avg_access, 2),
            }
        
        return status


# ==============================
# 🤖 LLM 增强 Wrapper（核心新增）
# ==============================
class LLMBrainWrapper:
    def __init__(self, brain, ollama_model_name="qwen2.5:0.5b"):
        self.brain = brain
        self.model_name = ollama_model_name
        
        print(f"🤖 正在初始化 LangChain + Ollama，使用模型: {ollama_model_name}")
        self.llm = ChatOllama(
            model=ollama_model_name,
            temperature=0.7,
            num_predict=256,
        )
        print("✅ LangChain + Ollama 初始化完成！")

    def _call_llm(self, prompt, system_prompt=None, max_tokens=256, temperature=0.7):
        """🔥 修复版：直接创建新的 ChatOllama 实例，不用 configure"""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        try:
            # 每次调用都创建一个新的实例，传入想要的参数
            temp_llm = ChatOllama(
                model=self.model_name,
                temperature=temperature,
                num_predict=max_tokens,
            )
            response = temp_llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"⚠️ LangChain + Ollama 调用失败: {e}")
            return ""

    def learn(self, text):
        """用 Ollama 增强的学习：先理解，再存储"""
        print(f"\n📚 正在学习: {text[:40]}...")
        
        parse_prompt = f"""请分析下面这段知识，判断它属于什么类型，并提取关键词。

知识内容：{text}

请直接回答，按以下格式：
类型：[人物/案件/名言/视觉/抽象]
关键词：[关键词1, 关键词2]"""
        
        try:
            parse_result = self._call_llm(
                parse_prompt,
                max_tokens=64,
                temperature=0.3
            )
            print(f"🤖 Ollama 理解: {parse_result.strip()}")
            
            expert = self.brain.learn(text, is_fact=True)
            print(f"🧠 存入专家: [{expert}]")
            return expert
            
        except Exception as e:
            print(f"⚠️ Ollama 解析失败，直接存储: {e}")
            return self.brain.learn(text, is_fact=True)

#     def ask(self, query):
#         """用 Ollama 增强的问答：理解查询 -> 检索记忆 -> 生成自然回答"""
#         print(f"\n❓ 用户问题: {query}")
        
#         understand_prompt = f"""用户问了一个问题，请分析他想知道什么，并生成2-3个检索关键词。

# 用户问题：{query}

# 请直接回答，只输出关键词，用空格分隔："""
        
#         try:
#             keywords = self._call_llm(
#                 understand_prompt,
#                 max_tokens=32,
#                 temperature=0.3
#             ).strip()
#             print(f"🔍 检索关键词: {keywords}")
            
#             search_query = keywords if keywords else query
#             result, meta = self.brain.recall_compositional(search_query)
            
#             if not meta or meta.get('similarity', 0) < 0.25:
#                 result, meta = self.brain.recall_compositional(query)
            
#             if meta and meta.get('similarity', 0) > 0.25:
#                 print(f"✅ 找到记忆 (相似度: {meta['similarity']:.2f}, 专家: {meta['expert']})")
                
#                 system_prompt = """你是一个知识助手。请严格遵守以下规则：
#                                 1. 你只能基于"提供的记忆内容"回答，不得添加记忆外的任何信息。
#                                 2. 如果记忆内容无法完整回答问题，请直接说"记忆中只有部分信息"。
#                                 3. 不要解释、扩展、或推测。只复述记忆中的事实。
#                                 4. 回答长度不超过3句话。
#                                 5. 绝对禁止编造。"""
#                 user_prompt = f"记忆内容：{result}\n\n用户问题：{query}"
                
#                 final_answer = self._call_llm(
#                     user_prompt,
#                     system_prompt=system_prompt,
#                     max_tokens=256,
#                     temperature=0.7
#                 )
#                 return f"💬 {final_answer.strip()}"
#             else:
#                 print(f"🧠 没找到相关记忆，触发自主学习...")
#                 self.brain.learn(query)
#                 return "抱歉，我还不知道这个问题的答案，但我已经把它记下来了！"
                
#         except Exception as e:
#             print(f"⚠️ Ollama 处理失败，直接用原系统: {e}")
#             result, meta = self.brain.recall_compositional(query)
#             return result if result else "抱歉，我还不知道这个问题的答案。"

    def ask(self, query):
        print(f"\n❓ 用户问题: {query}")
        
        understand_prompt = f"""用户问了一个问题，请分析他想知道什么，并生成2-3个检索关键词。

    用户问题：{query}

    请直接回答，只输出关键词，用空格分隔："""
        
        try:
            keywords = self._call_llm(
                understand_prompt,
                max_tokens=32,
                temperature=0.3
            ).strip()
            print(f"🔍 检索关键词: {keywords}")
            
            search_query = keywords if keywords else query
            memories, meta = self.brain.recall_compositional(search_query)
            
            if not memories:
                memories, meta = self.brain.recall_compositional(query)
            
            if memories:
                print(f"✅ 找到 {len(memories)} 条相关记忆")
                
                # 🔥 关键修改：把所有记忆都传给大模型
                system_prompt = """你是一个知识助手。请严格遵守以下规则：
    1. 你只能基于下面提供的"相关记忆"回答问题。
    2. 从多条记忆中选择最相关的一条来回答。
    3. 如果没有任何一条记忆能回答问题，请直接说"抱歉，我没有这方面的信息"。
    4. 不要编造任何记忆中没有的信息。
    5. 不要提到"记忆"这个词。
    6. 回答要简洁准确，不超过2句话。"""
                
                # 把所有记忆拼接成上下文
                memory_context = "\n".join([f"- {mem}" for mem in memories])
                user_prompt = f"""相关记忆：
    {memory_context}

    用户问题：{query}"""
                
                final_answer = self._call_llm(
                    user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=256,
                    temperature=0.3  # 降低温度，让回答更准确
                )
                if final_answer.strip() == "抱歉，我没有这方面的信息。":
                    # is_statement = not any(user_prompt in query.lower() for q in ["?", "什么", "哪里", "谁", "怎么", "吗", "呢"])
                    is_statement = not any(q in query.lower() for q in ["?", "什么", "哪里", "谁", "怎么", "吗", "呢"])
            
                    if is_statement:
                        # 是陈述，直接学习
                        print(f"🧠 没找到相关记忆，触发自主学习...")
                        self.learn(query)
                        return "💬 好的，我记住了！"
                    else:
                        # 是问题，反问用户答案
                        return f"🧠 我不知道这个问题的答案..."                        
                return f"💬 {final_answer.strip()}"
                
            else:
                if final_answer.strip() == "抱歉，我没有这方面的信息。":
                    # is_statement = not any(user_prompt in user_input.lower() for q in ["?", "什么", "哪里", "谁", "怎么", "吗", "呢"])
                    is_statement = not any(q in query.lower() for q in ["?", "什么", "哪里", "谁", "怎么", "吗", "呢"])
            
                    if is_statement:
                        # 是陈述，直接学习
                        print(f"🧠 没找到相关记忆，触发自主学习...")
                        self.learn(query)
                        return "💬 好的，我记住了！"
                    else:
                        # 是问题，反问用户答案
                        return f"🧠 我不知道这个问题的答案..."                        
                return f"💬 {final_answer.strip()}"
                
        except Exception as e:
            print(f"⚠️ Ollama 处理失败: {e}")
            return "抱歉，我遇到了一些问题，请稍后再试。"

# ==============================
# 使用示例
# ==============================
if __name__ == "__main__":
    import shutil
    import json
    import os

    storage_dir = "brain_v3_demo"
    os.makedirs(storage_dir, exist_ok=True)

    # # # 清理旧数据
    if os.path.exists("brain_v3_demo"):
        shutil.rmtree("brain_v3_demo")

    # 1. 初始化类脑记忆系统
    # print("=" * 60)
    # print("🧠 初始化类脑记忆系统...")
    # print("=" * 60)
    # brain = AdvancedBrainV2(dim=768, storage_dir="brain_v2_demo")

    # # 2. 包装 LLM 增强层
    # print("\n" + "=" * 60)
    # llm_brain = LLMBrainWrapper(brain, ollama_model_name='gemma3:4b')
    # print("=" * 60)

    # 3. 从外部 JSON 加载福尔摩斯数据集
    # print("\n" + "=" * 60)
    # print("📚 加载福尔摩斯数据集 JSON...")
    # print("=" * 60)

    # dataset_path = "sherlock_holmes_dataset_chinese.json"
    # if not os.path.exists(dataset_path):
    #     print(f"❌ 未找到文件: {dataset_path}")
    #     exit(1)

    # with open(dataset_path, "r", encoding="utf-8") as f:
    #     holmes = json.load(f)["sherlock_holmes_dataset"]

    # # 构造学习文本列表
    # samples = []

    # # 基本信息
    # basic = holmes["basic_information"]
    # samples.append(f"人物: {basic['chinese_name']}({basic['full_name']})，职业{basic['occupation']}，住址{basic['residence']}，作者{basic['creator']}")
    # samples.append(f"技能: 福尔摩斯擅长{', '.join(basic['special_skills'])}")
    # samples.append(f"案件: 福尔摩斯首案《{basic['first_case']}》，终案《{basic['last_case']}》")

    # # 时间线
    # for item in holmes["timeline"]:
    #     samples.append(f"事件: {item['year']}年 {item['event']}")

    # # 经典案件
    # for case in holmes["classic_cases"]:
    #     samples.append(f"案件: {case['case_name']}，类型{case['type']}，关键线索{case.get('clue', '')}，凶手{case.get('perpetrator', case.get('solution', ''))}")

    # # 主要人物
    # for char in holmes["characters"]:
    #     samples.append(f"人物: {char['name']}，关系{char['relation']}，身份{char['role']}")

    # # 名言
    # for q in holmes["famous_quotes"]:
    #     samples.append(f"名言: {q['chinese']} —— 福尔摩斯")

    # # 推理方法
    # for m in holmes["deduction_methods"]:
    #     samples.append(f"方法: {m['method']}，{m['description']}")

    # # 开始学习
    # print("\n📚 开始学习福尔摩斯知识...")
    # for text in samples:
    #     llm_brain.learn(text)

    # # 4. 测试问答（福尔摩斯主题）
    # print("\n" + "=" * 60)
    # print("💬 开始福尔摩斯问答测试...")
    # print("=" * 60)

    # queries = [
    #     "福尔摩斯会什么技能?",
    #     "斑点带子案的凶手是谁?",
    #     "华生是谁？",
    #     "莫里亚蒂是什么人？",
    #     "福尔摩斯的名言是什么？",
    #     "巴斯克维尔的猎犬是怎么回事？"
    #     # "What skills does Sherlock Holmes have？",
    #     # "Who is the murderer in The Adventure of the Speckled Band?",
    #     # "Who is John Watson?",
    #     # "Who is James Moriarty?",
    #     # "What are Sherlock Holmes' famous quotes?",
    #     # "What is the story of The Hound of the Baskervilles?"
    # ]

    # for q in queries:
    #     answer = llm_brain.ask(q)
    #     print(f"\n{answer}")

    # 1. 初始化类脑记忆系统
    print("=" * 60)
    print("🧠 初始化类脑记忆系统...")
    print("=" * 60)
    brain = AdvancedBrainV2(dim=768, storage_dir=storage_dir)

    # 2. 包装 LLM 增强层
    print("\n" + "=" * 60)
    llm_brain = LLMBrainWrapper(brain, ollama_model_name='gemma3:4b')
    print("=" * 60)

    # 3. 一键导入通用知识数据集（仅首次运行）
    dataset_path = "general_knowledge.txt"
    first_run_flag = os.path.join(storage_dir, "general_knowledge_imported")
    
    if os.path.exists(dataset_path) and not os.path.exists(first_run_flag):
        print("\n" + "=" * 60)
        print("📚 首次运行，正在导入通用知识数据集...")
        print("=" * 60)
        
        from tqdm import tqdm
        
        # 解析TXT文件
        with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        knowledge_list = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            knowledge_list.append(line)
        
        print(f"✅ 解析完成，共找到 {len(knowledge_list)} 条知识")
        
        # 批量导入
        print(f"\n📚 开始导入...")
        success_count = 0
        for knowledge in tqdm(knowledge_list, desc="导入进度"):
            try:
                llm_brain.learn(knowledge)
                success_count += 1
            except Exception as e:
                print(f"\n❌ 导入失败: {knowledge[:50]}... 错误: {e}")
        
        # 保存并创建标记文件
        print("\n💾 正在保存...")
        brain.cortex.save()
        with open(first_run_flag, "w") as f:
            f.write("通用知识数据集已导入")
        
        print(f"\n🎉 导入完成！成功: {success_count} 条")
    else:
        print("\n✅ 通用知识数据集已导入，跳过")

    # 4. 显示初始大脑状态
    print("\n" + "=" * 60)
    print("📊 初始大脑状态")
    print("=" * 60)
    status = brain.get_brain_status()
    print(json.dumps(
        status, indent=2, ensure_ascii=False,
        default=lambda x: x.item() if hasattr(x, 'item') else x
    ))

    # 4. 交互式对话循环
    print("\n" + "=" * 60)
    print("💬 进入对话模式 (输入 'exit' 退出)")
    print("=" * 60)

    try:
        while True:
            user_input = input("\n你: ").strip()
            if user_input.lower() == "exit":
                # ==============================================
                # 🔥 强制在break之前立即保存所有记忆
                # ==============================================
                print("\n" + "=" * 60)
                print("💾 正在保存所有记忆...")
                print("=" * 60)
                
                # 1. 保存皮层记忆索引（最重要！）
                brain.cortex.save()
                print("✅ 皮层记忆索引已保存")
                
                # 2. 保存所有专家权重
                for name, exp in brain.experts.items():
                    exp.save_weights(os.path.join(storage_dir, f"{name}_weights.pt"))
                print("✅ 专家权重已保存")
                
                # 3. 保存SDR编码器
                brain.hippo.encoder.save(os.path.join(storage_dir, "sdr_encoder.pt"))
                print("✅ SDR编码器已保存")
                
                # 4. 保存海马体路由
                brain.hippo.save_projections()
                print("✅ 海马体路由已保存")
                
                print("\n✅ 所有记忆已安全保存！")
                print("\n再见！下次再聊~")
                break
                
            if not user_input:
                print("请输入你的问题")
                continue
                
            answer = llm_brain.ask(user_input)
            print(f"\n{answer}")

    except KeyboardInterrupt:
        # 处理Ctrl+C强制退出，也自动保存
        print("\n\n⚠️ 检测到强制退出，正在紧急保存记忆...")
        brain.cortex.save()
        print("✅ 记忆已紧急保存，再见！")
    
    # 5. 显示最终大脑状态
    print("\n" + "=" * 60)
    print("📊 最终大脑状态")
    print("=" * 60)
    status = brain.get_brain_status()
    print(json.dumps(
        status, indent=2, ensure_ascii=False,
        default=lambda x: x.item() if hasattr(x, 'item') else x
    ))