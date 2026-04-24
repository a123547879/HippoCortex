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

from PersistentCortexV2 import PersistentCortexV2
from HippocampusRouterV2 import HippocampusRouterV2
from DynamicExpert import DynamicExpert

from langchain_ollama import OllamaEmbeddings

class AdvancedBrainV3:
    def __init__(self, dim=1024, storage_dir="brain_v2"):
        # 🔥 关键修改：bge-m3 原生输出维度为 1024
        self.dim = 1024
        self.storage_dir = storage_dir
        self.expert_names = ["视觉", "概念", "空间", "抽象"]
        os.makedirs(storage_dir, exist_ok=True)
        
        self.cortex = PersistentCortexV2(storage_dir)
        self.hippo = HippocampusRouterV2(
            self.expert_names, storage_dir, input_dim=self.dim, sdr_dim=2048, active_size=60
        )
        
        self.experts = {}
        for name in self.expert_names:
            exp = DynamicExpert(name, initial_dim=2048, max_dim=8192)
            exp.load_weights(os.path.join(storage_dir, f"{name}_weights.pt"))
            self.experts[name] = exp
        
        # 🔥 关键修改：加载 bge-m3 中文嵌入模型
        print("🤖 正在加载 bge-m3 中文嵌入模型...")
        self.embedder = OllamaEmbeddings(model="bge-m3")
        print("✅ bge-m3 中文嵌入模型加载完成！")
        
        self.blur_counter = 0
        self.learn_stats = {'total_samples': 0, 'expert_distribution': {n: 0 for n in self.expert_names}}
        
        # 初始化专家语义原型（自动适配bge-m3编码）
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
        # 🔥 关键修改：用 bge-m3 生成中文语义向量
        emb = self.embedder.embed_query(text)
        # 归一化，保证余弦相似度计算准确
        return F.normalize(torch.tensor(emb, dtype=torch.float32), p=2, dim=-1)

    def learn(self, text, is_fact=False, force_expert=None):
        """
        学习一条知识
        is_fact: 是否是事实性知识（用于赫布学习）
        force_expert: 强制指定存入的专家分区，None则自动路由（优先级最高）
        """
        # ==============================================
        # 🔥 优先级1：强制指定专家（最高优先级）
        # ==============================================
        if force_expert and force_expert in self.expert_names:
            expert_name = force_expert
        else:
            # ==============================================
            # 优先级2：三层路由逻辑（完全保留你的原有逻辑）
            # ==============================================
            # ========== 第一层：强制前缀路由（扩展版）==========
            prefix_map = {
                "人物:": "概念", "人物：": "概念",  # 兼容中文冒号
                "技能:": "概念", "技能：": "概念",
                "案件:": "空间", "案件：": "空间",
                "事件:": "空间", "事件：": "空间",
                "名言:": "抽象", "名言：": "抽象",
                "方法:": "抽象", "方法：": "抽象",
                "知识:": "抽象", "知识：": "抽象",
                "视觉:": "视觉", "视觉：": "视觉",
                "图像:": "视觉", "图像：": "视觉",
                "图片:": "视觉", "图片：": "视觉",
            }
            
            # 先检查硬前缀
            expert_name = None
            for prefix, expert in prefix_map.items():
                if text.startswith(prefix):
                    expert_name = expert
                    break
            
            if not expert_name:
                # ========== 第二层：关键词路由 ==========
                keyword_map = {
                    "概念": ["人物", "职业", "身份", "关系", "朋友", "家人", "名字", "医生", "教授", "侦探", "作家", "诗人"],
                    "空间": ["案件", "事件", "地点", "位置", "地图", "现场", "谋杀", "盗窃", "年份", "时间", "历史", "发生"],
                    "抽象": ["名言", "方法", "逻辑", "推理", "理论", "观察", "细节", "思维", "知识", "概念", "定义"],
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
                    # ========== 第三层：向量路由（保底）==========
                    clip_vec = self.encode_text(text)
                    clip_vec = F.normalize(clip_vec.detach(), p=2, dim=-1)
                    expert_name = self.hippo.route(clip_vec)
        
        # ==============================================
        # 后续逻辑：完全保留你的原有完整逻辑
        # ==============================================
        # 确保 clip_vec 存在
        if 'clip_vec' not in locals():
            clip_vec = self.encode_text(text)
            clip_vec = F.normalize(clip_vec.detach(), p=2, dim=-1)
        
        sdr = self.hippo.encode(clip_vec)
        self.hippo.encoder.online_learn(clip_vec)
        
        # 暂时禁用原型更新，防止滚雪球
        self.hippo.update_expert_prototype(expert_name, sdr)
        
        self.experts[expert_name].hebbian_update(sdr, sdr, is_fact)
        self.cortex.store_detailed_memory(expert_name, sdr, clip_vec, text, metadata={'is_fact': is_fact})
        
        # 注释掉频繁保存，只在退出时统一保存，提升性能
        # self.experts[expert_name].save_weights(os.path.join(self.storage_dir, f"{expert_name}_weights.pt"))
        # self.hippo.encoder.save(os.path.join(self.storage_dir, "sdr_encoder.pt"))
        # self.hippo.save_projections()
        
        self.learn_stats['total_samples'] += 1
        self.learn_stats['expert_distribution'][expert_name] += 1
        
        # 打印存入的专家（方便调试）
        print(f"🧠 存入专家: [{expert_name}]")
        
        return expert_name

    def recall_compositional(self, text, target_expert=None):
        clip_vec = self.encode_text(text)
        clip_vec = F.normalize(clip_vec.detach().squeeze(), p=2, dim=-1)
        
        print(f"🔍 遍历所有专家，寻找最佳匹配...")
        results = self.cortex.search_memories(
            clip_vec,
            expert_name=target_expert,
            min_similarity=0.0
        )
        
        print(f"  找到 {len(results)} 条候选记忆")
        if len(results) > 5:
            # r = np.random.randint(0, len(results) - 5)
            for i, (mem_id, sim, content, meta) in enumerate(results[:5]):
                print(f"    候选 {i+1}: 相似度={sim:.3f}, 专家={meta.get('expert', '?')}, 内容={content[:40]}...")
        else:
            for i, (mem_id, sim, content, meta) in enumerate(results):
                print(f"    候选 {i+1}: 相似度={sim:.3f}, 专家={meta.get('expert', '?')}, 内容={content[:40]}...")
        
        if not results:
            return [], None
        
        # 返回所有匹配记忆，而不是只返回第一条
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

    def get_brain_status(self):
        # 从实际存储中统计，完全抛弃统计变量
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