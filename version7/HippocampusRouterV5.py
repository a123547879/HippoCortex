import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import deque
import logging

logger = logging.getLogger("HippocampusRouter")

class HippocampusRouterV5(nn.Module):
    """
    海马体路由：基于向量相似度的智能专家路由
    ✅ 新增【身份】独立专家分区，最高优先级
    ✅ 彻底分离：身份认知、概念、空间、抽象、视觉
    替代硬编码关键词规则
    """
    def __init__(self, input_dim=1024, expert_names=None, learning_rate=1e-3):
        super().__init__()
        self.input_dim = input_dim
        # 🔥 固定五大脑区：身份(最高优先级)、概念、空间、抽象、视觉
        self.expert_names = expert_names or ["身份", "概念", "空间", "抽象", "视觉"]
        self.num_experts = len(self.expert_names)
        
        # 路由网络
        self.router = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, self.num_experts)
        )
        
        # 在线学习buffer
        self.training_buffer = deque(maxlen=1000)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # 初始化专家原型向量
        self.expert_prototypes = {}
        for name in self.expert_names:
            self.expert_prototypes[name] = torch.zeros(input_dim)
        
        # 原型初始化标志
        self._prototypes_initialized = False

    def _initialize_prototypes_with_embedding(self, embedding_model):
        """
        🔥 关键修复：新增【身份专家】原型向量初始化
        专门学习：你/我、名字、主人、关系、身份认知
        """
        if self._prototypes_initialized:
            return
        
        logger.info("🧭 正在初始化专家原型向量...")

        # ===================== 【新增】身份专家典型样本 =====================
        identity_samples = [
            "你是谁？我是小白",
            "我是谁？你是邓尧",
            "我的名字叫小白",
            "你的主人是邓尧",
            "我是你的主人，你是我的宠物",
            "我们的关系是主人和伙伴",
            "你是小白，我是邓尧",
            "身份认知，自我定义"
        ]
        
        # 概念专家典型样本：人物、职业、事物定义
        concept_samples = [
            "人物：阿尔伯特·爱因斯坦，德国物理学家，相对论提出者",
            "职业：医生、老师、工程师",
            "这是什么东西？物体定义",
            "身份证号码有多少位？"
        ]
        
        # 空间专家典型样本：事件、地点、历史
        space_samples = [
            "事件：第二次世界大战，1939年-1945年",
            "中国的首都是北京",
            "历史上的今天发生了什么？",
            "秦始皇统一六国，建立秦朝",
            "这个地方在哪里？"
        ]
        
        # 抽象专家典型样本：名言、知识、道理
        abstract_samples = [
            "名言：三人行，必有我师焉",
            "知识：水的沸点是100摄氏度",
            "什么是人工智能？",
            "知识：地球是圆的"
        ]
        
        # 视觉专家典型样本：图像、视觉信息
        visual_samples = [
            "这张图片里有什么？",
            "看这张照片",
            "图像识别",
            "视觉信息",
            "图片描述"
        ]
        
        # 编码并初始化所有原型
        alpha = 0.8
        
        # 1. 初始化身份专家（核心新增）
        for sample in identity_samples:
            try:
                emb = torch.tensor(embedding_model.embed_query(sample), dtype=torch.float32)
                self.expert_prototypes["身份"] = (1 - alpha) * self.expert_prototypes["身份"] + alpha * emb
            except:
                pass

        # 2. 概念专家
        for sample in concept_samples:
            try:
                emb = torch.tensor(embedding_model.embed_query(sample), dtype=torch.float32)
                self.expert_prototypes["概念"] = (1 - alpha) * self.expert_prototypes["概念"] + alpha * emb
            except:
                pass
        
        # 3. 空间专家
        for sample in space_samples:
            try:
                emb = torch.tensor(embedding_model.embed_query(sample), dtype=torch.float32)
                self.expert_prototypes["空间"] = (1 - alpha) * self.expert_prototypes["空间"] + alpha * emb
            except:
                pass
        
        # 4. 抽象专家
        for sample in abstract_samples:
            try:
                emb = torch.tensor(embedding_model.embed_query(sample), dtype=torch.float32)
                self.expert_prototypes["抽象"] = (1 - alpha) * self.expert_prototypes["抽象"] + alpha * emb
            except:
                pass
        
        # 5. 视觉专家
        for sample in visual_samples:
            try:
                emb = torch.tensor(embedding_model.embed_query(sample), dtype=torch.float32)
                self.expert_prototypes["视觉"] = (1 - alpha) * self.expert_prototypes["视觉"] + alpha * emb
            except:
                pass
        
        self._prototypes_initialized = True
        logger.info("✅ 专家原型向量初始化完成（含身份专家）")

    def forward(self, x):
        """前向传播：输出每个专家的概率"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return F.softmax(self.router(x), dim=-1)

    def route(self, clip_vec, text=None):
        """
        路由决策：选择最适合的专家
        🔥 核心修复：【身份专家】最高优先级，彻底分离人称/关系
        """
        with torch.no_grad():
            # 第一步：计算与各专家原型的相似度
            expert_scores = {}
            for name, prototype in self.expert_prototypes.items():
                if prototype.norm() > 0:
                    sim = F.cosine_similarity(clip_vec, prototype, dim=-1).item()
                    expert_scores[name] = sim
                else:
                    expert_scores[name] = 0.0
            
            # 第二步：🔥 关键词兜底（身份 > 视觉 > 概念 > 空间 > 抽象）
            if text:
                text_lower = text.lower()
                # ===================== 最高优先级：身份/人称/主人/名字 =====================
                if any(keyword in text_lower for keyword in [
                    "我是谁", "你是谁", "我叫", "你叫", "名字", "身份", "主人", 
                    "你是", "我是", "邓尧", "小白", "关系", "自己", "本人"
                ]):
                    return "身份"
                # 视觉
                if any(keyword in text_lower for keyword in ["图片", "图像", "照片", "视觉", "看"]):
                    return "视觉"
                # 概念
                if any(keyword in text_lower for keyword in ["人物", "职业", "东西"]):
                    return "概念"
                # 空间
                if any(keyword in text_lower for keyword in ["案件", "事件", "地点", "历史", "哪里"]):
                    return "空间"
                # 抽象
                if any(keyword in text_lower for keyword in ["名言", "知识", "概念", "定义"]):
                    return "抽象"
            
            # 第三步：选择相似度最高的专家
            if expert_scores:
                best_expert = max(expert_scores.items(), key=lambda x: x[1])[0]
                return best_expert
            
            # 兜底：网络输出
            probs = self.forward(clip_vec).squeeze(0)
            expert_idx = torch.argmax(probs).item()
            return self.expert_names[expert_idx]

    def online_learn(self, clip_vec, expert_name):
        """在线学习：更新路由网络和专家原型"""
        if expert_name not in self.expert_names:
            return
        
        # 更新专家原型
        alpha = 0.05
        self.expert_prototypes[expert_name] = (
            (1 - alpha) * self.expert_prototypes[expert_name] 
            + alpha * clip_vec.detach().cpu()
        )
        
        # 添加到训练buffer
        expert_idx = self.expert_names.index(expert_name)
        self.training_buffer.append((clip_vec.detach().cpu(), expert_idx))
        
        # 批量训练
        if len(self.training_buffer) >= 32:
            self._train_step()

    def _train_step(self):
        """单步训练"""
        batch = list(self.training_buffer)[-32:]
        xs = torch.stack([x for x, y in batch])
        ys = torch.tensor([y for x, y in batch], dtype=torch.long)
        
        self.optimizer.zero_grad()
        logits = self.router(xs)
        loss = F.cross_entropy(logits, ys)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_buffer.clear()
        logger.debug(f"🧭 海马体路由训练完成，损失: {loss.item():.4f}")

    def save(self, path):
        """保存路由模型"""
        torch.save({
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'expert_prototypes': self.expert_prototypes,
            'expert_names': self.expert_names,
            '_prototypes_initialized': self._prototypes_initialized
        }, path)

    def load(self, path):
        """加载路由模型"""
        if not os.path.exists(path):
            return
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            self.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.expert_prototypes = checkpoint.get('expert_prototypes', self.expert_prototypes)
            self.expert_names = checkpoint.get('expert_names', self.expert_names)
            self._prototypes_initialized = checkpoint.get('_prototypes_initialized', False)
            logger.info("✅ 海马体路由加载完成")
        except Exception as e:
            logger.error(f"❌ 海马体路由加载失败: {e}")