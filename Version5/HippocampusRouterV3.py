import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import deque
import logging

logger = logging.getLogger("HippocampusRouter")

class HippocampusRouterV3(nn.Module):
    """
    海马体路由：基于向量相似度的智能专家路由
    替代硬编码关键词规则
    """
    def __init__(self, input_dim=1024, expert_names=None, learning_rate=1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.expert_names = expert_names or ["概念", "空间", "抽象", "视觉"]
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
        
        # 专家原型向量
        self.expert_prototypes = {}
        for name in self.expert_names:
            self.expert_prototypes[name] = torch.zeros(input_dim)

    def forward(self, x):
        """
        前向传播：输出每个专家的概率
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return F.softmax(self.router(x), dim=-1)

    def route(self, clip_vec, text=None):
        """
        路由决策：选择最适合的专家
        """
        with torch.no_grad():
            # 基于向量的路由
            probs = self.forward(clip_vec).squeeze(0)
            expert_idx = torch.argmax(probs).item()
            expert_name = self.expert_names[expert_idx]
            
            # 关键词兜底（处理极端情况）
            if text:
                text_lower = text.lower()
                if any(keyword in text_lower for keyword in ["图片", "图像", "照片", "视觉", "看"]):
                    return "视觉"
                if any(keyword in text_lower for keyword in ["名字", "叫什么", "我是", "你是", "身份", "称呼", "主人", "人物"]):
                    return "概念"
                if any(keyword in text_lower for keyword in ["案件", "事件", "地点", "历史", "什么时候", "哪里"]):
                    return "空间"
            
            return expert_name

    def online_learn(self, clip_vec, expert_name):
        """
        在线学习：更新路由网络和专家原型
        """
        if expert_name not in self.expert_names:
            return
        
        # 更新专家原型
        alpha = 0.1  # 学习率
        self.expert_prototypes[expert_name] = (
            (1 - alpha) * self.expert_prototypes[expert_name] 
            + alpha * clip_vec.detach().cpu()
        )
        
        # 添加到训练buffer
        expert_idx = self.expert_names.index(expert_name)
        self.training_buffer.append((clip_vec.detach().cpu(), expert_idx))
        
        # 当buffer足够大时训练
        if len(self.training_buffer) >= 32:
            self._train_step()

    def _train_step(self):
        """
        单步训练
        """
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
        """
        保存路由模型
        """
        torch.save({
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'expert_prototypes': self.expert_prototypes,
            'expert_names': self.expert_names
        }, path)

    def load(self, path):
        """
        加载路由模型
        """
        if not os.path.exists(path):
            return
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            self.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.expert_prototypes = checkpoint.get('expert_prototypes', self.expert_prototypes)
            self.expert_names = checkpoint.get('expert_names', self.expert_names)
            logger.info("✅ 海马体路由加载完成")
        except Exception as e:
            logger.error(f"❌ 海马体路由加载失败: {e}")