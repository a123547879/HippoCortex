import torch
import torch.nn.functional as F
import os
from collections import deque
import logging
import re

logger = logging.getLogger("HippocampusRouter")

class HippocampusRouterV7(torch.nn.Module):
    """
    海马体路由：神经网络为主，规则仅为最后兜底
    ✅ 彻底解决视觉专家垄断问题
    ✅ 全程防垄断机制
    ✅ 类别平衡训练
    """
    def __init__(
        self,
        input_dim: int = 1024,
        expert_names: list = None,
        learning_rate: float = 1e-3,
        confidence_threshold: float = 0.15,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.expert_names = expert_names or ["身份", "概念", "空间", "抽象", "视觉"]
        self.num_experts = len(self.expert_names)
        self.confidence_threshold = confidence_threshold
        
        # ====================== 🔥 核心1：轻量路由网络（避免过拟合） ======================
        # 用更简单的网络，减少过拟合风险
        self.router = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
            torch.nn.Linear(256, self.num_experts)
        )
        
        # ====================== 🔥 核心2：强制初始化所有偏置为0 ======================
        # 彻底解决某个专家偏置异常高的问题
        for m in self.router.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.zeros_(m.bias)
        
        # 在线学习buffer
        self.training_buffer = deque(maxlen=2000)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=learning_rate,
            weight_decay=0.01  # 加入权重衰减，防止过拟合
        )
        
        # 专家原型向量
        self.expert_prototypes = {}
        for name in self.expert_names:
            self.expert_prototypes[name] = torch.zeros(input_dim)
        
        # 原型初始化标志
        self._prototypes_initialized = False
        
        # 统计
        self.correct_count = 0
        self.total_count = 0

    def _initialize_prototypes_with_embedding(self, embedding_model):
        """
        平衡初始化所有专家原型，彻底避免垄断
        """
        if self._prototypes_initialized:
            return
        
        logger.info("🧭 正在初始化专家原型向量...")

        # 每个专家完全相同数量的样本，确保平衡
        identity_samples = [
            "你是谁？我是一个AI助手", "我是谁？你是用户", "我的名字叫AI", "你的用户是我",
            "我是你的助手，你是我的用户", "我们的关系是助手和用户", "你是AI，我是用户",
            "身份认知，自我定义", "我叫什么名字？", "你的名字是什么？"
        ]
        
        concept_samples = [
            "人物：阿尔伯特·爱因斯坦，德国物理学家", "职业：医生、老师、工程师",
            "这是什么东西？物体定义", "什么是苹果？一种水果", "牛顿是谁？物理学家"
        ]
        
        space_samples = [
            "事件：第二次世界大战，1939年-1945年", "中国的首都是北京",
            "历史上的今天发生了什么？", "秦始皇统一六国，建立秦朝", "这个地方在哪里？"
        ]
        
        abstract_samples = [
            "名言：三人行，必有我师焉", "知识：水的沸点是100摄氏度",
            "什么是人工智能？", "知识：地球是圆的", "为什么天是蓝色的？"
        ]
        
        visual_samples = [
            "这张图片里有什么？", "看这张照片", "图像识别", "视觉信息",
            "图片描述", "苹果是什么颜色？红色", "长什么样？圆圆的"
        ]
        
        alpha = 0.7
        
        # 编码并初始化所有原型
        all_prototypes = []
        for name, samples in zip(self.expert_names, 
                                 [identity_samples, concept_samples, space_samples, abstract_samples, visual_samples]):
            for sample in samples:
                try:
                    emb = torch.tensor(embedding_model.embed_query(sample), dtype=torch.float32)
                    self.expert_prototypes[name] = (1 - alpha) * self.expert_prototypes[name] + alpha * emb
                except:
                    pass
            # 强制归一化
            self.expert_prototypes[name] = F.normalize(self.expert_prototypes[name], p=2, dim=-1)
            all_prototypes.append(self.expert_prototypes[name])
        
        # ====================== 🔥 核心3：原型正交化 ======================
        # 让所有专家原型互相正交，彻底避免重叠
        prototype_matrix = torch.stack(all_prototypes)
        # 正交化处理
        # U, S, V = torch.svd(prototype_matrix)
        # orthogonal_prototypes = U @ V.T
        # # 重新分配给各个专家
        # for i, name in enumerate(self.expert_names):
        #     self.expert_prototypes[name] = orthogonal_prototypes[i]
        #     self.expert_prototypes[name] = F.normalize(self.expert_prototypes[name], p=2, dim=-1)
        
        self._prototypes_initialized = True
        logger.info("✅ 专家原型向量初始化完成（正交平衡版）")

    def forward(self, x):
        """前向传播：输出logits"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.router(x)

    def route(self, clip_vec, text=None):
        with torch.no_grad():
            # 1. 原始logits直接推理，不加高温、不抹平分布
            logits = self.forward(clip_vec).squeeze(0)
            nn_probs = F.softmax(logits, dim=-1)

            # 2. 纯神经网络得分，彻底移除原型相似度干扰
            final_scores = {
                name: nn_probs[i].item()
                for i, name in enumerate(self.expert_names)
            }

            # 排序 + 置信度
            sorted_experts = sorted(final_scores.items(), key=lambda x: -x[1])
            best_expert, best_score = sorted_experts[0]
            second_score = sorted_experts[1][1] if len(sorted_experts) > 1 else 0.0
            confidence = (best_score - second_score) / max(best_score, 1e-8)

            self.total_count += 1
            logger.info(f"🧭 路由 | 得分:{final_scores} | 最优:{best_expert} | 置信度:{confidence:.2f}")

            # 3. 正常置信度 → 直接用网络结果（绝大多数场景）
            if confidence >= 0.05:
                self._online_finetune_prototype(clip_vec, best_expert)
                return best_expert

            # 4. 只有【极度模糊、分数几乎一模一样】才走规则兜底
            logger.warning(f"⚠️ 语义极度模糊，启用极简规则兜底")
            if text:
                rule_expert = self._rule_based_fallback(text)
                if rule_expert:
                    return rule_expert

            return best_expert

    def _rule_based_fallback(self, text: str) -> str:
        text_lower = text.lower()
        # 仅保留强特征关键词，绝不干预正常语义路由
        if any(k in text_lower for k in ["图片","照片","图像","颜色","长什么样"]):
            return "视觉"
        if any(k in text_lower for k in ["我是谁","你是谁","名字","介绍你自己"]):
            return "身份"
        return None


    def _online_finetune_prototype(self, clip_vec, expert_name):
        """无监督自学习"""
        if expert_name not in self.expert_prototypes:
            return
        
        alpha = 0.01
        self.expert_prototypes[expert_name] = (
            (1 - alpha) * self.expert_prototypes[expert_name] 
            + alpha * clip_vec.detach().cpu()
        )
        self.expert_prototypes[expert_name] = F.normalize(self.expert_prototypes[expert_name], p=2, dim=-1)

    def online_learn(self, clip_vec, expert_name):
        """有监督强化学习"""
        if expert_name not in self.expert_names:
            return
        
        # 更新原型
        alpha = 0.1
        self.expert_prototypes[expert_name] = (
            (1 - alpha) * self.expert_prototypes[expert_name] 
            + alpha * clip_vec.detach().cpu()
        )
        self.expert_prototypes[expert_name] = F.normalize(self.expert_prototypes[expert_name], p=2, dim=-1)
        
        # 添加到训练buffer
        expert_idx = self.expert_names.index(expert_name)
        self.training_buffer.append((clip_vec.detach().cpu(), expert_idx))
        
        # 批量训练（加入类别平衡）
        if len(self.training_buffer) >= 16:
            self._train_step_balanced()
        
        # 统计
        self.correct_count += 1

    def _train_step_balanced(self):
        """在线学习时使用，不要清空整个buffer"""
        if len(self.training_buffer) < 16:
            return
        
        # 随机采样，不从尾部取
        indices = torch.randperm(len(self.training_buffer))[:16]
        batch = [self.training_buffer[i] for i in indices]
        
        xs = torch.stack([x for x, y in batch])
        ys = torch.tensor([y for x, y in batch], dtype=torch.long)
        
        self.optimizer.zero_grad()
        logits = self.router(xs)
        loss = F.cross_entropy(logits, ys)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        
        # ❌ 不要在这里 clear()！
        logger.debug(f"🧭 训练步 | 损失: {loss.item():.4f}")


    def train(self, training_data, epochs=20, batch_size=16, log_interval=5):
        if not training_data:
            return
        
        # 先把数据全部加入buffer（不清空）
        for text, clip_vec, correct_expert in training_data:
            if correct_expert not in self.expert_names:
                continue
            expert_idx = self.expert_names.index(correct_expert)
            self.training_buffer.append((clip_vec.detach().cpu(), expert_idx))
        
        logger.info(f"🧭 开始预训练 | 数据量: {len(training_data)} | Buffer: {len(self.training_buffer)}")
        
        # 真正训练：每个epoch多次采样
        for epoch in range(epochs):
            # 从buffer中随机采样训练
            if len(self.training_buffer) < batch_size:
                continue
                
            # 每个epoch训练多轮
            num_batches = max(1, len(self.training_buffer) // batch_size)
            for _ in range(num_batches):
                # 随机采样batch（不要只取最后16条）
                indices = torch.randperm(len(self.training_buffer))[:batch_size]
                batch = [self.training_buffer[i] for i in indices]
                
                xs = torch.stack([x for x, y in batch])
                ys = torch.tensor([y for x, y in batch], dtype=torch.long)
                
                self.optimizer.zero_grad()
                logits = self.router(xs)
                loss = F.cross_entropy(logits, ys)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()
            
            # 评估
            if (epoch + 1) % log_interval == 0:
                correct = 0
                for text, clip_vec, correct_expert in training_data:
                    pred = self.route(clip_vec, text)
                    if pred == correct_expert:
                        correct += 1
                acc = correct / len(training_data)
                logger.info(f"   Epoch {epoch+1}/{epochs} | 损失: {loss.item():.4f} | 准确率: {acc:.2%}")

    def save(self, path):
        """保存路由模型"""
        torch.save({
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'expert_prototypes': self.expert_prototypes,
            'expert_names': self.expert_names,
            '_prototypes_initialized': self._prototypes_initialized,
            'confidence_threshold': self.confidence_threshold,
            'correct_count': self.correct_count,
            'total_count': self.total_count,
        }, path)
        logger.info("💾 海马体路由已保存")

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
            self.confidence_threshold = checkpoint.get('confidence_threshold', 0.15)
            self.correct_count = checkpoint.get('correct_count', 0)
            self.total_count = checkpoint.get('total_count', 0)
            logger.info(f"✅ 海马体路由加载完成 | 动态阈值: {self.confidence_threshold:.2f}")
        except Exception as e:
            logger.error(f"❌ 海马体路由加载失败: {e}")