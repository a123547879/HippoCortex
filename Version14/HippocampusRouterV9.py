import torch
import torch.nn.functional as F
import os
from collections import deque
import logging
import re
import time

logger = logging.getLogger("HippocampusRouter")

class HippocampusRouter(torch.nn.Module):
    """
    🔥 仿生海马体终极版：路由功能 + 记忆编码/巩固/模式分离/补全
    完全兼容V8所有接口 + 新增真实海马体核心功能
    1. 快速编码新记忆（临时缓冲区）
    2. 睡眠系统巩固（转移到皮层长期存储）
    3. 模式分离（避免相似记忆混淆）
    4. 模式补全（部分输入恢复完整记忆）
    5. 索引式记忆提取（海马体存索引，皮层存内容）
    """
    def __init__(
        self,
        input_dim: int = 1024,
        expert_names: list = None,
        experts=None,
        learning_rate: float = 1e-3,
        confidence_threshold: float = 0.15,
        buffer_size: int = 20,        # 海马体临时记忆容量
        consolidation_rate: float = 0.2, # 睡眠巩固进度
        separation_threshold: float = 0.85 # 模式分离阈值
    ):
        super().__init__()
        # ====================== 保留V8所有原有属性 ======================
        self.input_dim = input_dim
        self.expert_names = expert_names or ["身份", "概念", "空间", "抽象", "视觉"]
        self.num_experts = len(self.expert_names)
        self.confidence_threshold = confidence_threshold
        self.experts = experts if experts is not None else {}
        
        # 轻量路由网络（V8原有）
        self.router = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
            torch.nn.Linear(256, self.num_experts)
        )
        
        # 初始化权重（V8原有）
        for m in self.router.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.zeros_(m.bias)
        
        # 在线学习buffer（V8原有）
        self.training_buffer = deque(maxlen=2000)
        self.optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # 专家原型向量（V8原有）
        self.expert_prototypes = {name: torch.zeros(input_dim) for name in self.expert_names}
        self._prototypes_initialized = False
        self.correct_count = 0
        self.total_count = 0
        self.last_confidence = 0.0
        self.last_nn_scores = {}
        self.last_semantic_scores = {}

        # ====================== 🔥 新增：海马体核心仿生功能 ======================
        self.buffer_size = buffer_size
        self.consolidation_rate = consolidation_rate
        self.separation_threshold = separation_threshold

        # 1. 海马体临时记忆缓冲区（CA3区：快速存储短期记忆）
        self.hippocampal_buffer = deque(maxlen=buffer_size)
        # 2. 海马体→皮层索引表（仅存索引，不存完整记忆）
        self.cortex_index_map = {}
        # 3. 全局记忆ID
        self.next_mem_id = 1

    # ====================== 保留V8所有原有方法：初始化/路由/学习/保存/加载 ======================
    def _initialize_prototypes_with_embedding(self, embedding_model):
        """V8原有原型初始化，完全不变"""
        if self._prototypes_initialized:
            return
        
        logger.info("🧭 正在初始化专家原型向量...")
        identity_samples = ["你是谁？我是一个AI助手", "我是谁？你是用户", "我的名字叫AI", "你的用户是我","我是你的助手，你是我的用户", "我们的关系是助手和用户", "你是AI，我是用户","身份认知，自我定义", "我叫什么名字？", "你的名字是什么？"]
        concept_samples = ["人物：阿尔伯特·爱因斯坦，德国物理学家", "职业：医生、老师、工程师","这是什么东西？物体定义", "什么是苹果？一种水果", "牛顿是谁？物理学家"]
        space_samples = ["事件：第二次世界大战，1939年-1945年", "中国的首都是北京","历史上的今天发生了什么？", "秦始皇统一六国，建立秦朝", "这个地方在哪里？"]
        abstract_samples = ["名言：三人行，必有我师焉", "知识：水的沸点是100摄氏度","什么是人工智能？", "知识：地球是圆的", "为什么天是蓝色的？"]
        visual_samples = ["这张图片里有什么？", "看这张照片", "图像识别", "视觉信息","图片描述", "苹果是什么颜色？红色", "长什么样？圆圆的"]
        
        alpha = 0.7
        for name, samples in zip(self.expert_names, [identity_samples, concept_samples, space_samples, abstract_samples, visual_samples]):
            for sample in samples:
                try:
                    emb = torch.tensor(embedding_model.embed_query(sample), dtype=torch.float32)
                    self.expert_prototypes[name] = (1 - alpha) * self.expert_prototypes[name] + alpha * emb
                except:
                    pass
            self.expert_prototypes[name] = F.normalize(self.expert_prototypes[name], p=2, dim=-1)
        
        self._prototypes_initialized = True
        logger.info("✅ 专家原型向量初始化完成（平衡版）")

    def forward(self, x):
        """V8原有前向传播，完全不变"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.router(x)

    def route(self, clip_vec, text=None):
        """V8原有路由逻辑 + 🔥 新增模式补全功能"""
        with torch.no_grad():
            # 1. 原有神经网络+语义网络融合路由
            logits = self.forward(clip_vec).squeeze(0)
            nn_probs = F.softmax(logits, dim=-1)

            semantic_scores = {}
            for i, name in enumerate(self.expert_names):
                expert = self.experts[name]
                semantic_score = 0.0
                if text and hasattr(expert, 'kg'):
                    try:
                        top_nodes = expert.kg.get_top_related_nodes(text, top_k=5)
                        if top_nodes:
                            semantic_score = sum(node.get("similarity", 0.0) for node in top_nodes) / len(top_nodes)
                            semantic_score = max(0.0, min(1.0, semantic_score))
                    except Exception as e:
                        logger.debug(f"[{name}] 语义网络匹配失败: {e}")
                semantic_scores[name] = semantic_score

            # 融合得分
            final_scores = {name: nn_probs[i].item()*0.4 + semantic_scores.get(name,0.0)*0.6 for i, name in enumerate(self.expert_names)}
            sorted_experts = sorted(final_scores.items(), key=lambda x: -x[1])
            best_expert, best_score = sorted_experts[0]
            second_score = sorted_experts[1][1] if len(sorted_experts) > 1 else 0.0
            confidence = (best_score - second_score) / max(best_score, 1e-8)

            # 保存状态
            self.last_confidence = confidence
            self.last_scores = final_scores
            self.last_nn_scores = {name: nn_probs[i].item() for i, name in enumerate(self.expert_names)}
            self.last_semantic_scores = semantic_scores
            self.total_count += 1

            # 日志
            logger.info(f"🧭 路由 | 神经网络得分:{self.last_nn_scores}")
            logger.info(f"🧭 路由 | 语义网络得分:{semantic_scores}")
            logger.info(f"🧭 路由 | 最终融合得分:{final_scores} | 最优:{best_expert} | 置信度:{confidence:.2f}")

            # 🔥 核心新增：路由前触发模式补全（部分输入恢复完整记忆）
            completed_mem = self._pattern_completion(clip_vec)
            if completed_mem:
                logger.info(f"🧠 海马体模式补全成功：{completed_mem['content'][:30]}...")
                best_expert = completed_mem['expert']
                self.last_confidence = 0.95

            # 原有置信度判断+规则兜底
            if confidence >= 0.05:
                self._online_finetune_prototype(clip_vec, best_expert)
                return best_expert

            logger.warning(f"⚠️ 语义极度模糊，启用极简规则兜底")
            if text:
                rule_expert = self._rule_based_fallback(text)
                if rule_expert:
                    return rule_expert

            return best_expert

    def _rule_based_fallback(self, text: str) -> str:
        """V8原有规则兜底，完全不变"""
        text_lower = text.lower()
        if any(k in text_lower for k in ["图片","照片","图像","颜色","长什么样"]):
            return "视觉"
        if any(k in text_lower for k in ["我是谁","你是谁","名字","介绍你自己"]):
            return "身份"
        return None

    def _online_finetune_prototype(self, clip_vec, expert_name):
        """V8原有原型微调，完全不变"""
        if expert_name not in self.expert_prototypes:
            return
        alpha = 0.01
        self.expert_prototypes[expert_name] = (1 - alpha) * self.expert_prototypes[expert_name] + alpha * clip_vec.detach().cpu()
        self.expert_prototypes[expert_name] = F.normalize(self.expert_prototypes[expert_name], p=2, dim=-1)

    def online_learn(self, clip_vec, expert_name):
        """V8原有在线学习，完全不变"""
        if expert_name not in self.expert_names:
            return
        alpha = 0.1
        self.expert_prototypes[expert_name] = (1 - alpha) * self.expert_prototypes[expert_name] + alpha * clip_vec.detach().cpu()
        self.expert_prototypes[expert_name] = F.normalize(self.expert_prototypes[expert_name], p=2, dim=-1)
        
        expert_idx = self.expert_names.index(expert_name)
        self.training_buffer.append((clip_vec.detach().cpu(), expert_idx))
        
        if len(self.training_buffer) >= 16:
            self._train_step_balanced()
        self.correct_count += 1

    def _train_step_balanced(self):
        """V8原有平衡训练，完全不变"""
        if len(self.training_buffer) < 16:
            return
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
        logger.debug(f"🧭 训练步 | 损失: {loss.item():.4f}")

    def train(self, training_data, epochs=20, batch_size=16, log_interval=5):
        """V8原有预训练，完全不变"""
        if not training_data:
            return
        for text, clip_vec, correct_expert in training_data:
            if correct_expert not in self.expert_names:
                continue
            expert_idx = self.expert_names.index(correct_expert)
            self.training_buffer.append((clip_vec.detach().cpu(), expert_idx))
        
        logger.info(f"🧭 开始预训练 | 数据量: {len(training_data)} | Buffer: {len(self.training_buffer)}")
        
        for epoch in range(epochs):
            if len(self.training_buffer) < batch_size:
                continue
            num_batches = max(1, len(self.training_buffer) // batch_size)
            for _ in range(num_batches):
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
            
            if (epoch + 1) % log_interval == 0:
                correct = 0
                for text, clip_vec, correct_expert in training_data:
                    pred = self.route(clip_vec, text)
                    if pred == correct_expert:
                        correct += 1
                acc = correct / len(training_data)
                logger.info(f"   Epoch {epoch+1}/{epochs} | 损失: {loss.item():.4f} | 准确率: {acc:.2%}")

    # ====================== 🔥 新增：海马体核心仿生方法 ======================
    def encode(self, clip_vec: torch.Tensor, sdr: torch.Tensor, content: str, metadata: dict, expert: str) -> int:
        """
        海马体快速编码：新记忆优先存入临时缓冲区
        """
        # 模式分离：避免相似记忆混淆
        separated_sdr = self._pattern_separation(clip_vec, sdr)
        mem_id = self.next_mem_id
        self.next_mem_id += 1

        # 构建记忆单元
        memory = {
            "mem_id": mem_id,
            "clip_vec": clip_vec,
            "sdr": separated_sdr,
            "content": content,
            "metadata": metadata,
            "expert": expert,
            "created_at": time.time(),
            "consolidation_level": 0.0,
            "replay_count": 0
        }

        self.hippocampal_buffer.append(memory)
        logger.info(f"🧠 海马体快速编码 | ID:{mem_id} | 专家:{expert} | 内容:{content[:30]}...")
        return mem_id

    def _pattern_separation(self, clip_vec: torch.Tensor, sdr: torch.Tensor) -> torch.Tensor:
        """
        模式分离：相似输入生成不同表征，防止记忆干扰
        """
        for mem in self.hippocampal_buffer:
            sim = F.cosine_similarity(clip_vec, mem["clip_vec"], dim=-1).item()
            if sim > self.separation_threshold:
                flip_mask = torch.rand_like(sdr) < 0.1
                sdr[flip_mask] = 1 - sdr[flip_mask]
                logger.debug(f"🧠 触发模式分离 | 相似度:{sim:.2f}")
                break
        return F.normalize(sdr, p=2, dim=-1)

    def _pattern_completion(self, query_vec: torch.Tensor, threshold: float = 0.7) -> dict:
        """
        模式补全：部分输入恢复完整记忆
        """
        best_sim = 0.0
        best_mem = None
        for mem in self.hippocampal_buffer:
            sim = F.cosine_similarity(query_vec, mem["clip_vec"], dim=-1).item()
            if sim > best_sim and sim > threshold:
                best_sim = sim
                best_mem = mem
        return best_mem

    def consolidate(self, mem: dict, cortex) -> bool:
        """
        单条记忆巩固：回放后转移到皮层长期存储
        """
        if mem["consolidation_level"] >= 1.0:
            return True
        
        mem["replay_count"] += 1
        mem["consolidation_level"] += self.consolidation_rate
        logger.debug(f"🧠 记忆回放 | ID:{mem['mem_id']} | 回放次数:{mem['replay_count']} | 进度:{mem['consolidation_level']:.2f}")

        # 巩固完成：写入皮层
        if mem["consolidation_level"] >= 1.0:
            cortex.store_detailed_memory(
                expert_name=mem["expert"],
                sdr=mem["sdr"],
                clip_vec=mem["clip_vec"],
                content=mem["content"],
                metadata=mem["metadata"]
            )
            self.cortex_index_map[mem["mem_id"]] = mem["expert"]
            logger.info(f"✅ 记忆巩固完成 | ID:{mem['mem_id']} | 已存入皮层")
            return True
        return False

    def consolidate_all(self, cortex) -> int:
        """
        睡眠全量巩固：回放所有临时记忆，转移到皮层
        """
        logger.info("\n🌙 海马体睡眠巩固（记忆回放）...")
        consolidated = 0
        # 倒序回放：新记忆优先巩固
        for mem in reversed(self.hippocampal_buffer):
            if self.consolidate(mem, cortex):
                consolidated += 1
        
        # 移除已巩固记忆
        self.hippocampal_buffer = deque(
            [m for m in self.hippocampal_buffer if m["consolidation_level"] < 1.0],
            maxlen=self.buffer_size
        )
        logger.info(f"✅ 海马体巩固完成 | 共巩固{consolidated}条 | 缓冲区剩余{len(self.hippocampal_buffer)}条")
        return consolidated

    def get_memory(self, mem_id: int, cortex) -> dict:
        """
        索引式记忆提取：海马体查索引，皮层取内容
        """
        # 先查临时缓冲区
        for mem in self.hippocampal_buffer:
            if mem["mem_id"] == mem_id:
                return mem
        
        # 再查皮层索引
        if mem_id in self.cortex_index_map:
            return cortex.index.get_memory(mem_id)
        return None

    # ====================== 重写：保存/加载（兼容原有+新增状态） ======================
    def save(self, path):
        """保存：兼容原有路由 + 海马体状态"""
        save_data = {
            # V8原有数据
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'expert_prototypes': self.expert_prototypes,
            'expert_names': self.expert_names,
            '_prototypes_initialized': self._prototypes_initialized,
            'confidence_threshold': self.confidence_threshold,
            'correct_count': self.correct_count,
            'total_count': self.total_count,
            # 🔥 海马体新增数据
            'next_mem_id': self.next_mem_id,
            'cortex_index_map': self.cortex_index_map,
            'hippocampal_buffer': list(self.hippocampal_buffer)
        }
        torch.save(save_data, path)
        logger.info("💾 海马体路由+记忆状态已保存")

    def load(self, path):
        """加载：兼容原有路由 + 海马体状态"""
        if not os.path.exists(path):
            return
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            # 加载V8原有数据
            self.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.expert_prototypes = checkpoint.get('expert_prototypes', self.expert_prototypes)
            self.expert_names = checkpoint.get('expert_names', self.expert_names)
            self._prototypes_initialized = checkpoint.get('_prototypes_initialized', False)
            self.confidence_threshold = checkpoint.get('confidence_threshold', 0.15)
            self.correct_count = checkpoint.get('correct_count', 0)
            self.total_count = checkpoint.get('total_count', 0)
            
            # 加载🔥海马体数据
            self.next_mem_id = checkpoint.get('next_mem_id', 1)
            self.cortex_index_map = checkpoint.get('cortex_index_map', {})
            self.hippocampal_buffer = deque(checkpoint.get('hippocampal_buffer', []), maxlen=self.buffer_size)
            
            logger.info(f"✅ 海马体加载完成 | 临时记忆:{len(self.hippocampal_buffer)} | 皮层索引:{len(self.cortex_index_map)}")
        except Exception as e:
            logger.error(f"❌ 海马体加载失败: {e}")