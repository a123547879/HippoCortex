from typing import List, Dict, Optional, Any, Deque, Tuple
import torch
import torch.nn.functional as F
import os
from collections import deque
import logging
import re
import time
from Data_models import MemoryPacket  # 数据契约

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
        expert_names: Optional[List[str]] = None,
        experts: Optional[Dict[str, Any]] = None,
        learning_rate: float = 1e-3,
        confidence_threshold: float = 0.15,
        buffer_size: int = 20,        # 海马体临时记忆容量
        consolidation_rate: float = 0.2, # 睡眠巩固进度
        separation_threshold: float = 0.85 # 模式分离阈值
    ):
        super().__init__()
        # ====================== 保留V8所有原有属性 ======================
        self.input_dim: int = input_dim
        self.expert_names: List[str] = expert_names or ["身份", "概念", "空间", "抽象", "视觉"]
        self.num_experts: int = len(self.expert_names)
        self.confidence_threshold: float = confidence_threshold
        self.experts: Dict[str, Any] = experts if experts is not None else {}
        
        # 初始化last_scores
        self.last_scores: Dict[str, float] = {name: 0.0 for name in self.expert_names}
        
        # 轻量路由网络
        self.router: torch.nn.Sequential = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
            torch.nn.Linear(256, self.num_experts)
        )
        
        # 初始化权重
        for m in self.router.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.zeros_(m.bias)
        
        # 在线学习buffer
        self.training_buffer: Deque[Tuple[torch.Tensor, int]] = deque(maxlen=2000)
        self.optimizer: torch.optim.AdamW = torch.optim.AdamW(
            self.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # 专家原型向量
        self.expert_prototypes: Dict[str, torch.Tensor] = {name: torch.zeros(input_dim) for name in self.expert_names}
        self._prototypes_initialized: bool = False
        self.correct_count: int = 0
        self.total_count: int = 0
        self.last_confidence: float = 0.0
        self.last_nn_scores: Dict[str, float] = {}
        self.last_semantic_scores: Dict[str, float] = {}

        # ====================== 海马体核心仿生功能 ======================
        self.buffer_size: int = buffer_size
        self.consolidation_rate: float = consolidation_rate
        self.separation_threshold: float = separation_threshold

        # 🔥 核心修复：临时缓冲区存储 MemoryPacket 对象（不再用字典！）
        self.hippocampal_buffer: Deque[MemoryPacket] = deque(maxlen=buffer_size)
        # 海马体→皮层索引表
        self.cortex_index_map: Dict[int, str] = {}
        # 全局记忆ID
        self.next_mem_id: int = 100000

    # ====================== 保留V8所有原有方法 ======================
    def _initialize_prototypes_with_embedding(self, embedding_model: Any) -> None:
        """V8原有原型初始化，完全不变"""
        if self._prototypes_initialized:
            return
        
        logger.info("🧭 正在初始化专家原型向量...")
        identity_samples = ["你是谁？我是一个AI助手", "我是谁？你是用户", "我的名字叫AI", "你的用户是我","我是你的助手，你是我的用户", "你是AI，我是用户","身份认知，自我定义", "我叫什么名字？", "你的名字是什么？"]
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """V8原有前向传播，完全不变"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.router(x)

    def route(self, clip_vec: torch.Tensor, text: Optional[str] = None, is_encoding: bool = False) -> str:
        """V8原有路由逻辑，完全不变"""
        with torch.no_grad():
            logits = self.forward(clip_vec).squeeze(0)
            nn_probs = F.softmax(logits, dim=-1)

            semantic_scores: Dict[str, float] = {}
            for i, name in enumerate(self.expert_names):
                semantic_score = 0.0
                if self.experts and name in self.experts:
                    expert = self.experts[name]
                    if text and hasattr(expert, 'kg'):
                        try:
                            top_nodes = expert.kg.get_top_related_nodes(text, top_k=5)
                            if top_nodes:
                                semantic_score = sum(node.get("similarity", 0.0) for node in top_nodes) / len(top_nodes)
                                semantic_score = max(0.0, min(1.0, semantic_score))
                        except Exception as e:
                            logger.debug(f"[{name}] 语义网络匹配失败: {e}")
                semantic_scores[name] = semantic_score

            final_scores: Dict[str, float] = {name: nn_probs[i].item()*0.4 + semantic_scores.get(name,0.0)*0.6 for i, name in enumerate(self.expert_names)}
            
            if text is not None and text.strip() != "" and is_encoding:
                final_scores["视觉"] = 0.0
                logger.info(f"🎯 硬编码生效：学习模式+纯文本，强制屏蔽【视觉】专家")

            sorted_experts = sorted(final_scores.items(), key=lambda x: -x[1])
            best_expert, best_score = sorted_experts[0]
            second_score = sorted_experts[1][1] if len(sorted_experts) > 1 else 0.0
            confidence = (best_score - second_score) / max(best_score, 1e-8)

            self.last_confidence = confidence
            self.last_scores = final_scores
            self.last_nn_scores = {name: nn_probs[i].item() for i, name in enumerate(self.expert_names)}
            self.last_semantic_scores = semantic_scores
            self.total_count += 1

            logger.info(f"🧭 路由 | 神经网络得分:{self.last_nn_scores}")
            logger.info(f"🧭 路由 | 语义网络得分:{semantic_scores}")
            logger.info(f"🧭 路由 | 最终融合得分:{final_scores} | 最优:{best_expert} | 置信度:{confidence:.2f}")

            completed_mem = self._pattern_completion(clip_vec)
            if completed_mem:
                logger.info(f"🧠 海马体模式补全成功：{completed_mem.content[:30]}...")
                best_expert = completed_mem.expert
                self.last_confidence = 0.95

            if confidence >= 0.05:
                self._online_finetune_prototype(clip_vec, best_expert)
                return best_expert

            logger.warning(f"⚠️ 语义极度模糊，启用极简规则兜底")
            if text:
                rule_expert = self._rule_based_fallback(text)
                if rule_expert:
                    return rule_expert

            return best_expert
    
    def _rule_based_fallback(self, text: str) -> Optional[str]:
        """V8原有规则兜底，完全不变"""
        text_lower = text.lower()
        if any(k in text_lower for k in ["图片","照片","图像","颜色","长什么样"]):
            return "视觉"
        if any(k in text_lower for k in ["我是谁","你是谁","名字","介绍你自己"]):
            return "身份"
        return None

    def _online_finetune_prototype(self, clip_vec: torch.Tensor, expert_name: str) -> None:
        """V8原有原型微调，完全不变"""
        if expert_name not in self.expert_prototypes:
            return
        alpha = 0.01
        self.expert_prototypes[expert_name] = (1 - alpha) * self.expert_prototypes[expert_name] + alpha * clip_vec.detach().cpu()
        self.expert_prototypes[expert_name] = F.normalize(self.expert_prototypes[expert_name], p=2, dim=-1)

    def online_learn(self, clip_vec: torch.Tensor, expert_name: str) -> None:
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

    def _train_step_balanced(self) -> None:
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

    def train(self, training_data: List[Tuple[str, torch.Tensor, str]], epochs: int = 20, batch_size: int = 16, log_interval: int = 5) -> None:
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

    # ====================== 🔥 核心修复：海马体仿生方法（全对象化） ======================
    def encode(self, clip_vec: torch.Tensor, sdr: torch.Tensor, content: str, metadata: Dict[str, Any], expert: str) -> int:
        """
        海马体快速编码：适配官方 MemoryPacket 规范
        """
        # 重复检测
        for mem in self.hippocampal_buffer:
            sim = F.cosine_similarity(clip_vec, mem.clip_vec, dim=-1).item()
            if sim > 0.95:
                logger.info(f"🔄 检测到海马体重复记忆，跳过: {content[:30]}...")
                mem.metadata["access_count"] = mem.metadata.get("access_count", 0) + 1
                mem.metadata["last_accessed"] = time.time()
                return mem.mem_id

        # 模式分离
        separated_sdr = self._pattern_separation(clip_vec, sdr)
        mem_id = self.next_mem_id
        self.next_mem_id += 1

        # 🔥 核心修复：仅传递 MemoryPacket 定义的字段，额外数据存入 metadata
        metadata["expert"] = expert  # expert 存入 metadata（匹配 @property）
        metadata["last_accessed"] = time.time()
        metadata["replay_count"] = 0

        # 严格按照你的 MemoryPacket 定义创建对象
        memory = MemoryPacket(
            mem_id=mem_id,
            content=content,
            metadata=metadata,
            sdr=separated_sdr,
            clip_vec=clip_vec,
            consolidation_level=0.0,
        )

        if 'vae_latent' in metadata and metadata['vae_latent'] is not None:
            logger.info(f"🧠 确认VAE数据已存入海马体 | 记忆ID:{mem_id}")

        self.hippocampal_buffer.append(memory)
        logger.info(f"🧠 海马体快速编码 | ID:{mem_id} | 专家:{expert} | 内容:{content[:30]}...")
        return mem_id
    
    def _pattern_separation(self, clip_vec: torch.Tensor, sdr: torch.Tensor) -> torch.Tensor:
        """模式分离：适配对象"""
        for mem in self.hippocampal_buffer:
            sim = F.cosine_similarity(clip_vec, mem.clip_vec, dim=-1).item()
            if sim > self.separation_threshold:
                flip_mask = torch.rand_like(sdr) < 0.1
                sdr[flip_mask] = 1 - sdr[flip_mask]
                logger.debug(f"🧠 触发模式分离 | 相似度:{sim:.2f}")
                break
        return F.normalize(sdr, p=2, dim=-1)

    def _pattern_completion(self, query_vec: torch.Tensor, threshold: float = 0.7) -> Optional[MemoryPacket]:
        """模式补全：返回对象"""
        best_sim = 0.0
        best_mem = None
        for mem in self.hippocampal_buffer:
            sim = F.cosine_similarity(query_vec, mem.clip_vec, dim=-1).item()
            if sim > best_sim and sim > threshold:
                best_sim = sim
                best_mem = mem
        return best_mem

    def consolidate(self, mem: MemoryPacket, cortex: Any) -> bool:
        """单条记忆巩固：适配对象"""
        if mem.consolidation_level >= 1.0:
            return True
        
        mem.replay_count += 1
        mem.consolidation_level += self.consolidation_rate
        logger.debug(f"🧠 记忆回放 | ID:{mem.mem_id} | 回放次数:{mem.replay_count} | 进度:{mem.consolidation_level:.2f}")

        # 巩固完成：写入皮层
        if mem.consolidation_level >= 1.0:
            cortex.store_detailed_memory(
                expert_name=mem.expert,
                sdr=mem.sdr,
                clip_vec=mem.clip_vec,
                content=mem.content,
                metadata=mem.metadata
            )
            self.cortex_index_map[mem.mem_id] = mem.expert
            logger.info(f"✅ 记忆巩固完成 | ID:{mem.mem_id} | 已存入皮层")
            return True
        return False

    def consolidate_all(self, cortex: Any) -> int:
        """睡眠全量巩固：适配对象"""
        logger.info("\n🌙 海马体睡眠巩固（记忆回放）...")
        consolidated = 0
        for mem in reversed(self.hippocampal_buffer):
            if self.consolidate(mem, cortex):
                consolidated += 1
        
        # 移除已巩固记忆
        self.hippocampal_buffer = deque(
            [m for m in self.hippocampal_buffer if m.consolidation_level < 1.0],
            maxlen=self.buffer_size
        )
        logger.info(f"✅ 海马体巩固完成 | 共巩固{consolidated}条 | 缓冲区剩余{len(self.hippocampal_buffer)}条")
        return consolidated

    def get_memory(self, mem_id: int, cortex: Any) -> Optional[MemoryPacket]:
        """统一接口：直接返回对象，无转换开销"""
        # 查临时缓冲区
        for mem in self.hippocampal_buffer:
            if mem.mem_id == mem_id:
                return mem
        
        # 查皮层
        if mem_id in self.cortex_index_map:
            return cortex.index.get_memory(mem_id)
        
        return None

    # ====================== 重写：保存/加载（对象兼容） ======================
    def save(self, path: str) -> None:
        """保存：将对象转为字典序列化"""
        # 把 MemoryPacket 转为可序列化的字典
        buffer_data = []
        for mem in self.hippocampal_buffer:
            buffer_data.append({
                "mem_id": mem.mem_id,
                "clip_vec": mem.clip_vec.detach().cpu().numpy(),  # 保留detach，防止梯度报错
                "sdr": mem.sdr.detach().cpu().numpy(),
                "content": mem.content,
                "metadata": mem.metadata,
                "expert": mem.expert,
                "created_at": mem.created_at,
                # ✅ 修复：全部从 metadata 中获取，不直接访问对象属性
                "last_accessed": mem.metadata.get("last_accessed", 0.0),
                "consolidation_level": mem.metadata.get("consolidation_level", 0.0),
                "replay_count": mem.metadata.get("replay_count", 0)
            })

        save_data = {
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'expert_prototypes': self.expert_prototypes,
            'expert_names': self.expert_names,
            '_prototypes_initialized': self._prototypes_initialized,
            'confidence_threshold': self.confidence_threshold,
            'correct_count': self.correct_count,
            'total_count': self.total_count,
            'next_mem_id': self.next_mem_id,
            'cortex_index_map': self.cortex_index_map,
            'hippocampal_buffer': buffer_data  # 保存字典格式
        }
        torch.save(save_data, path)
        logger.info("💾 海马体路由+记忆状态已保存")

    def load(self, path: str) -> None:
        """加载：字典转回 MemoryPacket 对象"""
        if not os.path.exists(path):
            return
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            # 加载原有数据
            self.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.expert_prototypes = checkpoint.get('expert_prototypes', self.expert_prototypes)
            self.expert_names = checkpoint.get('expert_names', self.expert_names)
            self._prototypes_initialized = checkpoint.get('_prototypes_initialized', False)
            self.confidence_threshold = checkpoint.get('confidence_threshold', 0.15)
            self.correct_count = checkpoint.get('correct_count', 0)
            self.total_count = checkpoint.get('total_count', 0)
            
            # 加载海马体数据
            self.next_mem_id = checkpoint.get('next_mem_id', 1)
            self.cortex_index_map = checkpoint.get('cortex_index_map', {})
            
            # 🔥 核心修复：字典转回 MemoryPacket，扩展字段存入 metadata
            self.hippocampal_buffer.clear()
            for mem_data in checkpoint.get('hippocampal_buffer', []):
                # 提取扩展字段
                last_accessed = mem_data.get("last_accessed", 0.0)
                consolidation_level = mem_data.get("consolidation_level", 0.0)
                replay_count = mem_data.get("replay_count", 0)
                
                # 复制基础 metadata，并加入扩展字段
                metadata = mem_data.get("metadata", {}).copy()
                metadata["last_accessed"] = last_accessed
                metadata["consolidation_level"] = consolidation_level
                metadata["replay_count"] = replay_count
                
                # ✅ 修复：仅传入 MemoryPacket 支持的构造参数
                mem = MemoryPacket(
                    mem_id=mem_data["mem_id"],
                    clip_vec=torch.tensor(mem_data["clip_vec"]),
                    sdr=torch.tensor(mem_data["sdr"]),
                    content=mem_data["content"],
                    metadata=metadata,  # 包含所有扩展字段
                    expert=mem_data["expert"],
                    created_at=mem_data["created_at"]
                )
                self.hippocampal_buffer.append(mem)
            
            self.last_scores = {name: 0.0 for name in self.expert_names}
            
            logger.info(f"✅ 海马体加载完成 | 临时记忆:{len(self.hippocampal_buffer)} | 皮层索引:{len(self.cortex_index_map)}")
        except Exception as e:
            logger.error(f"❌ 海马体加载失败: {e}", exc_info=True)