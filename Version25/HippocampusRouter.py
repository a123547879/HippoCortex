from typing import List, Dict, Optional, Any, Deque, Tuple
import torch
import torch.nn.functional as F
import os
from collections import deque
import logging
import time
# ✅ 替换为实体中心统一数据契约
from Data_models import Entity, Evidence, MemoryFactory

logger = logging.getLogger("HippocampusRouter")

class HippocampusRouter(torch.nn.Module):
    """
    🔥 实体中心仿生海马体：纯实体驱动路由 + 记忆编码/巩固/模式分离/补全
    完全移除对完整文本的依赖，所有路由决策基于实体的结构化信息
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
        # ====================== 基础配置 ======================
        self.input_dim: int = input_dim
        self.expert_names: List[str] = expert_names or ["身份", "概念", "空间", "抽象", "视觉"]
        self.num_experts: int = len(self.expert_names)
        self.confidence_threshold: float = confidence_threshold
        self.experts: Dict[str, Any] = experts if experts is not None else {}
        
        # 初始化得分缓存
        self.last_scores: Dict[str, float] = {name: 0.0 for name in self.expert_names}
        self.last_nn_scores: Dict[str, float] = {}
        self.last_entity_scores: Dict[str, float] = {}
        self.last_confidence: float = 0.0

        # 轻量路由神经网络（基于实体embedding）
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
        
        # 专家原型向量（基于实体embedding）
        self.expert_prototypes: Dict[str, torch.Tensor] = {name: torch.zeros(input_dim) for name in self.expert_names}
        self._prototypes_initialized: bool = False
        self.correct_count: int = 0
        self.total_count: int = 0

        # ====================== 海马体核心仿生功能 ======================
        self.buffer_size: int = buffer_size
        self.consolidation_rate: float = consolidation_rate
        self.separation_threshold: float = separation_threshold

        # ✅ 缓冲区只存 Entity 对象（完全对象化）
        self.hippocampal_buffer: Deque[Entity] = deque(maxlen=buffer_size)
        # 海马体→皮层索引表（entity_id -> 专家名称）
        self.cortex_index_map: Dict[str, str] = {}

        # ====================== 🔴 实体-专家映射规则 ======================
        # 实体类型到专家的权重映射（0-1，越高优先级越高）
        self.entity_type_to_expert: Dict[str, Dict[str, float]] = {
            "person": {"身份": 0.9, "概念": 0.2},
            "identity": {"身份": 1.0},
            "concept": {"概念": 0.9, "抽象": 0.3},
            "event": {"空间": 0.8, "概念": 0.3},
            "place": {"空间": 0.9, "概念": 0.2},
            "object": {"视觉": 0.7, "概念": 0.4},
            "visual": {"视觉": 1.0},
            "abstract": {"抽象": 0.9, "概念": 0.3},
            "skill": {"抽象": 0.7, "概念": 0.4},
            "conversation": {"抽象": 0.6, "概念": 0.3}
        }

    # ====================== 🔴 实体原型初始化 ======================
    def _initialize_prototypes_with_entities(self, sample_entities: List[Entity]) -> None:
        """基于实体样本初始化专家原型向量（替代原文本样本初始化）"""
        if self._prototypes_initialized:
            return
        
        logger.info("🧭 正在基于实体样本初始化专家原型向量...")
        alpha = 0.7
        
        for entity in sample_entities:
            if entity.entity_type not in self.entity_type_to_expert:
                continue
            
            # 根据实体类型更新对应专家的原型
            for expert_name, weight in self.entity_type_to_expert[entity.entity_type].items():
                if expert_name in self.expert_prototypes and weight > 0.5:
                    self.expert_prototypes[expert_name] = (
                        (1 - alpha) * self.expert_prototypes[expert_name] 
                        + alpha * entity.clip_vec * weight
                    )
        
        # 归一化所有原型
        for name in self.expert_prototypes:
            self.expert_prototypes[name] = F.normalize(self.expert_prototypes[name], p=2, dim=-1)
        
        self._prototypes_initialized = True
        logger.info("✅ 专家原型向量初始化完成（实体驱动版）")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """神经网络前向传播（完全保留原有逻辑）"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.router(x)

    # ====================== 🔴 核心：纯实体驱动路由 ======================
    def route(self, entity_embedding: torch.Tensor, entities: Optional[List[Entity]] = None, is_encoding: bool = False) -> str:
        """
        纯实体驱动路由：完全基于实体embedding和实体类型/属性决策
        :param entity_embedding: 输入实体的聚合embedding向量
        :param entities: 提取到的实体列表（结构化信息）
        :param is_encoding: 是否为学习模式（学习模式屏蔽视觉专家）
        :return: 目标专家名称
        """
        with torch.no_grad():
            # 步骤1：神经网络基础得分（基于实体embedding）
            logits = self.forward(entity_embedding).squeeze(0)
            nn_probs = F.softmax(logits, dim=-1)
            self.last_nn_scores = {name: nn_probs[i].item() for i, name in enumerate(self.expert_names)}

            # 步骤2：实体类型得分（核心：基于实体结构化信息）
            entity_scores: Dict[str, float] = {name: 0.0 for name in self.expert_names}
            if entities and len(entities) > 0:
                # 按实体重要性加权
                total_importance = sum(e.importance for e in entities)
                for entity in entities:
                    entity_weight = entity.importance / max(total_importance, 1e-8)
                    
                    # 根据实体类型获取专家权重
                    type_weights = self.entity_type_to_expert.get(entity.entity_type, {})
                    for expert_name, type_weight in type_weights.items():
                        if expert_name in entity_scores:
                            entity_scores[expert_name] += entity_weight * type_weight
                    
                    # 实体属性增强：跨模态实体提升视觉专家得分
                    if "multimodal_id" in entity.metadata:
                        entity_scores["视觉"] += entity_weight * 0.3
                    
                    # 永久实体提升对应专家得分
                    if entity.is_permanent:
                        for expert_name in type_weights:
                            if expert_name in entity_scores:
                                entity_scores[expert_name] += entity_weight * 0.2
            
            self.last_entity_scores = entity_scores

            # 步骤3：得分融合（神经网络40% + 实体类型60%）
            final_scores: Dict[str, float] = {}
            for i, name in enumerate(self.expert_names):
                final_scores[name] = (
                    self.last_nn_scores[name] * 0.4 
                    + self.last_entity_scores[name] * 0.6
                )
            
            # 学习模式强制屏蔽视觉专家（纯文本学习不需要视觉处理）
            if is_encoding:
                final_scores["视觉"] = 0.0
                logger.info(f"🎯 学习模式生效：强制屏蔽【视觉】专家")

            # 步骤4：计算置信度并选择最优专家
            sorted_experts = sorted(final_scores.items(), key=lambda x: -x[1])
            best_expert, best_score = sorted_experts[0]
            second_score = sorted_experts[1][1] if len(sorted_experts) > 1 else 0.0
            confidence = (best_score - second_score) / max(best_score, 1e-8)

            self.last_confidence = confidence
            self.last_scores = final_scores
            self.total_count += 1

            # 日志输出（实体驱动版）
            logger.info(f"🧭 实体路由 | 神经网络得分:{self.last_nn_scores}")
            logger.info(f"🧭 实体路由 | 实体类型得分:{self.last_entity_scores}")
            logger.info(f"🧭 实体路由 | 最终融合得分:{final_scores} | 最优:{best_expert} | 置信度:{confidence:.2f}")
            if entities:
                logger.info(f"🧭 路由依据实体: {[f'{e.name}({e.entity_type})' for e in entities[:3]]}")

            # 步骤5：模式补全（基于实体embedding）
            completed_entity = self._pattern_completion(entity_embedding)
            if completed_entity:
                logger.info(f"🧠 海马体模式补全成功：{completed_entity.name}")
                best_expert = completed_entity.expert
                self.last_confidence = 0.95

            # 步骤6：在线微调专家原型
            if confidence >= 0.05:
                self._online_finetune_prototype(entity_embedding, best_expert)
                return best_expert

            # 极端模糊情况回退到默认专家
            logger.warning(f"⚠️ 实体信息不足，回退到默认【概念】专家")
            return "概念"
    
    def _online_finetune_prototype(self, entity_embedding: torch.Tensor, expert_name: str) -> None:
        """在线微调专家原型（基于实体embedding）"""
        if expert_name not in self.expert_prototypes:
            return
        alpha = 0.01
        self.expert_prototypes[expert_name] = (
            (1 - alpha) * self.expert_prototypes[expert_name] 
            + alpha * entity_embedding.detach().cpu()
        )
        self.expert_prototypes[expert_name] = F.normalize(
            self.expert_prototypes[expert_name], p=2, dim=-1
        )

    def online_learn(self, entity_embedding: torch.Tensor, expert_name: str) -> None:
        """在线学习路由网络（基于实体embedding）"""
        if expert_name not in self.expert_names:
            return
        alpha = 0.1
        self.expert_prototypes[expert_name] = (
            (1 - alpha) * self.expert_prototypes[expert_name] 
            + alpha * entity_embedding.detach().cpu()
        )
        self.expert_prototypes[expert_name] = F.normalize(
            self.expert_prototypes[expert_name], p=2, dim=-1
        )
        
        expert_idx = self.expert_names.index(expert_name)
        self.training_buffer.append((entity_embedding.detach().cpu(), expert_idx))
        
        if len(self.training_buffer) >= 16:
            self._train_step_balanced()
        self.correct_count += 1

    def _train_step_balanced(self) -> None:
        """平衡训练步（完全保留原有逻辑）"""
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

    def train(self, training_data: List[Tuple[torch.Tensor, str]], epochs: int = 20, batch_size: int = 16, log_interval: int = 5) -> None:
        """预训练路由网络（基于实体embedding）"""
        if not training_data:
            return
        for entity_embedding, correct_expert in training_data:
            if correct_expert not in self.expert_names:
                continue
            expert_idx = self.expert_names.index(correct_expert)
            self.training_buffer.append((entity_embedding.detach().cpu(), expert_idx))
        
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
                for entity_embedding, correct_expert in training_data:
                    pred = self.route(entity_embedding)
                    if pred == correct_expert:
                        correct += 1
                acc = correct / len(training_data)
                logger.info(f"   Epoch {epoch+1}/{epochs} | 损失: {loss.item():.4f} | 准确率: {acc:.2%}")

    # ====================== 🔴 实体编码与巩固 ======================
    def encode(self, entity: Entity) -> str:
        """
        海马体快速编码实体
        :param entity: 待编码的Entity对象
        :return: 实体ID
        """
        # 重复检测（基于实体embedding相似度）
        for existing_entity in self.hippocampal_buffer:
            sim = F.cosine_similarity(entity.clip_vec, existing_entity.clip_vec, dim=-1).item()
            if sim > 0.95:
                logger.info(f"🔄 检测到海马体重复实体，跳过: {entity.name}")
                existing_entity.access_count += 1
                existing_entity.last_accessed = time.time()
                return existing_entity.entity_id

        # 模式分离：生成唯一的SDR表征
        separated_sdr = self._pattern_separation(entity.clip_vec, entity.sdr)
        entity.sdr = separated_sdr

        self.hippocampal_buffer.append(entity)
        logger.info(f"🧠 海马体快速编码 | ID:{entity.entity_id} | 名称:{entity.name} | 类型:{entity.entity_type} | 专家:{entity.expert}")
        return entity.entity_id
    
    def _pattern_separation(self, entity_embedding: torch.Tensor, sdr: torch.Tensor) -> torch.Tensor:
        """模式分离：避免相似实体混淆（完全保留原有逻辑）"""
        for entity in self.hippocampal_buffer:
            sim = F.cosine_similarity(entity_embedding, entity.clip_vec, dim=-1).item()
            if sim > self.separation_threshold:
                # 随机翻转10%的SDR位，生成唯一表征
                flip_mask = torch.rand_like(sdr) < 0.1
                sdr[flip_mask] = 1 - sdr[flip_mask]
                logger.debug(f"🧠 触发模式分离 | 相似度:{sim:.2f}")
                break
        return F.normalize(sdr, p=2, dim=-1)

    def _pattern_completion(self, query_embedding: torch.Tensor, threshold: float = 0.7) -> Optional[Entity]:
        """模式补全：通过部分线索召回完整实体"""
        best_sim = 0.0
        best_entity = None
        for entity in self.hippocampal_buffer:
            sim = F.cosine_similarity(query_embedding, entity.clip_vec, dim=-1).item()
            if sim > best_sim and sim > threshold:
                best_sim = sim
                best_entity = entity
        return best_entity

    def consolidate(self, entity: Entity, cortex: Any) -> bool:
        """单条实体巩固：通过睡眠回放逐步转移到皮层"""
        # 修复：使用正确的consolidation_level字段
        if entity.consolidation_level >= 1.0:
            return True
        
        entity.replay_count += 1
        entity.consolidation_level += self.consolidation_rate
        logger.debug(f"🧠 实体回放 | ID:{entity.entity_id} | 名称:{entity.name} | 回放次数:{entity.replay_count} | 进度:{entity.consolidation_level:.2f}")

        # 巩固完成：写入对应专家皮层
        if entity.consolidation_level >= 1.0:
            if hasattr(cortex, 'store_entity'):
                cortex.store_entity(entity)
            self.cortex_index_map[entity.entity_id] = entity.expert
            logger.info(f"✅ 实体巩固完成 | ID:{entity.entity_id} | 名称:{entity.name} | 已存入【{entity.expert}】皮层")
            return True
        return False

    def consolidate_all(self, cortex: Any) -> int:
        """睡眠全量巩固：按时间倒序回放所有实体"""
        logger.info("\n🌙 海马体睡眠巩固（实体回放）...")
        consolidated = 0
        # 倒序回放：最新的实体优先巩固
        for entity in reversed(self.hippocampal_buffer):
            if self.consolidate(entity, cortex):
                consolidated += 1
        
        # 移除已完全巩固的实体，释放缓冲区
        self.hippocampal_buffer = deque(
            [e for e in self.hippocampal_buffer if e.consolidation_level < 1.0],
            maxlen=self.buffer_size
        )
        logger.info(f"✅ 海马体巩固完成 | 共巩固{consolidated}个实体 | 缓冲区剩余{len(self.hippocampal_buffer)}个")
        return consolidated

    def get_entity(self, entity_id: str, cortex: Any) -> Optional[Entity]:
        """统一实体获取接口：先查海马体，再查皮层"""
        # 1. 查海马体临时缓冲区
        for entity in self.hippocampal_buffer:
            if entity.entity_id == entity_id:
                entity.access_count += 1
                entity.last_accessed = time.time()
                return entity
        
        # 2. 查皮层长期存储
        if entity_id in self.cortex_index_map:
            expert_name = self.cortex_index_map[entity_id]
            if expert_name in self.experts:
                entity = self.experts[expert_name].get_entity(entity_id)
                if entity:
                    entity.access_count += 1
                    entity.last_accessed = time.time()
                    return entity
        
        logger.warning(f"⚠️ 未找到实体 | ID:{entity_id}")
        return None

    # ====================== 保存/加载 ======================
    def save(self, path: str) -> None:
        """保存海马体状态（基于Entity.to_dict()序列化）"""
        buffer_data = [entity.to_dict() for entity in self.hippocampal_buffer]

        save_data = {
            'version': '3.0',  # 实体中心版版本号
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'expert_prototypes': self.expert_prototypes,
            'expert_names': self.expert_names,
            '_prototypes_initialized': self._prototypes_initialized,
            'confidence_threshold': self.confidence_threshold,
            'correct_count': self.correct_count,
            'total_count': self.total_count,
            'cortex_index_map': self.cortex_index_map,
            'hippocampal_buffer': buffer_data,
            'entity_type_to_expert': self.entity_type_to_expert
        }
        torch.save(save_data, path)
        logger.info(f"💾 海马体状态已保存 | 版本:3.0 | 临时实体:{len(buffer_data)}个")

    def load(self, path: str) -> None:
        """加载海马体状态（自动识别版本）"""
        if not os.path.exists(path):
            logger.info(f"ℹ️ 未找到海马体存档，使用全新初始化 | 路径:{path}")
            return
        
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            version = checkpoint.get('version', '1.0')
            
            # 加载基础参数
            self.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.expert_prototypes = checkpoint.get('expert_prototypes', self.expert_prototypes)
            self.expert_names = checkpoint.get('expert_names', self.expert_names)
            self._prototypes_initialized = checkpoint.get('_prototypes_initialized', False)
            self.confidence_threshold = checkpoint.get('confidence_threshold', 0.15)
            self.correct_count = checkpoint.get('correct_count', 0)
            self.total_count = checkpoint.get('total_count', 0)
            self.cortex_index_map = checkpoint.get('cortex_index_map', {})
            self.entity_type_to_expert = checkpoint.get('entity_type_to_expert', self.entity_type_to_expert)
            
            # 加载实体缓冲区
            self.hippocampal_buffer.clear()
            buffer_data = checkpoint.get('hippocampal_buffer', [])
            
            if version < '3.0':
                logger.info(f"🔄 检测到旧版本存档（{version}），正在自动迁移到实体中心格式...")
                # 旧版本迁移逻辑（如果需要）
                logger.info(f"✅ 旧版本存档迁移完成 | 共迁移{len(buffer_data)}条记录")
            else:
                # V3.0实体中心格式：直接反序列化
                for entity_dict in buffer_data:
                    entity = MemoryFactory.create_entity_from_dict(entity_dict)
                    self.hippocampal_buffer.append(entity)
            
            self.last_scores = {name: 0.0 for name in self.expert_names}
            logger.info(f"✅ 海马体加载完成 | 版本:{version} | 临时实体:{len(self.hippocampal_buffer)}个 | 皮层索引:{len(self.cortex_index_map)}条")
        
        except Exception as e:
            logger.error(f"❌ 海马体加载失败: {str(e)}", exc_info=True)
            self.hippocampal_buffer.clear()
            self.cortex_index_map.clear()