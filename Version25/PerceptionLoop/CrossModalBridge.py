import os, time
import torch
import torch.nn.functional as F
import logging
import uuid
from typing import List, Dict, Tuple, Optional, Any

from ILifecycle import ILifecycle, IService
from ServiceContainer import ServiceContainer
from event_system import EventBus, Event, EventType
from BrainConfig import config

logger = logging.getLogger("CrossModalBridge")

class CrossModalBridge(ILifecycle, IService):
    def __init__(self):
        self.use_learnable_pons: bool = True
        self.bridge_learning_rate: float = 1e-4
        self.cycle_consistency_weight: float = 0.1
        self.contrastive_loss_weight: float = 0.3
        
        self.pons: Dict[str, Any] = {}
        self.cross_modal_bridge_optimizer: Optional[torch.optim.Adam] = None
        
        self.bridge_batch_size: int = 32
        self.bridge_pair_buffer: List[Tuple[torch.Tensor, torch.Tensor, str, str]] = []
        self.max_buffer_size: int = 1000
        
        self.bridge_train_steps: int = 0
        self.bridge_loss_history: Dict[str, List[float]] = {
            "total": [], "vision": [], "text": [], "cycle_text": [], 
            "cycle_vision": [], "contrastive": []
        }
        
        self.cold_start_steps: int = 100
        self.cold_start_lr_multiplier: float = 0.1
        
        self._container = None
    
    def initialize(self, container: 'ServiceContainer') -> None:
        self._container = container
        logger.info("🧠 初始化跨模态脑桥翻译器...")
        
        self.pons = {}
        bridge_params = []
        experts = container["experts"]
        
        for expert_name in experts.keys():
            if self.use_learnable_pons:
                text_to_expert = torch.nn.Linear(2048, 2048, bias=True)
                expert_to_text = torch.nn.Linear(2048, 2048, bias=True)
                
                torch.nn.init.eye_(text_to_expert.weight)
                torch.nn.init.eye_(expert_to_text.weight)
                torch.nn.init.zeros_(text_to_expert.bias)
                torch.nn.init.zeros_(expert_to_text.bias)
                
                self.pons[expert_name] = {
                    "text_to_expert": text_to_expert,
                    "expert_to_text": expert_to_text
                }
                
                bridge_params.extend(text_to_expert.parameters())
                bridge_params.extend(expert_to_text.parameters())
            else:
                self.pons[expert_name] = torch.nn.Linear(2048, 2048, bias=False)
                torch.nn.init.eye_(self.pons[expert_name].weight)
        
        if self.use_learnable_pons and bridge_params:
            self.cross_modal_bridge_optimizer = torch.optim.Adam(
                bridge_params, lr=self.bridge_learning_rate
            )
            logger.info(f"✅ 可学习双向脑桥初始化完成，共 {len(self.pons)} 个专家翻译器")
        else:
            logger.info(f"✅ 固定脑桥初始化完成，共 {len(self.pons)} 个专家翻译器")
    
    def start(self) -> None:
        pass
    
    def stop(self) -> None:
        pass
    
    def save(self, storage_dir: str) -> None:
        self.save_bridges(storage_dir)
        self.save_bridge_dataset(os.path.join(storage_dir, "cross_modal_dataset.pt"))
    
    def load(self, storage_dir: str) -> None:
        self.load_bridges(storage_dir)
        self.load_bridge_dataset(os.path.join(storage_dir, "cross_modal_dataset.pt"))
    
    def cross_modal_learning_step(self, 
                                text_features: torch.Tensor, 
                                vision_features: torch.Tensor,
                                target_expert: str = "视觉",
                                multimodal_id: str = None) -> float:
        if not self.use_learnable_pons or target_expert not in self.pons:
            return 0.0
        
        bridge = self.pons[target_expert]
        if not isinstance(bridge, dict):
            return 0.0
        
        text_features = text_features.detach().cpu()
        vision_features = vision_features.detach().cpu()
        
        if len(self.bridge_pair_buffer) < self.max_buffer_size:
            self.bridge_pair_buffer.append((text_features, vision_features, target_expert, multimodal_id))
            logger.debug(f"🧠 添加跨模态配对 | 缓冲区: {len(self.bridge_pair_buffer)}/{self.bridge_batch_size} | 专家: {target_expert}")
    
        if len(self.bridge_pair_buffer) >= self.bridge_batch_size:
            return self._train_bridge_batch()
        
        return 0.0
    
    def _train_bridge_batch(self) -> float:
        if len(self.bridge_pair_buffer) < self.bridge_batch_size:
            return 0.0
        
        batch = self.bridge_pair_buffer[:self.bridge_batch_size]
        self.bridge_pair_buffer = self.bridge_pair_buffer[self.bridge_batch_size:]
        
        expert_batches = {}
        for text_sdr, vision_sdr, expert_name, multimodal_id in batch:
            if expert_name not in expert_batches:
                expert_batches[expert_name] = ([], [])
            expert_batches[expert_name][0].append(text_sdr)
            expert_batches[expert_name][1].append(vision_sdr)
        
        total_loss = 0.0
        
        for expert_name, (text_list, vision_list) in expert_batches.items():
            if expert_name not in self.pons or not isinstance(self.pons[expert_name], dict):
                continue
            
            bridge = self.pons[expert_name]
            text_sdrs = torch.stack(text_list)
            vision_sdrs = torch.stack(vision_list)
            
            device = next(bridge["text_to_expert"].parameters()).device
            text_sdrs = text_sdrs.to(device)
            vision_sdrs = vision_sdrs.to(device)
            
            self.cross_modal_bridge_optimizer.zero_grad()
            
            predicted_vision = bridge["text_to_expert"](text_sdrs)
            predicted_text = bridge["expert_to_text"](vision_sdrs)
            
            loss_vision = 1 - F.cosine_similarity(predicted_vision, vision_sdrs, dim=1).mean()
            loss_text = 1 - F.cosine_similarity(predicted_text, text_sdrs, dim=1).mean()
            
            cycle_text = bridge["expert_to_text"](predicted_vision)
            cycle_vision = bridge["text_to_expert"](predicted_text)
            loss_cycle_text = 1 - F.cosine_similarity(cycle_text, text_sdrs, dim=1).mean()
            loss_cycle_vision = 1 - F.cosine_similarity(cycle_vision, vision_sdrs, dim=1).mean()
            
            batch_size = text_sdrs.shape[0]
            similarity_matrix = predicted_vision @ vision_sdrs.T
            
            pos_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
            neg_mask = ~pos_mask
            
            pos_similarity = similarity_matrix[pos_mask].view(batch_size, 1)
            neg_similarity = similarity_matrix[neg_mask].view(batch_size, -1)
            
            temperature = 0.1
            logits = torch.cat([pos_similarity, neg_similarity], dim=1) / temperature
            labels = torch.zeros(batch_size, dtype=torch.long, device=device)
            loss_contrastive = F.cross_entropy(logits, labels)
            
            batch_total_loss = (
                loss_vision + loss_text + 
                self.cycle_consistency_weight * (loss_cycle_text + loss_cycle_vision) +
                self.contrastive_loss_weight * loss_contrastive
            )
            
            current_lr = self.bridge_learning_rate
            if self.bridge_train_steps < self.cold_start_steps:
                current_lr *= self.cold_start_lr_multiplier
                for param_group in self.cross_modal_bridge_optimizer.param_groups:
                    param_group['lr'] = current_lr
            
            batch_total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(bridge["text_to_expert"].parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(bridge["expert_to_text"].parameters(), max_norm=1.0)
            
            self.cross_modal_bridge_optimizer.step()
            
            if self.bridge_train_steps == self.cold_start_steps and current_lr != self.bridge_learning_rate:
                for param_group in self.cross_modal_bridge_optimizer.param_groups:
                    param_group['lr'] = self.bridge_learning_rate
                logger.info(f"🧠 脑桥冷启动阶段结束，恢复正常学习率: {self.bridge_learning_rate:.6f}")
            
            self.bridge_loss_history["total"].append(batch_total_loss.item())
            self.bridge_loss_history["vision"].append(loss_vision.item())
            self.bridge_loss_history["text"].append(loss_text.item())
            self.bridge_loss_history["cycle_text"].append(loss_cycle_text.item())
            self.bridge_loss_history["cycle_vision"].append(loss_cycle_vision.item())
            self.bridge_loss_history["contrastive"].append(loss_contrastive.item())
            
            self.bridge_train_steps += 1
            total_loss += batch_total_loss.item()
            
            logger.info(
                f"🧠 脑桥批量训练完成 | 专家: {expert_name} | 步骤: {self.bridge_train_steps} | "
                f"总损失: {batch_total_loss.item():.4f} | 视觉损失: {loss_vision.item():.4f} | "
                f"对比损失: {loss_contrastive.item():.4f} | 学习率: {current_lr:.6f}"
            )
        
        EventBus().emit(Event(
            event_type=EventType.BRIDGE_TRAINED,
            data={"steps": self.bridge_train_steps, "loss": total_loss / len(expert_batches) if expert_batches else 0.0},
            timestamp=time.time()
        ))
        
        return total_loss / len(expert_batches) if expert_batches else 0.0
    
    def process_image_with_caption(self, image_tensor: torch.Tensor, caption: str) -> Tuple[Optional[int], Optional[int]]:
        try:
            embedding_model = self._container.embedding_model
            vae_manager = self._container["vae_manager"].manager
            hippocampus_router = self._container["hippocampus_router"].router
            experts = self._container["experts"]
            sdr_encoders = self._container["sdr_encoders"]
            
            text_clip_vec = torch.as_tensor(embedding_model.embed_query(caption), dtype=torch.float32)
            vision_clip_vec = vae_manager.encode_image(image_tensor)
            
            text_sdr = sdr_encoders["概念"].encode(text_clip_vec)
            vision_sdr = sdr_encoders["视觉"].encode(vision_clip_vec)
            
            multimodal_id = str(uuid.uuid4())
            bridge_loss = self.cross_modal_learning_step(text_sdr, vision_sdr, target_expert="视觉")
            logger.info(f"🧠 图文跨模态关联学习完成 | 损失: {bridge_loss:.4f}")
            
            text_mem_id = hippocampus_router.encode(
                clip_vec=text_clip_vec, sdr=text_sdr, content=caption,
                metadata={"multimodal_id": multimodal_id, "type": "text", "source": "user_input", "has_visual": True},
                expert="概念"
            )
            
            visual_content = f"[用户上传图片] 绑定ID:{multimodal_id}"
            vision_mem_id = hippocampus_router.encode(
                clip_vec=vision_clip_vec, sdr=vision_sdr, content=visual_content,
                metadata={"multimodal_id": multimodal_id, "type": "visual", "source": "user_upload", "caption": caption},
                expert="视觉"
            )
            
            concept_expert = experts["概念"]
            visual_expert = experts["视觉"]
            
            predicted_vision = self.pons["视觉"]["text_to_expert"](text_sdr)
            prediction_error = 1 - F.cosine_similarity(predicted_vision, vision_sdr, dim=0).item()
            rpe = prediction_error - concept_expert.expected_error
            concept_expert.predictive_std_update(text_sdr, predicted_vision, rpe)
            
            predicted_text = self.pons["视觉"]["expert_to_text"](vision_sdr)
            prediction_error_vision = 1 - F.cosine_similarity(predicted_text, text_sdr, dim=0).item()
            rpe_vision = prediction_error_vision - visual_expert.expected_error
            visual_expert.predictive_std_update(vision_sdr, predicted_text, rpe_vision)
            
            logger.info(f"✅ 图文联合处理完成 | 文本ID: {text_mem_id} | 视觉ID: {vision_mem_id} | 绑定ID: {multimodal_id}")
            
            EventBus().emit(Event(
                event_type=EventType.IMAGE_PROCESSED,
                data={"text_mem_id": text_mem_id, "vision_mem_id": vision_mem_id, "multimodal_id": multimodal_id},
                timestamp=time.time()
            ))
            
            return (text_mem_id, vision_mem_id)
            
        except Exception as e:
            logger.error(f"❌ 图文联合处理失败: {e}", exc_info=True)
            return (None, None)
    
    def save_bridges(self, save_dir: str) -> None:
        if not self.use_learnable_pons:
            return
        
        os.makedirs(save_dir, exist_ok=True)
        bridge_path = os.path.join(save_dir, "cross_modal_bridges.pt")
        
        bridge_state_dict = {}
        for expert_name, bridge in self.pons.items():
            if isinstance(bridge, dict):
                bridge_state_dict[f"{expert_name}_text_to_expert"] = bridge["text_to_expert"].state_dict()
                bridge_state_dict[f"{expert_name}_expert_to_text"] = bridge["expert_to_text"].state_dict()
        
        torch.save(bridge_state_dict, bridge_path)
        logger.info(f"💾 跨模态脑桥参数已保存: {bridge_path}")
    
    def load_bridges(self, load_dir: str) -> None:
        if not self.use_learnable_pons:
            return
        
        bridge_path = os.path.join(load_dir, "cross_modal_bridges.pt")
        if not os.path.exists(bridge_path):
            logger.info(f"🔄 未找到脑桥参数文件，使用初始脑桥: {bridge_path}")
            return
        
        try:
            bridge_state_dict = torch.load(bridge_path, map_location='cpu', weights_only=False)
            
            for expert_name, bridge in self.pons.items():
                if isinstance(bridge, dict):
                    text_to_expert_key = f"{expert_name}_text_to_expert"
                    expert_to_text_key = f"{expert_name}_expert_to_text"
                    
                    if text_to_expert_key in bridge_state_dict:
                        bridge["text_to_expert"].load_state_dict(bridge_state_dict[text_to_expert_key])
                    if expert_to_text_key in bridge_state_dict:
                        bridge["expert_to_text"].load_state_dict(bridge_state_dict[expert_to_text_key])
            
            logger.info(f"✅ 跨模态脑桥参数加载成功: {bridge_path}")
        except Exception as e:
            logger.error(f"❌ 脑桥参数加载失败，使用初始脑桥: {e}")
    
    def save_bridge_dataset(self, path: str = "./data/cross_modal_dataset.pt") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        serializable_data = []
        for text_sdr, vision_sdr, expert_name, multimodal_id in self.bridge_pair_buffer:
            serializable_data.append({
                "text_sdr": text_sdr.tolist(),
                "vision_sdr": vision_sdr.tolist(),
                "expert_name": expert_name,
                "multimodal_id": multimodal_id
            })
        
        save_data = {
            "pairs": serializable_data,
            "train_steps": self.bridge_train_steps,
            "loss_history": self.bridge_loss_history
        }
        
        torch.save(save_data, path)
        logger.info(f"✅ 保存跨模态数据集 | 配对数: {len(serializable_data)} | 路径: {path}")
    
    def load_bridge_dataset(self, path: str = "./data/cross_modal_dataset.pt") -> int:
        if not os.path.exists(path):
            logger.info(f"🔄 未找到跨模态数据集文件: {path}")
            return 0
        
        try:
            load_data = torch.load(path, map_location='cpu', weights_only=False)
            
            for item in load_data["pairs"]:
                text_sdr = torch.tensor(item["text_sdr"], dtype=torch.float32)
                vision_sdr = torch.tensor(item["vision_sdr"], dtype=torch.float32)
                expert_name = item["expert_name"]
                multimodal_id = item.get("multimodal_id", None)
                
                if len(self.bridge_pair_buffer) < self.max_buffer_size:
                    self.bridge_pair_buffer.append((text_sdr, vision_sdr, expert_name, multimodal_id))
            
            self.bridge_train_steps = load_data.get("train_steps", 0)
            if "loss_history" in load_data:
                self.bridge_loss_history = load_data["loss_history"]
            
            logger.info(f"✅ 加载跨模态数据集 | 配对数: {len(load_data['pairs'])} | 已训练步数: {self.bridge_train_steps}")
            return len(load_data["pairs"])
            
        except Exception as e:
            logger.error(f"❌ 加载跨模态数据集失败: {e}")
            return 0
    
    def plot_bridge_loss_curve(self, save_path: str = "./output/bridge_loss_curve.png") -> None:
        if not self.bridge_loss_history["total"]:
            logger.info("⚠️ 没有训练数据，无法绘制损失曲线")
            return
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        plt.plot(self.bridge_loss_history["total"], label="总损失", linewidth=2)
        plt.plot(self.bridge_loss_history["vision"], label="视觉损失", alpha=0.7)
        plt.plot(self.bridge_loss_history["text"], label="文本损失", alpha=0.7)
        plt.plot(self.bridge_loss_history["cycle_text"], label="循环文本损失", alpha=0.5)
        plt.plot(self.bridge_loss_history["contrastive"], label="对比损失", alpha=0.5)
        
        plt.title("跨模态脑桥训练损失曲线")
        plt.xlabel("训练步数")
        plt.ylabel("损失值")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 脑桥损失曲线已保存 | 路径: {save_path} | 总训练步数: {self.bridge_train_steps}")