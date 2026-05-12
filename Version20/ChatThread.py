from BrainConfig import config
from PyQt5.QtCore import QThread, pyqtSignal
import os
import logging
import uuid
import torch
import torch.nn.functional as F

# ================== 后台工作线程 ==================
import uuid
import torch
import torch.nn.functional as F
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ChatThread")

class ChatThread(QThread):
    response_received = pyqtSignal(str)
    
    def __init__(self, user_input, img_path=None, brain= None, mm_gateway= None, llm_brain= None):
        super().__init__()
        self.user_input = user_input
        self.img_path = img_path
        self.VISUAL_EXPERT = "视觉"
        self.TARGET_DIM = brain.dim
        self.brain = brain
        self.mm_gateway = mm_gateway
        self.llm_brain = llm_brain

    def run(self):
        try:
            multimodal_id = str(uuid.uuid4())
            vae_data = None  # 🔥 新增：存储VAE数据
            
            # ============== 核心：纯视觉向量存储 + 极简冲突标签 + VAE提取 ==============
            if self.img_path and os.path.exists(self.img_path):
                # ============== 文本记忆：极简存储，携带绑定ID ==============
                if self.user_input.strip():
                    text_content = f"{self.user_input} | 绑定ID:{multimodal_id}"
                    self.brain.learn(text=text_content)
                    logger.info(f"✅ 文本记忆已存入 | 绑定ID:{multimodal_id}")

                # 1. 提取纯图片CLIP视觉向量（用于语义检索）
                image_feat = self.mm_gateway.encode_image(self.img_path)
                
                # 2. 对齐维度（仅保留这个必要的技术修复）
                if image_feat.shape[-1] != self.TARGET_DIM:
                    proj = torch.nn.Linear(image_feat.shape[-1], self.TARGET_DIM, bias=False).to(image_feat.device)
                    image_feat = proj(image_feat)
                
                # 3. 归一化
                image_feat = F.normalize(image_feat.detach().squeeze(), p=2, dim=-1)

                # 4. 🔥 新增：提取VAE生成向量（用于大脑内部重塑图像）
                if hasattr(self.brain, 'vae_manager'):
                    vae_data = self.brain.vae_manager.encode_image(self.img_path)
                    logger.info(f"✅ VAE向量提取成功 | 大小: ~4KB")

                # 5. ✅ 极简冲突标签（只做区分，不做任何加工）
                if self.user_input and self.user_input.strip():
                    tag = self.user_input.strip()[:6].replace("|", "").replace("\n", "")
                else:
                    tag = os.path.splitext(os.path.basename(self.img_path))[0]

                # 6. ✅ 视觉记忆：纯向量 + 绑定ID + 冲突标签
                visual_content = f"[视觉记忆-{tag}] 绑定ID:{multimodal_id}"
                logger.info(f"✅ 视觉记忆生成 | 标签:{tag} | 绑定ID:{multimodal_id}")

                # 7. 原生丘脑过滤 + 海马体存储（🔥 新增VAE数据存入metadata）
                passed, info_packet = self.brain.thalamus.filter_and_relay(
                    input_vec=image_feat,
                    input_text=visual_content,
                    metadata={
                        "force_expert": self.VISUAL_EXPERT,
                        "multimodal_id": multimodal_id,
                        "type": "visual",
                        "image_path": self.img_path,
                        "tag": tag,
                        "vae_latent": vae_data  # 🔥 关键：把VAE数据存入metadata
                    }
                )
                
                if passed:
                    # ✅ 修复：info_packet是字典，用[]语法取值（唯一修改的地方）
                    image_feat = info_packet["vec"]
                    saliency = info_packet["saliency"]
                    
                    sdr_encoder = self.brain.sdr_encoders.get(self.VISUAL_EXPERT, self.brain.sdr_encoders["概念"])
                    sdr = sdr_encoder.encode(image_feat.unsqueeze(0))

                    # 🔥 核心修复：用 mem_id 接住 encode 的返回值（🔥 同时存VAE数据）
                    mem_id = self.brain.hippocampus_router.encode(
                        clip_vec=image_feat,
                        sdr=sdr,
                        content=visual_content,
                        metadata={
                            "expert": self.VISUAL_EXPERT,
                            "saliency": saliency,
                            "multimodal_id": multimodal_id,
                            "type": "visual",
                            "image_path": self.img_path,
                            "tag": tag,
                            "vae_latent": vae_data  # 🔥 关键：同时存入海马体记忆的metadata
                        },
                        expert=self.VISUAL_EXPERT
                    )
                    logger.info(f"✅ 纯视觉向量已存入视觉专家 | 记忆ID:{mem_id}")

                    # ===================== 🔥 新增：调用大脑专门的绑定函数 =====================
                    # 只调用，不做复杂逻辑，绑定功能全在大脑里实现
                    self.brain.bind_related_memories(
                        new_mem_id=mem_id,
                        new_mem_vec=image_feat,
                        new_mem_text=visual_content,
                        target_expert=self.VISUAL_EXPERT,
                        user_input=self.user_input
                    )
                    # ============================================================================

            # ============== 问答（只保留一次调用，删除重复） ==============
            response = self.llm_brain.ask(self.user_input)
            self.response_received.emit(response)

        except Exception as e:
            logger.error(f"❌ 图文处理出错：{str(e)}", exc_info=True)
            self.response_received.emit("抱歉，图片学习失败了~")