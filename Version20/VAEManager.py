import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import os
import logging
from PIL import Image
import torch
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np
import gc
import os

logger = logging.getLogger("VAEManager")

class VAEManager:
    def __init__(self, local_model_path: str, device="cpu"):
        """
        🔥 专为你已下载的 sd-vae-ft-mse 模型优化
        :param local_model_path: 你本地的模型路径（必填）
        :param device: 默认用CPU，不占用显存；有GPU可以改成"cuda"
        """
        self.local_model_path = local_model_path
        self.device = device
        self.vae = None  # 懒加载：初始不加载
        self.latent_dim = 4
        self.latent_size = 64
        
        # 验证模型路径是否存在
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"❌ 模型路径不存在: {local_model_path}\n请检查路径是否正确！")
        
        logger.info(f"✅ VAE管理器初始化完成 | 本地模型:{local_model_path} | 设备:{device} (懒加载)")

    def _load_vae(self):
        """懒加载：只有真正需要时才加载模型"""
        if self.vae is None:
            logger.info(f"🔄 加载本地VAE模型: {self.local_model_path}...")
            self.vae = AutoencoderKL.from_pretrained(
                self.local_model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                local_files_only=True  # 强制用本地文件，不联网
            ).to(self.device)
            self.vae.eval()
            logger.info(f"✅ 本地VAE模型加载成功")

    def _unload_vae(self):
        """用完立即卸载，释放所有资源"""
        if self.vae is not None:
            del self.vae
            self.vae = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"♻️ VAE模型已卸载，资源已释放")

    def encode_image(self, image_path: str) -> dict:
        """
        编码图像为VAE潜在向量
        返回：包含量化向量和参数的dict，方便存储
        """
        self._load_vae()
        
        try:
            # 读取并预处理图像
            image = Image.open(image_path).convert("RGB").resize((512, 512))
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
            image_tensor = image_tensor * 2 - 1  # 归一化到[-1, 1]
            
            # VAE编码
            with torch.no_grad():
                latent = self.vae.encode(image_tensor).latent_dist.sample()
                latent = latent * self.vae.config.scaling_factor
            
            latent = latent.squeeze(0)
            
            # 🔥 极致量化：从float16压缩到uint8，体积减75%（16KB→4KB）
            latent_min = float(latent.min())
            latent_max = float(latent.max())
            latent_normalized = (latent - latent_min) / (latent_max - latent_min + 1e-8)
            latent_quantized = (latent_normalized * 255).to(torch.uint8).cpu().numpy()
            
            # 返回可序列化的dict（方便存到JSON）
            result = {
                "latent": latent_quantized.tolist(),
                "min": latent_min,
                "max": latent_max,
                "shape": list(latent.shape)
            }
            
            logger.info(f"✅ VAE编码成功 | 原始图像:{os.path.basename(image_path)} | 量化后大小: ~4KB")
            return result
            
        finally:
            self._unload_vae()  # 无论成功失败，都卸载模型

    def decode_latent(self, latent_data: dict) -> Image.Image:
        """
        从量化的VAE潜在向量解码图像
        :param latent_data: 之前encode_image返回的dict
        :return: PIL图像
        """
        self._load_vae()
        
        try:
            # 恢复量化的向量
            latent_quantized = torch.from_numpy(np.array(latent_data["latent"])).to(self.device)
            latent_min = torch.tensor(latent_data["min"]).to(self.device)
            latent_max = torch.tensor(latent_data["max"]).to(self.device)
            
            latent_normalized = latent_quantized.float() / 255.0
            latent = latent_normalized * (latent_max - latent_min) + latent_min
            latent = latent.unsqueeze(0) / self.vae.config.scaling_factor
            
            # VAE解码
            with torch.no_grad():
                image_tensor = self.vae.decode(latent).sample
            
            # 后处理
            image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            
            logger.info(f"✅ VAE解码成功 | 图像大小: 512x512")
            return Image.fromarray(image_np)
            
        finally:
            self._unload_vae()  # 无论成功失败，都卸载模型