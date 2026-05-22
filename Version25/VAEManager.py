import torch
import torch.nn.functional as F
import numpy as np
import os
import logging
from PIL import Image
from diffusers import AutoencoderKL
import gc

logger = logging.getLogger("VAEManager")

class VAEManager:
    def __init__(self, local_model_path: str, device="cpu", proj_save_dir: str = "./models/vae_proj"):
        """
        🔥 适配跨模态脑桥的VAE管理器
        :param local_model_path: 本地sd-vae-ft-mse模型路径
        :param device: 运行设备（cpu/cuda）
        :param proj_save_dir: SDR↔VAE投影层参数保存目录
        """
        self.local_model_path = local_model_path
        self.device = device
        self.vae = None  # VAE保持懒加载
        self.latent_dim = 4
        self.latent_size = 64
        self.sdr_dim = 2048  # 与你的视觉专家SDR维度一致
        
        # 投影层保存路径
        self.proj_save_dir = proj_save_dir
        os.makedirs(self.proj_save_dir, exist_ok=True)
        self.sdr_to_latent_proj_path = os.path.join(self.proj_save_dir, "sdr_to_latent.pt")
        self.latent_to_sdr_proj_path = os.path.join(self.proj_save_dir, "latent_to_sdr.pt")
        
        # ✅ SDR↔VAE投影层（常驻内存，仅33MB，不占用显存）
        self.sdr_to_latent_proj = self._init_projection(
            self.sdr_dim, 
            self.latent_dim * self.latent_size * self.latent_size,
            self.sdr_to_latent_proj_path
        )
        self.latent_to_sdr_proj = self._init_projection(
            self.latent_dim * self.latent_size * self.latent_size,
            self.sdr_dim,
            self.latent_to_sdr_proj_path
        )
        
        # 验证模型路径
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"❌ VAE模型路径不存在: {local_model_path}")
        
        logger.info(f"✅ VAEManager初始化完成")
        logger.info(f"   - VAE模型: {local_model_path} (懒加载)")
        logger.info(f"   - 设备: {device}")
        logger.info(f"   - 投影层: 已加载 | SDR维度: {self.sdr_dim} | VAE维度: {self.latent_dim}x{self.latent_size}x{self.latent_size}")

    def _init_projection(self, in_dim: int, out_dim: int, save_path: str) -> torch.nn.Linear:
        """初始化投影层，自动加载已保存的参数"""
        proj = torch.nn.Linear(in_dim, out_dim)
        
        if os.path.exists(save_path):
            try:
                proj.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=False))
                logger.debug(f"✅ 加载投影层参数: {save_path}")
            except Exception as e:
                logger.warning(f"⚠️ 投影层参数加载失败，使用初始值: {e}")
                # 小标准差初始化，避免初始输出过大
                torch.nn.init.normal_(proj.weight, mean=0.0, std=0.01)
                torch.nn.init.zeros_(proj.bias)
        else:
            # 小标准差初始化，符合VAE latent分布
            torch.nn.init.normal_(proj.weight, mean=0.0, std=0.01)
            torch.nn.init.zeros_(proj.bias)
            logger.debug(f"🔄 初始化新投影层: {save_path}")
        
        return proj

    def save_projections(self) -> None:
        """保存SDR↔VAE投影层参数（训练后调用）"""
        torch.save(self.sdr_to_latent_proj.state_dict(), self.sdr_to_latent_proj_path)
        torch.save(self.latent_to_sdr_proj.state_dict(), self.latent_to_sdr_proj_path)
        logger.debug(f"💾 投影层参数已保存到: {self.proj_save_dir}")

    def _load_vae(self):
        """懒加载VAE模型，同步投影层设备"""
        if self.vae is None:
            logger.info(f"🔄 加载VAE模型: {self.local_model_path}...")
            self.vae = AutoencoderKL.from_pretrained(
                self.local_model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                local_files_only=True
            ).to(self.device)
            self.vae.eval()
            
            # ✅ 同步投影层到VAE相同设备
            self.sdr_to_latent_proj = self.sdr_to_latent_proj.to(self.device)
            self.latent_to_sdr_proj = self.latent_to_sdr_proj.to(self.device)
            
            logger.info(f"✅ VAE模型加载成功 | 缩放因子: {self.vae.config.scaling_factor}")

    def _unload_vae(self):
        """卸载VAE模型，保留投影层在内存"""
        if self.vae is not None:
            del self.vae
            self.vae = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"♻️ VAE模型已卸载，投影层保留在内存")

    # ===================== 🔥 核心新增：SDR转VAE潜在向量（脑桥想象必备） =====================
    def sdr_to_latent(self, sdr: torch.Tensor) -> torch.Tensor:
        """
        将视觉专家SDR转换为标准VAE潜在向量
        输出格式与encode_image完全一致，可直接用于decode_latent
        :param sdr: 视觉专家输出的2048维SDR向量
        :return: 4x64x64 VAE潜在向量（已乘以缩放因子）
        """
        # 确保输入在正确设备上
        sdr = sdr.to(self.device)
        
        # 投影并reshape为VAE标准形状
        latent_flat = self.sdr_to_latent_proj(sdr)
        latent = latent_flat.reshape(self.latent_dim, self.latent_size, self.latent_size)
        
        # ✅ 关键修复：对齐VAE latent分布和缩放因子
        # 1. tanh限制在(-1,1)，乘以3对齐真实VAE latent的(-3,3)分布
        # 2. 乘以VAE缩放因子，与encode_image输出格式完全一致
        latent = torch.tanh(latent) * 3.0
        latent = latent * self.vae.config.scaling_factor
        
        return latent

    # ===================== 🔥 新增：VAE潜在向量转SDR（双向脑桥预留） =====================
    def latent_to_sdr(self, latent: torch.Tensor) -> torch.Tensor:
        """
        将VAE潜在向量转换为视觉专家SDR
        用于将真实图片编码结果输入视觉专家
        :param latent: 4x64x64 VAE潜在向量（来自encode_image）
        :return: 2048维视觉专家SDR向量
        """
        # 确保输入在正确设备上
        latent = latent.to(self.device)
        
        # 展平并投影到SDR维度
        latent_flat = latent.flatten()
        sdr = self.latent_to_sdr_proj(latent_flat)
        
        # 稀疏化处理，符合SDR特性
        sdr = F.relu(sdr)
        top_values, top_indices = torch.topk(sdr, k=60)  # 与你的active_size一致
        sparse_sdr = torch.zeros_like(sdr)
        sparse_sdr[top_indices] = top_values
        
        return sparse_sdr

    # ===================== 重载：支持PIL Image直接编码 =====================
    def encode_image(self, image_input: str | Image.Image) -> dict:
        """
        编码图像为VAE潜在向量（支持文件路径或PIL Image）
        返回：可序列化的量化字典，大小约4KB
        """
        self._load_vae()
        
        try:
            # 处理输入类型
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")
            else:
                raise TypeError(f"不支持的输入类型: {type(image_input)}")
            
            # 预处理
            image = image.resize((512, 512))
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
            image_tensor = image_tensor * 2 - 1  # 归一化到[-1, 1]
            
            # VAE编码
            with torch.no_grad():
                latent = self.vae.encode(image_tensor).latent_dist.sample()
                latent = latent * self.vae.config.scaling_factor
            
            latent = latent.squeeze(0)
            
            # 极致量化：float16→uint8，体积减75%
            latent_min = float(latent.min())
            latent_max = float(latent.max())
            latent_normalized = (latent - latent_min) / (latent_max - latent_min + 1e-8)
            latent_quantized = (latent_normalized * 255).to(torch.uint8).cpu().numpy()
            
            result = {
                "latent": latent_quantized.tolist(),
                "min": latent_min,
                "max": latent_max,
                "shape": list(latent.shape)
            }
            
            if isinstance(image_input, str):
                logger.info(f"✅ VAE编码成功 | 文件: {os.path.basename(image_input)} | 大小: ~4KB")
            else:
                logger.info(f"✅ VAE编码成功 | PIL图像 | 大小: ~4KB")
            
            return result
            
        finally:
            self._unload_vae()

    def decode_latent(self, latent_data: dict) -> Image.Image:
        """
        从量化字典解码为PIL图像
        :param latent_data: encode_image或sdr_to_latent生成的量化字典
        :return: 512x512 RGB图像
        """
        self._load_vae()
        
        try:
            # 恢复量化向量
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
            self._unload_vae()