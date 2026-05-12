import torch
import logging
import re
import json
from PIL import Image
from langchain_ollama import OllamaEmbeddings, ChatOllama
from transformers import (
    CLIPTokenizer, CLIPTextModel,
    CLIPVisionModel, CLIPImageProcessor
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("MultiModalInputGateway")
logging.getLogger("transformers").setLevel(logging.ERROR)

class MultiModalInputGateway:
    def __init__(self, 
                 device="cuda" if torch.cuda.is_available() else "cpu", 
                 ollama_model="nomic-embed-text",
                 ollama_chat_model="gemma3:4b",
                 CLIP_MODEL_PATH=r"D:\2250111005\HippoCortex\clip-vit-large-patch14"):
        self.device = device
        self.dim = 768
        
        # Ollama
        self.embedding_model = OllamaEmbeddings(model=ollama_model)
        self.chat_model = ChatOllama(model=ollama_chat_model, temperature=0)
        self.dim = len(self.embedding_model.embed_query("test"))
        logger.info(f"✅ Ollama 就绪！向量维度: {self.dim}")

        # CLIP-L/14
        self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_PATH)
        self.text_encoder = CLIPTextModel.from_pretrained(CLIP_MODEL_PATH).to(self.device).eval()
        self.vision_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_PATH)
        self.vision_encoder = CLIPVisionModel.from_pretrained(CLIP_MODEL_PATH).to(self.device).eval()
        self.image_proj = torch.nn.Linear(1024, self.dim).to(self.device)
        logger.info("✅ CLIP-L/14 加载完成！")

    def encode_text(self, text: str) -> torch.Tensor:
        emb = self.embedding_model.embed_query(text.strip())
        tensor = torch.tensor(emb, device=self.device).float()
        return torch.nn.functional.normalize(tensor, p=2, dim=-1)

    def encode_image(self, image_path: str) -> torch.Tensor:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.vision_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.vision_encoder(**inputs)
            img_feat = outputs.pooler_output
            return self.image_proj(img_feat).squeeze(0)
        except:
            return torch.zeros(self.dim, device=self.device)

    def fuse_text_image_vec(self, text_vec, img_vec, alpha=0.6):
        fused = alpha * text_vec + (1 - alpha) * img_vec
        return torch.nn.functional.normalize(fused, p=2, dim=-1)

    # ✅ 修复：大模型输出清洗 + 强约束JSON
    def auto_grounding(self, text: str):
        try:
            prompt = """
            你是实体语义提取器，严格遵守规则：
            1. 只输出一行标准JSON，不要解释、不要多余文字
            2. entity：提取句子核心实体名词
            3. behavior：精简为单个标准动词，不要后缀修饰
            4. property：归纳为大类属性，只能从下面选一个：
            宠物动物 / 水果植被 / 自然风景 / 建筑居所 / 生活用品 / 人物身份

            句子：%s
            输出严格格式：{"entity":"","behavior":"","property":""}
                    """ % text.strip()
            
            res = self.chat_model.invoke(prompt).content
            # 清洗输出，提取JSON
            json_str = re.search(r'\{.*\}', res, re.DOTALL).group()
            result = json.loads(json_str)
            return {
                "entity_list": [result["entity"]],
                "behavior_list": [result["behavior"]],
                "property_list": [result["property"]]
            }
        except:
            return {"entity_list":["未知"],"behavior_list":["未知"],"property_list":["未知"]}

    def build_grounded_memory(self, text: str, image_path=None):
        text_vec = self.encode_text(text)
        fused_vec = text_vec
        if image_path:
            img_vec = self.encode_image(image_path)
            fused_vec = self.fuse_text_image_vec(text_vec, img_vec)

        meta = self.auto_grounding(text)
        return {
            "content": text, "clip_vec": fused_vec,
            "metadata": {**meta, "is_multimodal": image_path is not None}
        }

if __name__ == "__main__":
    mm = MultiModalInputGateway()

    mem2 = mm.build_grounded_memory("四个苹果长在树上", image_path=r"apple.png")
    print("🐶 测试结果：", mem2)

    mem3 = mm.build_grounded_memory("小白狗坐在地上", image_path=r"HippoCortexV6-2\imgs\sit.png")
    print("🐶 测试结果：", mem3)