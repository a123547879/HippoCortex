# 必须在导入matplotlib之前设置后端
import matplotlib
matplotlib.use('Agg')

import sys
import os
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FinalImagePet")

# 导入大脑核心模块
from BrainConfig import config
from brain_core import BrainCore
from Cognitive_system import CognitiveSystem
from brain_interface import BrainInterface
from event_system import EventBus
from LLMBrainWrapper import LLMBrainWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from Multimodal_gateway import MultiModalInputGateway
from DummyBrain import DummyBrain

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 60)
    print("🧠 正在初始化小白大脑 + 多模态网关...")
    print("=" * 60)
    print("💡 提示：请确保 BrainConfig.py 中 local_bias_strength = 1.2，热力图效果更明显！")

    # 1. 初始化LLM和Embedding
    llm = ChatOllama(model=config.ollama_model_name)
    embedding_model = OllamaEmbeddings(model="bge-m3")

    # 2. 创建并启动大脑接口
    brain_interface = BrainInterface(embedding_model, llm, kg_enabled=True)
    brain_interface.start(storage_dir=config.storage_dir)

    # 3. 获取核心组件引用
    cognitive_system = brain_interface.cognitive_system
    core = brain_interface.core

    # 4. 创建兼容包装器
    brain = DummyBrain(cognitive_system, brain_interface)
    llm_brain = LLMBrainWrapper(brain)

    # 5. 初始化多模态网关
    mm_gateway = MultiModalInputGateway(device=config.device)
    logger.info("✅ 多模态图文网关初始化完成！支持图片+文字同时学习")

    # 6. 知识导入检查
    dataset_path = r"HippoCortexV6-2\data_text\all_know.csv"
    first_run_flag = os.path.join(config.storage_dir, "general_knowledge_imported")
    # try:
    if os.path.exists(dataset_path) and not os.path.exists(first_run_flag):
        from MainTest5 import import_knowledge_dataset
        import_knowledge_dataset(llm_brain, dataset_path, first_run_flag, use_kg=False)
    else:
        status = brain.get_brain_status() or {}
        final_memory_count = status.get("total_memories", 0)
        kg_status = "✅ 已启用" if status.get("kg_enabled", True) else "⚡ 已关闭（性能模式）"
        intention_count = status.get("intention_queue_size", 0)
        logger.info(f"✅ 通用知识已导入 | 总记忆数：{final_memory_count} | 知识图谱：{kg_status} | 意图队列：{intention_count}")
    # except Exception as e:
    #     logger.warning(f"⚠️ 知识导入跳过: {e}")

    print("\n" + "=" * 60)
    print("✅ 大脑+多模态网关初始化完成！现在启动桌宠界面...")
    print("=" * 60)


    from PyQt5.QtWidgets import QApplication
    # 导入UI模块
    from ui.brain_pet_window import XiaobaiBrainPet

    # 7. 启动UI
    app = QApplication(sys.argv)
    pet = XiaobaiBrainPet(brain, cognitive_system, core, llm_brain, mm_gateway, brain_interface)
    pet.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()