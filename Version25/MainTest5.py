import os
import json
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch

from BrainConfig import config
# ===================== 🔥 修改：导入新架构 =====================
from brain_core import BrainCore
from Cognitive_system import CognitiveSystem
from brain_interface import BrainInterface
from event_system import EventBus, Event, EventType, on_event
# ================================================================
# 🔥 修复1：修正LLM包装器导入名称（匹配你的文件名）
from LLMBrainWrapper import LLMBrainWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ===================== 🔥 新增：导入多模态输入网关 =====================
from Multimodal_gateway import MultiModalInputGateway
# =====================================================================
from ChatThread import ChatThread
from Data_models import SleepReport, Intention, ThoughtResult
from DummyBrain import DummyBrain

# 配置中文字体
plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_brain_structure(brain):
    """
    诊断专家网络的突触结构（修复版）
    增加了安全检查，防止权重未初始化报错
    """
    print("\n🔍 正在分析专家突触结构...")
    
    for name, expert in brain.experts.items():
        try:
            # 安全检查权重是否存在且有效
            if not hasattr(expert, 'synapse') or expert.synapse is None:
                print(f"  - 专家 [{name}]: 突触权重未初始化，跳过")
                continue
            
            # 获取权重矩阵
            weights = expert.synapse.detach().cpu().numpy()
            
            # 检查权重矩阵是否为空
            if weights.size == 0 or np.all(weights == 0):
                print(f"  - 专家 [{name}]: 突触权重为空（还未学习），跳过")
                continue
            
            # 1. 绘制热力图 (观察聚类块)
            plt.figure(figsize=(8, 6))
            # 只取前200x200的子矩阵，避免图太大看不清
            plot_size = min(200, weights.shape[0], weights.shape[1])
            sns.heatmap(
                weights[:plot_size, :plot_size], 
                cmap="viridis", 
                center=0, 
                xticklabels=False, 
                yticklabels=False
            )
            plt.title(f"Synapse Heatmap: {name} (Top {plot_size}x{plot_size})")
            plt.tight_layout()
            plt.savefig(f"synapse_map_{name}.png", dpi=150)
            plt.close()
            
            # 2. 计算权重统计信息
            sparsity = (np.abs(weights) < 0.01).mean()
            mean_abs = np.mean(np.abs(weights))
            max_weight = np.max(weights)
            min_weight = np.min(weights)
            
            print(f"  - 专家 [{name}]:")
            print(f"    突触稀疏度: {sparsity:.2%}")
            print(f"    平均绝对权重: {mean_abs:.4f}")
            print(f"    权重范围: [{min_weight:.4f}, {max_weight:.4f}]")
            print(f"    ✅ 热力图已保存: synapse_map_{name}.png")
            
        except Exception as e:
            print(f"  - 专家 [{name}] 分析失败: {e}")
            continue
        
    print("\n✅ 分析完成！")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Main")

from BrainConfig import config
from DummyBrain import DummyBrain
from LLMBrainWrapper import LLMBrainWrapper
from langchain_ollama import ChatOllama

def import_knowledge_dataset(llm_brain, dataset_path, flag_path, use_kg: bool = False):
    """
    导入CSV格式实体-关系知识库到大脑皮层（Pandas优化版）
    :param use_kg: 是否在导入时启用知识图谱（默认False，提升导入速度）
    """
    if not os.path.exists(dataset_path):
        logger.warning(f"⚠️ 找不到数据集: {dataset_path}，跳过导入。")
        return

    logger.info(f"📚 开始从 {dataset_path} 导入CSV格式知识库...")
    
    try:
        # ✅ Pandas 读取CSV（自动处理编码、空值、注释）
        df = pd.read_csv(
            dataset_path,
            encoding="utf-8",
            comment="#",  # 自动跳过所有以#开头的注释行
            skip_blank_lines=True,  # 自动跳过空行
            on_bad_lines="warn",  # 错误行只警告不中断
            dtype=str  # 全部按字符串读取，避免类型转换错误
        )
        
        # 校验必要列是否存在
        required_columns = ["主体实体", "关系谓词", "客体实体"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"❌ CSV缺少必要列: {', '.join(missing_columns)}")
            return
        
        # ✅ 数据清洗：删除关键列有空值的行
        df_clean = df.dropna(subset=required_columns).copy()
        invalid_count = len(df) - len(df_clean)
        
        if len(df_clean) == 0:
            logger.warning("⚠️ 数据集为空或所有行都无效，跳过导入。")
            return
        
        logger.info(f"✅ CSV解析完成，共找到 {len(df_clean)} 条有效知识，跳过 {invalid_count} 条无效行")
        
        # ✅ 批量拼接自然语言句子（Pandas向量化操作，比循环快10倍）
        df_clean["knowledge_sentence"] = (
            df_clean["主体实体"].str.strip() + " " + 
            df_clean["关系谓词"].str.strip() + " " + 
            df_clean["客体实体"].str.strip()
        )
        
        # 转换为列表供批量学习使用
        knowledge_list = df_clean["knowledge_sentence"].tolist()
        
    except Exception as e:
        logger.error(f"❌ CSV文件读取失败: {str(e)}", exc_info=True)
        return
    
    # ✅ 正确获取原始知识图谱状态（修复原函数硬编码bug）
    original_kg_state = False
    if hasattr(llm_brain.brain, 'is_kg_enabled'):
        original_kg_state = llm_brain.brain.is_kg_enabled()
    
    # 导入时临时关闭知识图谱（大幅提升速度）
    if not use_kg and hasattr(llm_brain.brain, 'disable_kg'):
        llm_brain.brain.disable_kg()
        logger.info("⚡ 导入模式：知识图谱已临时关闭，提升导入速度")
    
    success_count = 0
    failed_batches = 0
    
    # 批量导入（每批32条，平衡速度和稳定性）
    batch_size = 32
    for i in tqdm(range(0, len(knowledge_list), batch_size), desc="导入进度"):
        batch_texts = knowledge_list[i:i+batch_size]
        try:
            # 优先使用批量接口
            if hasattr(llm_brain.brain, 'batch_learn'):
                llm_brain.brain.batch_learn(batch_texts)
            else:
                # 回退到单条导入
                for text in batch_texts:
                    llm_brain.learn(text)
            success_count += len(batch_texts)
        except Exception as e:
            failed_batches += 1
            logger.error(f"❌ 第{i//batch_size + 1}批导入失败: {str(e)}")
    
    # 恢复知识图谱状态
    if original_kg_state and not use_kg and hasattr(llm_brain.brain, 'enable_kg'):
        llm_brain.brain.enable_kg()
        logger.info("✅ 导入完成，知识图谱已恢复")
    
    # 保存并创建标记文件
    logger.info("\n💾 正在保存导入的知识...")
    llm_brain.brain.save_all()
    
    with open(flag_path, "w", encoding="utf-8") as f:
        f.write(f"通用知识数据集已导入\n成功: {success_count}/{len(knowledge_list)} 条\n失败批次: {failed_batches}")
    
    logger.info(f"\n🎉 导入完成！成功: {success_count}/{len(knowledge_list)} 条，失败批次: {failed_batches}")

if __name__ == "__main__":
    # 1. 先初始化LLM
    print("=" * 60)
    print("🤖 正在初始化LLM...")
    print("=" * 60)
    lm = ChatOllama(model=config.ollama_model_name)

    # ===================== 🔥 修改：初始化新架构大脑 =====================
    # 1. 初始化Embedding模型
    embedding_model = OllamaEmbeddings(model="bge-m3")

    # 2. 创建大脑接口
    brain_interface = BrainInterface(embedding_model, lm, kg_enabled=True)

    # 3. 启动大脑
    brain_interface.start(storage_dir=config.storage_dir)

    # 4. 获取认知系统引用（用于直接访问内部组件）
    cognitive_system = brain_interface.cognitive_system
    core = brain_interface.core

    # 创建兼容对象
    brain = DummyBrain(cognitive_system, brain_interface)
    llm_brain = LLMBrainWrapper(brain)

    # 3. 包装 LLM 增强层
    print("\n" + "=" * 60)
    llm_brain = LLMBrainWrapper(brain)
    print("=" * 60)

    # 4. 一键导入通用知识数据集（仅首次运行）
    dataset_path = r"HippoCortexV6-2\data_text\all_know.txt"
    first_run_flag = os.path.join(config.storage_dir, "general_knowledge")
    
    if os.path.exists(dataset_path) and not os.path.exists(first_run_flag):
        import_knowledge_dataset(llm_brain, dataset_path, first_run_flag, use_kg=False)
    else:
        final_memory_count = brain.get_brain_status()["total_memories"]
        kg_status = "✅ 已启用" if brain.get_brain_status().get("kg_enabled", True) else "⚡ 已关闭（性能模式）"
        logger.info(f"\n✅ 通用知识数据集已导入，当前总记忆数：{final_memory_count}，知识图谱：{kg_status}")

    # 5. 打印初始大脑状态
    print("\n" + "=" * 60)
    print("📊 初始大脑状态")
    print("=" * 60)
    status = brain.get_brain_status()
    print(json.dumps(
        status, indent=2, ensure_ascii=False,
        default=lambda x: x.item() if hasattr(x, 'item') else x
    ))

    # 6. 交互式对话循环
    print("\n" + "=" * 60)
    print("💬 进入对话模式")
    print("  - 输入 'exit' 退出并保存")
    print("  - 输入 'analyze' 分析专家突触结构")
    print("  - 输入 'sleep' 手动触发睡眠巩固")
    print("  - 输入 'add_entity <名字>' 添加重要实体（如：add_entity 邓尧）")
    print("  - 输入 'remove_entity <名字>' 删除重要实体")
    print("  - 输入 'list_entities' 列出所有重要实体")
    print("  - 输入 'enable_kg' 启用知识图谱")
    print("  - 输入 'disable_kg' 禁用知识图谱（性能模式）")
    print("=" * 60)

    try:
        while True:
            user_input = input("\n你: ").strip()

            if user_input.lower() == "analyze":
                analyze_brain_structure(brain)
                continue
            
            if user_input.lower() == "sleep":
                brain.sleep_consolidate_all()
                continue
            
            # 动态重要实体管理命令
            if user_input.lower().startswith("add_entity "):
                entity_name = user_input[len("add_entity "):].strip()
                if entity_name:
                    brain.add_important_entity(entity_name)
                    print(f"✅ 已添加重要实体: {entity_name}")
                else:
                    print("⚠️  请输入实体名称，如：add_entity 邓尧")
                continue
            
            if user_input.lower().startswith("remove_entity "):
                entity_name = user_input[len("remove_entity "):].strip()
                if entity_name:
                    brain.remove_important_entity(entity_name)
                    print(f"✅ 已删除重要实体: {entity_name}")
                else:
                    print("⚠️  请输入实体名称，如：remove_entity 邓尧")
                continue
            
            if user_input.lower() == "list_entities":
                entities = brain.list_important_entities()
                if entities:
                    print(f"📋 当前重要实体: {', '.join(entities)}")
                else:
                    print("📋 当前无重要实体")
                continue
            
            # 🔥 新增：知识图谱开关命令
            if user_input.lower() == "enable_kg":
                brain.enable_kg()
                print("✅ 知识图谱已启用")
                continue
            
            if user_input.lower() == "disable_kg":
                brain.disable_kg()
                print("✅ 知识图谱已禁用（性能模式）")
                continue
            
            if user_input.lower() == "exit":
                print("\n" + "=" * 60)
                print("🌙 正在进行睡眠巩固...")
                brain.sleep_consolidate_all()
                print("💾 正在保存所有大脑数据...")
                print("=" * 60)
                brain.save_all()
                print("\n✅ 所有数据已安全保存！")
                print("\n再见！下次再聊~")
                break

            if not user_input:
                print("请输入你的问题")
                continue
                
            # 调用封装好的大模型进行问答
            answer = llm_brain.ask(user_input)
            print(f"\n{answer}")

    except KeyboardInterrupt:
        print("\n\n⚠️ 检测到强制退出 (Ctrl+C)，正在紧急保存所有数据...")
        try:
            brain.save_all()
            print("✅ 数据已安全保存。")
        except Exception as e:
            logger.error(f"❌ 紧急保存失败: {e}")
        print("\n再见！下次再聊~")
    
    # 7. 显示最终大脑状态
    print("\n" + "=" * 60)
    print("📊 最终大脑状态")
    print("=" * 60)
    status = brain.get_brain_status()
    print(json.dumps(
        status, indent=2, ensure_ascii=False,
        default=lambda x: x.item() if hasattr(x, 'item') else x
    ))