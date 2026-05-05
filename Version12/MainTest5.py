import os
import json
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

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
from AdvancedBrainV10 import AdvancedBrainV10
from LLMBrainWrapperV5 import LLMBrainWrapperV5
from langchain_ollama import ChatOllama

def import_knowledge_dataset(llm_brain, dataset_path, flag_path, use_kg: bool = False):
    """
    导入通用知识库到大脑皮层（优化版：支持批量导入和知识图谱开关）
    :param use_kg: 是否在导入时启用知识图谱（默认False，提升导入速度）
    """
    if not os.path.exists(dataset_path):
        logger.warning(f"⚠️ 找不到数据集: {dataset_path}，跳过导入。")
        return

    logger.info(f"📚 开始从 {dataset_path} 导入知识...")
    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    
    knowledge_list = []
    for line in lines:
        text = line.strip()
        if text and not text.startswith('#'):
            knowledge_list.append(text)
    
    if not knowledge_list:
        logger.warning("⚠️ 数据集为空，跳过导入。")
        return
    
    logger.info(f"✅ 解析完成，共找到 {len(knowledge_list)} 条知识")
    
    # 🔥 优化：导入时临时关闭知识图谱（提升速度）
    original_kg_state = True
    # if not use_kg:
    #     llm_brain.brain.disable_kg()
    #     logger.info("⚡ 导入模式：知识图谱已临时关闭，提升导入速度")
    
    success_count = 0
    
    # 🔥 优化：批量导入（每批32条）
    batch_size = 32
    for i in tqdm(range(0, len(knowledge_list), batch_size), desc="导入进度"):
        batch_texts = knowledge_list[i:i+batch_size]
        try:
            # 如果有批量接口，用批量接口
            if hasattr(llm_brain.brain, 'batch_learn'):
                llm_brain.brain.batch_learn(batch_texts)
            else:
                # 否则回退到单条
                for text in batch_texts:
                    llm_brain.learn(text)
            success_count += len(batch_texts)
        except Exception as e:
            logger.error(f"❌ 批量导入失败: {e}")
    
    # 恢复知识图谱状态
    if original_kg_state and not use_kg:
        llm_brain.brain.enable_kg()
        logger.info("✅ 导入完成，知识图谱已恢复")
    
    # 保存并创建标记文件
    logger.info("\n💾 正在保存导入的知识...")
    llm_brain.brain.save_all()
    
    with open(flag_path, "w", encoding="utf-8") as f:
        f.write(f"通用知识数据集已导入，成功{success_count}条")
    
    logger.info(f"\n🎉 导入完成！成功: {success_count}/{len(knowledge_list)} 条")

if __name__ == "__main__":
    # 1. 先初始化LLM
    print("=" * 60)
    print("🤖 正在初始化LLM...")
    print("=" * 60)
    llm = ChatOllama(model=config.ollama_model_name)
    
    # 2. 初始化类脑记忆系统（🔥 可选：关闭知识图谱提升性能）
    print("\n" + "=" * 60)
    print("🧠 初始化类脑记忆系统...")
    print("=" * 60)
    brain = AdvancedBrainV10(
        dim=config.dim, 
        storage_dir=config.storage_dir, 
        ollama_model="bge-m3",
        llm=llm,
        kg_enabled=True  # 🔥 可设置为False关闭知识图谱
    )

    # 3. 包装 LLM 增强层
    print("\n" + "=" * 60)
    llm_brain = LLMBrainWrapperV5(brain)
    print("=" * 60)

    # 4. 一键导入通用知识数据集（仅首次运行）
    dataset_path = "general_knowledge.txt"
    first_run_flag = os.path.join(config.storage_dir, "general_knowledge")
    
    # if os.path.exists(dataset_path) and not os.path.exists(first_run_flag):
    import_knowledge_dataset(llm_brain, dataset_path, first_run_flag, use_kg=False)
    # else:
        # final_memory_count = brain.get_brain_status()["total_memories"]
        # kg_status = "✅ 已启用" if brain.get_brain_status().get("kg_enabled", True) else "⚡ 已关闭（性能模式）"
        # logger.info(f"\n✅ 通用知识数据集已导入，当前总记忆数：{final_memory_count}，知识图谱：{kg_status}")

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