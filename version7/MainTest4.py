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
from AdvancedBrainV6 import AdvancedBrainV6
from LLMBrainWrapperV4 import LLMBrainWrapperV4

def import_knowledge_dataset(llm_brain, dataset_path, flag_path):
    """
    导入通用知识库到大脑皮层（修复版）
    用 LLMBrainWrapper 的 learn 方法，确保路由正确
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
    
    success_count = 0
    for text in tqdm(knowledge_list, desc="导入进度"):
        try:
            llm_brain.learn(text)
            success_count += 1
        except Exception as e:
            logger.error(f"❌ 导入失败: {text[:50]}... 错误: {e}")
    
    # 保存并创建标记文件
    logger.info("\n💾 正在保存导入的知识...")
    llm_brain.brain.save_all()
    
    with open(flag_path, "w", encoding="utf-8") as f:
        f.write(f"通用知识数据集已导入，成功{success_count}条")
    
    logger.info(f"\n🎉 导入完成！成功: {success_count}/{len(knowledge_list)} 条")

if __name__ == "__main__":
    # 1. 初始化类脑记忆系统
    print("=" * 60)
    print("🧠 初始化类脑记忆系统...")
    print("=" * 60)
    brain = AdvancedBrainV6(dim=config.dim, storage_dir=config.storage_dir, ollama_model= "bge-m3")

    # 2. 包装 LLM 增强层
    print("\n" + "=" * 60)
    llm_brain = LLMBrainWrapperV4(brain)
    print("=" * 60)

    # 3. 一键导入通用知识数据集（仅首次运行）
    dataset_path = "chinese_history.txt"
    first_run_flag = os.path.join(config.storage_dir, "chinese_history")
    
    if os.path.exists(dataset_path) and not os.path.exists(first_run_flag):
        import_knowledge_dataset(llm_brain, dataset_path, first_run_flag)
    else:
        final_memory_count = brain.get_brain_status()["total_memories"]
        logger.info(f"\n✅ 通用知识数据集已导入，当前总记忆数：{final_memory_count}")

    # 4. 打印初始大脑状态
    print("\n" + "=" * 60)
    print("📊 初始大脑状态")
    print("=" * 60)
    status = brain.get_brain_status()
    print(json.dumps(
        status, indent=2, ensure_ascii=False,
        default=lambda x: x.item() if hasattr(x, 'item') else x
    ))

    # 5. 交互式对话循环
    print("\n" + "=" * 60)
    print("💬 进入对话模式")
    print("  - 输入 'exit' 退出并保存")
    print("  - 输入 'analyze' 分析专家突触结构")
    print("  - 输入 'sleep' 手动触发睡眠巩固")
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
    
    # 6. 显示最终大脑状态
    print("\n" + "=" * 60)
    print("📊 最终大脑状态")
    print("=" * 60)
    status = brain.get_brain_status()
    print(json.dumps(
        status, indent=2, ensure_ascii=False,
        default=lambda x: x.item() if hasattr(x, 'item') else x
    ))