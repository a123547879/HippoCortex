import torch
import os
import sys
import logging
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("HippocampusTrainer")

# 项目根目录
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

from HippocampusRouter import HippocampusRouter
from Data_models import Entity, MemoryFactory, Evidence
from BrainConfig import config

# ====================== 🔴 核心配置 ======================
EXPERT_TO_ENTITY_TYPE = {
    "身份": "identity",
    "概念": "concept",
    "空间": "place",
    "抽象": "emotion",
    "视觉": "visual"
}

ENTITY_TYPE_TO_EXPERT = {v: k for k, v in EXPERT_TO_ENTITY_TYPE.items()}
ENTITY_TYPE_TO_EXPERT.update({
    "person": "身份",
    "object": "概念",
    "system": "概念",
    "skill": "概念"
})

SDR_DIM = 2048
SDR_ACTIVE_COUNT = 60
BATCH_ENCODE_SIZE = 32
DEVICE = torch.device("cpu")  # 统一使用torch.device类型
GENERATED_TEST_SIZE = 200
TOTAL_EPOCHS = 500
EVAL_INTERVAL = 20  # 每20轮评估一次
PATIENCE = 100  # 连续100轮无提升则早停
logger.info(f"🖥️ 使用设备: {DEVICE}")

# ====================== 🔴 实体生成器 ======================
def create_simple_entity(text: str, expert_name: str, embedding: torch.Tensor) -> Entity:
    entity_type = EXPERT_TO_ENTITY_TYPE.get(expert_name, "concept")
    entity_name = text.strip()[:20]

    normalized_emb = torch.nn.functional.normalize(embedding, p=2, dim=-1)
    _, top_indices = torch.topk(normalized_emb, k=SDR_ACTIVE_COUNT)
    sdr = torch.zeros(SDR_DIM, dtype=torch.float32, device=DEVICE)
    sdr[top_indices] = 1.0

    evi = Evidence(
        content=text,
        source="学习",
        confidence=1.0,
        sdr=sdr.clone().detach(),
        clip_vec=embedding.clone().detach()
    )

    entity = MemoryFactory.create_entity(
        name=entity_name,
        entity_type=entity_type,
        expert=expert_name,
        importance=0.7,
        clip_vec=embedding,
        sdr=sdr
    )
    
    entity.add_evidence(evi)
    return entity

# ====================== 🔴 从CSV加载训练数据 ======================
def load_training_data_from_csv(csv_path: Path):
    if not csv_path.exists():
        logger.error(f"❌ 找不到CSV训练集: {csv_path}")
        sys.exit(1)

    logger.info(f"📚 正在从CSV知识库加载训练数据: {csv_path}")
    
    try:
        df = pd.read_csv(
            csv_path,
            encoding="utf-8",
            comment="#",
            skip_blank_lines=True,
            on_bad_lines="warn",
            dtype=str
        )
        
        required_columns = ["主体实体", "实体类型", "关系谓词", "客体实体", "客体类型"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"❌ CSV缺少必要列: {', '.join(missing_columns)}")
            sys.exit(1)
        
        df_clean = df.dropna(subset=required_columns).copy()
        
        knowledge_base = defaultdict(lambda: {"subjects": set(), "predicates": set(), "objects": set(), "samples": []})
        all_samples = []
        
        for _, row in df_clean.iterrows():
            subject = row["主体实体"].strip()
            subject_type = row["实体类型"].strip()
            predicate = row["关系谓词"].strip()
            obj = row["客体实体"].strip()
            
            expert = ENTITY_TYPE_TO_EXPERT.get(subject_type, "概念")
            text = f"{subject} {predicate} {obj}"
            
            all_samples.append((text, expert))
            knowledge_base[expert]["subjects"].add((subject, subject_type))
            knowledge_base[expert]["predicates"].add(predicate)
            knowledge_base[expert]["objects"].add((obj, row["客体类型"].strip()))
            knowledge_base[expert]["samples"].append(text)
        
        category_counts = defaultdict(int)
        for _, expert in all_samples:
            category_counts[expert] += 1
        
        logger.info("✅ CSV训练集加载完成！")
        logger.info(f"总训练样本数: {len(all_samples)}")
        logger.info("训练集类别分布:")
        for expert, count in sorted(category_counts.items()):
            logger.info(f"  [{expert}]: {count} 条")
        
        # 视觉类无文本样本提示
        if category_counts.get("视觉", 0) == 0:
            logger.info("\nℹ️ 视觉类无文本训练样本（正常，视觉类仅通过图像训练）")
        
        return all_samples, knowledge_base
    
    except Exception as e:
        logger.error(f"❌ CSV文件读取失败: {str(e)}", exc_info=True)
        sys.exit(1)

# ====================== 🔴 生成随机合成测试集 ======================
def generate_synthetic_test_data(knowledge_base, test_size: int = 200):
    logger.info(f"\n🎲 正在生成 {test_size} 条随机合成测试数据...")
    
    test_cases = []
    expert_list = list(knowledge_base.keys())
    total_samples = sum(len(kb["samples"]) for kb in knowledge_base.values())
    expert_weights = [len(knowledge_base[exp]["samples"]) / total_samples for exp in expert_list]
    
    for _ in range(test_size):
        expert = random.choices(expert_list, weights=expert_weights, k=1)[0]
        kb = knowledge_base[expert]
        
        subject, _ = random.choice(list(kb["subjects"]))
        predicate = random.choice(list(kb["predicates"]))
        obj, _ = random.choice(list(kb["objects"]))
        
        test_cases.append((f"{subject} {predicate} {obj}", expert))
    
    category_counts = defaultdict(int)
    for _, expert in test_cases:
        category_counts[expert] += 1
    
    logger.info("✅ 合成测试集生成完成！")
    logger.info("测试集类别分布:")
    for expert, count in sorted(category_counts.items()):
        logger.info(f"  [{expert}]: {count} 条")
    
    return test_cases

# ====================== 🔴 通用工具函数 ======================
def encode_and_create_entities(samples, embedding_model):
    logger.info(f"🔤 正在批量生成文本向量编码（批量大小: {BATCH_ENCODE_SIZE}）...")
    
    entity_samples = []
    training_data = []
    failed_count = 0
    
    texts = [s[0] for s in samples]
    experts = [s[1] for s in samples]
    
    for i in range(0, len(texts), BATCH_ENCODE_SIZE):
        batch_texts = texts[i:i+BATCH_ENCODE_SIZE]
        batch_experts = experts[i:i+BATCH_ENCODE_SIZE]
        
        try:
            batch_embs = embedding_model.embed_documents(batch_texts)
            
            for text, expert, emb in zip(batch_texts, batch_experts, batch_embs):
                emb_tensor = torch.tensor(emb, dtype=torch.float32, device=DEVICE)
                entity = create_simple_entity(text, expert, emb_tensor)
                entity_samples.append((entity, expert))
                training_data.append((emb_tensor, expert))
                
        except Exception as e:
            failed_count += len(batch_texts)
            logger.warning(f"批量编码失败 第{i//BATCH_ENCODE_SIZE + 1}批 | 错误: {str(e)}")
            continue
        
        if (i + BATCH_ENCODE_SIZE) % 100 == 0:
            logger.info(f"处理进度: {min(i+BATCH_ENCODE_SIZE, len(texts))}/{len(texts)}")

    logger.info(f"✅ 处理完成！成功：{len(training_data)} 条，失败：{failed_count} 条")
    return entity_samples, training_data

def evaluate_router(router, embedding_model, test_cases, test_name="测试", save_confusion_matrix: bool = True):
    logger.info("="*50)
    logger.info(f"🧪 {test_name}")
    logger.info("="*50)

    y_true = []
    y_pred = []
    expert_to_idx = {expert: idx for idx, expert in enumerate(router.expert_names)}

    for text, expected in test_cases:
        try:
            emb = torch.tensor(embedding_model.embed_query(text), dtype=torch.float32, device=DEVICE)
            entity = create_simple_entity(text, expected, emb)
            
            expert = router.route(
                entity_embedding=emb,
                entities=[entity],
                is_encoding=False
            )
            
            y_true.append(expert_to_idx[expected])
            y_pred.append(expert_to_idx[expert])
            
            status = "✅ 正确" if expert == expected else "❌ 错误"
            logger.info(f"{status} | 内容：{text:40} → 路由：{expert:5} | 期望：{expected:5}")
        
        except Exception as e:
            logger.warning(f"评估失败 | 文本：{text} | 错误：{str(e)}", exc_info=True)

    if not y_true:
        logger.warning("⚠️ 没有有效的评估结果")
        return 0.0

    all_labels = list(range(len(router.expert_names)))
    
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    acc = correct / total if total > 0 else 0.0
    logger.info(f"\n📊 {test_name} 总体准确率：{acc:.1%} ({correct}/{total})")

    logger.info(f"\n📊 {test_name} 详细分类报告：")
    report = classification_report(
        y_true, y_pred, 
        labels=all_labels,
        target_names=router.expert_names,
        zero_division=0
    )
    logger.info("\n" + report)

    if save_confusion_matrix and len(router.expert_names) > 1:
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=router.expert_names,
            yticklabels=router.expert_names
        )
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.title(f'{test_name} 混淆矩阵')
        plt.tight_layout()
        cm_path = f'{test_name.replace(" ", "_")}_confusion_matrix.png'
        plt.savefig(cm_path, dpi=300)
        logger.info(f"📈 混淆矩阵已保存至: {cm_path}")
        plt.close()

    return acc

# ====================== 🔴 主训练流程（纯原生train函数版） ======================
if __name__ == "__main__":
    logger.info("="*60)
    logger.info("🧠 海马体路由预训练程序（极简原生版）")
    logger.info("="*60)

    # 加载Embedding模型
    logger.info("🤖 正在加载 Embedding 模型（bge-m3）...")
    try:
        from langchain_ollama import OllamaEmbeddings
        embedding_model = OllamaEmbeddings(
            model="bge-m3",
            base_url="http://localhost:11434",
            temperature=0.0
        )
        test_emb = embedding_model.embed_query("test")
        emb_dim = len(test_emb)
        logger.info(f"✅ 模型加载成功 | 向量维度：{emb_dim}")
    except ImportError:
        logger.error("❌ 缺少依赖：pip install langchain-ollama scikit-learn seaborn matplotlib pandas")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 模型加载失败：{e}")
        logger.error("💡 请启动Ollama，并执行：ollama pull bge-m3")
        sys.exit(1)

    # 初始化路由模型（纯原生参数）
    expert_names = ["身份", "概念", "空间", "抽象", "视觉"]
    router = HippocampusRouter(
        input_dim=emb_dim,
        expert_names=expert_names,
        experts={}
    )

    # 加载已有权重
    save_dir = Path("brain_benchmark_test")
    save_dir.mkdir(exist_ok=True)
    best_model_path = save_dir / "hippocampus_router_best.pt"
    latest_model_path = save_dir / "hippocampus_router_latest.pt"
    
    if latest_model_path.exists():
        try:
            router.load(str(latest_model_path))
            logger.info("✅ 已有实体中心权重加载成功")
        except Exception as e:
            logger.warning(f"⚠️ 旧版本权重加载失败，将重新训练: {e}")

    # 加载数据
    csv_path = ROOT_DIR / "data_text" / "all_know.csv"
    all_training_samples, knowledge_base = load_training_data_from_csv(csv_path)
    synthetic_test_cases = generate_synthetic_test_data(knowledge_base, GENERATED_TEST_SIZE)
    
    # 划分训练集/验证集
    train_samples, val_samples = train_test_split(
        all_training_samples, 
        test_size=0.15, 
        random_state=42, 
        stratify=[s[1] for s in all_training_samples]
    )
    logger.info(f"\n📊 最终数据集划分:")
    logger.info(f"  - 训练集: {len(train_samples)} 条")
    logger.info(f"  - 验证集: {len(val_samples)} 条")
    logger.info(f"  - 合成测试集: {len(synthetic_test_cases)} 条")

    # 编码数据
    train_entity_samples, train_training_data = encode_and_create_entities(train_samples, embedding_model)
    val_entity_samples, val_training_data = encode_and_create_entities(val_samples, embedding_model)

    # 初始化专家原型
    logger.info("🧭 正在基于训练集实体样本初始化专家原型向量...")
    train_entities = [entity for entity, _ in train_entity_samples]
    router._initialize_prototypes_with_entities(train_entities)

    # 🔥 纯原生训练流程（只调用你有的router.train）
    logger.info("\n" + "="*60)
    logger.info("🚀 开始训练（5分类实体驱动模型）")
    logger.info("="*60)
    
    best_val_acc = 0.0
    no_improvement_count = 0
    
    # 逐轮训练+评估（完全适配你只有train函数的情况）
    for current_epoch in range(0, TOTAL_EPOCHS, EVAL_INTERVAL):
        # 训练EVAL_INTERVAL轮
        logger.info(f"\n📅 训练轮次: {current_epoch+1} ~ {current_epoch+EVAL_INTERVAL}")
        router.train(
            training_data=train_training_data,
            epochs=EVAL_INTERVAL,
            batch_size=16,
            log_interval=10  # 每10轮打印一次训练日志
        )
        
        # 评估验证集
        val_acc = evaluate_router(
            router, embedding_model, val_samples, 
            f"验证集 Epoch {current_epoch+EVAL_INTERVAL}", 
            save_confusion_matrix=False
        )
        
        # 保存最新模型
        router.save(str(latest_model_path))
        logger.info(f"💾 最新模型已保存至: {latest_model_path}")
        
        # 早停逻辑
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improvement_count = 0
            router.save(str(best_model_path))
            logger.info(f"🏆 新的最佳验证准确率: {best_val_acc:.1%}，最佳模型已保存")
        else:
            no_improvement_count += EVAL_INTERVAL
            logger.info(f"⏳ 验证准确率未提升，连续无提升轮数: {no_improvement_count}/{PATIENCE}")
            
            if no_improvement_count >= PATIENCE:
                logger.info(f"🛑 早停触发！连续{PATIENCE}个epoch无提升")
                break

    # 加载最佳模型进行最终评估
    logger.info("\n" + "="*60)
    logger.info("🏁 训练完成，加载最佳模型进行全面评估")
    logger.info("="*60)
    router.load(str(best_model_path))

    # 最终评估
    train_acc = evaluate_router(router, embedding_model, train_samples, "训练集最终测试")
    val_acc = evaluate_router(router, embedding_model, val_samples, "验证集最终测试")
    synthetic_acc = evaluate_router(
        router, embedding_model, synthetic_test_cases, 
        "随机合成测试集评估", save_confusion_matrix=True
    )

    # 保存结果
    with open(save_dir / "training_results.txt", "w", encoding="utf-8") as f:
        f.write("海马体路由训练结果\n")
        f.write("="*50 + "\n")
        f.write(f"训练集准确率: {train_acc:.1%}\n")
        f.write(f"验证集准确率: {val_acc:.1%}\n")
        f.write(f"随机合成测试集准确率: {synthetic_acc:.1%}\n")
        f.write(f"最佳模型路径: {best_model_path}\n")
        f.write(f"训练总轮数: {current_epoch+EVAL_INTERVAL}\n")
        f.write(f"生成测试集大小: {GENERATED_TEST_SIZE}\n")

    logger.info("\n" + "="*60)
    logger.info("✅ 所有训练和测试完成！")
    logger.info(f"🏆 最佳验证准确率: {best_val_acc:.1%}")
    logger.info(f"📊 随机合成测试集准确率: {synthetic_acc:.1%}")
    logger.info(f"💾 最佳模型已保存至: {best_model_path}")
    logger.info("="*60)