import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import datetime
from transformers import (
    CLIPTokenizer, CLIPTextModel,
    AutoTokenizer, AutoModelForCausalLM
)
from typing import List, Tuple, Optional
import numpy as np
from collections import deque, defaultdict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from AdvancedBrainV2 import AdvancedBrainV2
from LLMBrainWrapper import LLMBrainWrapper

import shutil
import json
import os

if __name__ == "__main__":
    import shutil
    import json
    import os

    storage_dir = "brain_v3_demo"
    # os.makedirs(storage_dir, exist_ok=True)

    # # # 清理旧数据
    # if os.path.exists("brain_v3_demo"):
    #     shutil.rmtree("brain_v3_demo")

    # 1. 初始化类脑记忆系统
    # print("=" * 60)
    # print("🧠 初始化类脑记忆系统...")
    # print("=" * 60)
    # brain = AdvancedBrainV2(dim=768, storage_dir="brain_v2_demo")

    # # 2. 包装 LLM 增强层
    # print("\n" + "=" * 60)
    # llm_brain = LLMBrainWrapper(brain, ollama_model_name='gemma3:4b')
    # print("=" * 60)

    # 3. 从外部 JSON 加载福尔摩斯数据集
    # print("\n" + "=" * 60)
    # print("📚 加载福尔摩斯数据集 JSON...")
    # print("=" * 60)

    # dataset_path = "sherlock_holmes_dataset_chinese.json"
    # if not os.path.exists(dataset_path):
    #     print(f"❌ 未找到文件: {dataset_path}")
    #     exit(1)

    # with open(dataset_path, "r", encoding="utf-8") as f:
    #     holmes = json.load(f)["sherlock_holmes_dataset"]

    # # 构造学习文本列表
    # samples = []

    # # 基本信息
    # basic = holmes["basic_information"]
    # samples.append(f"人物: {basic['chinese_name']}({basic['full_name']})，职业{basic['occupation']}，住址{basic['residence']}，作者{basic['creator']}")
    # samples.append(f"技能: 福尔摩斯擅长{', '.join(basic['special_skills'])}")
    # samples.append(f"案件: 福尔摩斯首案《{basic['first_case']}》，终案《{basic['last_case']}》")

    # # 时间线
    # for item in holmes["timeline"]:
    #     samples.append(f"事件: {item['year']}年 {item['event']}")

    # # 经典案件
    # for case in holmes["classic_cases"]:
    #     samples.append(f"案件: {case['case_name']}，类型{case['type']}，关键线索{case.get('clue', '')}，凶手{case.get('perpetrator', case.get('solution', ''))}")

    # # 主要人物
    # for char in holmes["characters"]:
    #     samples.append(f"人物: {char['name']}，关系{char['relation']}，身份{char['role']}")

    # # 名言
    # for q in holmes["famous_quotes"]:
    #     samples.append(f"名言: {q['chinese']} —— 福尔摩斯")

    # # 推理方法
    # for m in holmes["deduction_methods"]:
    #     samples.append(f"方法: {m['method']}，{m['description']}")

    # # 开始学习
    # print("\n📚 开始学习福尔摩斯知识...")
    # for text in samples:
    #     llm_brain.learn(text)

    # # 4. 测试问答（福尔摩斯主题）
    # print("\n" + "=" * 60)
    # print("💬 开始福尔摩斯问答测试...")
    # print("=" * 60)

    # queries = [
    #     "福尔摩斯会什么技能?",
    #     "斑点带子案的凶手是谁?",
    #     "华生是谁？",
    #     "莫里亚蒂是什么人？",
    #     "福尔摩斯的名言是什么？",
    #     "巴斯克维尔的猎犬是怎么回事？"
    #     # "What skills does Sherlock Holmes have？",
    #     # "Who is the murderer in The Adventure of the Speckled Band?",
    #     # "Who is John Watson?",
    #     # "Who is James Moriarty?",
    #     # "What are Sherlock Holmes' famous quotes?",
    #     # "What is the story of The Hound of the Baskervilles?"
    # ]

    # for q in queries:
    #     answer = llm_brain.ask(q)
    #     print(f"\n{answer}")

    # 1. 初始化类脑记忆系统
    print("=" * 60)
    print("🧠 初始化类脑记忆系统...")
    print("=" * 60)
    brain = AdvancedBrainV2(dim=768, storage_dir=storage_dir)

    # 2. 包装 LLM 增强层
    print("\n" + "=" * 60)
    llm_brain = LLMBrainWrapper(brain, ollama_model_name='gemma3:4b')
    print("=" * 60)

    # 3. 一键导入通用知识数据集（仅首次运行）
    # dataset_path = "general_knowledge.txt"
    # first_run_flag = os.path.join(storage_dir, "general_knowledge_imported")
    
    # if os.path.exists(dataset_path) and not os.path.exists(first_run_flag):
    #     print("\n" + "=" * 60)
    #     print("📚 首次运行，正在导入通用知识数据集...")
    #     print("=" * 60)
        
    #     from tqdm import tqdm
        
    #     # 解析TXT文件
    #     with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
    #         lines = f.readlines()
        
    #     knowledge_list = []
    #     for line in lines:
    #         line = line.strip()
    #         if not line or line.startswith('#'):
    #             continue
    #         knowledge_list.append(line)
        
    #     print(f"✅ 解析完成，共找到 {len(knowledge_list)} 条知识")
        
    #     # 批量导入
    #     print(f"\n📚 开始导入...")
    #     success_count = 0
    #     for knowledge in tqdm(knowledge_list, desc="导入进度"):
    #         try:
    #             llm_brain.learn(knowledge)
    #             success_count += 1
    #         except Exception as e:
    #             print(f"\n❌ 导入失败: {knowledge[:50]}... 错误: {e}")
        
    #     # 保存并创建标记文件
    #     print("\n💾 正在保存...")
    #     brain.cortex.save()
    #     with open(first_run_flag, "w") as f:
    #         f.write("通用知识数据集已导入")
        
    #     print(f"\n🎉 导入完成！成功: {success_count} 条")
    # else:
    #     print("\n✅ 通用知识数据集已导入，跳过")

    # 4. 显示初始大脑状态
    print("\n" + "=" * 60)
    print("📊 初始大脑状态")
    print("=" * 60)
    status = brain.get_brain_status()
    print(json.dumps(
        status, indent=2, ensure_ascii=False,
        default=lambda x: x.item() if hasattr(x, 'item') else x
    ))

    # 4. 交互式对话循环
    print("\n" + "=" * 60)
    print("💬 进入对话模式 (输入 'exit' 退出)")
    print("=" * 60)

    try:
        while True:
            user_input = input("\n你: ").strip()
            if user_input.lower() == "exit":
                # ==============================================
                # 🔥 强制在break之前立即保存所有记忆
                # ==============================================
                print("\n" + "=" * 60)
                print("💾 正在保存所有记忆...")
                print("=" * 60)
                
                # 1. 保存皮层记忆索引（最重要！）
                brain.cortex.save()
                print("✅ 皮层记忆索引已保存")
                
                # 2. 保存所有专家权重
                for name, exp in brain.experts.items():
                    exp.save_weights(os.path.join(storage_dir, f"{name}_weights.pt"))
                print("✅ 专家权重已保存")
                
                # 3. 保存SDR编码器
                brain.hippo.encoder.save(os.path.join(storage_dir, "sdr_encoder.pt"))
                print("✅ SDR编码器已保存")
                
                # 4. 保存海马体路由
                brain.hippo.save_projections()
                print("✅ 海马体路由已保存")
                
                print("\n✅ 所有记忆已安全保存！")
                print("\n再见！下次再聊~")
                break
                
            if not user_input:
                print("请输入你的问题")
                continue
                
            answer = llm_brain.ask(user_input)
            print(f"\n{answer}")

    except KeyboardInterrupt:
        # 处理Ctrl+C强制退出，也自动保存
        print("\n\n⚠️ 检测到强制退出，正在紧急保存记忆...")
        brain.cortex.save()
        print("✅ 记忆已紧急保存，再见！")
    
    # 5. 显示最终大脑状态
    print("\n" + "=" * 60)
    print("📊 最终大脑状态")
    print("=" * 60)
    status = brain.get_brain_status()
    print(json.dumps(
        status, indent=2, ensure_ascii=False,
        default=lambda x: x.item() if hasattr(x, 'item') else x
    ))