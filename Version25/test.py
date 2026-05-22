import json
import os
from BrainConfig import config

# 加载皮层记忆索引
index_file = os.path.join(config.storage_dir, "cortex_memory_index.json")
with open(index_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 修复所有记忆的元数据
for mem_id_str, mem in data["memories"].items():
    meta = mem["metadata"]
    content = mem["content"]
    
    # 1. 修复专家字段
    if content.startswith("名言："):
        meta["expert"] = "抽象"
        meta["subject"] = content[3:].split("——")[0].strip()
    elif content.startswith("知识："):
        meta["expert"] = "概念"
        meta["subject"] = content[3:].split("是")[0].strip()
    elif content.startswith("人物："):
        meta["expert"] = "概念"
        meta["subject"] = content[3:].split("，")[0].strip()
    elif content.startswith("事件："):
        meta["expert"] = "空间"
        meta["subject"] = content[3:].split("，")[0].strip()
    elif content.startswith("身份："):
        meta["expert"] = "身份"
        meta["subject"] = "我"
    
    print(f"修复记忆 {mem_id_str}: 专家={meta['expert']} | 主体={meta['subject']} | 内容={content[:30]}...")

# 保存修复后的索引
with open(index_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ 所有旧记忆元数据修复完成！")