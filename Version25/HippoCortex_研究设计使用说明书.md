# 🧠 HippoCortex 类脑神经网络系统
## 研究设计与使用说明书

---

## 一、系统概述

### 1.1 项目背景
HippoCortex 是一个受神经科学启发的**类脑人工智能系统**，模拟人脑海马体-皮层记忆系统的信息处理机制。系统通过仿生神经网络实现**持续学习、记忆巩固、多模态感知和自主认知**等高级智能功能。

### 1.2 核心设计理念
| 脑区原型 | 系统模块 | 功能对应 |
|---------|---------|---------|
| 海马体 (Hippocampus) | `HippocampusRouter` | 快速编码、模式分离、路由决策 |
| 大脑皮层 (Cortex) | `PersistentCortex` + `EntityIndex` | 长期记忆存储、语义检索 |
| 丘脑 (Thalamus) | `Thalamus` | 信息过滤、注意力调控、中继站 |
| 多巴胺系统 | `DopamineSystem` | 奖励预测误差、突触可塑性调节 |
| 专家网络 | `DynamicExpert` × 5 | 分区专业化处理（身份/概念/空间/抽象/视觉）|
| 稀疏编码 | `LearnableSparseEncoder` | SDR (Sparse Distributed Representation) |

### 1.3 系统规模
- **总代码量**: 10,612 行 Python 代码
- **核心模块**: 34 个 Python 文件
- **神经网络参数**: 可扩展至 8192 维 SDR 空间
- **记忆容量**: 理论无上限（基于 FAISS 向量索引）

---

## 二、系统架构

### 2.1 分层架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    【应用层】                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  LLMBrainWrapper │  │  ChatThread │  │  XiaobaiBrainPet    │  │
│  │  (对话封装)   │  │  (Qt线程)   │  │  (PyQt5桌宠UI)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    【接口层】                                │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              BrainInterface (大脑接口)                │  │
│  │  ├─ chat()        文本对话                           │  │
│  │  ├─ learn_text()   学习文本记忆                       │  │
│  │  ├─ process_image() 处理图像记忆                      │  │
│  │  ├─ trigger_sleep() 触发睡眠巩固                     │  │
│  │  └─ load_plugins()  加载扩展插件                     │  │
│  └─────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    【认知系统层】                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │           CognitiveSystem (认知系统外观)             │  │
│  │  ├─ think()              思考推理                    │  │
│  │  ├─ learn()              学习记忆                    │  │
│  │  ├─ batch_learn()        批量导入                    │  │
│  │  ├─ sleep_consolidate_all() 全脑睡眠巩固             │  │
│  │  └─ generate_dream()     梦境生成                    │  │
│  └─────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    【服务容器层】                          │
│  ┌─────────────────────────────────────────────────────┐  │
│  │           ServiceContainer (IoC容器)                 │  │
│  │  管理 17 个核心服务的生命周期：初始化→启动→停止→保存  │  │
│  └─────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    【核心循环层】                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │ PerceptionLoop │ │ LearningLoop │ │ Consolidation│      │
│  │   (感知循环)   │ │   (学习循环)  │ │   (巩固循环)  │      │
│  └──────────────┘ └──────────────┘ └──────────────┘      │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │ DreamingLoop │ │ ThinkEngine  │ │ CrossModal   │      │
│  │   (梦境循环)   │ │   (思考引擎)  │ │   (跨模态桥)  │      │
│  └──────────────┘ └──────────────┘ └──────────────┘      │
├─────────────────────────────────────────────────────────────┤
│                    【神经计算层】                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐  │
│  │DynamicExpert│ │LearnableSparse│ │ CognitiveEnergy    │  │
│  │  (动态专家)  │ │  Encoder    │ │    Field           │  │
│  │  × 5 个专家 │ │  (稀疏编码)  │ │  (认知能量场)        │  │
│  └─────────────┘ └─────────────┘ └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    【记忆存储层】                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐  │
│  │EntityIndex  │ │Hippocampus  │ │ ConversationMemory   │  │
│  │ (实体索引)   │ │  Router     │ │   (对话记忆)         │  │
│  │ FAISS向量库  │ │ (海马体路由) │ │ 艾宾浩斯衰减         │  │
│  └─────────────┘ └─────────────┘ └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    【基础组件层】                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │  BrainCore   │ │   Thalamus   │ │ SymbolicCore│          │
│  │  (大脑核心)  │ │   (丘脑)     │ │  (符号核心)  │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │DopamineSystem│ │Metacognition │ │  Curiosity  │          │
│  │ (多巴胺系统)  │ │  (元认知)    │ │  (好奇心)    │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │  VAEManager  │ │MultiModal    │ │  EventBus   │          │
│  │  (VAE管理)   │ │  Gateway     │ │  (事件总线)  │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 数据流架构

```
用户输入/图像
    │
    ▼
┌─────────────────┐
│  Thalamus       │ ←── 信息过滤 (显著性计算)
│  (丘脑中继)     │ ←── 注意力调控
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ 文本编码 │ │ 图像编码 │
│(BGE-M3) │ │(CLIP/VAE)│
└───┬────┘ └───┬────┘
    │          │
    ▼          ▼
┌─────────────────┐
│ HippocampusRouter│ ←── 实体提取 (LLM驱动)
│  (海马体路由)    │ ←── 专家分配决策
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│海马体缓冲区│ │ 专家网络  │
│(短期记忆) │ │ (SDR编码) │
└───┬────┘ └───┬────┘
    │          │
    ▼          ▼
┌─────────────────┐
│ PersistentCortex │ ←── FAISS向量索引
│  (皮层长期记忆)  │ ←── 实体关系图谱
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLMBrainWrapper │ ←── 上下文组装
│  (回复生成)      │ ←── 记忆增强Prompt
└─────────────────┘
```

---

## 三、核心模块详解

### 3.1 记忆系统 (Memory System)

#### 3.1.1 实体中心记忆模型
系统采用**实体中心 (Entity-Centric)** 架构，所有知识以 `Entity` 对象为核心组织：

```python
class Entity(BaseDataModel):
    entity_id: str      # 全局唯一ID
    name: str           # 实体名称
    entity_type: str    # 类型: person/place/concept/visual/...
    sdr: torch.Tensor   # 稀疏分布式表示 (神经检索)
    clip_vec: torch.Tensor  # 语义向量 (语义检索)
    importance: float   # 重要性 (0-1)
    is_permanent: bool  # 永久记忆标记
    evidences: List[Evidence]  # 支持证据链
    expert: str         # 所属专家分区
```

#### 3.1.2 三级记忆体系
| 记忆类型 | 存储位置 | 容量 | 保持时间 | 编码方式 |
|---------|---------|------|---------|---------|
| **感觉记忆** | Thalamus过滤器 | 瞬时 | 毫秒级 | 向量显著性计算 |
| **短期记忆** | Hippocampal Buffer | 20条 | 分钟-小时 | SDR快速编码 |
| **长期记忆** | PersistentCortex + Expert | 无上限 | 永久 | 突触权重+向量索引 |

#### 3.1.3 记忆生命周期
```
输入 → 丘脑过滤 → 海马体编码 → 专家SDR学习 → 皮层存储
                                          ↓
                                    睡眠巩固 (回放+STDP)
                                          ↓
                                    长期存储 + 突触修剪
```

### 3.2 神经计算系统 (Neural Computation)

#### 3.2.1 动态专家网络 (DynamicExpert)
基于 **STDP (Spike-Timing-Dependent Plasticity)** 的脉冲神经网络：

```python
class DynamicExpert(nn.Module):
    # 核心参数
    dim: int = 2048              # SDR维度
    synapse: nn.Parameter        # 突触权重矩阵 (dim×dim)

    # STDP学习规则
    tau_plus: float = 20.0       # LTP时间窗口
    tau_minus: float = 20.0      # LTD时间窗口
    A_plus: float = 0.01         # 增强幅度
    A_minus: float = 0.012       # 抑制幅度
```

**学习机制**：
- **Hebbian学习**: 同步激活的神经元连接增强
- **STDP时序学习**: 脉冲先后顺序决定权重增减
- **预测性STDP**: 基于预测误差调节可塑性
- **突触修剪**: 定期移除弱连接 (10%分位数)
- **突触新生**: 高共激活神经元对建立新连接

#### 3.2.2 稀疏编码器 (LearnableSparseEncoder)
实现 **竞争性稀疏激活**：

```
输入向量 (1024d)
    │
    ▼
┌─────────────────┐
│  Encoder (MLP)   │
│  1024 → 4096 → 2048 │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│核心偏置 │ │ 侧抑制  │
│(+0.5)  │ │ (WTA)   │
└───┬────┘ └───┬────┘
    │          │
    ▼          ▼
┌─────────────────┐
│ Top-K 硬激活    │ ←── 只保留60个最强激活位
│ Softmax 软激活  │ ←── 梯度传播用
│ 直通估计器      │ ←── 反向传播优化
└─────────────────┘
    │
    ▼
SDR (2048d, 60位激活)
```

### 3.3 认知增强系统 (Cognitive Enhancement)

#### 3.3.1 多巴胺奖励系统
基于 **Rescorla-Wagner 模型** 的奖励预测误差 (RPE) 计算：

```
RPE = 实际奖励 - 期望奖励

多巴胺浓度更新:
  如果 RPE > 0: 浓度上升 (意外奖励)
  如果 RPE < 0: 浓度下降 (预测失误)

突触可塑性门控:
  |RPE| > 0.05 时才触发显著权重变化
```

**多维度奖励源**：
- 好奇心奖励 (信息增益)
- 预测奖励 (预测准确率)
- 实体发现奖励 (新知识)
- 关系建立奖励 (结构完善)
- 外部反馈奖励 (用户评分)
- 情绪奖励 (情感效价)

#### 3.3.2 元认知监控
```python
class Metacognition:
    # 知识置信度评估 (四维度)
    confidence = (
        0.35 * activation_strength +      # 激活强度
        0.25 * association_count +        # 关联数量
        0.25 * consistency +              # 一致性
        0.15 * time_decay                # 时间衰减 (艾宾浩斯曲线)
    )

    # 自适应功能
    - 承认无知判断 (confidence < 0.2)
    - 学习优先级排序 (倒U型曲线)
    - 复习提醒触发
```

### 3.4 睡眠巩固系统 (Sleep Consolidation)

#### 3.4.1 三阶段睡眠模型
```
┌─────────────────────────────────────────────┐
│ 阶段1: 浅睡 (Light Sleep)                    │
│ ├── 整理重要对话记忆                         │
│ └── 标记待巩固实体                           │
├─────────────────────────────────────────────┤
│ 阶段2: 深睡 (Deep Sleep)                     │
│ ├── 海马体→皮层转移 (回放)                   │
│ ├── 专家网络STDP巩固                         │
│ ├── 突触修剪与新生                           │
│ ├── 跨模态关联强化                           │
│ └── 多巴胺离线重放                           │
├─────────────────────────────────────────────┤
│ 阶段3: REM睡眠 (Rapid Eye Movement)          │
│ ├── 神经随机激活 → 实体联想                  │
│ ├── 梦境片段提取                             │
│ ├── LLM重构连贯梦境                          │
│ └── 梦境记忆存储                             │
└─────────────────────────────────────────────┘
```

#### 3.4.2 睡眠质量评估
综合评分维度：
- 巩固率 (35分): 成功巩固记忆比例
- 遗忘健康度 (25分): 最佳遗忘率 10-20%
- 跨模态关联 (10分): 多模态配对巩固数
- 突触重塑 (10分): 修剪+新生平衡
- 能量消耗 (20分): 最佳范围 5-15

---

## 四、数据模型规范

### 4.1 核心数据契约

| 模型 | 用途 | 关键字段 |
|------|------|---------|
| `Entity` | 知识原子单位 | entity_id, name, type, sdr, clip_vec, importance, expert |
| `EntityRelation` | 实体间关系 | relation_id, subject_id, predicate, object_id, synapse_weight |
| `Evidence` | 全文本证据 | evidence_id, content, source, sdr, clip_vec, confidence |
| `ConversationTurn` | 对话轮次 | turn_id, user_input, ai_response, activation, is_important |
| `SleepReport` | 睡眠报告 | stages, consolidated_count, quality_score, dream_content |
| `DreamResult` | 梦境生成 | success, content, fragments, dominant_experts |

### 4.2 序列化规范
- **格式**: JSON + PyTorch张量
- **版本**: V3.0 (实体中心格式)
- **兼容性**: 自动识别并迁移旧版本数据
- **原子写入**: temp文件 → replace，防止文件损坏

---

## 五、配置系统

### 5.1 核心配置参数 (`BrainConfig`)

```python
class BrainConfig(BaseSettings):
    # 向量维度
    dim: int = 1024              # BGE-M3 原生维度
    sdr_dim: int = 2048          # SDR空间维度
    sdr_active_size: int = 60    # 稀疏激活位数

    # 专家配置
    expert_names = ["身份", "视觉", "概念", "空间", "抽象"]
    max_expert_dim: int = 8192

    # 检索配置
    top_k: int = 10
    min_similarity: float = 0.4

    # LLM配置
    ollama_model_name: str = "gemma3:4b"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 256

    # 认知状态
    mind_wandering_enabled: bool = True
    fatigue_sleep_threshold: float = 0.85

    # 专家STDP差异化配置
    EXPERT_CONFIG = {
        "身份": {"sparsity": 0.05, "tau_plus": 25.0, "A_plus": 0.015},
        "概念": {"sparsity": 0.02, "tau_plus": 20.0, "A_plus": 0.01},
        "空间": {"sparsity": 0.03, "tau_plus": 30.0, "A_plus": 0.012},
        "抽象": {"sparsity": 0.025, "tau_plus": 15.0, "A_plus": 0.018},
        "视觉": {"sparsity": 0.04, "tau_plus": 10.0, "A_plus": 0.02}
    }
```

---

## 六、使用指南

### 6.1 环境准备

```bash
# 1. 安装依赖
pip install torch numpy faiss-cpu pydantic langchain-ollama 
pip install transformers diffusers pillow PyQt5

# 2. 启动 Ollama 服务
ollama pull gemma3:4b
ollama pull bge-m3
ollama pull nomic-embed-text

# 3. 准备模型路径 (修改 BrainConfig)
# vae_model_path: 本地 sd-vae-ft-mse 路径
# CLIP_MODEL_PATH: 本地 clip-vit-large-patch14 路径
```

### 6.2 快速启动

```python
from BrainConfig import config
from brain_interface import BrainInterface
from langchain_ollama import ChatOllama, OllamaEmbeddings

# 1. 初始化模型
llm = ChatOllama(model=config.ollama_model_name)
embedding_model = OllamaEmbeddings(model="bge-m3")

# 2. 创建大脑
brain = BrainInterface(embedding_model, llm, kg_enabled=True)
brain.start(storage_dir="my_brain")

# 3. 学习知识
brain.learn_text("身份：我是小明，今年25岁，住在北京")
brain.learn_text("我喜欢吃苹果和香蕉")

# 4. 图像学习
brain.process_image("photo.jpg", description="这是我家的小狗")

# 5. 对话
response = brain.chat("我是谁？")
print(response)

# 6. 睡眠巩固
report = brain.trigger_sleep(is_manual=True)
print(f"睡眠质量: {report.quality_rating}")

# 7. 关闭
brain.stop()
```

### 6.3 批量知识导入

```python
from MainTest5 import import_knowledge_dataset

# CSV格式: 主体实体,关系谓词,客体实体
import_knowledge_dataset(
    llm_brain, 
    dataset_path="knowledge.csv",
    flag_path="imported.flag",
    use_kg=False  # 导入时关闭KG提升速度
)
```

### 6.4 交互式命令

在 `MainTest5.py` 交互模式下支持：

| 命令 | 功能 |
|------|------|
| `sleep` | 手动触发睡眠巩固 |
| `analyze` | 分析专家突触结构热力图 |
| `add_entity <名字>` | 添加重要实体 |
| `remove_entity <名字>` | 删除重要实体 |
| `list_entities` | 列出所有重要实体 |
| `enable_kg` / `disable_kg` | 知识图谱开关 |
| `exit` | 退出并保存 |

---

## 七、测试与验证

### 7.1 基准测试 (`BrainBenchmark`)

四阶段端到端验证：

```
测试1: 学习身份 → 验证海马体缓冲区
测试2: 睡眠巩固 → 验证皮层转移
测试3: 查询身份 → 验证长期记忆检索
测试4: 多轮对话 → 验证上下文关联
```

**通过标准**: 所有测试通过 → 可安全进入 Phase 2 开发

### 7.2 性能监控

```python
# 获取大脑状态
status = brain.get_status()
print(status)
# {
#   "is_running": True,
#   "fatigue_level": 0.15,
#   "is_mind_wandering": False,
#   "intention_queue_size": 0
# }
```

---

## 八、扩展开发

### 8.1 插件系统

```python
from plugin_system import PerceptionPlugin

class VoicePlugin(PerceptionPlugin):
    def get_modality(self): return "voice"
    def perceive(self):
        # 实现语音感知逻辑
        return audio_data

# 加载插件
brain.load_plugins(["voice_plugin.py"])
brain.perceive_from_modality("voice")
```

### 8.2 自定义专家

在 `ComponentFactory.create_experts()` 中添加新专家类型，并配置对应的 `EXPERT_CONFIG` 参数。

---

## 九、技术亮点

| 特性 | 实现机制 | 科学基础 |
|------|---------|---------|
| **仿生记忆巩固** | 三阶段睡眠 + STDP回放 | 海马体索引理论 (Teyler & DiScenna) |
| **稀疏分布式表示** | 竞争性WTA + 侧抑制 | 大脑皮层稀疏编码 (Olshausen & Field) |
| **多巴胺强化学习** | RPE驱动的突触可塑性 | 奖励预测误差假说 (Schultz et al.) |
| **跨模态绑定** | 文本-视觉 SDR 关联学习 | 多感官整合理论 (Stein & Stanford) |
| **元认知监控** | 置信度四维度评估 | 元认知理论 (Nelson & Narens) |
| **梦境生成** | 神经随机激活 + LLM重构 | 激活-合成假说 (Hobson & McCarley) |
| **实体中心架构** | 知识图谱 + 神经向量双索引 | 语义网络理论 (Collins & Loftus) |

---

## 十、版本历史

| 版本 | 日期 | 重大变更 |
|------|------|---------|
| V1.0 | - | 基础记忆存储与检索 |
| V2.0 | - | 引入专家网络与STDP |
| V3.0 | 当前 | **实体中心架构重构**，统一数据契约，跨模态增强 |

---

**文档版本**: V3.0  
**生成日期**: 2026-05-22  
**系统代号**: HippoCortex  
**总代码规模**: 10,612 行 | 34 个模块 | 5 大子系统
