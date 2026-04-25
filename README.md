# Brain-Inspired LLM Long-Term Memory System
## 类脑启发式大模型长期记忆系统







> 浙大城市学院研究生个人兴趣探索项目
> 原创创意驱动，大模型辅助工程实现，纯学习与科研实验用途

---

## 📌 项目简介（真实、客观、不夸大）

本项目由本人**独立提出完整创意与顶层设计思路**，
聚焦**类脑智能、仿生认知、大模型长期记忆**前沿方向，
参考人脑「海马体–大脑皮层」认知工作机制，
搭建了一套**实验性仿生长期记忆系统**。

### 真实项目定位（重要）
1. **核心创意、整体架构思想、研究方向、功能规划均为本人原创**
2. **工程代码编写、底层逻辑实现、问题排查、迭代优化由大模型辅助完成**
3. 本人全程主导需求设计、方案筛选、效果验收、系统调试与项目推进
4. 本项目为**个人学习型、实验型科研探索项目**，用于研究生技术积累与类脑AI入门研究

本项目**不宣称完全手工自研算法**，如实采用「人类创意设计 + AI工程辅助」的现代AI开发模式。

### 目前最新版本为Version5

---

## ✨ 项目核心亮点

本系统模拟人脑生物认知机制，区别于传统粗暴的向量RAG：

### 1. 海马体智能路由机制
模拟人脑海马体信息筛选功能，自动对用户问题进行认知分类，
智能分发至不同认知专家分区，解决传统RAG知识混杂、检索混乱问题。

### 2. 多专家认知皮层分区
构建四大仿生认知分区，模拟大脑不同功能区域：
- 概念专家：负责人物、身份、名称、实体信息
- 空间专家：负责事件、时间、地点、历史信息
- 抽象专家：负责知识、名言、定义、理论内容
- 视觉专家：预留视觉图像认知接口

### 3. 仿生赫布联想学习
模拟人脑突触可塑性，高频关联记忆自动强化连接，
实现**越用越牢、相关记忆自动联想**的类脑效果。

### 4. 稀疏神经编码 SDR
采用仿生稀疏表征方式存储记忆，
降低内存占用、提升抗干扰能力、模拟人脑稀疏激活特性。

### 5. 睡眠记忆巩固机制
模拟人脑夜间睡眠修剪机制：
自动修剪无效弱突触、强化重要记忆、梳理知识结构。

### 6. 分层记忆演化机制
实现记忆生命周期管理：
短期记忆 → 长期记忆 → 永久记忆标记
同时搭载自然记忆衰减，模拟人类「正常遗忘」能力。

---

## 📊 系统运行效果

系统已完整落地并稳定运行，当前实验数据：
- 总记忆容量：1008 条有效仿生记忆
- 自动分区分布合理：抽象571 / 概念253 / 空间184
- 路由准确率高，可精准匹配对应认知专家
- 支持记忆自动修复、索引重建、睡眠巩固、突触修剪
- 全程本地部署，基于 Ollama 离线运行，无需云端API

具备完整日志输出、突触稀疏度统计、大脑状态可视化能力。

---

## 🧠 学术参考与研究对标

本项目学习、参考国内顶尖类脑团队研究方向：
- 清华大学类脑计算研究中心：联想记忆、突触可塑性、稀疏神经计算
- 浙江大学计算机学院：AGI认知架构、智能路由、分层记忆系统
- 中科院自动化所：类脑智能、长期记忆机制、认知仿生系统

同时参考国际前沿研究：
`LLM long-term memory`、`hippocampus routing`、`Hebbian learning AI`、`sparse coding cognition`

---

## 🚀 快速部署运行

### 1. 环境依赖
```bash
pip install torch faiss-cpu numpy langchain-ollama

## 模型架构图
graph TD
    %% 外部输入层
    UserInput[用户输入] --> Embedding[Ollama Embedding 模型]
    Embedding --> |维度: 1024| InputVector(输入向量)

    %% 核心中枢
    subgraph "中枢认知层 (AdvancedBrainV5)"
        Hippo[HippocampusRouterV4<br/>海马体路由]
        Cortex[PersistentCortexV5<br/>皮层与向量索引]
    end

    %% 专家网络层
    subgraph "专家网络 (DynamicExpertV3)"
        ExpVis[视觉专家]
        ExpCon[概念专家]
        ExpSpa[空间专家]
        ExpAbs[抽象专家]
    end

    %% 核心逻辑流
    InputVector --> |分类与权重| Hippo
    Hippo --> |路由/分配| ExpVis
    Hippo --> |路由/分配| ExpCon
    Hippo --> |路由/分配| ExpSpa
    Hippo --> |路由/分配| ExpAbs

    %% 存储与反馈流
    ExpVis & ExpCon & ExpSpa & ExpAbs <--> |Heabbian Update| Cortex
    Cortex <--> |FAISS检索| UserInput

    %% LLM生成层
    MemoryContext[检索到的关联记忆] --> LLM[LLMBrainWrapperV3<br/>LLM 生成层]
    InputVector --> LLM
    LLM --> Response[最终输出]

    %% 配置层
    Config[(BrainConfig.py<br/>配置管理中心)] -.-> Hippo
    Config -.-> Cortex
    Config -.-> LLM

    %% 睡眠/自我修复流
    Sleep[Sleep Consolidate] -.->|修剪/巩固| ExpVis
    Sleep -.->|修剪/巩固| ExpCon
    Sleep -.->|修剪/巩固| ExpSpa
    Sleep -.->|修剪/巩固| ExpAbs
