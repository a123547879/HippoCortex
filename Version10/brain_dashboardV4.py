# ================== 第一阶段：只初始化大脑 ==================
import matplotlib
matplotlib.use('Agg')  # 必须在第一行

import sys
import os
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import random  # 新增随机库
plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from BrainConfig import config
from AdvancedBrainV10 import AdvancedBrainV10
from LLMBrainWrapperV5 import LLMBrainWrapperV5
# from MainTest5 import analyze_brain_structure
from langchain_ollama import ChatOllama
from datetime import datetime
from collections import defaultdict  # 必须加这个！

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FinalImagePet")

def plot_local_connectivity_heatmap(expert, name, brain, save_dir="heatmaps/V8"):
    """
    ✅ 终极修复版：
    1. 直接从 brain.cortex.index 读取记忆（最准确）
    2. 按专家名称过滤，只显示该专家的记忆
    3. 彻底解决记忆串扰问题
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}_local_connectivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    # 1. 取出突触权重矩阵
    synapse = expert.synapse.data.cpu().numpy()
    dim = expert.dim
    
    # 2. 生成「局部连接掩码」：假设前20%神经元属于核心功能区
    partition_size = int(dim * 0.2)
    mask = np.zeros((dim, dim))
    mask[:partition_size, :partition_size] = 1  # 核心功能区高亮
    
    # 3. 计算「局部连接率」
    local_connections = np.sum(np.abs(synapse[:partition_size, :partition_size]) > 0.1)
    total_local = max(partition_size * partition_size, 1)
    local_connectivity_rate = (local_connections / total_local) * 100
    
    # 4. 生成热力图数据：局部连接 * 2.5 强化，跨区连接 * 0.2 淡化
    heatmap_data = synapse * mask * 2.5 + synapse * (1 - mask) * 0.2
    
    # 5. 🔥 终极修复：直接从 brain.cortex.index 读取该专家的记忆
    neuron_activation_count = np.zeros(dim)
    neuron_to_memories = defaultdict(list)  # 神经元ID -> 记忆内容列表
    
    # 遍历所有记忆，只处理属于当前专家的
    for mem_id, mem in brain.cortex.index.memories.items():
        mem_expert = mem["metadata"].get("expert", "未知")
        if mem_expert != name:
            continue  # 只处理当前专家的记忆
        
        # 取出SDR
        if "sdr" in mem:
            sdr = mem["sdr"]
            content = mem["content"]
            
            # 找出这个SDR中激活的神经元
            active_neurons = torch.where(sdr > 0.1)[0].numpy()
            for neuron_id in active_neurons:
                if neuron_id < dim:  # 只统计在画图范围内的神经元
                    neuron_activation_count[neuron_id] += 1
                    # 只记录前50个字符，避免太长
                    neuron_to_memories[neuron_id].append(content[:50] + "..." if len(content) > 50 else content)
    
    # 6. 画图（只画前150x150，看得更清楚）
    plt.figure(figsize=(14, 12))
    
    # 6.1 画突触权重热力图
    plot_size = min(200, dim)
    im = plt.imshow(heatmap_data[:plot_size, :plot_size], cmap='coolwarm', vmin=-2.5, vmax=2.5)
    plt.colorbar(im, label='突触权重 (强化局部连接)', shrink=0.8)
    
    # 6.2 叠加激活位置（白色十字标记）
    # 只显示激活次数最多的前15个神经元，避免太乱
    top_activated_neurons = np.argsort(neuron_activation_count)[::-1][:15]
    # plot_size = min(np.max(top_activated_neurons), dim)
    # print(plot_size)

    cleaned_neuron_mem = {}
    for nid, mem_list in neuron_to_memories.items():
        unique_mem = list(dict.fromkeys(mem_list))  # 保持顺序的去重
        cleaned_neuron_mem[nid] = unique_mem[:3]

    # 5.2 优先选核心区激活的神经元（让核心区不那么空）
    core_neurons = np.where((neuron_activation_count > 1) & (np.arange(dim) < partition_size))[0]
    top_core_neurons = core_neurons[np.argsort(neuron_activation_count[core_neurons])[::-1][:5]]
    
    # 非核心区激活最多的神经元（过滤激活次数>1，避免无效标记）
    non_core_neurons = np.where((neuron_activation_count > 1) & (np.arange(dim) >= partition_size))[0]
    top_non_core_neurons = non_core_neurons[np.argsort(neuron_activation_count[non_core_neurons])[::-1][:5]]
    
    # 合并神经元列表，优先核心区
    top_activated_neurons = np.concatenate([top_core_neurons, top_non_core_neurons])
    top_activated_neurons = np.unique(top_activated_neurons)  # 去重
    
    used_labels = set()
    legend_added = False

    for neuron_id in top_activated_neurons:
        if neuron_id < plot_size and neuron_activation_count[neuron_id] > 0:
            # 画白色十字标记（图例只加一次，避免重复）
            plt.scatter(neuron_id, neuron_id, s=120, c='white', marker='+', linewidths=2.5, 
                       label=f'神经元 {neuron_id} (激活{int(neuron_activation_count[neuron_id])}次)')
            legend_added = True
            
            # 标签处理：终极去重+防重叠
            if neuron_id in cleaned_neuron_mem and len(cleaned_neuron_mem[neuron_id]) > 0:
                mem_list = cleaned_neuron_mem[neuron_id]
                
                selected_mem = None
                # 优先选未使用过的记忆
                for mem in mem_list:
                    if mem not in used_labels:
                        selected_mem = mem
                        break
                # 处理同质化记忆：给重复标签加序号和神经元ID区分
                if selected_mem is None:
                    base_mem = mem_list[0]
                    count = sum(1 for label in used_labels if label.startswith(base_mem))
                    selected_mem = f"{base_mem} [{neuron_id}-{count+1}]"
                
                used_labels.add(selected_mem)
                
                # 随机偏移标签位置，避免重叠
                # offset_x = random.randint(5, 15)
                # offset_y = random.randint(-5, 5)
                plt.text(neuron_id + 3, neuron_id, selected_mem, fontsize=7, 
                        ha='left', va='center', 
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))
    
    plt.title(f'[{name}] 专家局部连接热力图\n局部连接率: {local_connectivity_rate:.2f}% | 记忆数: {len([m for m in brain.cortex.index.memories.values() if m["metadata"].get("expert") == name])}', fontsize=14)
    plt.xlabel('神经元索引 (前200)', fontsize=12)
    plt.ylabel('神经元索引 (前200)', fontsize=12)
    
    # 核心功能区高亮
    plt.gca().add_patch(plt.Rectangle((-0.5, -0.5), min(partition_size, plot_size), min(partition_size, plot_size), 
                                       linewidth=3, edgecolor='yellow', linestyle='--', facecolor='none',
                                       label='核心功能区'))
    
    # 只显示前6个图例，避免太乱
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(handles[:6], labels[:6], fontsize=10, loc='upper right')
    
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path, dpi=200)
    # plt.close()
    
    # 统计该专家的记忆数
    expert_mem_count = len([m for m in brain.cortex.index.memories.values() if m["metadata"].get("expert") == name])
    
    print(f"\n✅ [{name}] 终极修复版热力图已保存到: {save_path}")
    print(f"   - 专家记忆数: {expert_mem_count}")
    print(f"   - 激活神经元标记数: {min(15, len(top_activated_neurons))}")
    print(f"   - 神经元-知识映射数: {len(neuron_to_memories)}")
    
    return local_connectivity_rate

# import random  # 确保文件顶部导入random

# def plot_local_connectivity_heatmap(expert, name, brain, save_dir="heatmaps/V8"):
#     """
#     ✅ 终极修复版：
#     1. 解决标签重复+同质化问题
#     2. 解决标签重叠问题
#     3. 强化核心功能区显示
#     4. 优化热力图可读性
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, f"{name}_local_connectivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
#     # 1. 取出突触权重矩阵
#     synapse = expert.synapse.data.cpu().numpy()
#     dim = expert.dim
    
#     # 2. 生成「局部连接掩码」：前20%神经元为核心功能区
#     partition_size = int(dim * 0.2)
#     mask = np.zeros((dim, dim))
#     mask[:partition_size, :partition_size] = 1
    
#     # 3. 计算局部连接率
#     local_connections = np.sum(np.abs(synapse[:partition_size, :partition_size]) > 0.1)
#     total_local = max(partition_size * partition_size, 1)
#     local_connectivity_rate = (local_connections / total_local) * 100
    
#     # 4. 热力图数据优化：调整非核心区淡化系数，让整体结构可见
#     heatmap_data = synapse * mask * 2.5 + synapse * (1 - mask) * 0.5  # 0.5比0.2更能看到整体
    
#     # 5. 读取该专家的记忆
#     neuron_activation_count = np.zeros(dim)
#     neuron_to_memories = defaultdict(list)
    
#     for mem_id, mem in brain.cortex.index.memories.items():
#         mem_expert = mem["metadata"].get("expert", "未知")
#         if mem_expert != name:
#             continue
        
#         if "sdr" in mem:
#             sdr = mem["sdr"]
#             content = mem["content"]
#             active_neurons = torch.where(sdr > 0.1)[0].numpy()
#             for neuron_id in active_neurons:
#                 if neuron_id < dim:
#                     neuron_activation_count[neuron_id] += 1
#                     # 限制标签长度，避免过长
#                     short_content = content[:40] + "..." if len(content) > 40 else content
#                     neuron_to_memories[neuron_id].append(short_content)

#     # 5.1 神经元记忆去重 + 限制数量（每个神经元最多3条）
#     cleaned_neuron_mem = {}
#     for nid, mem_list in neuron_to_memories.items():
#         unique_mem = list(dict.fromkeys(mem_list))  # 保持顺序的去重
#         cleaned_neuron_mem[nid] = unique_mem[:3]

#     # 5.2 优先选核心区激活的神经元（让核心区不那么空）
#     core_neurons = np.where((neuron_activation_count > 1) & (np.arange(dim) < partition_size))[0]
#     top_core_neurons = core_neurons[np.argsort(neuron_activation_count[core_neurons])[::-1][:5]]
    
#     # 非核心区激活最多的神经元（过滤激活次数>1，避免无效标记）
#     non_core_neurons = np.where((neuron_activation_count > 1) & (np.arange(dim) >= partition_size))[0]
#     top_non_core_neurons = non_core_neurons[np.argsort(neuron_activation_count[non_core_neurons])[::-1][:5]]
    
#     # 合并神经元列表，优先核心区
#     top_activated_neurons = np.concatenate([top_core_neurons, top_non_core_neurons])
#     top_activated_neurons = np.unique(top_activated_neurons)  # 去重

#     # 6. 画图
#     plt.figure(figsize=(14, 12))
#     plot_size = min(200, dim)  # 扩大到200，同时显示核心区和部分非核心区
#     im = plt.imshow(heatmap_data[:plot_size, :plot_size], cmap='coolwarm', vmin=-1, vmax=1)  # 缩小颜色范围，差异更明显
#     plt.colorbar(im, label='突触权重 (强化局部连接)', shrink=0.8)
#     # plot_size = min(np.max(top_activated_neurons), dim)
    
#     used_labels = set()
#     legend_added = False

#     for neuron_id in top_activated_neurons:
#         if neuron_id < plot_size and neuron_activation_count[neuron_id] > 0:
#             # 画白色十字标记（图例只加一次，避免重复）
#             plt.scatter(neuron_id, neuron_id, s=120, c='white', marker='+', linewidths=2.5, 
#                        label="激活神经元" if not legend_added else "")
#             legend_added = True
            
#             # 标签处理：终极去重+防重叠
#             if neuron_id in cleaned_neuron_mem and len(cleaned_neuron_mem[neuron_id]) > 0:
#                 mem_list = cleaned_neuron_mem[neuron_id]
                
#                 selected_mem = None
#                 # 优先选未使用过的记忆
#                 for mem in mem_list:
#                     if mem not in used_labels:
#                         selected_mem = mem
#                         break
#                 # 处理同质化记忆：给重复标签加序号和神经元ID区分
#                 if selected_mem is None:
#                     base_mem = mem_list[0]
#                     count = sum(1 for label in used_labels if label.startswith(base_mem))
#                     selected_mem = f"{base_mem} [{neuron_id}-{count+1}]"
                
#                 used_labels.add(selected_mem)
                
#                 # 随机偏移标签位置，避免重叠
#                 offset_x = random.randint(5, 15)
#                 offset_y = random.randint(-5, 5)
#                 plt.text(neuron_id + offset_x, neuron_id + offset_y, selected_mem, fontsize=7, 
#                         ha='left', va='center', 
#                         bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))
    
#     plt.title(f'[{name}] 专家局部连接热力图\n局部连接率: {local_connectivity_rate:.2f}% | 记忆数: {len([m for m in brain.cortex.index.memories.values() if m["metadata"].get("expert") == name])}', fontsize=14)
#     plt.xlabel('神经元索引 (前200)', fontsize=12)
#     plt.ylabel('神经元索引 (前200)', fontsize=12)
    
#     # 核心功能区高亮
#     plt.gca().add_patch(plt.Rectangle((-0.5, -0.5), min(partition_size, plot_size), min(partition_size, plot_size), 
#                                        linewidth=3, edgecolor='yellow', linestyle='--', facecolor='none',
#                                        label='核心功能区'))
    
#     plt.legend(fontsize=10, loc='upper right')
#     plt.tight_layout()
    
#     # 只保存图片，不显示（彻底消除Agg后端警告）
#     plt.savefig(save_path, dpi=200, bbox_inches='tight')
#     plt.close()
    
#     # 打印统计信息
#     expert_mem_count = len([m for m in brain.cortex.index.memories.values() if m["metadata"].get("expert") == name])
#     print(f"\n✅ [{name}] 热力图已保存到: {save_path}")
#     print(f"   - 专家记忆数: {expert_mem_count}")
#     print(f"   - 激活神经元标记数: {len(top_activated_neurons)}")
#     print(f"   - 神经元-知识映射数: {len(cleaned_neuron_mem)}")
    
#     return local_connectivity_rate
# ========================================================================

# ================== 100% 完全复制 MainTest 的大脑初始化 ==================
print("=" * 60)
print("🧠 正在初始化小白大脑...")
print("=" * 60)
print("💡 提示：请确保 BrainConfig.py 中 local_bias_strength = 1.2，热力图效果更明显！")

# 先初始化LLM
llm = ChatOllama(model=config.ollama_model_name)

# 初始化类脑记忆系统（🔥 可选：关闭知识图谱提升性能）
brain = AdvancedBrainV10(
    dim=config.dim, 
    storage_dir=config.storage_dir, 
    ollama_model="bge-m3",
    llm=llm,
    kg_enabled=True  # 🔥 可设置为False关闭知识图谱
)
llm_brain = LLMBrainWrapperV5(brain)

# 知识导入检查
dataset_path = "general_knowledge.txt"
first_run_flag = os.path.join(config.storage_dir, "general_knowledge_imported")
if os.path.exists(dataset_path) and not os.path.exists(first_run_flag):
    from MainTest5 import import_knowledge_dataset
    import_knowledge_dataset(llm_brain, dataset_path, first_run_flag, use_kg=False)
else:
    final_memory_count = brain.get_brain_status()["total_memories"]
    kg_status = "✅ 已启用" if brain.get_brain_status().get("kg_enabled", True) else "⚡ 已关闭（性能模式）"
    logger.info(f"✅ 通用知识已导入，当前总记忆数：{final_memory_count}，知识图谱：{kg_status}")

print("\n" + "=" * 60)
print("✅ 大脑初始化完成！现在启动桌宠界面...")
print("=" * 60)

# ================== 第二阶段：大脑初始化完成后，才导入 PyQt5 ==================
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QMenu, QLineEdit, QTextEdit, QAction, QFrame)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QFont, QTextCursor, QImage

# ================== 后台工作线程 ==================
class ChatThread(QThread):
    response_received = pyqtSignal(str)
    
    def __init__(self, user_input):
        super().__init__()
        self.user_input = user_input
        
    def run(self):
        try:
            response = llm_brain.ask(self.user_input)
            self.response_received.emit(response)
        except Exception as e:
            self.response_received.emit(f"❌ 出错了：{str(e)[:50]}")

class SleepThread(QThread):
    finish_signal = pyqtSignal(str)
    
    def run(self):
        try:
            brain.sleep_consolidate_all()
            brain.save_all()
            self.finish_signal.emit("✅ 睡眠巩固完成！记忆已保存")
        except Exception as e:
            self.finish_signal.emit(f"❌ 睡眠失败：{str(e)[:30]}")

# ================== 小白大脑桌宠主界面 ==================
class XiaobaiBrainPet(QWidget):
    def __init__(self):
        super().__init__()
        self.drag_pos = QPoint()
        self.is_chat_expanded = False
        self.is_sleeping = False
        
        # 获取项目绝对路径
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        print(f"\n📂 项目根目录：{self.project_root}")
        
        # 头像配置（使用绝对路径）
        self.avatar_paths = {
            "awake": os.path.join(self.project_root, "imgs", "stand.png"),
            "working": os.path.join(self.project_root, "imgs", "sit.png"),
            "sleep": os.path.join(self.project_root, "imgs", "sleep.png"),
            "error": os.path.join(self.project_root, "imgs", "error.png"),
        }
        
        self.current_avatar = None
        self.check_all_avatars()
        self.initUI()
        
        # 定时更新状态
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_brain_status)
        self.status_timer.start(3000)

        # 自动睡眠定时器
        self.auto_sleep_timer = QTimer()
        self.auto_sleep_timer.timeout.connect(self.check_auto_sleep)
        self.auto_sleep_timer.start(5000)

    def check_all_avatars(self):
        """检查所有头像文件，并打印详细信息"""
        print("\n🔍 正在检查所有头像文件：")
        for state, full_path in self.avatar_paths.items():
            print(f"  状态 [{state}]: {full_path}")
            if os.path.exists(full_path):
                print(f"    ✅ 文件存在，大小：{os.path.getsize(full_path)} 字节")
                img = QImage(full_path)
                if img.isNull():
                    print(f"    ❌ 文件存在，但不是有效的图片文件！")
                else:
                    print(f"    ✅ 图片有效，尺寸：{img.width()}x{img.height()}")
            else:
                print(f"    ❌ 文件不存在！")
        print("=" * 60)

    def check_auto_sleep(self):
        """仅在未聊天、未睡眠时，随机触发自动睡眠"""
        if self.is_chat_expanded or self.is_sleeping:
            return
        if random.randint(1, 1000) <= 10:
            print("😴 空闲状态，自动触发睡眠...")
            self.trigger_sleep()

    def initUI(self):
        self.setWindowTitle("🧠 小白大脑桌宠")
        self.setFixedSize(220, 260)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # 头像容器
        avatar_frame = QFrame()
        avatar_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(30, 41, 59, 200);
                border-radius: 10px;
            }
        """)
        avatar_layout = QVBoxLayout(avatar_frame)
        avatar_layout.setContentsMargins(10, 10, 10, 10)
        
        self.avatar_label = QLabel()
        self.avatar_label.setFixedSize(120, 120)
        self.avatar_label.setAlignment(Qt.AlignCenter)
        self.update_avatar("awake")
        avatar_layout.addWidget(self.avatar_label)
        main_layout.addWidget(avatar_frame)

        # 状态显示
        self.status_label = QLabel("🧠 大脑已唤醒")
        self.status_label.setFont(QFont("SimHei", 10))
        self.status_label.setStyleSheet("color: #000000;")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        # 记忆数 + 知识图谱状态
        self.memory_label = QLabel("📚 记忆数：加载中...")
        self.memory_label.setFont(QFont("SimHei", 9))
        self.memory_label.setStyleSheet("color: #94a3b8;")
        self.memory_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.memory_label)

        # 功能按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        
        self.chat_btn = QPushButton("💬 聊天")
        self.chat_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        self.chat_btn.clicked.connect(self.toggle_chat)
        
        self.sleep_btn = QPushButton("🌙 睡眠")
        self.sleep_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        self.sleep_btn.clicked.connect(self.trigger_sleep)
        
        btn_layout.addWidget(self.chat_btn)
        btn_layout.addWidget(self.sleep_btn)
        main_layout.addLayout(btn_layout)

        # 聊天区域（默认隐藏）
        self.chat_frame = QWidget()
        chat_layout = QVBoxLayout(self.chat_frame)
        chat_layout.setSpacing(8)
        
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setFixedHeight(120)
        self.chat_history.setStyleSheet("""
            QTextEdit {
                background-color: #1e293b;
                border: none;
                border-radius: 6px;
                padding: 8px;
                font-size: 10px;
                color: #f1f5f9;
            }
        """)
        chat_layout.addWidget(self.chat_history)
        
        input_layout = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("和小白说点什么...")
        self.input_box.setStyleSheet("""
            QLineEdit {
                background-color: #334155;
                border: none;
                border-radius: 6px;
                padding: 6px;
                font-size: 10px;
                color: #f1f5f9;
            }
        """)
        self.input_box.returnPressed.connect(self.send_message)
        
        self.send_btn = QPushButton("发送")
        self.send_btn.setFixedSize(50, 24)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #4cc9f0;
                color: #0f172a;
                border: none;
                border-radius: 6px;
                font-size: 10px;
            }
        """)
        self.send_btn.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.input_box)
        input_layout.addWidget(self.send_btn)
        chat_layout.addLayout(input_layout)
        
        self.chat_frame.setVisible(False)
        main_layout.addWidget(self.chat_frame)

    def update_avatar(self, state):
        if state == self.current_avatar:
            return
            
        self.current_avatar = state
        full_path = self.avatar_paths.get(state, self.avatar_paths["error"])
        
        img = QImage(full_path)
        if not img.isNull():
            pixmap = QPixmap.fromImage(img)
            scaled_pixmap = pixmap.scaled(
                120, 120, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.avatar_label.setPixmap(scaled_pixmap)
            print(f"✅ 成功加载头像：{state}")
        else:
            self.avatar_label.setText(f"❌ 图片加载失败\n{os.path.basename(full_path)}")
            self.avatar_label.setStyleSheet("color: #ef4444; font-size: 10px;")
            print(f"❌ 头像加载失败：{full_path}")

    def update_brain_status(self):
        """🔥 优化：显示知识图谱状态"""
        try:
            status = brain.get_brain_status()
            total = status.get("total_memories", 0)
            kg_enabled = status.get("kg_enabled", True)
            kg_text = " | 🧠 KG: 已启用" if kg_enabled else " | ⚡ KG: 性能模式"
            self.memory_label.setText(f"📚 记忆数：{total} 条{kg_text}")
        except Exception as e:
            self.status_label.setText(f"❌ {str(e)[:10]}")

    def toggle_chat(self):
        self.is_chat_expanded = not self.is_chat_expanded
        self.chat_frame.setVisible(self.is_chat_expanded)
        
        if self.is_chat_expanded:
            self.setFixedSize(220, 460)
            self.update_avatar('working')
            self.chat_btn.setText("🙈 收起")
        else:
            self.setFixedSize(220, 260)
            self.update_avatar('awake')
            self.chat_btn.setText("💬 聊天")

    def append_message(self, sender, text):
        if sender == "你":
            color = "#4cc9f0"
            align = "right"
        else:
            color = "#f1f5f9"
            align = "left"
            
        html = f"""
        <div style='text-align: {align}; margin: 5px 0;'>
            <div style='display: inline-block; background-color: {"#334155" if sender == "你" else "#1e293b"}; 
                        padding: 6px 10px; border-radius: 8px; max-width: 90%;'>
                <span style='color: #94a3b8; font-size: 8px;'>{sender}</span><br>
                <span style='color: {color}; font-size: 10px;'>{text}</span>
            </div>
        </div>
        """
        self.chat_history.append(html)
        self.chat_history.moveCursor(QTextCursor.End)

    def send_message(self):
        user_input = self.input_box.text().strip()
        if not user_input:
            return
            
        self.append_message("你", user_input)
        self.input_box.clear()
        self.status_label.setText("🧠 正在思考...")
        
        self.chat_thread = ChatThread(user_input)
        self.chat_thread.response_received.connect(self.on_chat_response)
        self.chat_thread.start()

    def on_chat_response(self, response):
        if response == "抱歉，我没有这方面的信息":
            self.update_avatar('error')
        else:
            self.update_avatar('working')
        self.append_message("小白", response)
        self.status_label.setText("🧠 大脑已唤醒")

    def trigger_sleep(self):
        if self.is_sleeping:
            return
        self.is_sleeping = True
        self.update_avatar("sleep")
        self.status_label.setText("🌙 睡眠巩固中...")
        self.sleep_btn.setEnabled(False)
        self.chat_btn.setEnabled(False)
        
        self.sleep_thread = SleepThread()
        self.sleep_thread.finish_signal.connect(self.on_sleep_finish)
        self.sleep_thread.start()

    def on_sleep_finish(self, msg):
        self.append_message("系统", msg)
        self.update_avatar("awake")
        self.status_label.setText("🧠 大脑已唤醒")
        self.sleep_btn.setEnabled(True)
        self.chat_btn.setEnabled(True)
        self.is_sleeping = False

    # 窗口拖动
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_pos)
            event.accept()

    # 右键菜单（🔥 新增：知识图谱开关 + 局部连接热力图）
    def contextMenuEvent(self, event):
        menu = QMenu(self)
        
        # 知识图谱开关
        kg_status = brain.get_brain_status().get("kg_enabled", True)
        if kg_status:
            kg_action = QAction("⚡ 禁用知识图谱（性能模式）", self)
            kg_action.triggered.connect(lambda: self.toggle_kg(False))
        else:
            kg_action = QAction("🧠 启用知识图谱", self)
            kg_action.triggered.connect(lambda: self.toggle_kg(True))
        
        # 🔥 新增：局部连接高亮热力图
        analyze_action = QAction("🔍 分析局部连接热力图", self)
        analyze_action.triggered.connect(self.trigger_analyze_local)

        redistribute_action = QAction("🔍 重新分配历史记忆", self)
        redistribute_action.triggered.connect(self.redistribute_memories)

        clean_identity_action = QAction("🔍 清理专家记忆", self)
        clean_identity_action.triggered.connect(self.clean_identity)
        
        save_action = QAction("💾 保存大脑数据", self)
        save_action.triggered.connect(self.save_brain)
        
        exit_action = QAction("🚪 退出", self)
        exit_action.triggered.connect(self.exit_app)
        
        menu.addAction(kg_action)
        menu.addSeparator()
        menu.addAction(analyze_action)
        menu.addAction(redistribute_action)
        menu.addAction(clean_identity_action)
        menu.addAction(save_action)
        menu.addSeparator()
        menu.addAction(exit_action)
        
        menu.exec_(event.globalPos())

    def clean_identity(self):
        brain.force_clean_all_experts()

    def redistribute_memories(self):
        brain.redistribute_memories()

    def toggle_kg(self, enable: bool):
        """🔥 新增：切换知识图谱状态"""
        if enable:
            brain.enable_kg()
            self.append_message("系统", "✅ 知识图谱已启用")
        else:
            brain.disable_kg()
            self.append_message("系统", "✅ 知识图谱已禁用（性能模式）")
        self.update_brain_status()

    # ================== 🔥 新增：触发局部连接高亮热力图分析 ==================
    def trigger_analyze_local(self):
        self.append_message("系统", "🔍 正在分析局部连接热力图（终极修复版）...")
        try:
            # 遍历所有专家，生成专门的局部连接热力图
            heatmap_dir = os.path.join(self.project_root, "heatmaps/V9")
            total_rate = 0.0
            for name, expert in brain.experts.items():
                # 🔥 修复：传入 brain 对象
                rate = plot_local_connectivity_heatmap(expert, name, brain, save_dir=heatmap_dir)
                total_rate += rate
            
            avg_rate = total_rate / len(brain.experts)
            self.append_message("系统", f"✅ 分析完成！\n平均局部连接率: {avg_rate:.2f}%\n热力图已保存到: {heatmap_dir}")
        except Exception as e:
            self.append_message("系统", f"❌ 分析失败：{str(e)[:50]}")
            import traceback
            traceback.print_exc()
    # ========================================================================

    def save_brain(self):
        try:
            brain.save_all()
            self.append_message("系统", "✅ 大脑数据已保存！")
        except Exception as e:
            self.append_message("系统", f"❌ 保存失败：{str(e)[:30]}")

    def exit_app(self):
        try:
            brain.save_all()
            logger.info("✅ 程序正常退出，数据已保存")
        except Exception as e:
            logger.error(f"❌ 退出时保存失败：{e}")
        QApplication.quit()

# ================== 程序入口 ==================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    pet = XiaobaiBrainPet()
    pet.show()
    sys.exit(app.exec_())