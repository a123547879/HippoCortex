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
from AdvancedBrainV15 import AdvancedBrain  # ✅ 升级到V12（意图驱动版）
from LLMBrainWrapperV5 import LLMBrainWrapperV5
from langchain_ollama import ChatOllama
from datetime import datetime
from collections import defaultdict  # 必须加这个！

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FinalImagePet")

def plot_core_region_heatmap(expert, name, brain, save_dir="heatmaps/V13_core"):
    """
    🔥 最终版核心区热力图：文字100%紧贴神经元 + 绝对防重叠
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}_core_detail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    synapse = expert.synapse.data.cpu().numpy()
    dim = expert.dim
    plot_size = min(200, dim)
    partition_size = int(dim * 0.2)

    synapse_plot = synapse[:plot_size, :plot_size]

    # 统计神经元激活与记忆
    neuron_activation_count = np.zeros(dim)
    neuron_to_memories = defaultdict(list)
    for mem_id, mem in brain.cortex.index.memories.items():
        if mem["metadata"].get("expert") != name:
            continue
        if "sdr" in mem:
            sdr = mem["sdr"]
            content = mem["content"]
            active_neurons = torch.where(sdr > 0.1)[0].numpy()
            for neuron_id in active_neurons:
                if neuron_id < plot_size:
                    neuron_activation_count[neuron_id] += 1
                    neuron_to_memories[neuron_id].append(content[:35] + "...")

    # 绘图
    plt.figure(figsize=(12, 10))
    plt.rcParams['font.size'] = 9

    # 绘制热力图
    mask = np.zeros((plot_size, plot_size))
    mask[:min(partition_size, plot_size), :min(partition_size, plot_size)] = 1
    heatmap_data = synapse_plot * mask * 2.5 + synapse_plot * (1 - mask) * 0.2
    im = plt.imshow(heatmap_data, cmap='coolwarm', vmin=-2.5, vmax=2.5)
    cbar = plt.colorbar(im, label='突触权重（强化局部连接）', shrink=0.8)

    # 获取前10个激活最高的神经元，并排序（防止乱序标注重叠）
    top_neurons = np.argsort(neuron_activation_count)[::-1][:10]
    top_neurons = sorted(top_neurons)  # 按神经元ID排序，核心优化点
    
    cleaned_mem = {}
    for nid, mem_list in neuron_to_memories.items():
        unique_mem = list(dict.fromkeys(mem_list))[:1]  # 每个神经元只留1条记忆，更清爽
        cleaned_mem[nid] = unique_mem

    # ================== 核心优化：四方向循环偏移（绝对防重叠 + 紧贴神经元）==================
    # 偏移方向：右、下、左、上 循环，每个神经元用不同方向，永不重叠
    directions = [
        (5, 0, 'left'),    # 右侧
        (0, 5, 'center'),  # 下方
        (-5, 0, 'right'),  # 左侧
        (0, -5, 'center')  # 上方
    ]
    
    used_labels = set()

    for idx, neuron_id in enumerate(top_neurons):
        if neuron_id >= plot_size or neuron_activation_count[neuron_id] == 0:
            continue
        
        # 绘制神经元标记
        plt.scatter(neuron_id, neuron_id, s=90, c='white', marker='+', linewidths=2,
                   label=f'神经元{neuron_id}({int(neuron_activation_count[neuron_id])})')
        
        if neuron_id not in cleaned_mem or len(cleaned_mem[neuron_id]) == 0:
            continue

        # 选择不重复的记忆文本
        mem_text = cleaned_mem[neuron_id][0]
        while mem_text in used_labels:
            mem_text += f"[{neuron_id}]"
        used_labels.add(mem_text)

        # 循环获取偏移方向（核心：轮流换方向，彻底不重叠）
        offset_x, offset_y, ha = directions[idx % 4]

        # 绘制文本：紧贴神经元，绝对不重叠
        plt.text(
            neuron_id + offset_x,
            neuron_id + offset_y,
            mem_text,
            fontsize=7.5,
            ha=ha,
            va='center',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='yellow', alpha=0.95)
        )

    # 核心区框选
    plt.gca().add_patch(plt.Rectangle((-0.5, -0.5), min(partition_size, plot_size), min(partition_size, plot_size),
                                       linewidth=3, edgecolor='gold', linestyle='--', facecolor='none',
                                       label='核心功能区'))

    # 标题与标签
    core_connections = np.sum(np.abs(synapse[:partition_size, :partition_size]) > 0.1)
    local_rate = (core_connections / (partition_size * partition_size)) * 100
    plt.title(
        f'[{name}] 核心区细节热力图（前200维）\n'
        f'局部连接率: {local_rate:.2f}% | 记忆数: {len(neuron_to_memories)}\n'
        f'时间: [{datetime.now().strftime("%Y:%m:%d %H:%M:%S")}]',
        fontsize=14
    )
    plt.xlabel('神经元索引（前200）', fontsize=12)
    plt.ylabel('神经元索引（前200）', fontsize=12)
    
    # 图例优化
    plt.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.2, 1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n✅ [{name}] 核心区细节图已保存: {save_path}")
    return local_rate

def plot_local_connectivity_heatmap(expert, name, brain, save_dir="heatmaps/V13"):
    """
    ✅ 2048维全览 · 分块聚合热力图（超强可读性）
    完整观测2048维，不丢失信息，不混乱，突出局部连接+稀疏性
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}_blockwise_2048.png")

    synapse = expert.synapse.data.cpu().numpy()
    dim = expert.dim  # 2048
    full_dim = 2048

    # ================== 🔥 核心：分块聚合（32×32块，覆盖完整2048维） ==================
    block_size = 64  # 2048 / 32 = 64
    n_blocks = full_dim // block_size  # 32

    # 计算每个块的平均连接强度
    block_heatmap = np.zeros((n_blocks, n_blocks))
    for i in range(n_blocks):
        for j in range(n_blocks):
            si, ei = i*block_size, (i+1)*block_size
            sj, ej = j*block_size, (j+1)*block_size
            block = synapse[si:ei, sj:ej]
            # 用【平均绝对值】表示连接强度（突出稀疏性）
            block_heatmap[i, j] = np.mean(np.abs(block))

    # ================== 核心区标记（20% = 409维 → 对应前 7 个块） ==================
    core_size = int(full_dim * 0.2)  # ~409
    core_blocks = int(np.ceil(core_size / block_size))  # 7块

    # ================== 绘制 ==================
    plt.figure(figsize=(12, 10))
    plt.rcParams['font.size'] = 10

    # 画 32×32 块聚合热力图（完整2048维）
    im = plt.imshow(block_heatmap, cmap='coolwarm', vmin=0, vmax=np.percentile(block_heatmap, 99))
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Average Synapse Strength | 块平均连接强度', fontsize=12)

    # 黄色框标出【核心功能区】（前7×7块）
    plt.gca().add_patch(plt.Rectangle(
        (-0.5, -0.5), core_blocks, core_blocks,
        linewidth=3, edgecolor='gold', linestyle='--',
        facecolor='none', label='Core Region | 核心区'
    ))

    # 神经元范围标注
    plt.xticks(ticks=np.arange(0, n_blocks, 4), labels=[f"{i*64}" for i in range(0, n_blocks, 4)], fontsize=8)
    plt.yticks(ticks=np.arange(0, n_blocks, 4), labels=[f"{i*64}" for i in range(0, n_blocks, 4)], fontsize=8)
    plt.xlabel('Neuron Block (64 neurons/block) | 神经元块', fontsize=12)
    plt.ylabel('Neuron Block (64 neurons/block) | 神经元块', fontsize=12)

    # 统计信息
    total_connections = np.sum(np.abs(synapse) > 0.1)
    core_connections = np.sum(np.abs(synapse[:core_size, :core_size]) > 0.1)
    core_rate = (core_connections / total_connections) * 100 if total_connections > 0 else 0

    plt.title(
        f'[{name}] 2048维突触 · 分块聚合热力图\n'
        f'核心区连接占比: {core_rate:.1f}% | 总稀疏度: {1-total_connections/(2048*2048):.2%}',
        fontsize=14
    )
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"\n✅ [{name}] 2048维完整热力图已保存: {save_path}")
    return core_rate

# ========================================================================

# ================== 100% 完全复制 MainTest 的大脑初始化 ==================
print("=" * 60)
print("🧠 正在初始化小白大脑...")
print("=" * 60)
print("💡 提示：请确保 BrainConfig.py 中 local_bias_strength = 1.2，热力图效果更明显！")

# 先初始化LLM
llm = ChatOllama(model=config.ollama_model_name)

brain = AdvancedBrain(
    dim=config.dim, 
    storage_dir=config.storage_dir, 
    ollama_model="bge-m3",
    llm=llm,
    kg_enabled=True
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
    intention_count = brain.get_brain_status().get("intention_queue_size", 0)
    logger.info(f"✅ 通用知识已导入 | 总记忆数：{final_memory_count} | 知识图谱：{kg_status} | 意图队列：{intention_count}")

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
            print(f"❌ 出错了：{str(e)}")

class SleepThread(QThread):
    finish_signal = pyqtSignal(str)
    
    def run(self):
        try:
            brain.sleep_consolidate_all()
            brain.save_all()
            self.finish_signal.emit("✅ 睡眠巩固完成！记忆已保存")
        except Exception as e:
            print(f"❌ 睡眠失败：{str(e)}")

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
        
        # 头像配置（🔥 双走神头像轮播）
        self.avatar_paths = {
            "awake": os.path.join(self.project_root, "imgs", "stand.png"),      # 清醒
            "working": os.path.join(self.project_root, "imgs", "sit.png"),      # 工作中
            "sleep": os.path.join(self.project_root, "imgs", "sleep.png"),      # 睡眠
            "wandering01": os.path.join(self.project_root, "imgs", "wandering01.png"),  # 走神1
            "wandering02": os.path.join(self.project_root, "imgs", "wandering02.png"),  # 走神2
            "error": os.path.join(self.project_root, "imgs", "error.png")       # 错误
        }
        
        # ================== 走神头像轮播状态 ==================
        self.current_wandering_index = 0
        self.wandering_avatar_list = ["wandering01", "wandering02"]
        # ================================================================
        
        self.current_avatar = None
        self.check_all_avatars()
        self.initUI()
        
        # 定时更新状态
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_brain_status)
        self.status_timer.start(3000)

        # ================== 走神状态定时器（2秒，配合头像轮播） ==================
        self.mind_wandering_timer = QTimer()
        self.mind_wandering_timer.timeout.connect(self.check_mind_wandering)
        self.mind_wandering_timer.start(2000)

        # ================== 🔥 新增：主动意图检查定时器 ==================
        self.intention_check_timer = QTimer()
        self.intention_check_timer.timeout.connect(self.check_and_execute_intentions)
        self.intention_check_timer.start(3000)  # 每3秒检查一次
        logger.info("✅ 主动意图检查定时器已启动")
        # ====================================================================

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

    # ================== 🔥 修复：主动意图检查方法 ==================
    def check_and_execute_intentions(self):
        """检查大脑是否有需要主动说的意图（修复聊天展开时不显示的问题）"""
        try:
            # 只有睡眠中才完全屏蔽主动意图
            if self.is_sleeping:
                return
                
            # 增加调试日志，方便排查问题
            logger.debug(f"🔍 检查主动意图 | 聊天展开: {self.is_chat_expanded} | 睡眠中: {self.is_sleeping}")
                
            intention = brain.get_pending_social_intention()
            if intention:
                logger.info(f"💬 显示主动意图: {intention['content']}")
                
                # 显示主动消息
                self.append_message("小白", intention["content"])
                
                # 短暂显示工作状态
                self.update_avatar('working')
                self.status_label.setText("💬 有话想对你说...")
                
                # 3秒后恢复走神状态
                QTimer.singleShot(3000, self._resume_wandering_state)
                
        except Exception as e:
            logger.error(f"❌ 意图执行失败: {e}", exc_info=True)

    def _resume_wandering_state(self):
        """恢复走神状态（修复状态不一致问题）"""
        try:
            if brain.is_mind_wandering and not self.is_sleeping:
                # 恢复走神头像
                current_wandering_state = self.wandering_avatar_list[self.current_wandering_index]
                self.update_avatar(current_wandering_state)
                self.status_label.setText("🌙 走神中...")
                # 强制刷新状态显示
                self.update_brain_status()
            else:
                # 如果已经不在走神状态，恢复清醒头像
                self.update_avatar('awake')
                self.status_label.setText("🧠 大脑已唤醒")
        except Exception as e:
            logger.debug(f"恢复走神状态失败: {e}")
    # ====================================================================

    def check_mind_wandering(self):
        """检查是否应该触发走神、是否有睡眠请求，并更新界面显示"""
        try:
            # 睡眠中 → 不处理走神
            if self.is_sleeping:
                return
            
            # 1. 优先检查：是否有来自大脑的睡眠请求
            if brain.check_and_consume_sleep_request():
                logger.info("🎯 收到大脑睡眠请求，界面接管睡眠流程...")
                self.trigger_sleep()
                return
            
            # 2. 让大脑检查是否应该触发走神
            brain._check_mind_wandering_trigger()
            
            # 3. 更新界面显示（走神状态）
            if brain.is_mind_wandering:
                # 走神头像轮播逻辑
                self.current_wandering_index = 1 - self.current_wandering_index
                current_wandering_state = self.wandering_avatar_list[self.current_wandering_index]
                
                if self.current_avatar != current_wandering_state:
                    self.update_avatar(current_wandering_state)
                    self.status_label.setText("🌙 走神中...")  # 🔥 主标签只显示状态
                
                # 🔥 次级标签显示所有详细信息
                fatigue_pct = int(brain.fatigue_level * 100)
                intention_count = brain.get_brain_status().get("intention_queue_size", 0)
                self.sub_status_label.setText(f"💪 疲劳：{fatigue_pct}% | 💡 想法：{intention_count}")
            else:
                # 清醒状态
                self.update_brain_status()
                if self.current_avatar not in ['awake', 'working', 'sleep', 'error']:
                    self.update_avatar('awake')
                    self.current_wandering_index = 0
                    
        except Exception as e:
            logger.debug(f"走神状态检查失败: {e}")

    def initUI(self):
        self.setWindowTitle("🧠 小白大脑桌宠 (意图驱动版)")
        self.setFixedSize(220, 280)  # 🔥 高度增加20像素，给新标签留出空间
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(8)  # 🔥 间距从10改为8，更紧凑

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

        # 🔥 主状态标签（显示核心信息）
        self.status_label = QLabel("🧠 大脑已唤醒")
        self.status_label.setFont(QFont("SimHei", 11))  # 🔥 字体稍微放大
        self.status_label.setStyleSheet("color: #000000; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        # 🔥 新增：次级状态标签（显示次要信息，解决拥挤问题）
        self.sub_status_label = QLabel("📚 记忆：0 | 🧠 KG：开")
        self.sub_status_label.setFont(QFont("SimHei", 9))
        self.sub_status_label.setStyleSheet("color: #94a3b8;")
        self.sub_status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.sub_status_label)

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
        
        if self.is_sleeping:
            return
    
        """🔥 优化：双标签分层显示，解决内容拥挤问题"""
        try:
            # 如果在走神或睡眠，不覆盖状态显示
            if brain.is_mind_wandering or self.is_sleeping:
                return
                
            status = brain.get_brain_status()
            total = status.get("total_memories", 0)
            kg_enabled = status.get("kg_enabled", True)
            intention_count = status.get("intention_queue_size", 0)
            
            # 🔥 主标签显示核心状态
            self.status_label.setText("🧠 大脑已唤醒")
            
            # 🔥 次级标签显示所有详细信息
            kg_text = "开" if kg_enabled else "关"
            fatigue_text = ""
            if brain.fatigue_level > 0:
                fatigue_pct = int(brain.fatigue_level * 100)
                fatigue_text = f" | 💪 疲劳：{fatigue_pct}%"
                
            self.sub_status_label.setText(f"📚 记忆：{total} | 🧠 KG：{kg_text} | 💡 想法：{intention_count}{fatigue_text}")
        except Exception as e:
            self.status_label.setText(f"❌ {str(e)[:10]}")

    def toggle_chat(self):
        self.is_chat_expanded = not self.is_chat_expanded
        self.chat_frame.setVisible(self.is_chat_expanded)
        
        if self.is_chat_expanded:
            self.setFixedSize(220, 480)  # 🔥 从460改为480，增加20像素
            self.update_avatar('working')
            self.chat_btn.setText("🙈 收起")
        else:
            self.setFixedSize(220, 280)  # 🔥 从260改为280，增加20像素
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
            
        # 强制回神并重置所有状态
        if brain.is_mind_wandering:
            brain._stop_mind_wandering()
            self.current_wandering_index = 0
        
        # 清空待执行的社交意图
        if hasattr(brain, 'pending_social_intention'):
            brain.pending_social_intention = None
        
        brain._update_interaction_time()
        brain.fatigue_level = max(0.0, brain.fatigue_level - 0.1)
        
        self.append_message("你", user_input)
        self.input_box.clear()
        self.status_label.setText("🧠 正在思考...")  # 🔥 主标签显示思考状态
        self.sub_status_label.setText("🔍 检索记忆中...")  # 🔥 次级标签显示提示
        self.update_avatar('working')
        
        self.chat_thread = ChatThread(user_input)
        self.chat_thread.response_received.connect(self.on_chat_response)
        self.chat_thread.start()

    def on_chat_response(self, response):
        if response == "抱歉，我没有这方面的信息":
            self.update_avatar('error')
            self.status_label.setText("❌ 思考失败")
        else:
            self.update_avatar('working')
            self.status_label.setText("🧠 大脑已唤醒")
        
        self.append_message("小白", response)
        self.update_brain_status()  # 🔥 自动恢复双标签显示

    def trigger_sleep(self):
        """界面原有的睡眠触发方法，完整控制所有面板状态"""
        if self.is_sleeping:
            return
            
        # 强制停止走神
        if brain.is_mind_wandering:
            brain._stop_mind_wandering()
            self.current_wandering_index = 0
        
        self.is_sleeping = True
        self.update_avatar("sleep")
        self.status_label.setText("🌙 睡眠巩固中...")  # 🔥 主标签显示睡眠状态
        self.sub_status_label.setText("💤 正在整理记忆...")  # 🔥 次级标签显示提示
        self.sleep_btn.setEnabled(False)
        self.chat_btn.setEnabled(False)
        
        # 启动后台睡眠线程
        self.sleep_thread = SleepThread()
        self.sleep_thread.finish_signal.connect(self.on_sleep_finish)
        self.sleep_thread.start()

    def on_sleep_finish(self, msg):
        self.append_message("系统", msg)
        
        # 强制重置所有状态
        self.is_sleeping = False
        brain.is_mind_wandering = False
        brain._mind_wandering_running = False
        brain.needs_sleep_request = False
        self.current_wandering_index = 0
        
        # 强制更新界面显示
        self.update_avatar('awake')
        self.update_brain_status()  # 🔥 自动更新双标签
        
        self.sleep_btn.setEnabled(True)
        self.chat_btn.setEnabled(True)
        
        logger.info("✅ 睡眠结束，所有界面状态已重置")
    
    # 窗口拖动
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_pos)
            event.accept()

    # 右键菜单
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
        
        # 局部连接高亮热力图
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
        if enable:
            brain.enable_kg()
            self.append_message("系统", "✅ 知识图谱已启用")
        else:
            brain.disable_kg()
            self.append_message("系统", "✅ 知识图谱已禁用（性能模式）")
        self.update_brain_status()

    def trigger_analyze_local(self):
        self.append_message("系统", "🔍 正在分析局部连接热力图...")
        try:
            heatmap_dir = os.path.join(self.project_root, "heatmaps/V13_both")
            core_dir = os.path.join(heatmap_dir, "core_details")
            global_dir = os.path.join(heatmap_dir, "global_blockwise")
            
            avg_global_rate = 0.0
            avg_core_rate = 0.0
            for name, expert in brain.experts.items():
                # 1. 生成全局分块聚合图
                global_rate = plot_local_connectivity_heatmap(expert, name, brain, save_dir=global_dir)
                avg_global_rate += global_rate
                
                # 2. 生成核心区细节图（带神经元-记忆标注）
                core_rate = plot_core_region_heatmap(expert, name, brain, save_dir=core_dir)
                avg_core_rate += core_rate
            
            avg_global_rate /= len(brain.experts)
            avg_core_rate /= len(brain.experts)
            
            self.append_message("系统", 
                f"✅ 双视图分析完成！\n"
                f"• 全局平均连接率: {avg_global_rate:.2f}%\n"
                f"• 核心区平均连接率: {avg_core_rate:.2f}%\n"
                f"已保存到: {heatmap_dir}")
        except Exception as e:
            self.append_message("系统", f"❌ 分析失败：{str(e)}")
            import traceback
            traceback.print_exc()

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