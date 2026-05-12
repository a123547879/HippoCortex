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
import random
from datetime import datetime
from collections import defaultdict
from PIL import Image
from pathlib import Path

# 新增：文件对话框
from PyQt5.QtWidgets import QFileDialog

plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from BrainConfig import config
# ===================== 🔥 修改：导入新架构 =====================
from brain_core import BrainCore
from Cognitive_systemV2 import CognitiveSystem
from brain_interface import BrainInterface
from event_system import EventBus, Event, EventType, on_event
# ================================================================
# 🔥 修复1：修正LLM包装器导入名称（匹配你的文件名）
from LLMBrainWrapperV7 import LLMBrainWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ===================== 🔥 新增：导入多模态输入网关 =====================
from Multimodal_gateway import MultiModalInputGateway
# =====================================================================
from ChatThread import ChatThread
from Data_models import SleepReport, Intention, ThoughtResult
from DummyBrain import DummyBrain

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FinalImagePet")

def plot_core_region_heatmap(expert, name, cognitive_system, save_dir="heatmaps/V15_core"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}_core_detail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    synapse = expert.synapse.data.cpu().numpy()
    dim = expert.dim
    plot_size = min(200, dim)
    partition_size = int(dim * 0.2)

    synapse_plot = synapse[:plot_size, :plot_size]

    neuron_activation_count = np.zeros(dim)
    neuron_to_memories = defaultdict(list)
    for mem_id, mem in cognitive_system.cortex.index.memories.items():
        # ✅ 修复1：字典[] → 对象.属性
        if mem.metadata.get("expert") != name:
            continue
        # ✅ 修复2：字典in判断 → 对象hasattr判断（MemoryPacket必含sdr，可直接删除此行）
        if hasattr(mem, 'sdr'):
            # ✅ 修复3：字典[] → 对象.属性
            sdr = mem.sdr
            content = mem.content
            active_neurons = torch.where(sdr > 0.1)[0].numpy()
            for neuron_id in active_neurons:
                if neuron_id < plot_size:
                    neuron_activation_count[neuron_id] += 1
                    neuron_to_memories[neuron_id].append(content[:35] + "...")

    plt.figure(figsize=(12, 10))
    plt.rcParams['font.size'] = 9

    mask = np.zeros((plot_size, plot_size))
    mask[:min(partition_size, plot_size), :min(partition_size, plot_size)] = 1
    heatmap_data = synapse_plot * mask * 2.5 + synapse_plot * (1 - mask) * 0.2
    im = plt.imshow(heatmap_data, cmap='coolwarm', vmin=-2.5, vmax=2.5)
    cbar = plt.colorbar(im, label='突触权重（强化局部连接）', shrink=0.8)

    top_neurons = np.argsort(neuron_activation_count)[::-1][:10]
    top_neurons = sorted(top_neurons)
    
    cleaned_mem = {}
    for nid, mem_list in neuron_to_memories.items():
        unique_mem = list(dict.fromkeys(mem_list))[:1]
        cleaned_mem[nid] = unique_mem

    directions = [
        (5, 0, 'left'),
        (0, 5, 'center'),
        (-5, 0, 'right'),
        (0, -5, 'center')
    ]
    
    used_labels = set()

    for idx, neuron_id in enumerate(top_neurons):
        if neuron_id >= plot_size or neuron_activation_count[neuron_id] == 0:
            continue
        
        plt.scatter(neuron_id, neuron_id, s=90, c='white', marker='+', linewidths=2,
                   label=f'神经元{neuron_id}({int(neuron_activation_count[neuron_id])})')
        
        if neuron_id not in cleaned_mem or len(cleaned_mem[neuron_id]) == 0:
            continue

        mem_text = cleaned_mem[neuron_id][0]
        while mem_text in used_labels:
            mem_text += f"[{neuron_id}]"
        used_labels.add(mem_text)

        offset_x, offset_y, ha = directions[idx % 4]

        plt.text(
            neuron_id + offset_x,
            neuron_id + offset_y,
            mem_text,
            fontsize=7.5,
            ha=ha,
            va='center',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='yellow', alpha=0.95)
        )

    plt.gca().add_patch(plt.Rectangle((-0.5, -0.5), min(partition_size, plot_size), min(partition_size, plot_size),
                                       linewidth=3, edgecolor='gold', linestyle='--', facecolor='none',
                                       label='核心功能区'))

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
    
    plt.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.2, 1))

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n✅ [{name}] 核心区细节图已保存: {save_path}")
    return local_rate


# ✅ 此函数完全不需要修改（没有访问MemoryPacket对象）
def plot_local_connectivity_heatmap(expert, name, cognitive_system, save_dir="heatmaps/V15"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}_blockwise_2048_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    synapse = expert.synapse.data.cpu().numpy()
    dim = expert.dim
    full_dim = 2048

    block_size = 64
    n_blocks = full_dim // block_size

    block_heatmap = np.zeros((n_blocks, n_blocks))
    for i in range(n_blocks):
        for j in range(n_blocks):
            si, ei = i*block_size, (i+1)*block_size
            sj, ej = j*block_size, (j+1)*block_size
            block = synapse[si:ei, sj:ej]
            block_heatmap[i, j] = np.mean(np.abs(block))

    core_size = int(full_dim * 0.2)
    core_blocks = int(np.ceil(core_size / block_size))

    plt.figure(figsize=(12, 10))
    plt.rcParams['font.size'] = 10

    im = plt.imshow(block_heatmap, cmap='coolwarm', vmin=0, vmax=np.percentile(block_heatmap, 99))
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Average Synapse Strength | 块平均连接强度', fontsize=12)

    plt.gca().add_patch(plt.Rectangle(
        (-0.5, -0.5), core_blocks, core_blocks,
        linewidth=3, edgecolor='gold', linestyle='--',
        facecolor='none', label='Core Region | 核心区'
    ))

    plt.xticks(ticks=np.arange(0, n_blocks, 4), labels=[f"{i*64}" for i in range(0, n_blocks, 4)], fontsize=8)
    plt.yticks(ticks=np.arange(0, n_blocks, 4), labels=[f"{i*64}" for i in range(0, n_blocks, 4)], fontsize=8)
    plt.xlabel('Neuron Block (64 neurons/block) | 神经元块', fontsize=12)
    plt.ylabel('Neuron Block (64 neurons/block) | 神经元块', fontsize=12)

    total_connections = np.sum(np.abs(synapse) > 0.1)
    core_connections = np.sum(np.abs(synapse[:core_size, :core_size]) > 0.1)
    core_rate = (core_connections / total_connections) * 100 if total_connections > 0 else 0

    plt.title(
        f'[{name}] 2048维突触 · 分块聚合热力图\n'
        f'核心区连接占比: {core_rate:.1f}% | 总稀疏度: {1-total_connections/(2048*2048):.2%}\n'
        f'时间: [{datetime.now().strftime("%Y:%m:%d %H:%M:%S")}]',
        fontsize=14
    )
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"\n✅ [{name}] 2048维完整热力图已保存: {save_path}")
    return core_rate

# ========================================================================

print("=" * 60)
print("🧠 正在初始化小白大脑 + 多模态网关...")
print("=" * 60)
print("💡 提示：请确保 BrainConfig.py 中 local_bias_strength = 1.2，热力图效果更明显！")

# 初始化LLM
llm = ChatOllama(model=config.ollama_model_name)

# ===================== 🔥 修改：初始化新架构大脑 =====================
# 1. 初始化Embedding模型
embedding_model = OllamaEmbeddings(model="bge-m3")

# 2. 创建大脑接口
brain_interface = BrainInterface(embedding_model, llm, kg_enabled=True)

# 3. 启动大脑
brain_interface.start(storage_dir=config.storage_dir)

# 4. 获取认知系统引用（用于直接访问内部组件）
cognitive_system = brain_interface.cognitive_system
core = brain_interface.core

# 创建兼容对象
brain = DummyBrain(cognitive_system, brain_interface)
llm_brain = LLMBrainWrapper(brain)
# =====================================================================

# ===================== 🔥 初始化多模态网关（从config读取设备） =====================
mm_gateway = MultiModalInputGateway(device=config.device)
logger.info("✅ 多模态图文网关初始化完成！支持图片+文字同时学习")
# =================================================================

# 知识导入检查（增加异常防护）
dataset_path = r"HippoCortexV6-2\data_text\all_know.txt"
first_run_flag = os.path.join(config.storage_dir, "general_knowledge_imported")
try:
    if os.path.exists(dataset_path) and not os.path.exists(first_run_flag):
        from MainTest5 import import_knowledge_dataset
        import_knowledge_dataset(llm_brain, dataset_path, first_run_flag, use_kg=False)
    else:
        # 🔥 修复3：空值安全防护
        status = brain.get_brain_status() or {}
        final_memory_count = status.get("total_memories", 0)
        kg_status = "✅ 已启用" if status.get("kg_enabled", True) else "⚡ 已关闭（性能模式）"
        intention_count = status.get("intention_queue_size", 0)
        logger.info(f"✅ 通用知识已导入 | 总记忆数：{final_memory_count} | 知识图谱：{kg_status} | 意图队列：{intention_count}")
except Exception as e:
    logger.warning(f"⚠️ 知识导入跳过: {e}")

print("\n" + "=" * 60)
print("✅ 大脑+多模态网关初始化完成！现在启动桌宠界面...")
print("=" * 60)

# ================== 第二阶段：导入PyQt5 ==================
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QMenu, QLineEdit, QTextEdit, QAction, QFrame)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QFont, QTextCursor, QImage

# ================== 后台工作线程 ==================
class SleepThread(QThread):
    progress_signal = pyqtSignal(int, str)
    finish_signal = pyqtSignal(SleepReport)  # 明确发射SleepReport对象
    
    def __init__(self, is_manual: bool = False):
        super().__init__()
        self.is_manual = is_manual
        
    def run(self):
        try:
            def on_sleep_progress(event):
                self.progress_signal.emit(event.data["progress"], event.data["message"])
            
            brain_interface.event_bus.subscribe(EventType.SLEEP_PROGRESS_UPDATED, on_sleep_progress)
            
            sleep_report = brain_interface.trigger_sleep(is_manual=self.is_manual)
            brain.save_all()
            
            self.finish_signal.emit(sleep_report)
            
        except Exception as e:
            logger.error(f"❌ 睡眠失败：{str(e)}", exc_info=True)
            self.finish_signal.emit(SleepReport(error=str(e)))
        finally:
            brain_interface.event_bus.unsubscribe(EventType.SLEEP_PROGRESS_UPDATED, on_sleep_progress)

# ================== 小白大脑桌宠主界面 ==================
class XiaobaiBrainPet(QWidget):
    def __init__(self):
        super().__init__()
        self.drag_pos = QPoint()
        self.is_chat_expanded = False
        self.is_sleeping = False
        self.selected_image_path = None
        
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        print(f"\n📂 项目根目录：{self.project_root}")
        
        self.avatar_paths = {
            "awake": os.path.join(self.project_root, "imgs", "stand.png"),
            "working": os.path.join(self.project_root, "imgs", "sit.png"),
            "sleep": os.path.join(self.project_root, "imgs", "sleep.png"),
            "wandering01": os.path.join(self.project_root, "imgs", "wandering01.png"),
            "wandering02": os.path.join(self.project_root, "imgs", "wandering02.png"),
            "error": os.path.join(self.project_root, "imgs", "error.png")
        }
        
        self.current_wandering_index = 0
        self.wandering_avatar_list = ["wandering01", "wandering02"]
        self.current_avatar = None
        self.check_all_avatars()
        self.initUI()
        
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_brain_status)
        self.status_timer.start(3000)

        self.mind_wandering_timer = QTimer()
        self.mind_wandering_timer.timeout.connect(self.check_mind_wandering)
        self.mind_wandering_timer.start(2000)

        self.intention_check_timer = QTimer()
        self.intention_check_timer.timeout.connect(self.check_and_execute_intentions)
        self.intention_check_timer.start(3000)
        logger.info("✅ 主动意图检查定时器已启动")

    def check_all_avatars(self):
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

    def check_and_execute_intentions(self):
        try:
            if self.is_sleeping:
                return
            logger.debug(f"🔍 检查主动意图 | 聊天展开: {self.is_chat_expanded} | 睡眠中: {self.is_sleeping}")
            intention = brain.get_pending_social_intention()
            if intention:
                logger.info(f"💬 显示主动意图: {intention.content}")
                self.append_message("小白", intention.content)
                self.update_avatar('working')
                self.status_label.setText("💬 有话想对你说...")
                QTimer.singleShot(3000, self._resume_wandering_state)
        except Exception as e:
            logger.error(f"❌ 意图执行失败: {e}", exc_info=True)

    def _resume_wandering_state(self):
        try:
            if brain.is_mind_wandering and not self.is_sleeping:
                current_wandering_state = self.wandering_avatar_list[self.current_wandering_index]
                self.update_avatar(current_wandering_state)
                self.status_label.setText("🌙 走神中...")
                self.update_brain_status()
            else:
                self.update_avatar('awake')
                self.status_label.setText("🧠 大脑已唤醒")
        except Exception as e:
            logger.debug(f"恢复走神状态失败: {e}")

    def check_mind_wandering(self):
        try:
            if self.is_sleeping:
                return
            if brain.check_and_consume_sleep_request():
                logger.info("🎯 收到大脑睡眠请求，界面接管睡眠流程...")
                self.is_sleeping = True
                self.update_avatar("sleep")
                self.status_label.setText("🌙 睡眠巩固中...")
                self.sub_status_label.setText("💤 正在进入浅睡...")
                self.sleep_btn.setEnabled(False)
                self.chat_btn.setEnabled(False)
                
                self.sleep_thread = SleepThread(is_manual=False)
                self.sleep_thread.progress_signal.connect(self.update_sleep_progress)
                self.sleep_thread.finish_signal.connect(self.on_sleep_finish)
                self.sleep_thread.start()
                return
            brain._check_mind_wandering_trigger()
            if brain.is_mind_wandering:
                self.current_wandering_index = 1 - self.current_wandering_index
                current_wandering_state = self.wandering_avatar_list[self.current_wandering_index]
                if self.current_avatar != current_wandering_state:
                    self.update_avatar(current_wandering_state)
                    self.status_label.setText("🌙 走神中...")
                fatigue_pct = int(brain.fatigue_level * 100)
                intention_count = brain.get_brain_status().get("intention_queue_size", 0)
                self.sub_status_label.setText(f"💪 疲劳：{fatigue_pct}% | 💡 想法：{intention_count}")
            else:
                self.update_brain_status()
                if self.current_avatar not in ['awake', 'working', 'sleep', 'error']:
                    self.update_avatar('awake')
                    self.current_wandering_index = 0
        except Exception as e:
            logger.debug(f"走神状态检查失败: {e}")

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.selected_image_path = file_path
            pix = QPixmap(file_path).scaled(80,80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img_preview_label.setPixmap(pix)
            self.img_preview_label.setVisible(True)
            logger.info(f"📷 已选择图片: {file_path}")
        else:
            self.clear_selected_image()

    def clear_selected_image(self):
        self.selected_image_path = None
        self.img_preview_label.clear()
        self.img_preview_label.setVisible(False)

    def initUI(self):
        self.setWindowTitle("🧠 小白大脑桌宠 (图文多模态版)")
        self.setFixedSize(220, 300)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(8)

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

        self.status_label = QLabel("🧠 大脑已唤醒")
        self.status_label.setFont(QFont("SimHei", 11))
        self.status_label.setStyleSheet("color: #000000; font-weight: bold;")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        self.sub_status_label = QLabel("📚 记忆：0 | 🧠 KG：开")
        self.sub_status_label.setFont(QFont("SimHei", 9))
        self.sub_status_label.setStyleSheet("color: #94a3b8;")
        self.sub_status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.sub_status_label)

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

        # 聊天区域
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

        # 图文输入行
        input_layout = QHBoxLayout()
        input_layout.setSpacing(5)

        self.img_preview_label = QLabel()
        self.img_preview_label.setFixedSize(30,30)
        self.img_preview_label.setVisible(False)

        self.select_img_btn = QPushButton("📷")
        self.select_img_btn.setFixedSize(30,24)
        self.select_img_btn.setStyleSheet("""
            QPushButton {
                background-color: #6366f1;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 10px;
            }
        """)
        self.select_img_btn.clicked.connect(self.select_image)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("文字 + 可选图片...")
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
        
        input_layout.addWidget(self.select_img_btn)
        input_layout.addWidget(self.img_preview_label)
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
        try:
            if brain.is_mind_wandering or self.is_sleeping:
                return
                
            status = brain.get_brain_status() or {}
            total = status.get("total_memories", 0)
            kg_enabled = status.get("kg_enabled", True)
            intention_count = status.get("intention_queue_size", 0)
            
            self.status_label.setText("🧠 大脑已唤醒")
            
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
            self.setFixedSize(220, 480)
            self.update_avatar('working')
            self.chat_btn.setText("🙈 收起")
        else:
            self.setFixedSize(220, 300)
            self.update_avatar('awake')
            self.chat_btn.setText("💬 聊天")
            self.clear_selected_image()

    def append_message(self, sender, text, has_img=False):
        if sender == "你":
            color = "#4cc9f0"
            align = "right"
        else:
            color = "#f1f5f9"
            align = "left"
            
        img_tip = " [附带图片]" if has_img else ""
        html = f"""
        <div style='text-align: {align}; margin: 5px 0;'>
            <div style='display: inline-block; background-color: {"#334155" if sender == "你" else "#1e293b"}; 
                        padding: 6px 10px; border-radius: 8px; max-width: 90%;'>
                <span style='color: #94a3b8; font-size: 8px;'>{sender}{img_tip}</span><br>
                <span style='color: {color}; font-size: 10px;'>{text}</span>
            </div>
        </div>
        """
        self.chat_history.append(html)
        self.chat_history.moveCursor(QTextCursor.End)

    def send_message(self):
        user_input = self.input_box.text().strip()
        if not user_input and not self.selected_image_path:
            return
            
        if brain.is_mind_wandering:
            brain._stop_mind_wandering()
            self.current_wandering_index = 0
        
        if hasattr(cognitive_system, 'pending_social_intention'):
            cognitive_system.pending_social_intention = None
        
        brain._update_interaction_time()
        brain.fatigue_level = max(0.0, brain.fatigue_level - 0.1)
        
        self.append_message("你", user_input if user_input else "发送了一张图片", has_img=bool(self.selected_image_path))
        self.input_box.clear()
        self.status_label.setText("🧠 正在思考...")
        self.sub_status_label.setText("🔍 检索记忆中...")
        self.update_avatar('working')
        
        self.chat_thread = ChatThread(user_input, self.selected_image_path, brain=brain, mm_gateway=mm_gateway, llm_brain=llm_brain)
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
        self.update_brain_status()
        self.clear_selected_image()

    def trigger_sleep(self):
        if self.is_sleeping:
            return
            
        if brain.is_mind_wandering:
            brain._stop_mind_wandering()
            self.current_wandering_index = 0
        
        self.is_sleeping = True
        self.update_avatar("sleep")
        self.status_label.setText("🌙 睡眠巩固中...")
        self.sub_status_label.setText("💤 正在进入浅睡...")
        self.sleep_btn.setEnabled(False)
        self.chat_btn.setEnabled(False)
        
        self.sleep_thread = SleepThread(is_manual=True)
        self.sleep_thread.progress_signal.connect(self.update_sleep_progress)
        self.sleep_thread.finish_signal.connect(self.on_sleep_finish)
        self.sleep_thread.start()

    def update_sleep_progress(self, progress: int, message: str):
        self.status_label.setText(f"🌙 睡眠中... {progress}%")
        self.sub_status_label.setText(message)
        logger.debug(f"睡眠进度: {progress}% - {message}")

    # 🔥 修复4：核心！修正睡眠完成接收逻辑（最关键的崩溃点）
    def on_sleep_finish(self, sleep_report: SleepReport):
        if sleep_report.error:
            self.append_message("系统", f"❌ 睡眠失败：{sleep_report.error}")
        else:
            self.append_message("系统", 
                f"✅ 睡眠巩固完成！\n"
                f"• 总记忆数：{sleep_report.total_memories}\n"
                f"• 巩固成功：{sleep_report.consolidated_count}条\n"
                f"• 主动遗忘：{sleep_report.forgotten_count}条\n"
                f"• 睡眠质量：{sleep_report.quality_rating} ({sleep_report.quality_score}分)\n"
                f"• 睡眠时长：{sleep_report.sleep_duration}秒")
            
            if sleep_report.dream_content:
                self.append_message("小白", f"😴 我刚刚做了一个梦：\n{sleep_report.dream_content}")
        
        # 重置状态
        self.is_sleeping = False
        core.is_mind_wandering = False
        cognitive_system._mind_wandering_running = False
        core.needs_sleep_request = False
        self.current_wandering_index = 0
        
        self.update_avatar('awake')
        self.update_brain_status()
        
        self.sleep_btn.setEnabled(True)
        self.chat_btn.setEnabled(True)
        
        logger.info("✅ 睡眠结束，所有界面状态已重置")
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_pos)
            event.accept()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        
        kg_status = brain.get_brain_status().get("kg_enabled", True)
        if kg_status:
            kg_action = QAction("⚡ 禁用知识图谱（性能模式）", self)
            kg_action.triggered.connect(lambda: self.toggle_kg(False))
        else:
            kg_action = QAction("🧠 启用知识图谱", self)
            kg_action.triggered.connect(lambda: self.toggle_kg(True))
        
        analyze_action = QAction("🔍 分析局部连接热力图", self)
        analyze_action.triggered.connect(self.trigger_analyze_local)
        
        save_action = QAction("💾 保存大脑数据", self)
        save_action.triggered.connect(self.save_brain)
        
        exit_action = QAction("🚪 退出", self)
        exit_action.triggered.connect(self.exit_app)
        
        menu.addAction(kg_action)
        menu.addSeparator()
        menu.addAction(analyze_action)
        menu.addAction(save_action)
        menu.addSeparator()
        menu.addAction(exit_action)
        
        menu.exec_(event.globalPos())

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
            heatmap_dir = os.path.join(self.project_root, "heatmaps/V15_both")
            core_dir = os.path.join(heatmap_dir, "core_details")
            global_dir = os.path.join(heatmap_dir, "global_blockwise")
            
            avg_global_rate = 0.0
            avg_core_rate = 0.0
            expert_count = len(cognitive_system.experts)
            if expert_count == 0:
                self.append_message("系统", "❌ 无专家模块，无法分析")
                return
                
            for name, expert in cognitive_system.experts.items():
                global_rate = plot_local_connectivity_heatmap(expert, name, cognitive_system, save_dir=global_dir)
                avg_global_rate += global_rate
                
                core_rate = plot_core_region_heatmap(expert, name, cognitive_system, save_dir=core_dir)
                avg_core_rate += core_rate
            
            avg_global_rate /= expert_count
            avg_core_rate /= expert_count
            
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
            brain_interface.stop()
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