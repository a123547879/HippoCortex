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
from AdvancedBrainV6 import AdvancedBrainV6
from LLMBrainWrapperV4 import LLMBrainWrapperV4
from MainTest4 import analyze_brain_structure

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FinalImagePet")

# ================== 100% 完全复制 MainTest 的大脑初始化 ==================
print("=" * 60)
print("🧠 正在初始化小白大脑...")
print("=" * 60)

# 完全照搬 MainTest 的代码
brain = AdvancedBrainV6(
    dim=config.dim, 
    storage_dir=config.storage_dir, 
    ollama_model="bge-m3"
)
llm_brain = LLMBrainWrapperV4(brain)

# 知识导入检查
dataset_path = "general_knowledge.txt"
first_run_flag = os.path.join(config.storage_dir, "general_knowledge_imported")
if os.path.exists(dataset_path) and not os.path.exists(first_run_flag):
    from MainTest4 import import_knowledge_dataset
    import_knowledge_dataset(llm_brain, dataset_path, first_run_flag)
else:
    final_memory_count = brain.get_brain_status()["total_memories"]
    logger.info(f"✅ 通用知识已导入，当前总记忆数：{final_memory_count}")

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
        self.is_sleeping = False  # 标记是否正在睡眠
        
        # 获取项目绝对路径
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        print(f"\n📂 项目根目录：{self.project_root}")
        
        # 头像配置（使用绝对路径）
        self.avatar_paths = {
            "awake": os.path.join(self.project_root, "imgs", "stand.png"),
            "working": os.path.join(self.project_root, "imgs", "sit.png"),
            "sleep": os.path.join(self.project_root, "imgs", "sleep.png"),
            "error": os.path.join(self.project_root, "imgs", "error.png"),
            # "study": os.path.join(self.project_root, "imgs", "work.png")
        }
        
        # 🔥 修复：初始值设为None，强制第一次加载
        self.current_avatar = None
        
        # 检查所有头像文件
        self.check_all_avatars()
        
        self.initUI()
        
        # 定时更新状态
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_brain_status)
        self.status_timer.start(3000)

        # ================== 新增：自动睡眠定时器 ==================
        self.auto_sleep_timer = QTimer()
        self.auto_sleep_timer.timeout.connect(self.check_auto_sleep)
        self.auto_sleep_timer.start(5000)  # 每5秒检查一次

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

    # ================== 新增：自动睡眠逻辑 ==================
    def check_auto_sleep(self):
        """仅在未聊天、未睡眠时，随机触发自动睡眠"""
        if self.is_chat_expanded or self.is_sleeping:
            return
        # 10% 概率触发睡眠（可自行修改概率）
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
        
        # 🔥 修复：强制第一次加载awake状态
        self.update_avatar("awake")
        
        avatar_layout.addWidget(self.avatar_label)
        main_layout.addWidget(avatar_frame)

        # 状态显示
        self.status_label = QLabel("🧠 大脑已唤醒")
        self.status_label.setFont(QFont("SimHei", 10))
        self.status_label.setStyleSheet("color: #000000;")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        # 记忆数
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
        # 只有当状态真的变化时才更新
        if state == self.current_avatar:
            return
            
        self.current_avatar = state
        full_path = self.avatar_paths.get(state, self.avatar_paths["error"])
        
        # 强制用QImage加载，然后转QPixmap
        img = QImage(full_path)
        if not img.isNull():
            pixmap = QPixmap.fromImage(img)
            # 保持比例缩放
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
        try:
            status = brain.get_brain_status()
            total = status.get("total_memories", 0)
            self.memory_label.setText(f"📚 总记忆数：{total} 条")
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
        # self.update_avatar("working")
        self.status_label.setText("🧠 正在思考...")
        
        self.chat_thread = ChatThread(user_input)
        self.chat_thread.response_received.connect(self.on_chat_response)
        self.chat_thread.start()

    def on_chat_response(self, response):
        self.append_message("小白", response)
        # self.update_avatar("awake")
        self.status_label.setText("🧠 大脑已唤醒")

    def trigger_sleep(self):
        if self.is_sleeping:
            return
        self.is_sleeping = True  # 标记睡眠中
        self.update_avatar("sleep")
        self.status_label.setText("🌙 睡眠巩固中...")
        self.sleep_btn.setEnabled(False)
        self.chat_btn.setEnabled(False)  # 睡眠时禁用聊天
        
        self.sleep_thread = SleepThread()
        self.sleep_thread.finish_signal.connect(self.on_sleep_finish)
        self.sleep_thread.start()

    def on_sleep_finish(self, msg):
        self.append_message("系统", msg)
        self.update_avatar("awake")
        self.status_label.setText("🧠 大脑已唤醒")
        self.sleep_btn.setEnabled(True)
        self.chat_btn.setEnabled(True)  # 恢复聊天
        self.is_sleeping = False  # 解除睡眠标记

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
        
        analyze_action = QAction("🔍 分析突触结构", self)
        analyze_action.triggered.connect(self.trigger_analyze)
        
        save_action = QAction("💾 保存大脑数据", self)
        save_action.triggered.connect(self.save_brain)
        
        exit_action = QAction("🚪 退出", self)
        exit_action.triggered.connect(self.exit_app)
        
        menu.addAction(analyze_action)
        menu.addAction(save_action)
        menu.addSeparator()
        menu.addAction(exit_action)
        
        menu.exec_(event.globalPos())

    def trigger_analyze(self):
        self.append_message("系统", "🔍 正在分析突触结构...")
        analyze_brain_structure(brain)
        self.append_message("系统", "✅ 分析完成！热力图已保存")

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