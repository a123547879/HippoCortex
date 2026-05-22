import os
import logging
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QMenu, QLineEdit, QTextEdit, QAction, QFrame, QFileDialog)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QPixmap, QFont, QTextCursor, QImage

from ChatThread import ChatThread
from .sleep_thread import SleepThread
from utils.heatmap_generator import plot_core_region_heatmap, plot_local_connectivity_heatmap

logger = logging.getLogger("BrainPetWindow")


class XiaobaiBrainPet(QWidget):
    def __init__(self, brain, cognitive_system, core, llm_brain, mm_gateway, brain_interface):
        super().__init__()
        # 注入所有业务依赖（无全局变量）
        self.brain = brain
        self.cognitive_system = cognitive_system
        self.core = core
        self.llm_brain = llm_brain
        self.mm_gateway = mm_gateway
        self.brain_interface = brain_interface
        
        # UI状态
        self.drag_pos = QPoint()
        self.is_chat_expanded = False
        self.is_sleeping = False
        self.selected_image_path = None
        
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        
        self.initUI()
        self.check_all_avatars()
        self.init_timers()

    def init_timers(self):
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
            intention = self.brain.get_pending_social_intention()
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
            if self.brain.is_mind_wandering and not self.is_sleeping:
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
            if self.brain.check_and_consume_sleep_request():
                logger.info("🎯 收到大脑睡眠请求，界面接管睡眠流程...")
                self.trigger_sleep(is_manual=False)
                return
            self.brain._check_mind_wandering_trigger()
            if self.brain.is_mind_wandering:
                self.current_wandering_index = 1 - self.current_wandering_index
                current_wandering_state = self.wandering_avatar_list[self.current_wandering_index]
                if self.current_avatar != current_wandering_state:
                    self.update_avatar(current_wandering_state)
                    self.status_label.setText("🌙 走神中...")
                fatigue_pct = int(self.brain.fatigue_level * 100)
                intention_count = self.brain.get_brain_status().get("intention_queue_size", 0)
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
        self.sleep_btn.clicked.connect(lambda: self.trigger_sleep(is_manual=True))
        
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
            logger.info(f"✅ 成功加载头像：{state}")
        else:
            self.avatar_label.setText(f"❌ 图片加载失败\n{os.path.basename(full_path)}")
            self.avatar_label.setStyleSheet("color: #ef4444; font-size: 10px;")
            logger.warning(f"❌ 头像加载失败：{full_path}")

    def update_brain_status(self):
        if self.is_sleeping:
            return
        try:
            if self.brain.is_mind_wandering or self.is_sleeping:
                return
                
            status = self.brain.get_brain_status() or {}
            total = status.get("total_memories", 0)
            kg_enabled = status.get("kg_enabled", True)
            intention_count = status.get("intention_queue_size", 0)
            
            self.status_label.setText("🧠 大脑已唤醒")
            
            kg_text = "开" if kg_enabled else "关"
            fatigue_text = ""
            if self.brain.fatigue_level > 0:
                fatigue_pct = int(self.brain.fatigue_level * 100)
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
            
        if self.brain.is_mind_wandering:
            self.brain._stop_mind_wandering()
            self.current_wandering_index = 0
        
        if hasattr(self.cognitive_system, 'pending_social_intention'):
            self.cognitive_system.pending_social_intention = None
        
        self.brain._update_interaction_time()
        self.brain.fatigue_level = max(0.0, self.brain.fatigue_level - 0.1)
        
        self.append_message("你", user_input if user_input else "发送了一张图片", has_img=bool(self.selected_image_path))
        self.input_box.clear()
        self.status_label.setText("🧠 正在思考...")
        self.sub_status_label.setText("🔍 检索记忆中...")
        self.update_avatar('working')
        
        self.chat_thread = ChatThread(
            user_input, 
            self.selected_image_path, 
            brain=self.brain, 
            mm_gateway=self.mm_gateway, 
            llm_brain=self.llm_brain
        )
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

    def trigger_sleep(self, is_manual: bool):
        if self.is_sleeping:
            return
            
        if self.brain.is_mind_wandering:
            self.brain._stop_mind_wandering()
            self.current_wandering_index = 0
        
        self.is_sleeping = True
        self.update_avatar("sleep")
        self.status_label.setText("🌙 睡眠巩固中...")
        self.sub_status_label.setText("💤 正在进入浅睡...")
        self.sleep_btn.setEnabled(False)
        self.chat_btn.setEnabled(False)
        
        self.sleep_thread = SleepThread(self.brain_interface, is_manual=is_manual)
        self.sleep_thread.progress_signal.connect(self.update_sleep_progress)
        self.sleep_thread.finish_signal.connect(self.on_sleep_finish)
        self.sleep_thread.start()

    def update_sleep_progress(self, progress: int, message: str):
        self.status_label.setText(f"🌙 睡眠中... {progress}%")
        self.sub_status_label.setText(message)
        logger.debug(f"睡眠进度: {progress}% - {message}")

    def on_sleep_finish(self, sleep_report):
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
        self.core.is_mind_wandering = False
        self.cognitive_system._mind_wandering_running = False
        self.core.needs_sleep_request = False
        self.current_wandering_index = 0
        
        self.brain.save_all()
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
        
        kg_status = self.brain.get_brain_status().get("kg_enabled", True)
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
            self.brain.enable_kg()
            self.append_message("系统", "✅ 知识图谱已启用")
        else:
            self.brain.disable_kg()
            self.append_message("系统", "✅ 知识图谱已禁用（性能模式）")
        self.update_brain_status()

    def trigger_analyze_local(self):
        self.append_message("系统", "🔍 正在分析局部连接热力图...")
        try:
            heatmap_dir = os.path.join(self.project_root, "heatmaps/V16_both")
            core_dir = os.path.join(heatmap_dir, "core_details")
            global_dir = os.path.join(heatmap_dir, "global_blockwise")
            
            avg_global_rate = 0.0
            avg_core_rate = 0.0
            expert_count = len(self.cognitive_system.experts)
            if expert_count == 0:
                self.append_message("系统", "❌ 无专家模块，无法分析")
                return
                
            for name, expert in self.cognitive_system.experts.items():
                global_rate = plot_local_connectivity_heatmap(expert, name, self.cognitive_system, save_dir=global_dir)
                avg_global_rate += global_rate
                
                core_rate = plot_core_region_heatmap(expert, name, self.cognitive_system, save_dir=core_dir)
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
            self.brain.save_all()
            self.append_message("系统", "✅ 大脑数据已保存！")
        except Exception as e:
            self.append_message("系统", f"❌ 保存失败：{str(e)[:30]}")

    def exit_app(self):
        try:
            self.brain.save_all()
            self.brain_interface.stop()
            logger.info("✅ 程序正常退出，数据已保存")
        except Exception as e:
            logger.error(f"❌ 退出时保存失败：{e}")
        QApplication.quit()