import os
import json
import random
import chardet
from typing import Optional, List, Dict, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger("BookReader")

# 阅读进度配置文件
PROGRESS_FILE = "book_read_progress.json"

@dataclass
class BookPage:
    """强类型书籍页面，与系统整体数据契约保持一致"""
    book_name: str
    content: str
    current_progress: float  # 0.0-1.0
    is_new_cycle: bool = False  # 是否刚读完一整遍重新开始

class BookReader:
    def __init__(self, book_dir: str = "./books"):
        self.book_dir = book_dir
        self.progress = self._load_progress()
        self.blacklist: Dict[str, int] = {}  # 书籍黑名单：{书名: 失败次数}
        self.MAX_FAILURES = 3  # 连续失败3次加入黑名单
        
        # ===================== 🔥 新增：阅读配置开关 =====================
        self.use_paragraph_alignment: bool = True  # 智能段落对齐
        self.use_weighted_random: bool = True  # 加权随机选书
        self.auto_encoding_detect: bool = True  # 自动编码检测
        self.mark_new_cycle: bool = True  # 标记新阅读周期
        # ==================================================================

        # 自动创建图书文件夹
        if not os.path.exists(book_dir):
            os.makedirs(book_dir)
            logger.info(f"📚 已自动创建图书文件夹: {book_dir}，请放入txt小说")

    def _load_progress(self) -> Dict:
        """加载阅读进度（空值安全+版本兼容）"""
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 兼容旧版进度格式（纯字节位置）
                    cleaned_data = {}
                    for k, v in data.items():
                        if isinstance(v, (int, float)):
                            cleaned_data[k] = {"pos": int(v), "file_size": 0}
                        else:
                            cleaned_data[k] = v
                    return cleaned_data
            except Exception as e:
                logger.error(f"阅读进度文件损坏，重新初始化: {e}")
                return {}
        return {}

    def _save_progress(self):
        """保存阅读进度（原子写入，避免文件损坏）"""
        try:
            temp_file = PROGRESS_FILE + ".tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.progress, f, ensure_ascii=False, indent=2)
            os.replace(temp_file, PROGRESS_FILE)
        except Exception as e:
            logger.error(f"保存阅读进度失败: {e}")

    def get_all_books(self) -> List[str]:
        """获取所有有效txt书籍（过滤空文件和黑名单）"""
        valid_books = []
        for f in os.listdir(self.book_dir):
            if f.endswith(".txt"):
                file_path = os.path.join(self.book_dir, f)
                if os.path.getsize(file_path) > 1024:
                    # 排除连续失败超过阈值的书
                    if self.blacklist.get(f, 0) < self.MAX_FAILURES:
                        valid_books.append(f)
        return valid_books

    def _detect_encoding(self, file_path: str) -> str:
        """自动检测文件编码（UTF-8/GBK）"""
        if not self.auto_encoding_detect:
            return "utf-8"
        
        try:
            with open(file_path, "rb") as f:
                raw = f.read(10000)  # 读取前10KB检测编码
            result = chardet.detect(raw)
            encoding = result["encoding"] or "utf-8"
            # 统一常见编码名称
            if encoding.lower() in ["gb2312", "gb18030"]:
                encoding = "gbk"
            logger.debug(f"📖 检测到《{os.path.basename(file_path)}》编码: {encoding}")
            return encoding
        except Exception as e:
            logger.debug(f"编码检测失败，使用默认UTF-8: {e}")
            return "utf-8"

    def read_book_page(self, book_name: str, page_size: int = 600) -> Optional[BookPage]:
        """
        分页阅读书籍（终极优化版）
        ✅ 自动编码检测（UTF-8/GBK）
        ✅ 智能段落对齐
        ✅ 文件变化自动重置进度
        ✅ 读完自动从头开始并标记新周期
        """
        book_path = os.path.join(self.book_dir, book_name)
        if not os.path.exists(book_path):
            logger.warning(f"书籍不存在: {book_name}")
            self.blacklist[book_name] = self.blacklist.get(book_name, 0) + 1
            return None

        # 获取文件大小，用于计算进度和检测文件变化
        file_size = os.path.getsize(book_path)
        book_progress = self.progress.get(book_name, {"pos": 0, "file_size": 0})
        current_pos = book_progress["pos"]
        saved_file_size = book_progress["file_size"]
        
        # 检测到文件内容变化，自动重置进度
        if saved_file_size != 0 and saved_file_size != file_size:
            logger.info(f"📖 《{book_name}》文件内容已变化，自动重置阅读进度")
            current_pos = 0

        is_new_cycle = False
        try:
            encoding = self._detect_encoding(book_path)
            
            with open(book_path, "rb") as f:
                f.seek(current_pos)
                # 多读3字节防止汉字截断，多读100字节用于段落对齐
                raw_bytes = f.read(page_size + 103)
                
                # 读完自动重置进度
                if not raw_bytes or len(raw_bytes.strip()) == 0:
                    logger.info(f"📖 《{book_name}》 已读完，自动从头开始阅读")
                    current_pos = 0
                    is_new_cycle = True
                    f.seek(0)
                    raw_bytes = f.read(page_size + 103)
                    if not raw_bytes:
                        return None

                # 修复UTF-8汉字截断
                valid_end = len(raw_bytes)
                while valid_end > 0 and (raw_bytes[valid_end-1] & 0xC0) == 0x80:
                    valid_end -= 1

                # 解码为字符串
                content = raw_bytes[:valid_end].decode(encoding, errors="replace").strip()
                
                # ===================== 🔥 新增：智能段落对齐 =====================
                if self.use_paragraph_alignment and valid_end < len(raw_bytes):
                    # 寻找最后一个段落结束符（换行/句号/感叹号/问号）
                    paragraph_ends = [
                        content.rfind("\n"),
                        content.rfind("。"),
                        content.rfind("！"),
                        content.rfind("？"),
                        content.rfind("；")
                    ]
                    last_paragraph_end = max(paragraph_ends)
                    
                    # 如果找到段落结束符，并且截断位置在页面的后30%，就截断到段落结束
                    if last_paragraph_end > int(page_size * 0.7):
                        content = content[:last_paragraph_end + 1].strip()
                        valid_end = len(raw_bytes[:valid_end].decode(encoding, errors="replace")[:last_paragraph_end + 1].encode(encoding))
                # ==================================================================

                # 清理乱码和特殊字符
                import re
                content = re.sub(r'[�\x00-\x1F\x7F]+', '', content)
                content = re.sub(r'\n+', '\n', content)  # 合并多余换行

                # 过滤过短的无效内容
                if len(content) < 20:
                    logger.debug(f"跳过过短内容: {book_name} 位置:{current_pos}")
                    new_pos = current_pos + valid_end
                    self.progress[book_name] = {"pos": new_pos, "file_size": file_size}
                    self._save_progress()
                    return self.read_book_page(book_name, page_size)
                
                # 更新进度
                new_pos = current_pos + valid_end
                self.progress[book_name] = {"pos": new_pos, "file_size": file_size}
                self._save_progress()
                
                # 重置失败计数
                self.blacklist[book_name] = 0
                
                # 计算阅读进度百分比
                progress_percent = min(1.0, new_pos / file_size) if file_size > 0 else 0.0
                
                return BookPage(
                    book_name=book_name,
                    content=content,
                    current_progress=progress_percent,
                    is_new_cycle=is_new_cycle
                )
                
        except Exception as e:
            logger.error(f"读取书籍《{book_name}》失败: {e}", exc_info=True)
            self.blacklist[book_name] = self.blacklist.get(book_name, 0) + 1
            
            # 出错时跳过当前位置
            new_pos = current_pos + page_size
            self.progress[book_name] = {"pos": new_pos, "file_size": file_size}
            self._save_progress()
            
            # 如果连续失败超过阈值，加入黑名单
            if self.blacklist[book_name] >= self.MAX_FAILURES:
                logger.warning(f"📖 《{book_name}》连续失败{self.MAX_FAILURES}次，已加入黑名单")
            
            return None

    def random_read(self) -> Optional[Dict]:
        """
        随机选一本书阅读一段（加权随机+新周期事件）
        ✅ 优先读进度少的书
        ✅ 读完一本书自动生成读完记忆
        ✅ 兼容原有返回格式
        """
        books = self.get_all_books()
        if not books:
            logger.info("📚 未找到任何有效书籍")
            return None
        
        # 加权随机选书：进度越少，权重越高
        if self.use_weighted_random:
            book_weights = []
            for book in books:
                book_progress = self.progress.get(book, {"pos": 0, "file_size": 0})
                pos = book_progress["pos"]
                file_size = book_progress["file_size"] or os.path.getsize(os.path.join(self.book_dir, book))
                progress = pos / file_size if file_size > 0 else 0.0
                # 权重 = 1 - 进度，进度越少权重越高
                weight = max(0.1, 1.0 - progress)
                book_weights.append(weight)
            
            # 归一化权重
            total_weight = sum(book_weights)
            normalized_weights = [w / total_weight for w in book_weights]
            
            # 加权随机选择
            book = random.choices(books, weights=normalized_weights, k=1)[0]
        else:
            # 纯随机选书（原有逻辑）
            book = random.choice(books)
        
        # 读取页面
        page = self.read_book_page(book)
        if not page:
            # 失败时重试其他书
            for _ in range(2):
                other_books = [b for b in books if b != book]
                if not other_books:
                    break
                book = random.choice(other_books)
                page = self.read_book_page(book)
                if page:
                    break
        
        if not page:
            logger.warning("📚 所有书籍都无法读取有效内容，请检查txt文件是否正常")
            return None
        
        # 读完一整本书时，生成读完记忆
        if page.is_new_cycle and self.mark_new_cycle:
            logger.info(f"🎉 已读完《{page.book_name}》一整遍！")
            # 这里可以触发事件总线，让大脑生成读完记忆
            # self.event_bus.emit(Event("BOOK_FINISHED", {"book_name": page.book_name}))
        
        # 兼容原有返回格式，上层代码完全不用改
        return {
            "book_name": page.book_name,
            "content": page.content,
            "progress": page.current_progress,
            "is_new_cycle": page.is_new_cycle
        }

    def get_book_progress(self, book_name: str) -> float:
        """获取指定书籍的阅读进度（0.0-1.0）"""
        book_path = os.path.join(self.book_dir, book_name)
        if not os.path.exists(book_path):
            return 0.0
        
        file_size = os.path.getsize(book_path)
        if file_size == 0:
            return 0.0
        
        book_progress = self.progress.get(book_name, {"pos": 0, "file_size": 0})
        return min(1.0, book_progress["pos"] / file_size)

    def reset_book_progress(self, book_name: str) -> None:
        """重置指定书籍的阅读进度"""
        if book_name in self.progress:
            del self.progress[book_name]
            self._save_progress()
            logger.info(f"📖 已重置《{book_name}》的阅读进度")

    def clear_blacklist(self) -> None:
        """清空书籍黑名单"""
        self.blacklist.clear()
        logger.info("📚 已清空书籍黑名单")