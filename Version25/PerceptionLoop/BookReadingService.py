import os
import torch
import torch.nn.functional as F
import logging
import re
import uuid
import random, datetime
import time
import json
from typing import List, Dict, Optional, Any

from ILifecycle import ILifecycle, IService
from ServiceContainer import ServiceContainer
from event_system import EventBus, Event, EventType
from BookReader import BookReader
from BrainConfig import config

logger = logging.getLogger("BookReadingService")

# 皮层外部存储目录（模拟大脑皮层，存储原始细节）
CORTEX_STORAGE_DIR = "./cortex_book_details"
# 书籍批次摘要缓存（用于生成全书级总结）
BOOK_BATCH_SUMMARIES: Dict[str, List[str]] = {}

class BookReadingService(ILifecycle, IService):
    def __init__(self):
        self.book_reader = BookReader()
        self.read_mode_prob = 0.7
        
        self._imagination_counter: int = 0
        self._book_read_counter: int = 0
        self._book_content_buffer: List[str] = []  # 片段级缓冲（不直接存）
        self._book_current_batch: List[Dict] = []  # 缓存当前批次的原始细节（含multimodal）
        self.reading_interval: int = 12
        
        self._container = None
        # 初始化皮层存储目录
        os.makedirs(CORTEX_STORAGE_DIR, exist_ok=True)
    
    def initialize(self, container: 'ServiceContainer') -> None:
        self._container = container
        EventBus().subscribe(EventType.MIND_WANDER_STOPPED, self._on_mind_wander_stopped)
    
    def start(self) -> None:
        pass
    
    def stop(self) -> None:
        pass
    
    def save(self, storage_dir: str) -> None:
        # 保存书籍批次摘要缓存，避免程序重启后丢失，无法生成全书总结
        try:
            cache_path = os.path.join(storage_dir, "book_batch_summaries.json")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(BOOK_BATCH_SUMMARIES, f, ensure_ascii=False, indent=2)
            logger.info("✅ 书籍批次摘要缓存已保存")
        except Exception as e:
            logger.error(f"保存书籍批次摘要缓存失败: {e}")
    
    def load(self, storage_dir: str) -> None:
        # 加载书籍批次摘要缓存
        try:
            cache_path = os.path.join(storage_dir, "book_batch_summaries.json")
            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    global BOOK_BATCH_SUMMARIES
                    BOOK_BATCH_SUMMARIES = json.load(f)
                logger.info("✅ 书籍批次摘要缓存已加载")
        except Exception as e:
            logger.error(f"加载书籍批次摘要缓存失败: {e}")
    
    def _on_mind_wander_started(self, event: Event):
        logger.info("📚 书籍阅读服务准备就绪")
    
    def _on_mind_wander_stopped(self, event: Event):
        self._imagination_counter = 0
        self._book_read_counter = 0
        self._book_content_buffer = []
        self._book_current_batch = []
        logger.info("📚 书籍阅读服务暂停")

    # ===================== 辅助方法：皮层存储路径生成 =====================
    def _get_cortex_storage_path(self, book_name: str) -> str:
        """生成皮层外部存储文件路径（模拟皮层存储原始细节）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 清理书名特殊字符，避免文件路径异常
        clean_book_name = re.sub(r'[\/:*?"<>|]', '', book_name)
        return os.path.join(CORTEX_STORAGE_DIR, f"{clean_book_name}_{timestamp}.json")

    # ===================== 辅助方法：摘要有效性校验 =====================
    def _validate_summary(self, summary: str, book_name: str) -> bool:
        """校验摘要有效性：必须包含至少1个实体，长度符合要求"""
        if len(summary) < 15 or len(summary) > 50:
            return False
        # 复用实体提取方法，校验摘要是否包含有效实体
        entity = self.extract_book_entity_llm(summary)
        return entity != "书籍内容"

    # ===================== 辅助方法：向量相似度计算（去重） =====================
    def _calculate_vec_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """计算两个向量的余弦相似度，用于重复摘要去重"""
        vec1_norm = F.normalize(vec1.squeeze(), p=2, dim=-1)
        vec2_norm = F.normalize(vec2.squeeze(), p=2, dim=-1)
        return torch.dot(vec1_norm, vec2_norm).item()

    # ===================== 三级压缩漏斗实现 =====================
    # 1. 片段级压缩（仅缓冲，不入海马体）
    def _summarize_segment(self, content: str, book_name: str) -> str:
        """单段600字内容 → 20-30字核心描述（仅缓冲，用于批次级压缩）"""
        try:
            llm = self._container.llm
            prompt = f"""
            对《{book_name}》的这段内容做极简核心描述（20-30字）：
            {content[:400]}
            
            规则：
            1. 只保留实体、关系，丢弃场景描写细节
            2. 不添加任何解释，语句通顺
            3. 格式：[主体] [动作/关系] [客体/结果]
            输出：
            """
            response = llm.invoke(prompt)
            seg_summary = response.content.strip()
            seg_summary = re.sub(r'[�\x00-\x1F\x7F\"\'\n]', '', seg_summary)
            # 兜底：若不符合长度，截取核心内容
            if not (20 <= len(seg_summary) <= 30):
                seg_summary = content[:28].strip() + "..."
            return seg_summary
        except Exception as e:
            logger.debug(f"片段级压缩失败，兜底处理: {e}")
            return content[:28].strip() + "..."

    # 2. 批次级压缩（5段合并 → 1-2句话摘要，存海马体）
    def _summarize_batch(self, book_name: str, segment_summaries: List[str]) -> str:
        """5段片段摘要 → 1-2句话摘要（≤50字，海马体核心索引）"""
        try:
            llm = self._container.llm
            merged_seg = " | ".join(segment_summaries)
            prompt = f"""
            对《{book_name}》的以下片段摘要，合并压缩为1-2句话（≤50字）：
            {merged_seg}
            
            规则：
            1. 保留核心实体、关系、情绪基调，丢弃冗余
            2. 格式固定：《书名》中，[主体] [做了什么/发生了什么]，[情绪/结果]。
            3. 不添加任何额外内容，语句通顺
            输出：
            """
            response = llm.invoke(prompt)
            batch_summary = response.content.strip()
            batch_summary = re.sub(r'[�\x00-\x1F\x7F\"\'\n]', '', batch_summary)
            # 校验有效性，无效则重试1次
            if not self._validate_summary(batch_summary, book_name):
                response = llm.invoke(prompt)
                batch_summary = response.content.strip()
                batch_summary = re.sub(r'[�\x00-\x1F\x7F\"\'\n]', '', batch_summary)
            return batch_summary
        except Exception as e:
            logger.debug(f"批次级压缩失败，回退到片段拼接: {e}")
            # 降级：拼接片段摘要前100字
            return "《{}》中，{}...".format(book_name, " | ".join(segment_summaries)[:40])

    # 3. 书籍级压缩（全书批次摘要 → 总摘要+关键词，海马体高优节点）
    def _summarize_book_full(self, book_name: str) -> Dict[str, str]:
        """全书所有批次摘要 → 总摘要（≤100字）+ 3-5个关键词（高重要性记忆）"""
        try:
            llm = self._container.llm
            batch_summaries = BOOK_BATCH_SUMMARIES.get(book_name, [])
            if len(batch_summaries) < 3:
                # 不足3个批次，不生成全书总摘要（避免过于片面）
                return {"full_summary": "", "keywords": ""}
            
            merged_batch = "\n".join(batch_summaries)
            prompt = f"""
            对《{book_name}》的所有阅读批次摘要，生成全书总摘要和关键词：
            批次摘要：
            {merged_batch}
            
            要求：
            1. 总摘要：≤100字，概括全书核心情节、核心人物关系、整体基调
            2. 关键词：3-5个，仅提取核心实体/主题，用逗号分隔
            3. 总摘要格式：《书名》讲述了[核心情节]，[核心人物]经历了[变化/结果]，整体基调[情绪]。
            输出格式（仅输出内容，无其他解释）：
            总摘要：XXX
            关键词：XXX
            """
            response = llm.invoke(prompt)
            # 解析输出
            lines = [line.strip() for line in response.content.split("\n") if line.strip()]
            full_summary = ""
            keywords = ""
            for line in lines:
                if line.startswith("总摘要："):
                    full_summary = line.replace("总摘要：", "").strip()
                elif line.startswith("关键词："):
                    keywords = line.replace("关键词：", "").strip()
            # 校验总摘要有效性
            if len(full_summary) < 30 or len(full_summary) > 100:
                full_summary = f"《{book_name}》讲述了多个核心情节，围绕关键人物展开，整体基调贴合故事主题。"
            if not keywords or len(keywords.split(",")) < 3:
                # 兜底提取关键词（复用实体提取）
                entities = [self.extract_book_entity_llm(s) for s in batch_summaries[:5]]
                entities = [e for e in entities if e != "书籍内容"]
                keywords = ",".join(list(set(entities[:5]))) if entities else "书籍,情节,人物"
            return {"full_summary": full_summary, "keywords": keywords}
        except Exception as e:
            logger.error(f"书籍级压缩失败: {e}")
            return {"full_summary": f"《{book_name}》全书阅读完成，包含多个核心情节和关键人物。", "keywords": "书籍,情节,人物"}

    # ===================== 额外建议：阅读感悟生成 =====================
    def _generate_reading_insight(self, book_name: str, batch_summary: str) -> str:
        """生成第一人称阅读感悟，与客观摘要并存，增强记忆的自我感知"""
        try:
            llm = self._container.llm
            prompt = f"""
            你刚刚读完《{book_name}》的一个章节片段，摘要如下：
            {batch_summary}
            
            请用第一人称写1句话阅读感悟（≤50字），格式要求：
            1. 贴合摘要内容，体现真实感受，提到核心实体
            2. 不解释、不分析，只写内心想法
            3. 示例："读完这段，我意识到XX的处境很艰难，也感受到了他的坚持。"
            输出：
            """
            response = llm.invoke(prompt)
            insight = response.content.strip()
            insight = re.sub(r'[�\x00-\x1F\x7F\"\'\n]', '', insight)
            return insight if len(insight) >= 15 else f"读完这段，我对《{book_name}》的核心内容有了更清晰的认识。"
        except Exception as e:
            logger.debug(f"阅读感悟生成失败，兜底处理: {e}")
            return f"读完这段，我对《{book_name}》的核心内容有了更清晰的认识。"

    # ===================== 核心修改：双轨存储实现（海马体+皮层） =====================
    def _save_book_memory(self, book_name: str, multimodal_id: Optional[str]) -> None:
        """
        双轨存储流程（核心）：
        1. 片段级压缩：对缓冲的5段原始内容做片段压缩
        2. 批次级压缩：合并片段摘要，生成批次摘要（海马体索引）
        3. 皮层存储：原始细节（5段内容+multimodal信息）落盘（模拟皮层）
        4. 海马体存储：批次摘要+摘要向量+SDR+皮层路径（索引）
        5. 重复去重：相邻批次摘要相似度>0.9则合并
        6. Hebbian更新：关联概念专家，强化抽象关联
        """
        # 1. 片段级压缩（仅缓冲，不入海马体）
        segment_summaries = [self._summarize_segment(content.split(" | ")[0], book_name) for content in self._book_content_buffer]
        
        # 2. 批次级压缩，生成海马体核心索引
        batch_summary = self._summarize_batch(book_name, segment_summaries)
        # 校验摘要，最终降级（若仍无效，回退到原文拼接）
        if not self._validate_summary(batch_summary, book_name):
            batch_summary = "《{}》中，{}...".format(book_name, " | ".join([c[:20] for c in self._book_content_buffer])[:30])
        logger.debug(f"📝 批次级摘要：{batch_summary}")

        # 3. 皮层存储：原始细节落盘（模拟大脑皮层存储）
        cortex_path = self._get_cortex_storage_path(book_name)
        cortex_data = {
            "book_name": book_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "batch_content": self._book_content_buffer,  # 原始5段内容（含multimodal_id）
            "segment_summaries": segment_summaries,      # 片段级摘要
            "multimodal_id": multimodal_id,              # 关联的视觉想象ID
            "detail_ref": cortex_path                    # 自引用，方便回溯
        }
        try:
            with open(cortex_path, "w", encoding="utf-8") as f:
                json.dump(cortex_data, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 皮层存储完成 | 路径：{cortex_path}")
        except Exception as e:
            logger.error(f"皮层存储失败: {e}")
            # 皮层存储失败不影响海马体索引，仅日志报警

        # 4. 海马体存储：摘要索引（不存原始细节）
        think_engine = self._container["think_engine"]
        sdr_encoders = self._container["sdr_encoders"]
        sdr_encoder = sdr_encoders["概念"]
        hippocampus_router = self._container["hippocampus_router"].router

        # 对批次摘要重新向量化（不复用单段vec，语义重心不同）
        summary_vec = think_engine.encode_text(batch_summary)
        summary_sdr = sdr_encoder.encode(summary_vec.unsqueeze(0))

        # 5. 重复摘要去重（同一本书相邻批次）
        need_save = True
        if book_name in BOOK_BATCH_SUMMARIES and len(BOOK_BATCH_SUMMARIES[book_name]) > 0:
            # 获取上一个批次的摘要向量（从海马体查询最近记忆，简化处理）
            last_summary = BOOK_BATCH_SUMMARIES[book_name][-1]
            last_summary_vec = think_engine.encode_text(last_summary)
            similarity = self._calculate_vec_similarity(summary_vec, last_summary_vec)
            if similarity > 0.9:
                logger.info(f"⚠️  批次摘要重复（相似度{similarity:.2f}），跳过存储，合并批次")
                need_save = False

        if need_save:
            # 生成阅读感悟（主观），与客观摘要并存
            reading_insight = self._generate_reading_insight(book_name, batch_summary)
            
            # 存入海马体（仅索引，不存原始细节）
            mem_id = hippocampus_router.encode(
                clip_vec=summary_vec,
                sdr=summary_sdr,
                content=reading_insight,  # 主观感悟（用于情绪/主题检索）
                metadata={
                    "source": "book",
                    "book_name": book_name,
                    "importance": 0.6,
                    "is_book_memory": True,
                    "multimodal_id": multimodal_id,
                    "summary": batch_summary,  # 客观摘要（用于事实检索）
                    "detail_ref": cortex_path, # 指向皮层存储的路径（回溯原始细节）
                    "segment_summaries": segment_summaries,
                    "type": "batch_summary"
                },
                expert="概念"
            )
            # 缓存批次摘要，用于后续生成全书级总结
            if book_name not in BOOK_BATCH_SUMMARIES:
                BOOK_BATCH_SUMMARIES[book_name] = []
            BOOK_BATCH_SUMMARIES[book_name].append(batch_summary)
            logger.info(f"✅ 海马体存储完成 | 记忆ID:{mem_id} | 摘要：{batch_summary[:40]}...")

            # 6. Hebbian更新：强化概念专家与摘要的关联
            try:
                experts = self._container["experts"]
                concept_expert = experts["概念"]
                concept_expert.hebbian_update(summary_sdr, summary_sdr, is_fact=True)
                logger.debug(f"✅ 概念专家Hebbian更新完成（批次摘要）")
            except Exception as e:
                logger.debug(f"概念专家Hebbian更新失败: {e}")

        # 重置批次缓存
        self._book_content_buffer = []
        self._book_current_batch = []

    # ===================== 核心修改：书籍级总结（读完一本书触发） =====================
    def _handle_book_finish(self, book_name: str) -> None:
        """读完一本书（is_new_cycle=True），生成书籍级总摘要，高重要性存入海马体"""
        try:
            # 生成书籍级总摘要+关键词
            full_book_info = self._summarize_book_full(book_name)
            full_summary = full_book_info["full_summary"]
            keywords = full_book_info["keywords"]
            if not full_summary:
                logger.info(f"📚 《{book_name}》批次不足，不生成全书总摘要")
                return

            # 向量化+SDR编码（书籍级摘要单独编码）
            think_engine = self._container["think_engine"]
            sdr_encoders = self._container["sdr_encoders"]
            sdr_encoder = sdr_encoders["概念"]
            hippocampus_router = self._container["hippocampus_router"].router

            full_summary_vec = think_engine.encode_text(full_summary)
            full_summary_sdr = sdr_encoder.encode(full_summary_vec.unsqueeze(0))

            # 高重要性存入海马体（书籍级索引节点）
            mem_id = hippocampus_router.encode(
                clip_vec=full_summary_vec,
                sdr=full_summary_sdr,
                content=f"📚 读完《{book_name}》：{full_summary}",
                metadata={
                    "source": "book",
                    "book_name": book_name,
                    "importance": 0.9,  # 高重要性，优先检索
                    "is_book_memory": True,
                    "is_book_summary": True,  # 标记为书籍总摘要
                    "full_summary": full_summary,
                    "keywords": keywords,
                    "batch_count": len(BOOK_BATCH_SUMMARIES.get(book_name, [])),
                    "type": "book_summary"
                },
                expert="抽象"  # 书籍总摘要存入抽象专家，符合抽象记忆定位
            )

            # Hebbian更新：强化抽象专家与书籍总摘要的关联
            try:
                experts = self._container["experts"]
                abstract_expert = experts["抽象"]
                abstract_expert.hebbian_update(full_summary_sdr, full_summary_sdr, is_fact=True)
                logger.debug(f"✅ 抽象专家Hebbian更新完成（书籍总摘要）")
            except Exception as e:
                logger.debug(f"抽象专家Hebbian更新失败: {e}")

            logger.info(f"🎉 《{book_name}》全书总结已存储 | 记忆ID:{mem_id} | 关键词：{keywords}")
            # 清空该书籍的批次摘要缓存（避免重复生成总摘要）
            BOOK_BATCH_SUMMARIES[book_name] = []
        except Exception as e:
            logger.error(f"书籍级总结存储失败: {e}", exc_info=True)

    # ===================== 原有方法修改：衔接新逻辑 =====================
    def read_book(self) -> None:
        try:
            book_data = self.book_reader.random_read()
            if not book_data:
                logger.info("📚 未找到任何书籍，回到正常走神~")
                return

            book_name = book_data["book_name"]
            content = book_data["content"].strip()
            if not content or len(content) < 20:
                return
            
            # 触发书籍级总结（读完一本书）
            if book_data.get("is_new_cycle", False):
                try:
                    # 新增：生成书籍级总摘要并存储
                    self._handle_book_finish(book_name)
                    # 保留原有逻辑：生成"读完书籍"的抽象记忆
                    learning_loop = self._container["learning_loop"]
                    learning_loop.learn(
                        f"我刚刚读完了《{book_name}》这本书，内容非常精彩。",
                        force_expert="抽象",
                        external_reward=0.3
                    )
                    logger.info(f"✅ 已生成读完《{book_name}》的抽象记忆")
                except Exception as e:
                    logger.debug(f"生成读完记忆失败: {e}")

            # 清理内容，避免乱码
            content = re.sub(r'[�\x00-\x1F\x7F]', '', content)
            content = content.lstrip('，。；：！？、"\'')
            if not content or len(content) < 20:
                return

            logger.info(f"📖 小白正在阅读学习：《{book_name}》")
            logger.info(f"📖 阅读片段：{content[:80]}...")

            # 向量化+SDR编码（单段内容，用于实时Hebbian更新）
            think_engine = self._container["think_engine"]
            clip_vec = think_engine.encode_text(content)
            sdr_encoders = self._container["sdr_encoders"]
            sdr_encoder = sdr_encoders["概念"]
            query_sdr = sdr_encoder.encode(clip_vec)

            self._imagination_counter += 1
            multimodal_id = None
            
            # 生成视觉想象（原有逻辑不变）
            if self._imagination_counter >= 5:
                multimodal_id = self._generate_imagination(book_name, content)
                self._imagination_counter = 0
            else:
                book_entity = self.extract_book_entity_llm(content)
                thought = f"🧠 正在阅读关于【{book_entity}】的精彩内容..."
                logger.info(f"🤔 阅读思考：{thought[:150]}...")

            # 缓存当前片段（用于后续批次压缩）
            self._book_read_counter += 1
            extra_info = f" | 绑定ID:{multimodal_id}" if multimodal_id is not None else ""
            self._book_content_buffer.append(content[:50] + extra_info)
            # 缓存原始细节（含multimodal信息，用于皮层存储）
            self._book_current_batch.append({
                "content": content,
                "multimodal_id": multimodal_id
            })
            
            # 累计5段，触发双轨存储（海马体+皮层）
            if self._book_read_counter >= 5:
                self._save_book_memory(book_name, multimodal_id)
                self._book_read_counter = 0

            # 单段内容的Hebbian更新（原有逻辑不变，强化实时关联）
            try:
                experts = self._container["experts"]
                concept_expert = experts["概念"]
                concept_expert.hebbian_update(query_sdr, query_sdr, is_fact=True)
            except:
                pass

            # 触发阅读完成事件（原有逻辑不变）
            EventBus().emit(Event(
                event_type=EventType.BOOK_READ_FINISHED,
                data={"book_name": book_name, "content_length": len(content)},
                timestamp=time.time()
            ))
            
            time.sleep(self.reading_interval)

        except Exception as e:
            logger.error(f"📚 阅读学习失败：{str(e)}", exc_info=True)
            time.sleep(5)

    # ===================== 原有方法：视觉想象（不变） =====================
    def _generate_imagination(self, book_name: str, content: str) -> Optional[str]:
        try:
            multimodal_id = str(uuid.uuid4())
            llm = self._container.llm
            
            scene_prompt = f"""
            你正在读小说《{book_name}》，刚刚读到这段内容：
            {content[:300]}

            请用简洁的语言描述这段文字最核心的视觉场景，不要有任何解释和分析。
            要求：
            1. 只描述能看到的东西，不要描述声音、情绪和心理活动
            2. 不超过30个字
            3. 不要出现"画面""场景""图"等词语

            输出：
            """
            scene_response = llm.invoke(scene_prompt)
            scene_description = scene_response.content.strip()
            scene_description = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9 ]', '', scene_description)
            logger.debug(f"🎨 生成阅读想象场景：{scene_description}")

            vae_manager = self._container["vae_manager"].manager
            cross_modal_bridge = self._container["cross_modal_bridge"]
            sdr_encoders = self._container["sdr_encoders"]
            think_engine = self._container["think_engine"]
            
            vae_data = None
            visual_content = None
            mem_id = None

            if (vae_manager is not None 
                and cross_modal_bridge.use_learnable_pons 
                and "视觉" in cross_modal_bridge.pons 
                and isinstance(cross_modal_bridge.pons["视觉"], dict)):
                try:
                    scene_clip_vec = think_engine.encode_text(scene_description)
                    text_sdr = sdr_encoders["概念"].encode(scene_clip_vec)
                    
                    bridge = cross_modal_bridge.pons["视觉"]
                    predicted_vision_sdr = bridge["text_to_expert"](text_sdr)
                    
                    brain_core = self._container["brain_core"]
                    bridge_maturity = min(1.0, brain_core.total_interactions / 1000.0)
                    noise_level = 0.1 * (1.0 - bridge_maturity * 0.5)
                    noise = torch.randn_like(predicted_vision_sdr) * noise_level
                    imagined_vision_sdr = predicted_vision_sdr + noise
                    
                    vae_latent = vae_manager.sdr_to_latent(imagined_vision_sdr)
                    
                    latent_min = float(vae_latent.min())
                    latent_max = float(vae_latent.max())
                    latent_normalized = (vae_latent - latent_min) / (latent_max - latent_min + 1e-8)
                    latent_quantized = (latent_normalized * 255).to(torch.uint8).cpu().numpy()
                    
                    vae_data = {
                        "latent": latent_quantized.tolist(),
                        "min": latent_min,
                        "max": latent_max,
                        "shape": list(vae_latent.shape)
                    }
                    
                    logger.info(f"✅ 基于脑桥的视觉想象生成成功 | 绑定ID:{multimodal_id} | 噪声水平:{noise_level:.2f}")
                    
                    cross_modal_bridge.cross_modal_learning_step(text_sdr, imagined_vision_sdr.detach(), target_expert="视觉")
                    
                    brain_core.dopamine_level = min(1.0, brain_core.dopamine_level + 0.05)
                    
                except Exception as e:
                    logger.debug(f"脑桥想象失败，回退到随机VAE: {str(e)}")
                    random_latent = torch.randn(4, 64, 64) * 0.8
                    latent_min = float(random_latent.min())
                    latent_max = float(random_latent.max())
                    latent_normalized = (random_latent - latent_min) / (latent_max - latent_min + 1e-8)
                    latent_quantized = (latent_normalized * 255).to(torch.uint8).cpu().numpy()
                    
                    vae_data = {
                        "latent": latent_quantized.tolist(),
                        "min": latent_min,
                        "max": latent_max,
                        "shape": [4, 64, 64]
                    }
            else:
                random_latent = torch.randn(4, 64, 64) * 0.8
                latent_min = float(random_latent.min())
                latent_max = float(random_latent.max())
                latent_normalized = (random_latent - latent_min) / (latent_max - latent_min + 1e-8)
                latent_quantized = (latent_normalized * 255).to(torch.uint8).cpu().numpy()
                
                vae_data = {
                    "latent": latent_quantized.tolist(),
                    "min": latent_min,
                    "max": latent_max,
                    "shape": [4, 64, 64]
                }
                
            tag = scene_description.strip()[:6].replace("|", "").replace("\n", "")
            if not tag:
                tag = "场景"

            visual_content = f"[阅读想象-{tag}] 绑定ID:{multimodal_id}"
            logger.info(f"✅ 视觉记忆生成 | 标签:{tag} | 绑定ID:{multimodal_id}")

            thalamus = self._container["thalamus"].thalamus
            hippocampus_router = self._container["hippocampus_router"].router
            
            image_feat = think_engine.encode_text(scene_description)
            if image_feat.shape[-1] != config.dim:
                proj = torch.nn.Linear(image_feat.shape[-1], config.dim, bias=False).to(image_feat.device)
                image_feat = proj(image_feat)
            image_feat = F.normalize(image_feat.detach().squeeze(), p=2, dim=-1)

            passed, info_packet = thalamus.filter_and_relay(
                input_vec=image_feat,
                input_text=visual_content,
                metadata={
                    "force_expert": "视觉",
                    "multimodal_id": multimodal_id,
                    "type": "visual",
                    "source": "book_imagination",
                    "book_name": book_name,
                    "scene": scene_description,
                    "tag": tag,
                    "vae_latent": vae_data
                }
            )
            
            if passed:
                image_feat = info_packet["vec"]
                saliency = info_packet["saliency"]
                
                visual_sdr_encoder = sdr_encoders.get("视觉", sdr_encoders["概念"])
                sdr = visual_sdr_encoder.encode(image_feat.unsqueeze(0))

                mem_id = hippocampus_router.encode(
                    clip_vec=image_feat,
                    sdr=sdr,
                    content=visual_content,
                    metadata={
                        "saliency": saliency,
                        "multimodal_id": multimodal_id,
                        "type": "visual",
                        "source": "book_imagination",
                        "book_name": book_name,
                        "scene": scene_description,
                        "tag": tag,
                        "vae_latent": vae_data
                    },
                    expert="视觉"
                )
                logger.info(f"✅ 阅读想象视觉记忆已存入视觉专家 | 记忆ID:{mem_id}")

                learning_loop = self._container["learning_loop"]
                if hasattr(learning_loop, 'bind_related_memories') and callable(learning_loop.bind_related_memories):
                    learning_loop.bind_related_memories(
                        new_mem_id=mem_id,
                        new_mem_vec=image_feat,
                        new_mem_text=visual_content,
                        target_expert="视觉",
                        user_input=f"阅读《{book_name}》时的想象：{scene_description}"
                    )

            thought_prompt = f"""
    你刚刚读了《{book_name}》里的这段内容：
    {content[:200]}

    你在脑海里想象出了这样的画面：{scene_description}

    请用第一人称写1句话你的感受，要提到你仿佛看到了什么。
    比如："读到这里，我仿佛看到了智子穿着和服在茶室里温柔鞠躬的样子"
    不要解释，不要分析，只写你的内心想法。
    """
            thought_response = llm.invoke(thought_prompt)
            imagination = thought_response.content.strip()
            imagination = re.sub(r'[\"\']', '', imagination)
            
            thought = f"💭 阅读想象：{imagination}"
            logger.info(f"🤔 阅读思考：{thought[:150]}...")

            EventBus().emit(Event(
                event_type=EventType.IMAGINATION_GENERATED,
                data={"book_name": book_name, "scene": scene_description, "multimodal_id": multimodal_id},
                timestamp=time.time()
            ))
            
            return multimodal_id
            
        except Exception as e:
            logger.debug(f"多模态想象生成失败: {str(e)}", exc_info=True)
            book_entity = self.extract_book_entity_llm(content)
            thought = f"🧠 正在阅读关于【{book_entity}】的精彩内容..."
            logger.info(f"🤔 阅读思考：{thought[:150]}...")
            return None

    # ===================== 原有方法：实体提取（不变，用于摘要校验） =====================
    def extract_book_entity_llm(self, text: str) -> str:
        try:
            short_text = text[:150].strip()
            if not short_text:
                return "书籍内容"

            llm = self._container.llm
            prompt = f"""
    你是书籍内容实体提取器，严格遵守规则：
    1. 只从下面的文本中提取**1个最核心、最具体的专有名词**
    2. 绝对不能提取代词（我/你/他/她/这/那）、副词、语气词、通用词
    3. 只输出实体本身，不要任何解释、标点、前缀后缀
    4. 如果没有明确实体，输出"书籍内容"

    文本：{short_text}
    输出：
            """

            response = llm.invoke(prompt)
            entity = response.content.strip()
            entity = re.sub(r'[^\u4e00-\u9fa5]', '', entity)

            stop_words = {'自己', '别人', '大家', '人们', '地方', '时间', '事情', '东西', '时候', '现在', '书籍内容'}
            if len(entity) < 2 or entity in stop_words:
                return self._extract_book_entity_fallback(short_text)
            return entity

        except Exception as e:
            logger.debug(f"LLM实体提取失败，使用兜底: {e}")
            return self._extract_book_entity_fallback(text)
    
    def _extract_book_entity_fallback(self, text: str) -> str:
        clean_text = re.sub(r'[\s\.\,\;\:\'\"\\\[\]\(\)\（\）\，\。\；\：\“\”\’\‘]+', '', text[:100])
        skip_pattern = r'^(我|你|他|她|它|我们|你们|他们|她们|它们|这|那|这个|那个|这些|那些|这里|那里|这时|那时|然后|接着|于是|但是|而且|因为|所以|虽然|如果|就|才|都|也|还|又|再|更|最|很|非常|特别|等等)+'
        clean_text = re.sub(skip_pattern, '', clean_text)
        
        if not clean_text:
            return "书籍内容"
        
        nouns = re.findall(r'[\u4e00-\u9fa5]{2,4}', clean_text)
        stop_words = {'自己', '别人', '大家', '人们', '男人', '女人', '老人', '孩子', '地方', '时间', '事情', '东西', '一天', '一年', '时候', '现在'}
        valid_nouns = [n for n in nouns if n not in stop_words]
        
        return valid_nouns[0] if valid_nouns else "书籍内容"
