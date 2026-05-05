import torch
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Optional
import logging
from torch.nn import functional as F

from BrainConfig import config

logger = logging.getLogger("LLMBrainWrapper")

class LLMBrainWrapperV5:
    def __init__(self, brain):
        self.brain = brain
        
        logger.info(f"🤖 正在初始化 LangChain + Ollama，使用模型: {config.ollama_model_name}")
        self.llm = ChatOllama(
            model=config.ollama_model_name,
            temperature=config.llm_temperature,
            num_predict=config.llm_max_tokens,
        )
        logger.info("✅ LangChain + Ollama 初始化完成！")

    def _call_llm(self, prompt, system_prompt=None, max_tokens=config.llm_max_tokens, temperature=config.llm_temperature):
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        try:
            temp_llm = ChatOllama(
                model=config.ollama_model_name,
                temperature=temperature,
                num_predict=max_tokens,
            )
            response = temp_llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"⚠️ LangChain + Ollama 调用失败: {e}")
            return ""

    def _person_convert(self, text: str) -> str:
        """
        用户视角 → 小白AI视角 纯净互换
        我 ↔ 你
        我的 ↔ 你的
        完全无占位符、无残留垃圾字符
        """
        mapping = [
            ("我", "@A@"),
            ("你", "@B@"),
            ("我的", "@C@"),
            ("你的", "@D@"),
        ]
        # 临时替换
        for old, tmp in mapping:
            text = text.replace(old, tmp)
        # 互换
        text = text.replace("@A@", "你")
        text = text.replace("@B@", "我")
        text = text.replace("@C@", "你的")
        text = text.replace("@D@", "我的")
        return text.strip()

    def _get_query_expert_local(self, query: str) -> str:
        """
        本地硬规则 + 身份判断 做意图分类（新增身份专家）
        完全不需要LLM，速度快、稳定
        """
        # 🔥 身份关键词（你是谁/我是谁/主人/名字）
        identity_words = [
            "你是谁", "我是谁", "名字", "叫什么", "主人", "你的主人",
            "我的名字", "你的名字", "身份", "你是", "我是"
        ]
        person_words = ["介绍", "人物", "哪位", "个人", "生平", "原名"]
        event_words = ["什么时候", "发生", "历史", "事件", "年份", "年代", "案件", "在哪里"]
        know_words = ["是什么", "答案", "原理", "含义", "来源", "意思", "方法", "名言", "知识"]

        query_lower = query.lower()
        # 优先级：身份 > 人物 > 事件 > 知识
        if any(w in query_lower for w in identity_words):
            return "身份"
        if any(w in query_lower for w in person_words):
            return "概念"
        if any(w in query_lower for w in event_words):
            return "空间"
        if any(w in query_lower for w in know_words):
            return "抽象"
        
        return "抽象"

    def learn(self, text):
        if hasattr(self.brain, '_update_interaction_time'):
            self.brain._update_interaction_time()

        logger.info(f"\n📚 正在学习: {text[:60]}...")
        
        text_lower = text.lower()
        target_expert = None

        # ===================== 🔥 核心改造：优先海马体路由智能判断 =====================
        try:
            # 调用海马体路由获取最优专家（神经网络自动分类，主逻辑）
            clip_vec = self.brain.encode_text(text)
            clip_vec = F.normalize(clip_vec.detach().squeeze(), p=2, dim=-1)
            target_expert = self.brain.hippocampus_router.route(clip_vec, text)
            logger.info(f"🧭 海马体路由自动分配: [{target_expert}]")
        except Exception as e:
            # 路由异常时打印日志
            logger.warning(f"⚠️ 海马体路由异常，启用兜底规则: {str(e)}")
            target_expert = None

        # ===================== 🛡️ 保底兜底：原有关键词规则（仅路由失效时使用） =====================
        if not target_expert:
            if text.startswith("身份："):
                processed_text = self._person_convert(text)
                logger.info(f"🔄 身份信息人称转换完成: {processed_text}")
                target_expert = "身份"
                # 身份信息强制学习
                self.brain.learn(processed_text, force_expert=target_expert)
                logger.info(f"🧠 兜底规则存入专家: [{target_expert}]")
                return target_expert
            elif any(keyword in text_lower for keyword in ["人物", "职业"]):
                target_expert = "概念"
            elif any(keyword in text_lower for keyword in ["案件", "事件", "地点", "历史"]):
                target_expert = "空间"
            else:
                target_expert = "抽象"
            logger.info(f"🛡️ 兜底规则分配: [{target_expert}]")

        # ===================== 统一学习逻辑 =====================
        # 身份信息单独做人称转换
        if target_expert == "身份" and text.startswith("身份："):
            processed_text = self._person_convert(text)
            logger.info(f"🔄 身份信息人称转换完成: {processed_text}")
            self.brain.learn(processed_text, force_expert=target_expert)
        else:
            # 普通内容按路由/兜底结果学习
            self.brain.learn(text, force_expert=target_expert)

        logger.info(f"🧠 最终存入专家: [{target_expert}]")
        return target_expert

    def ask(self, query):
        if hasattr(self.brain, '_update_interaction_time'):
            self.brain._update_interaction_time()

        # ===================== 工具函数：抽离重复判断逻辑 =====================
        def is_declarative_sentence(q: str) -> bool:
            """判断是否为陈述句（非疑问句，触发自主学习）"""
            question_words = ["？", "?", "什么", "哪里", "谁", "怎么", "吗", "呢"]
            return not any(word in q for word in question_words)

        # ===================== 1. 优先处理：带"记住了"的学习指令 =====================
        if "记住了" in query:
            if is_declarative_sentence(query):
                logger.info("🧠 检测到学习指令，触发自主学习...")
                self.learn(query.replace("记住了", "").strip())
                return "💬 好的，我记住了！"

        logger.info(f"\n❓ 用户问题: {query}")

        # ===================== 2. 本地路由目标专家 =====================
        target_expert = self._get_query_expert_local(query)
        logger.info(f"🎯 定向检索专家分区: [{target_expert}]")

        # ===================== 3. 核心：大脑类脑思考（激活+联想+传播） =====================
        thought_result = {}
        activated_memories = []
        try:
            # 调用大脑思考（修复原代码重复调用的BUG）
            thought_result = self.brain.think(query, steps=2)
            # 统一提取思考结果
            thought_chain = thought_result.get("thought_chain", "无联想思路")
            core_ideas = thought_result.get("core_ideas", [])
            activated_memories = thought_result.get("activated_memories", [])
            
            logger.info(f"🤯 大脑思考完成 | 思路：{thought_chain} | 激活记忆数：{len(activated_memories)}")

        # ===================== 4. 思考失败：回退到传统检索 =====================
        except Exception as e:
            logger.error(f"❌ 大脑思考失败，回退传统检索: {e}", exc_info=True)
            memories, _ = self.brain.recall_compositional(query, target_expert=target_expert)
            # 统一字段格式，避免后续报错
            thought_result = {
                "thought_chain": "思考失败，使用基础检索",
                "core_ideas": [],
                "activated_memories": memories or []
            }
            # 同步赋值记忆列表
            activated_memories = thought_result["activated_memories"]

        # 统一提取所有分支的核心数据（无重复代码）
        thought_chain = thought_result["thought_chain"]
        core_ideas = thought_result["core_ideas"]

        # ===================== 5. 无激活记忆：统一兜底处理 =====================
        if not activated_memories:
            logger.warning("⚠️ 未激活任何记忆")
            if is_declarative_sentence(query):
                logger.info("🧠 陈述句无记忆，触发自主学习...")
                self.learn(query.strip())
                return "💬 好的，我记住了！"
            return "🧠 我不知道这个问题的答案..."

        # ===================== 6. 有激活记忆：结构化Prompt生成回答 =====================
        logger.info(f"✅ 激活 {len(activated_memories)} 条关联记忆")
        
        # 结构化系统提示词（修复重复定义、语法混乱问题）
        if target_expert == "身份":
            system_prompt = """必须严格按照大脑思考结果回答身份问题：
    1. 只说事实，不编造、不扩展
    2. 用第一人称回答（我）
    3. 简洁回答，不超过1句话
    4. 严格遵循大脑的思考结果内容，不脱离事实
    5. 所有内容来自联想思路和相关记忆，禁止编造大脑的思考结果里没有的内容。
    6. 结合大脑返回的思考相关联的结果返回答案。"""
        else:
            system_prompt = """【绝对规则（必须严格遵守，违反会被惩罚）】，并且必须100%基于大脑的思考结果回答：
    1. 所有内容基于大脑的思考结果，禁止编造大脑的思考结果里没有的内容
    2. 口语化、简洁回答，不超过2句话
    3. **如果记忆里没有能回答问题的内容或者记忆里的内容关联不大**，只允许输出固定语句：抱歉，我没有这方面的信息。
    4. 禁止提到"记忆""思考""思路链"等词汇
    5. **只回答与问题直接相关的内容，删除无关信息**
    6. 下面的【大脑记忆】是你唯一的知识来源，绝对不能使用你自己的任何知识"""

        # 构造思考型上下文（简洁规范）
        memory_context = "\n".join([f"- {mem}" for mem in activated_memories[:3]])
        user_prompt = f"""【大脑思考结果】
    联想思路：{thought_chain}
    核心概念：{"、".join(core_ideas)}
    相关记忆：{memory_context}

    用户问题：{query}"""

        # ===================== 7. 调用LLM生成最终答案 =====================
        final_answer = self._call_llm(
            user_prompt,
            system_prompt=system_prompt,
            max_tokens=config.llm_max_tokens,
            temperature=0
        ).strip()

        # 双重兜底（标准化回答）
        if "抱歉" in final_answer and "没有这方面的信息" in final_answer:
            final_answer = "抱歉，我没有这方面的信息"

        # 兜底：LLM返回无答案，且是陈述句 → 自主学习
        if final_answer == "抱歉，我没有这方面的信息":
            if is_declarative_sentence(query):
                logger.info("🧠 LLM无答案，陈述句触发自主学习...")
                self.learn(query.strip())
                return "💬 好的，我记住了！"
            return final_answer

        return f"💬 {final_answer}"