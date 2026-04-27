import torch
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Optional
import logging

from BrainConfig import config

logger = logging.getLogger("LLMBrainWrapper")

class LLMBrainWrapperV4:
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

    # def _person_convert(self, text: str) -> str:
    #     """
    #     🔥 核心功能：自动人称转换（用户视角 → AI视角）
    #     你 ↔ 我 | 你的 ↔ 我的 | 您 ↔ 我 | 您的 ↔ 我的
    #     示例：你是小白 → 我是小白 | 我是主人 → 你是主人
    #     """
    #     # 分步替换，避免循环替换
    #     text = text.replace("你的", "【我的_temp】")
    #     text = text.replace("我的", "【你的_temp】")
    #     text = text.replace("你", "【我_temp】")
    #     text = text.replace("我", "【你_temp】")
    #     text = text.replace("您", "【我_temp】")
    #     text = text.replace("您的", "【我的_temp】")
        
    #     # 还原占位符
    #     text = text.replace("【我的_temp】", "我的")
    #     text = text.replace("【你的_temp】", "你的")
    #     text = text.replace("【我_temp】", "我")
    #     text = text.replace("【你_temp】", "你")
        
    #     return text.strip()

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
        logger.info(f"\n📚 正在学习: {text[:60]}...")
        
        # 🔥 完整的身份相关关键词
        text_lower = text.lower()
        # identity_keywords = [
        #     "名字", "叫什么", "我是", "你是", "身份", "称呼",
        #     "主人", "你的主人", "我的名字", "你的名字",
        #     "我叫", "你叫", "是谁", "你是谁", "我是谁",
        #     "你的身份", "我的身份", "你叫什么名字", "我叫什么名字"
        # ]
        
        # ===================== 核心：身份信息自动处理 =====================
        if text.startswith("身份："):
            # 1. 自动转换人称（用户视角→AI视角）
            processed_text = self._person_convert(text)
            logger.info(f"🔄 身份信息人称转换完成: {processed_text}")
            # 2. 强制存入【身份专家】
            target_expert = "身份"
            # 3. 学习转换后的文本
            self.brain.learn(processed_text, force_expert=target_expert)
        # ===================== 原有规则 =====================
        elif any(keyword in text_lower for keyword in ["人物", "职业"]):
            target_expert = "概念"
            self.brain.learn(text, force_expert=target_expert)
        elif any(keyword in text_lower for keyword in ["案件", "事件", "地点", "历史"]):
            target_expert = "空间"
            self.brain.learn(text, force_expert=target_expert)
        else:
            target_expert = "抽象"
            self.brain.learn(text, force_expert=target_expert)
        
        logger.info(f"🧠 存入专家: [{target_expert}]")
        return target_expert

    def ask(self, query):
        if "记住了" in query:
                is_statement = not any(q in query for q in ["？", "?", "什么", "哪里", "谁", "怎么", "吗", "呢"])
                if is_statement:
                    print(f"🧠 没找到相关记忆，触发自主学习...")
                    self.learn(query.replace("记住了", ""))
                    return "💬 好的，我记住了！"
        
        logger.info(f"\n❓ 用户问题: {query}")
        
        # ===================== 第一步：本地提取关键词 =====================
        search_query = query
        
        # ===================== 第二步：本地判断目标专家（含身份） =====================
        target_expert = self._get_query_expert_local(query)
        logger.info(f"🎯 定向检索专家分区: [{target_expert}]")

        # ===================== 第三步：记忆检索 =====================
        memories, meta = self.brain.recall_compositional(search_query, target_expert=target_expert)
        if not memories:
            memories, meta = self.brain.recall_compositional(query, target_expert=target_expert)
        
        # ===================== 第四步：生成回答 =====================
        if memories:
            logger.info(f"✅ 找到 {len(memories)} 条相关记忆")
            for i, mem in enumerate(memories[:5]):
                logger.info(f"    候选 {i+1}: {mem[:50]}...")
            
            # 身份专属Prompt（优先使用）
            if target_expert == "身份":
                system_prompt = """必须严格按照记忆回答身份问题：
1.  只说事实，不编造、不扩展
2.  用第一人称回答（我）
3.  简洁回答，不超过1句话
4.  不知道就说：我是【记忆回答里我的名字】
"""
            else:
                # 通用知识Prompt
                system_prompt = """你是一个温和友好的知识助手，必须100%遵守以下核心规则：
1.  所有回答只能基于下方提供的相关记忆，绝对不能添加、编造、联想任何记忆里没有的信息。
2.  从记忆里选择最匹配问题的内容，用自然流畅的口语化中文回答，不要生硬照搬原文。
3.  如果记忆里没有能回答问题的内容或则记忆里的内容关联不大，只允许输出固定语句：抱歉，我没有这方面的信息。
4.  绝对不能提到"记忆""参考内容"这类词，就像你本来就知道这些内容一样。
5.  回答要简洁，不超过2句话，不要多余的解释和扩展。
"""
            
            memory_context = "\n".join([f"- {mem}" for mem in memories])
            user_prompt = f"""相关记忆：
{memory_context}

用户问题：{query}"""
            
            final_answer = self._call_llm(
                user_prompt,
                system_prompt=system_prompt,
                max_tokens=config.llm_max_tokens,
                temperature=0  # 核心：设为0，彻底锁死
            ).strip()

            # 双重兜底
            if "抱歉" in final_answer and "没有这方面的信息" in final_answer:
                final_answer = "抱歉，我没有这方面的信息"
            
            if final_answer == "抱歉，我没有这方面的信息":
                is_statement = not any(q in query for q in ["？", "?", "什么", "哪里", "谁", "怎么", "吗", "呢"])
                if is_statement:
                    logger.info(f"🧠 没找到相关记忆，触发自主学习...")
                    self.learn(query)
                    return "💬 好的，我记住了！"
                else:
                    return "🧠 我不知道这个问题的答案..."
            
            return f"💬 {final_answer}"
            
        else:
            is_statement = not any(q in query for q in ["？", "?", "什么", "哪里", "谁", "怎么", "吗", "呢"])
            if is_statement:
                logger.info(f"🧠 没找到相关记忆，触发自主学习...")
                self.learn(query)
                return "💬 好的，我记住了！"
            else:
                return "🧠 我不知道这个问题的答案..."