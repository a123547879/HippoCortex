import torch
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Optional
import logging

from BrainConfig import config

logger = logging.getLogger("LLMBrainWrapper")

class LLMBrainWrapperV3:
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

    # 🔥 核心新增：本地意图分类，减少LLM调用
    def _get_query_expert_local(self, query: str) -> str:
        """
        本地硬规则 + bge向量相似度 做意图分类
        完全不需要LLM，速度快、稳定
        """
        person_words = ["是谁", "介绍", "人物", "哪位", "个人", "生平", "原名", "名字", "叫什么"]
        event_words = ["什么时候", "发生", "历史", "事件", "年份", "年代", "案件", "在哪里"]
        know_words = ["是什么", "答案", "原理", "含义", "来源", "意思", "方法", "名言", "知识"]

        query_lower = query.lower()
        if any(w in query_lower for w in person_words):
            return "概念"
        if any(w in query_lower for w in event_words):
            return "空间"
        if any(w in query_lower for w in know_words):
            return "抽象"
        
        # 本地规则判断不了，返回默认
        return "抽象"

    def learn(self, text):
        logger.info(f"\n📚 正在学习: {text[:60]}...")
        
        # 🔥 完整的身份相关关键词
        text_lower = text.lower()
        identity_keywords = [
            "名字", "叫什么", "我是", "你是", "身份", "称呼",
            "主人", "你的主人", "我的名字", "你的名字",
            "我叫", "你叫", "是谁", "你是谁", "我是谁",
            "你的身份", "我的身份", "你叫什么名字", "我叫什么名字"
        ]
        
        if any(keyword in text_lower for keyword in identity_keywords):
            target_expert = "概念"
        elif any(keyword in text_lower for keyword in ["人物", "职业", "身份"]):
            target_expert = "概念"
        elif any(keyword in text_lower for keyword in ["案件", "事件", "地点", "历史"]):
            target_expert = "空间"
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
                    self.learn(query)
                    return "💬 好的，我记住了！"
                else:
                    return "🧠 我不知道这个问题的答案..."
        logger.info(f"\n❓ 用户问题: {query}")
        
        # ===================== 第一步：本地提取关键词（简单规则） =====================
        # 直接用原问题作为搜索词，减少LLM调用
        search_query = query
        
        # ===================== 第二步：本地判断目标专家 =====================
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
            
            # 优化版Prompt
            system_prompt = """你是一个温和友好的知识助手，必须100%遵守以下核心规则：
1.  所有回答只能基于下方提供的相关记忆，绝对不能添加、编造、联想任何记忆里没有的信息。
2.  从记忆里选择最匹配问题的内容，用自然流畅的口语化中文回答，不要生硬照搬原文。
3.  如果记忆里没有能回答问题的内容，只允许输出固定语句：抱歉，我没有这方面的信息。
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