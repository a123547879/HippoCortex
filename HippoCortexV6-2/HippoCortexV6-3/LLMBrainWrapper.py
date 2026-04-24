# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # import os
# import json
# import datetime
# from transformers import (
#     CLIPTokenizer, CLIPTextModel,
#     AutoTokenizer, AutoModelForCausalLM
# )
# from typing import List, Tuple, Optional
# import numpy as np
# from collections import deque, defaultdict
# from langchain_ollama import ChatOllama
# from langchain_core.messages import HumanMessage, SystemMessage

# class LLMBrainWrapper:
#     def __init__(self, brain, ollama_model_name="qwen2.5:0.5b"):
#         self.brain = brain
#         self.model_name = ollama_model_name
        
#         print(f"🤖 正在初始化 LangChain + Ollama，使用模型: {ollama_model_name}")
#         self.llm = ChatOllama(
#             model=ollama_model_name,
#             temperature=0.7,
#             num_predict=256,
#         )
#         print("✅ LangChain + Ollama 初始化完成！")

#     def _call_llm(self, prompt, system_prompt=None, max_tokens=256, temperature=0.7):
#         """🔥 修复版：直接创建新的 ChatOllama 实例，不用 configure"""
#         messages = []
#         if system_prompt:
#             messages.append(SystemMessage(content=system_prompt))
#         messages.append(HumanMessage(content=prompt))
        
#         try:
#             # 每次调用都创建一个新的实例，传入想要的参数
#             temp_llm = ChatOllama(
#                 model=self.model_name,
#                 temperature=temperature,
#                 num_predict=max_tokens,
#             )
#             response = temp_llm.invoke(messages)
#             return response.content
#         except Exception as e:
#             print(f"⚠️ LangChain + Ollama 调用失败: {e}")
#             return ""

# #     def learn(self, text):
# #         """用 Ollama 增强的学习：先理解，再存储"""
# #         print(f"\n📚 正在学习: {text[:40]}...")
        
# #         parse_prompt = f"""请分析下面这段知识，判断它属于什么类型，并提取关键词。

# # 知识内容：{text}

# # 请直接回答，按以下格式：
# # 类型：[人物/案件/名言/视觉/抽象]
# # 关键词：[关键词1, 关键词2]"""
        
# #         try:
# #             parse_result = self._call_llm(
# #                 parse_prompt,
# #                 max_tokens=64,
# #                 temperature=0.3
# #             )
# #             print(f"🤖 Ollama 理解: {parse_result.strip()}")
            
# #             expert = self.brain.learn(text, is_fact=True)
# #             print(f"🧠 存入专家: [{expert}]")
# #             return expert
            
# #         except Exception as e:
# #             print(f"⚠️ Ollama 解析失败，直接存储: {e}")
# #             return self.brain.learn(text, is_fact=True)

#     def learn(self, text):
#         print(f"\n📚 正在学习: {text[:60]}...")
        
#         # 让Ollama分析知识类型和关键词
#         parse_prompt = f"""请分析下面这段知识，严格按要求输出：
#     1. 类型：只能从下面选择一个：人物/事件/案件/名言/方法/概念/知识
#     2. 关键词：3-5个核心关键词

#     知识内容：{text}

#     输出格式：
#     类型：[你的答案]
#     关键词：[关键词1, 关键词2, 关键词3]"""
        
#         try:
#             parse_result = self._call_llm(
#                 parse_prompt,
#                 max_tokens=64,
#                 temperature=0.1
#             ).strip()
            
#             # 解析Ollama返回的类型
#             knowledge_type = "知识"
#             for line in parse_result.split('\n'):
#                 if line.startswith("类型：") or line.startswith("类型:"):
#                     knowledge_type = line.replace("类型：", "").replace("类型:", "").strip()
#                     break
            
#             print(f"🤖 Ollama 理解: 类型：[{knowledge_type}]")
#             print(f"🤖 Ollama 理解: 关键词：[{parse_result.split('关键词：')[-1].strip()}]")
            
#             # ==============================================
#             # 🔥 核心修复：根据Ollama识别的类型，直接路由到对应专家分区
#             # ==============================================
#             type_to_expert = {
#                 # 人物 → 概念专家
#                 "人物": "概念",
#                 # 事件/案件 → 空间专家
#                 "事件": "空间",
#                 "案件": "空间",
#                 # 名言/方法/概念/知识 → 抽象专家
#                 "名言": "抽象",
#                 "方法": "抽象",
#                 "概念": "抽象",
#                 "知识": "抽象"
#             }
            
#             # 获取目标专家分区，默认抽象
#             target_expert = type_to_expert.get(knowledge_type, "抽象")
            
#             # 强制路由到目标专家分区
#             self.brain.learn(text, force_expert=target_expert)
#             print(f"🧠 存入专家: [{target_expert}]")
            
#             return target_expert
        
#         except Exception as e:
#             print(f"⚠️ 解析失败，默认存入抽象专家: {e}")
#             self.brain.learn(text, force_expert="抽象")
#             print(f"🧠 存入专家: [抽象]")
#             return "抽象"

#     def ask(self, query):
#         print(f"\n❓ 用户问题: {query}")
        
#         understand_prompt = f"""用户问了一个问题，请分析他想知道什么，并生成2-3个检索关键词。

#     用户问题：{query}

#     请直接回答，只输出关键词，用空格分隔："""
        
#         try:
#             keywords = self._call_llm(
#                 understand_prompt,
#                 max_tokens=32,
#                 temperature=0.3
#             ).strip()
#             print(f"🔍 检索关键词: {keywords}")
            
#             search_query = keywords if keywords else query
#             memories, meta = self.brain.recall_compositional(search_query)
            
#             if not memories:
#                 memories, meta = self.brain.recall_compositional(query)
            
#             if memories:
#                 print(f"✅ 找到 {len(memories)} 条相关记忆")
                
#                 # 🔥 关键修改：把所有记忆都传给大模型
#                 system_prompt = """你是一个知识助手。请严格遵守以下规则：
#     1. 你只能基于下面提供的"相关记忆"回答问题。
#     2. 从多条记忆中选择最相关的一条来回答。
#     3. 如果没有任何一条记忆能回答问题，请直接说"抱歉，我没有这方面的信息"。
#     4. 不要编造任何记忆中没有的信息。
#     5. 不要提到"记忆"这个词。
#     6. 回答要简洁准确，不超过2句话。"""
                
#                 # 把所有记忆拼接成上下文
#                 memory_context = "\n".join([f"- {mem}" for mem in memories])
#                 user_prompt = f"""相关记忆：
#     {memory_context}

#     用户问题：{query}"""
                
#                 final_answer = self._call_llm(
#                     user_prompt,
#                     system_prompt=system_prompt,
#                     max_tokens=256,
#                     temperature=0.3  # 降低温度，让回答更准确
#                 )
#                 if final_answer.strip() == "抱歉，我没有这方面的信息。":
#                     # is_statement = not any(user_prompt in user_input.lower() for q in ["?", "什么", "哪里", "谁", "怎么", "吗", "呢"])
#                     is_statement = not any(q in query.lower() for q in ["?", "什么", "哪里", "谁", "怎么", "吗", "呢"])
            
#                     if is_statement:
#                         # 是陈述，直接学习
#                         print(f"🧠 没找到相关记忆，触发自主学习...")
#                         self.learn(query)
#                         return "💬 好的，我记住了！"
#                     else:
#                         # 是问题，反问用户答案
#                         return f"🧠 我不知道这个问题的答案..."                        
#                 return f"💬 {final_answer.strip()}"
                
#             else:
#                 if final_answer.strip() == "抱歉，我没有这方面的信息。":
#                     # is_statement = not any(user_prompt in user_input.lower() for q in ["?", "什么", "哪里", "谁", "怎么", "吗", "呢"])
#                     is_statement = not any(q in query.lower() for q in ["?", "什么", "哪里", "谁", "怎么", "吗", "呢"])
            
#                     if is_statement:
#                         # 是陈述，直接学习
#                         print(f"🧠 没找到相关记忆，触发自主学习...")
#                         self.learn(query)
#                         return "💬 好的，我记住了！"
#                     else:
#                         # 是问题，反问用户答案
#                         return f"🧠 我不知道这个问题的答案..."                        
#                 return f"💬 {final_answer.strip()}"
                
#         except Exception as e:
#             print(f"⚠️ Ollama 处理失败: {e}")
#             return "抱歉，我遇到了一些问题，请稍后再试。"


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import datetime
from transformers import (
    CLIPTokenizer, CLIPTextModel,
    AutoTokenizer, AutoModelForCausalLM
)
from typing import List, Tuple, Optional
import numpy as np
from collections import deque, defaultdict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import re

class LLMBrainWrapper:
    def __init__(self, brain, ollama_model_name="qwen2.5:0.5b"):
        self.brain = brain
        self.model_name = ollama_model_name
        
        print(f"🤖 正在初始化 LangChain + Ollama，使用模型: {ollama_model_name}")
        self.llm = ChatOllama(
            model=ollama_model_name,
            temperature=0.3,
            num_predict=256,
        )
        print("✅ LangChain + Ollama 初始化完成！")

    def _call_llm(self, prompt, system_prompt=None, max_tokens=256, temperature=0.3):
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        try:
            temp_llm = ChatOllama(
                model=self.model_name,
                temperature=temperature,
                num_predict=max_tokens,
            )
            response = temp_llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"⚠️ LangChain + Ollama 调用失败: {e}")
            return ""

    def learn(self, text):
        print(f"\n📚 正在学习: {text[:60]}...")
        
        parse_prompt = f"""请分析下面这段知识，严格按要求输出：
1. 类型：只能从下面选择一个：人物/事件/案件/名言/方法/概念/知识
2. 关键词：3-5个核心关键词

知识内容：{text}

输出格式：
类型：[你的答案]
关键词：[关键词1, 关键词2, 关键词3]"""
        
        try:
            parse_result = self._call_llm(
                parse_prompt,
                max_tokens=64,
                temperature=0.1
            ).strip()
            
            knowledge_type = "知识"
            for line in parse_result.split('\n'):
                if line.startswith("类型：") or line.startswith("类型:"):
                    knowledge_type = line.replace("类型：", "").replace("类型:", "").strip()
                    break
            
            print(f"🤖 Ollama 理解: 类型：[{knowledge_type}]")
            print(f"🤖 Ollama 理解: 关键词：[{parse_result.split('关键词：')[-1].strip()}]")
            
            type_to_expert = {
                "人物": "概念",
                "事件": "空间",
                "案件": "空间",
                "名言": "抽象",
                "方法": "抽象",
                "概念": "抽象",
                "知识": "抽象"
            }
            
            target_expert = type_to_expert.get(knowledge_type, "抽象")
            self.brain.learn(text, force_expert=target_expert)
            print(f"🧠 存入专家: [{target_expert}]")
            
            return target_expert
        
        except Exception as e:
            print(f"⚠️ 解析失败，默认存入抽象专家: {e}")
            self.brain.learn(text, force_expert="抽象")
            print(f"🧠 存入专家: [抽象]")
            return "抽象"

    # 🔥 核心新增函数：问题 → 自动判断应该检索哪个专家
    def _get_query_expert(self, query):
        """
        根据用户问题，自动判断要检索的专家分区
        """
        prompt = f"""分析用户问题，判断它在询问什么类型，只能选一个：人物/事件/案件/方法/概念/其他

用户问题：{query}
直接输出答案，不要多余文字："""
        
        type_str = self._call_llm(prompt, temperature=0.1, max_tokens=16).strip()
        print(f"专家: {type_str}")
        
        type_to_expert = {
            "人物": "概念",
            "事件": "空间",
            "案件": "空间",
            "方法": "抽象",
            "概念": "抽象",
            "其他": "抽象"
        }
        return type_to_expert.get(type_str, "抽象")

    def ask(self, query):
        print(f"\n❓ 用户问题: {query}")
        
        # ===================== 🔥 第一步：提取检索关键词 =====================
        understand_prompt = """用户问了一个问题，请分析他想知道什么，并生成2-3个检索关键词。
用户问题：{query}
请直接回答，只输出关键词，用空格分隔："""
        
        try:
            if "记住了" in query:
                is_statement = not any(q in query for q in ["?", "什么", "哪里", "谁", "怎么", "吗", "呢"])
                if is_statement:
                    print(f"🧠 没找到相关记忆，触发自主学习...")
                    self.learn(query)
                    return "💬 好的，我记住了！"
                else:
                    return "🧠 我不知道这个问题的答案..."

            # 提取关键词
            keywords = self._call_llm(
                understand_prompt.format(query=query),
                max_tokens=32,
                temperature=0.1
            ).strip()
            print(f"🔍 检索关键词: {keywords}")
            
            search_query = keywords if keywords else query

            # ===================== 🔥 第二步：自动判断目标专家 =====================
            target_expert = self._get_query_expert(query)
            print(f"🎯 定向检索专家分区: [{target_expert}]")

            # ===================== 🔥 第三步：把专家传入 recall =====================
            # 先搜关键词
            memories, meta = self.brain.recall_compositional(search_query, target_expert=target_expert)
            
            # 没结果再搜原问题
            if not memories:
                memories, meta = self.brain.recall_compositional(query, target_expert=target_expert)
            
            # ===================== 后续回答逻辑（保持不变） =====================
            if memories:
                print(f"✅ 找到 {len(memories)} 条相关记忆")
                
                system_prompt = """你是一个知识助手。请严格遵守以下规则：
                1. 你只能基于下面提供的"相关记忆"回答问题。
                2. 从多条记忆中选择最相关的一条来回答。
                3. 如果没有任何一条记忆能回答问题，请直接说"抱歉，我没有这方面的信息"。
                4. 不要编造任何记忆中没有的信息。
                5. 不要提到"记忆"这个词。
                6. 回答要简洁准确，不超过2句话。
                7. 如果没有答案，请直接说"抱歉，我没有这方面的信息"，不要有多余的其他回复
                """

                # system_prompt = """你是一个严格的知识检索机器人，必须100%遵守以下规则：
                #         1. 只能根据下方提供的参考内容回答，**一字不差地提取原文信息**。
                #         2. **绝对禁止添加、编造、联想任何参考内容里没有的文字**。
                #         3. 如果参考内容里没有答案，**只允许输出固定语句**：抱歉，我没有这方面的信息。
                #         4. 不允许解释、不允许扩展、不允许润色、不允许自己造句。
                #         5. 回答必须**完全来自原文**，简洁、不超过2句话。
                #         6. 绝对不能输出参考内容以外的任何内容。
                #         7. 不允许使用自己的语言重新描述，只能摘抄原文答案。"""
                
                memory_context = "\n".join([f"- {mem}" for mem in memories])
                user_prompt = f"""相关记忆：
{memory_context}

用户问题：{query}"""
                
                final_answer = self._call_llm(
                    user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=256,
                    temperature=0
                )

                # if "抱歉，我没有找到" in final_answer:
                #     final_answer = "抱歉，我没有这方面的信息"
                
                if "抱歉，我没有这方面的信息" in final_answer:
                    is_statement = not any(q in query for q in ["?", "什么", "哪里", "谁", "怎么", "吗", "呢"])
                    if is_statement:
                        print(f"🧠 没找到相关记忆，触发自主学习...")
                        self.learn(query)
                        return "💬 好的，我记住了！"
                    else:
                        return "🧠 我不知道这个问题的答案..."
                
                return f"💬 {final_answer.strip()}"
                
            else:
                is_statement = not any(q in query for q in ["?", "什么", "哪里", "谁", "怎么", "吗", "呢"])
                if is_statement:
                    print(f"🧠 没找到相关记忆，触发自主学习...")
                    self.learn(query)
                    return "💬 好的，我记住了！"
                else:
                    return "🧠 我不知道这个问题的答案..."
                
        except Exception as e:
            print(f"⚠️ Ollama 处理失败: {e}")
            return "抱歉，我遇到了一些问题，请稍后再试。"

    # def ask(self, query):
    #     print(f"\n❓ 用户问题: {query}")
        
    #     # ===================== 🔥 第一步：提取检索关键词 =====================
    #     understand_prompt = """用户问了一个问题，请分析他想知道什么，并生成2-3个检索关键词。
    # 用户问题：{query}
    # 请直接回答，只输出关键词，用空格分隔："""
        
    #     try:
    #         # 提取关键词
    #         keywords = self._call_llm(
    #             understand_prompt.format(query=query),
    #             max_tokens=32,
    #             temperature=0.3
    #         ).strip()
    #         print(f"🔍 检索关键词: {keywords}")
            
    #         search_query = keywords if keywords else query

    #         # ===================== 🔥 第二步：自动判断目标专家 =====================
    #         target_expert = self._get_query_expert(query)
    #         print(f"🎯 定向检索专家分区: [{target_expert}]")

    #         # ===================== 🔥 第三步：把专家传入 recall =====================
    #         # 先搜关键词
    #         memories, meta = self.brain.recall_compositional(search_query, target_expert=target_expert)
            
    #         # 没结果再搜原问题
    #         if not memories:
    #             memories, meta = self.brain.recall_compositional(query, target_expert=target_expert)
            
    #         # ===================== 后续回答逻辑（保持不变） =====================
    #         if memories:
    #             print(f"✅ 找到 {len(memories)} 条相关记忆")
                
    #             system_prompt = """你是一个知识助手。请严格遵守以下规则：
    #                 1. 你只能基于下面提供的"相关记忆"回答问题。
    #                 2. 从多条记忆中选择最相关的一条来回答。
    #                 3. 如果没有任何一条记忆能回答问题，请直接说"抱歉，我没有这方面的信息"。
    #                 4. 绝对不要编造任何记忆中没有的信息。
    #                 5. 不要提到"记忆"这个词。
    #                 6. 回答要简洁准确，不超过2句话。
    #                 7. 回答开头请标注：【使用第N条记忆回答】
    #                 8. 绝对不能编造记忆里不存在的内容，作为答案。"
    #                 9. 不能回与记忆里无关的内容。
    #                 """
                
    #             memory_context = "\n".join([f"- {mem}" for mem in memories])
    #             user_prompt = f"""相关记忆：
    # {memory_context}

    # 用户问题：{query}"""
                
    #             final_answer = self._call_llm(
    #                 user_prompt,
    #                 system_prompt=system_prompt,
    #                 max_tokens=256,
    #                 temperature=0.3
    #             )

    #             # ===================== 🔥 新增：打印大模型使用了哪条记忆 =====================
    #             print(f"\�\n💡 大模型原始回答：{final_answer.strip()}")
    #             print(f"------------------------------------------------------------")

    #             # 解析大模型标记的第几条
    #             # import re
    #             # match = re.search(r'【使用第(\d+)条记忆回答】', final_answer)
    #             # if match:
    #             #     idx = int(match.group(1)) - 1
    #             #     if 0 <= idx < len(memories):
    #             #         # print(f"✅ 最终使用：第{idx+1}条记忆 → {memories[idx]}")
    #             #     # 去掉标记，只保留干净回答
    #             final_answer = final_answer.split('】')[-1].strip()

    #             if "抱歉，我没有这方面的信息" in final_answer:
    #                 is_statement = not any(q in query for q in ["?", "什么", "哪里", "谁", "怎么", "吗", "呢"])
    #                 if is_statement:
    #                     print(f"🧠 没找到相关记忆，触发自主学习...")
    #                     self.learn(query)
    #                     return "💬 好的，我记住了！"
    #                 else:
    #                     return "🧠 我不知道这个问题的答案..."
                
    #             return f"💬 {final_answer.strip()}"
                
    #         else:
    #             is_statement = not any(q in query for q in ["?", "什么", "哪里", "谁", "怎么", "吗", "呢"])
    #             if is_statement:
    #                 print(f"🧠 没找到相关记忆，触发自主学习...")
    #                 self.learn(query)
    #                 return "💬 好的，我记住了！"
    #             else:
    #                 return "🧠 我不知道这个问题的答案..."
                
    #     except Exception as e:
    #         print(f"⚠️ Ollama 处理失败: {e}")
    #         return "抱歉，我遇到了一些问题，请稍后再试。"