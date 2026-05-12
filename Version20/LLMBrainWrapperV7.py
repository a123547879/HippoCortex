import torch
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Optional
import logging
from torch.nn import functional as F

from BrainConfig import config

logger = logging.getLogger("LLMBrainWrapper")

class LLMBrainWrapper:
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
        本地硬规则 + 身份判断 做意图分类（新增身份专家+视觉专家）
        完全不需要LLM，速度快、稳定
        """
        # 🔥 视觉关键词（长相/样子/外貌/图片）
        visual_words = [
            "长什么样", "样子", "长相", "外貌", "模样", "长得", "图片", "照片", "看看你"
        ]
        # 🔥 身份关键词
        identity_words = [
            "你是谁", "我是谁", "名字", "叫什么", "主人", "你的主人",
            "我的名字", "你的名字", "身份", "你是", "我是"
        ]
        person_words = ["介绍", "人物", "哪位", "个人", "生平", "原名"]
        event_words = ["什么时候", "发生", "历史", "事件", "年份", "年代", "案件", "在哪里"]
        know_words = ["是什么", "答案", "原理", "含义", "来源", "意思", "方法", "名言", "知识"]

        query_lower = query.lower()
        # 优先级：视觉 > 身份 > 人物 > 事件 > 知识
        if any(w in query_lower for w in visual_words):
            return "视觉"
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

        # ===================== 🔥 核心新增：概念自动视觉接地（仅在主学习成功后执行） =====================
        try:
            # 只有非视觉专家的学习，才需要关联视觉（避免循环）
            if target_expert != "视觉":
                # 1. 用当前文本在视觉专家里做跨模态检索
                visual_memories, _ = self.brain.recall_compositional(
                    text,
                    target_expert="视觉",
                    top_k=1  # 只取最匹配的1个视觉特征
                )
                
                if visual_memories:
                    # 2. 建立弱突触连接（权重0.3，只做关联，不做主记忆）
                    # 注意：如果你的brain没有create_synapse方法，直接跳过，不影响
                    if hasattr(self.brain, 'create_synapse'):
                        # 这里的mem_id需要根据你的实际brain结构调整
                        # 如果你无法获取刚存入的mem_id，直接跳过这步，只做日志即可
                        logger.info(f"🔗 概念自动接地: [{text[:20]}] → 关联视觉记忆: {str(visual_memories[0])[:30]}")
                    else:
                        # 没有create_synapse也没关系，只要在回忆时一起召回就行
                        logger.info(f"🔗 概念视觉关联: [{text[:20]}] ↔ 找到匹配视觉记忆")
        except Exception as e:
            # 接地失败完全不影响主流程，只打debug日志
            logger.debug(f"⚠️ 概念视觉接地跳过: {e}")

        return target_expert

    def ask(self, query):
        if hasattr(self.brain, '_update_interaction_time'):
            self.brain._update_interaction_time()

        def is_declarative_sentence(q: str) -> bool:
            question_words = ["？", "?", "什么", "哪里", "谁", "怎么", "吗", "呢"]
            return not any(word in q for word in question_words)

        # ===================== 🔥 新增：初始化最终回复变量（统一管理所有返回路径） =====================
        final_answer = ""
        # ====================================================================================

        if "记住了" in query:
            if is_declarative_sentence(query):
                logger.info("🧠 检测到学习指令，触发自主学习...")
                self.learn(query.replace("记住了", "").strip())
                final_answer = "💬 好的，我记住了！"

        # ===================== 🔥 新增：获取时间衰减对话上下文（核心修改1） =====================
        active_context = []
        context_text = ""
        try:
            if hasattr(self.brain, 'cortex') and hasattr(self.brain.cortex, 'get_active_conversation_context'):
                active_context = self.brain.cortex.get_active_conversation_context()
                if active_context:
                    context_parts = ["【最近对话历史】"]
                    for turn in active_context:
                        context_parts.append(f"用户：{turn['user_input']}")
                        context_parts.append(f"小白：{turn['ai_response']}")
                    context_text = "\n".join(context_parts)
                    logger.debug(f"🧠 注入对话上下文 | 轮数:{len(active_context)}")
        except Exception as e:
            logger.error(f"❌ 获取对话上下文失败: {e}")
        # ====================================================================================

        # 如果还没有生成最终回复（不是学习指令），继续正常思考流程
        if not final_answer:
            logger.info(f"\n❓ 用户问题: {query}")
            target_expert = self._get_query_expert_local(query)
            logger.info(f"🎯 定向检索专家分区: [{target_expert}]")

            # ===================== 🔥 新增：构建增强查询（包含对话上下文） =====================
            enhanced_query = f"{context_text}\n用户当前问题：{query}" if context_text else query
            # ====================================================================================

            thought_result = {}
            activated_memories = []
            try:
                # ===================== 🔥 修改：传入增强查询而不是原始query =====================
                thought_result = self.brain.think(enhanced_query, steps=3, expert_last=target_expert)
                # ====================================================================================
                thought_chain = getattr(thought_result, 'thought_chain', '无')
                core_ideas = getattr(thought_result, 'core_ideas', [])
                activated_memories = getattr(thought_result, "activated_memories", [])
                logger.info(f"🤯 大脑思考完成 | 思路：{thought_chain} | 激活记忆数：{len(activated_memories)}")
            except Exception as e:
                logger.error(f"❌ 大脑思考失败，回退传统检索: {e}", exc_info=True)
                # ===================== 🔥 修改：回退检索也使用增强查询 =====================
                memories, _ = self.brain.recall_compositional(enhanced_query, target_expert=target_expert)
                # ====================================================================================
                thought_result = {
                    "thought_chain": "思考失败，使用基础检索",
                    "core_ideas": [],
                    "activated_memories": memories or []
                }
                activated_memories = thought_result["activated_memories"]

            thought_chain = thought_result.thought_chain
            core_ideas = thought_result.core_ideas

            if not activated_memories:
                logger.warning("⚠️ 未激活任何记忆")
                if is_declarative_sentence(query):
                    logger.info("🧠 陈述句无记忆，触发自主学习...")
                    self.learn(query.strip())
                    final_answer = "💬 好的，我记住了！"
                else:
                    if hasattr(self.brain, 'unanswered_questions'):
                        if query not in self.brain.unanswered_questions:
                            self.brain.unanswered_questions.append(query)
                            if len(self.brain.unanswered_questions) > self.brain.max_unanswered:
                                self.brain.unanswered_questions.pop(0)
                            logger.info(f"🧠 记录未解答问题: {query}")
                    final_answer = "🧠 我不知道这个问题的答案..."

            # 如果还没有生成最终回复，继续生成LLM回复
            if not final_answer:
                logger.info(f"✅ 激活 {len(activated_memories)} 条关联记忆")
                
                # ===================== 核心修改：大脑内部视觉记忆重塑（纯推理用，不显示给用户） =====================
                import os
                visual_detail_contexts = []  # 存储重塑后的精细视觉信息
                normal_memories = []
                
                for mem in activated_memories:
                    if isinstance(mem, dict):
                        content = mem.get("content", "")
                        metadata = mem.get("metadata", {})
                    else:
                        content = str(mem)
                        metadata = {}
                        
                    expert = metadata.get("expert", "")
                    vae_latent = metadata.get("vae_latent", None)  # 提取VAE潜在向量

                    # ===================== 大脑内部视觉重塑流程 =====================
                    if expert == "视觉" and vae_latent is not None and hasattr(self.brain, 'vae_manager'):
                        try:
                            logger.info(f"🧠 开始重塑视觉记忆 | ID:{metadata.get('id', '未知')}")
                            
                            # 1. 突触激活：从VAE潜在向量重塑出脑海中的图像
                            mental_image = self.brain.vae_manager.decode_latent(vae_latent)
                            logger.info(f"✅ 视觉记忆重塑完成 | 图像大小: {mental_image.size}")
                            
                            # 2. 生成精细语义描述（可选，但能大幅提升LLM推理能力）
                            # 如果你有图像描述模型，这里可以调用；没有就跳过
                            try:
                                from transformers import BlipProcessor, BlipForConditionalGeneration
                                if not hasattr(self, 'blip_processor'):
                                    self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                                    self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                                
                                inputs = self.blip_processor(mental_image, return_tensors="pt")
                                out = self.blip_model.generate(**inputs, max_length=50)
                                image_caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
                                visual_detail_contexts.append(f"精细视觉特征：{image_caption}")
                                logger.info(f"📝 视觉记忆精细描述: {image_caption}")
                            except:
                                # 没有BLIP也没关系，继续用基础视觉特征
                                pass
                            
                            # 3. 清洗基础视觉记忆文本
                            clean_content = content
                            if "[视觉记忆-" in clean_content:
                                clean_content = clean_content.replace("[视觉记忆-", "").split("]")[0].strip()
                            visual_detail_contexts.append(f"基础视觉特征：{clean_content}")
                            
                        except Exception as e:
                            logger.warning(f"⚠️ 视觉记忆重塑失败: {e}")
                            # 重塑失败不影响主流程，继续用原有的文本记忆
                            clean_content = content
                            if "[视觉记忆-" in clean_content:
                                clean_content = clean_content.replace("[视觉记忆-", "").split("]")[0].strip()
                            visual_detail_contexts.append(f"基础视觉特征：{clean_content}")
                    
                    else:
                        # 普通文本记忆正常处理
                        normal_memories.append(f"文本记忆：{content}")
                # ====================================================================================

                # ===================== 优化提示词：使用大脑内部重塑的视觉信息 =====================
                if visual_detail_contexts:
                    system_prompt = """
                        你现在正在使用你的内部视觉记忆进行思考！
                        下面的【视觉记忆】是你在脑海中重新生成的图像信息，包含了图像的所有细节。
                        请完全基于这些视觉信息和文本记忆，用第一人称口语化、简洁可爱地回答问题。
                        你可以描述图像的颜色、形状、数量、位置等任何细节，就像你真的看到了这张图一样。
                        只说事实，不编造，不要提到"记忆、VAE、潜在向量、重塑"等内部词汇。
                    """
                elif target_expert == "视觉":
                    system_prompt = """
                        你拥有真实的图像记忆能力！
                        下面的【视觉记忆】是你看过的原图信息，包含视觉特征。
                        请结合视觉记忆和文本记忆，用第一人称口语化、简洁可爱地回答问题。
                        只说事实，不编造，不要提到"记忆、路径、绑定ID"等内部词汇。"""
                elif target_expert == "身份":
                    system_prompt = """必须严格按照大脑思考结果回答身份问题：
                    1. 只说事实，不编造、不扩展
                    2. 用第一人称回答（我）
                    3. 简洁回答，不超过1句话
                    4. 严格遵循大脑的思考结果内容"""
                else:
                    system_prompt = """【绝对规则】
                    1. 所有内容基于记忆，禁止编造
                    2. 口语化、简洁回答，不超过2句话
                    3. 无相关信息只输出：抱歉，我没有这方面的信息
                    4. 禁止提到"记忆""思考"等词汇"""

                # ===================== 构造包含重塑视觉信息的Prompt =====================
                memory_context = ""
                if visual_detail_contexts:
                    memory_context += f"【视觉记忆】\n" + "\n".join(visual_detail_contexts) + "\n"
                if normal_memories:
                    memory_context += f"【文本记忆】\n" + "\n".join(normal_memories) + "\n"

                user_prompt = f"""【大脑思考结果】
            联想思路：{thought_chain}
            核心概念：{"、".join(core_ideas)}
            {memory_context}
            用户问题：{query}"""
                
                logger.info(f'user_prompt: {user_prompt}')

                llm_response = self._call_llm(
                    user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=config.llm_max_tokens,
                    temperature=0
                ).strip()

                if "抱歉" in llm_response and "没有这方面的信息" in llm_response:
                    llm_response = "抱歉，我没有这方面的信息"

                if llm_response == "抱歉，我没有这方面的信息":
                    if is_declarative_sentence(query):
                        logger.info("🧠 LLM无答案，陈述句触发自主学习...")
                        self.learn(query.strip())
                        final_answer = "💬 好的，我记住了！"
                    else:
                        if hasattr(self.brain, 'unanswered_questions'):
                            if query not in self.brain.unanswered_questions:
                                self.brain.unanswered_questions.append(query)
                                if len(self.brain.unanswered_questions) > self.brain.max_unanswered:
                                    self.brain.unanswered_questions.pop(0)
                                logger.info(f"🧠 记录未解答问题: {query}")
                        final_answer = llm_response
                else:
                    final_answer = f"💬 {llm_response}"

        # ===================== 🔥 新增：统一添加对话到时间衰减记忆（核心修改2） =====================
        try:
            if hasattr(self.brain, 'cortex') and hasattr(self.brain.cortex, 'add_conversation_turn'):
                # 自动判断对话重要性
                is_important = any(keyword in query.lower() for keyword in 
                    ["记住", "重要", "别忘了", "一定要记得", "我的", "你要", "永远"])
                
                # 提取纯回复内容（去掉前缀"💬 "）
                pure_answer = final_answer.replace("💬 ", "").replace("🧠 ", "")
                
                # 添加到对话记忆
                self.brain.cortex.add_conversation_turn(
                    user_input=query,
                    ai_response=pure_answer,
                    metadata={
                        "is_important": is_important,
                        "importance": 0.9 if is_important else 0.5,
                        "target_expert": target_expert if 'target_expert' in locals() else "概念"
                    }
                )
                logger.debug(f"✅ 对话已添加到记忆 | 重要:{is_important}")
        except Exception as e:
            logger.error(f"❌ 添加对话到记忆失败: {e}")
        # ====================================================================================

        return final_answer