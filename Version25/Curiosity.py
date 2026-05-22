import numpy as np
from collections import defaultdict

class Curiosity:
    def __init__(self, metacognition, dopamine_system):
        """
        初始化好奇心驱动模块
        :param metacognition: 元认知模块引用
        :param dopamine_system: 多巴胺系统引用
        """
        self.metacognition = metacognition
        self.dopamine = dopamine_system
        
        # 兴趣水平记录
        self.interest_level = defaultdict(lambda: 1.0)
        
        # 问题生成模板
        self.question_templates = [
            "什么是{concept}？",
            "{concept}和什么有关？",
            "{concept}有什么用？",
            "{concept}是如何工作的？",
            "{concept}和{related_concept}有什么区别？"
        ]
    
    def compute_curiosity(self, concept):
        """
        计算对某个概念的好奇心水平
        :param concept: 概念标识
        :return: 好奇心分数 (0-1)
        """
        confidence = self.metacognition.assess_knowledge_confidence(concept)
        
        # 好奇心抛物线：在"知道一点但不完全知道"时达到峰值
        # 公式：curiosity = -4*(confidence-0.5)^2 + 1
        base_curiosity = -4 * (confidence - 0.5) ** 2 + 1
        
        # 结合个人兴趣水平
        curiosity = base_curiosity * self.interest_level[concept]
        
        return np.clip(curiosity, 0.0, 1.0)
    
    def should_ask_question(self, concept, threshold=0.7):
        """
        判断是否应该主动提问
        :param concept: 概念标识
        :param threshold: 好奇心阈值
        :return: 是否应该提问
        """
        return self.compute_curiosity(concept) > threshold
    
    def generate_questions(self, concept, max_questions=3):
        """
        生成关于某个概念的问题
        :param concept: 概念标识
        :param max_questions: 最大问题数量
        :return: 问题列表
        """
        if not self.should_ask_question(concept):
            return []
        
        questions = []
        
        # 基础问题
        for template in self.question_templates[:3]:  # 先用前3个简单模板
            question = template.format(concept=concept)
            questions.append(question)
        
        # 如果有关联概念，生成比较问题
        try:
            related_concepts = self.metacognition.cortex.get_associations(concept)
            if related_concepts:
                related = related_concepts[0]  # 取第一个关联概念
                compare_question = self.question_templates[4].format(
                    concept=concept, 
                    related_concept=related
                )
                questions.append(compare_question)
        except:
            pass
        
        # 限制数量并返回
        return questions[:max_questions]
    
    def update_interest(self, concept, feedback):
        """
        根据用户反馈更新兴趣水平
        :param concept: 概念标识
        :param feedback: 用户反馈 (-1 到 1，1表示非常感兴趣)
        """
        # 正反馈增加兴趣，负反馈降低兴趣
        self.interest_level[concept] *= (1.0 + 0.2 * feedback)
        self.interest_level[concept] = np.clip(self.interest_level[concept], 0.1, 2.0)
        
        # 同时给多巴胺系统提供奖励信号
        if feedback > 0:
            # 用户感兴趣，给正奖励
            self.dopamine.compute_reward_prediction_error(0.5 * feedback)
    
    def get_exploration_targets(self, all_concepts, top_k=5):
        """
        获取最值得探索的概念列表
        :param all_concepts: 所有可用概念
        :param top_k: 返回数量
        :return: 按探索优先级排序的概念列表
        """
        exploration_scores = []
        for concept in all_concepts:
            curiosity = self.compute_curiosity(concept)
            # 探索分数 = 好奇心 * 新颖性 (这里简化为1-置信度)
            confidence = self.metacognition.assess_knowledge_confidence(concept)
            novelty = 1.0 - confidence
            score = curiosity * 0.7 + novelty * 0.3
            exploration_scores.append((concept, score))
        
        # 排序并返回top_k
        exploration_scores.sort(key=lambda x: x[1], reverse=True)
        return [concept for concept, _ in exploration_scores[:top_k]]