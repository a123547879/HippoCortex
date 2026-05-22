import numpy as np
from collections import defaultdict

class Metacognition:
    def __init__(self, cortex):
        """
        初始化元认知模块
        :param cortex: 大脑皮层对象的引用
        """
        self.cortex = cortex
        
        # 知识置信度字典
        self.knowledge_confidence = defaultdict(float)
        
        # 知识访问时间记录
        self.last_access_time = defaultdict(float)
        
        # 知识一致性记录
        self.knowledge_consistency = defaultdict(float)
        
        # 遗忘曲线参数 (艾宾浩斯遗忘曲线)
        self.forgetting_curve = {
            1: 0.67,   # 1天后
            2: 0.58,   # 2天后
            6: 0.44,   # 6天后
            30: 0.21   # 30天后
        }
    
    def assess_knowledge_confidence(self, concept, current_time=None):
        """
        综合评估对某个概念的知识置信度
        :param concept: 概念标识
        :param current_time: 当前时间戳 (默认使用time.time())
        :return: 置信度分数 (0-1)
        """
        if current_time is None:
            import time
            current_time = time.time()
        
        # 1. 基于激活强度的置信度
        activation_strength = self._get_activation_strength(concept)
        
        # 2. 基于关联数量的置信度
        association_count = self._get_association_count(concept)
        association_score = min(1.0, association_count / 10.0)  # 10个关联以上满分
        
        # 3. 基于激活一致性的置信度
        consistency = self._check_activation_consistency(concept)
        
        # 4. 基于遗忘曲线的时间衰减
        time_decay = self._apply_forgetting_curve(concept, current_time)
        
        # 综合置信度 (加权平均)
        confidence = (
            0.35 * activation_strength +
            0.25 * association_score +
            0.25 * consistency +
            0.15 * time_decay
        )
        
        # 更新记录
        self.knowledge_confidence[concept] = confidence
        self.last_access_time[concept] = current_time
        
        return confidence
    
    def should_admit_ignorance(self, concept, threshold=0.2):
        """
        判断是否应该承认"我不知道"
        :param concept: 概念标识
        :param threshold: 置信度阈值
        :return: 是否应该承认无知
        """
        confidence = self.assess_knowledge_confidence(concept)
        return confidence < threshold
    
    def should_review(self, concept, current_time=None, review_threshold=0.5, days_since_access=3):
        """
        判断是否需要复习某个概念
        :param concept: 概念标识
        :param current_time: 当前时间戳
        :param review_threshold: 复习置信度阈值
        :param days_since_access: 距离上次访问的天数阈值
        :return: 是否需要复习
        """
        if current_time is None:
            import time
            current_time = time.time()
        
        confidence = self.assess_knowledge_confidence(concept, current_time)
        days_passed = (current_time - self.last_access_time.get(concept, 0)) / (24 * 3600)
        
        # 条件：置信度低 且 有一段时间没访问了
        return confidence < review_threshold and days_passed > days_since_access
    
    def get_learning_priority(self, concepts):
        """
        获取概念的学习优先级排序
        :param concepts: 概念列表
        :return: 按优先级排序的概念列表 (优先级最高的在前)
        """
        priorities = []
        for concept in concepts:
            confidence = self.assess_knowledge_confidence(concept)
            # 优先级 = 1 - 置信度 (越不熟悉的优先级越高)
            # 但在"知道一点但不完全知道"时优先级最高
            priority = -4 * (confidence - 0.5) ** 2 + 1  # 抛物线函数
            priorities.append((concept, priority))
        
        # 按优先级降序排序
        priorities.sort(key=lambda x: x[1], reverse=True)
        return [concept for concept, _ in priorities]
    
    def _get_activation_strength(self, concept):
        """获取概念的激活强度 (0-1)"""
        # 这里需要连接到你的Cortex类的实际激活方法
        # 假设你的Cortex有get_activation方法
        try:
            activation = self.cortex.get_activation(concept)
            return min(1.0, activation)
        except:
            return 0.0
    
    def _get_association_count(self, concept):
        """获取概念的关联数量"""
        try:
            associations = self.cortex.get_associations(concept)
            return len(associations)
        except:
            return 0
    
    def _check_activation_consistency(self, concept):
        """检查激活模式的一致性 (0-1)"""
        # 比较当前激活模式与历史激活模式的相似度
        try:
            current_activation = self.cortex.get_activation_pattern(concept)
            historical_pattern = self.cortex.get_historical_activation_pattern(concept)
            
            if historical_pattern is None:
                return 0.5  # 没有历史数据，给中等分
            
            # 计算余弦相似度
            similarity = np.dot(current_activation, historical_pattern) / (
                np.linalg.norm(current_activation) * np.linalg.norm(historical_pattern)
            )
            return max(0.0, similarity)
        except:
            return 0.5
    
    def _apply_forgetting_curve(self, concept, current_time):
        """应用遗忘曲线计算时间衰减因子"""
        last_access = self.last_access_time.get(concept, current_time)
        days_passed = (current_time - last_access) / (24 * 3600)
        
        if days_passed <= 0:
            return 1.0
        
        # 找到最接近的遗忘曲线点
        closest_day = min(self.forgetting_curve.keys(), key=lambda x: abs(x - days_passed))
        if days_passed > closest_day:
            # 超过最大天数，使用最小保留率
            return min(self.forgetting_curve.values())
        
        return self.forgetting_curve.get(closest_day, 0.5)