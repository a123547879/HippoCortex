import numpy as np
from collections import defaultdict

class StructureMapper:
    def __init__(self, cortex):
        """
        初始化结构映射引擎
        :param cortex: 大脑皮层对象引用
        """
        self.cortex = cortex
        
        # 关系知识库：存储(实体1, 关系, 实体2)三元组
        self.relation_knowledge = []
        
        # 结构映射缓存
        self.structure_cache = {}
    
    def extract_relations(self, domain_concepts):
        """
        从一个领域中提取关系结构
        :param domain_concepts: 领域内的概念列表
        :return: 关系图 (邻接表形式)
        """
        relation_graph = defaultdict(list)
        
        for concept in domain_concepts:
            # 尝试获取该概念的关联概念和关系
            try:
                associations = self.cortex.get_associations(concept)
                for assoc in associations:
                    # 这里简化处理，实际应该从知识图谱中提取具体关系
                    # 假设我们有一个get_relation方法
                    relation = self._infer_relation(concept, assoc)
                    if relation:
                        relation_graph[concept].append((assoc, relation))
                        self.relation_knowledge.append((concept, relation, assoc))
            except:
                pass
        
        return relation_graph
    
    def find_analogy(self, source_domain, target_domain):
        """
        在源域和目标域之间寻找类比映射
        :param source_domain: 源域概念列表
        :param target_domain: 目标域概念列表
        :return: 类比映射字典 {源概念: 目标概念}
        """
        # 1. 提取两个领域的关系结构
        source_graph = self.extract_relations(source_domain)
        target_graph = self.extract_relations(target_domain)
        
        # 2. 计算结构相似度
        # 这里使用简化的结构匹配算法
        mapping = {}
        
        # 对于源域中的每个概念，寻找目标域中结构最相似的概念
        for source_concept in source_graph:
            best_match = None
            best_score = 0.0
            
            for target_concept in target_graph:
                score = self._structure_similarity(
                    source_concept, source_graph,
                    target_concept, target_graph
                )
                if score > best_score:
                    best_score = score
                    best_match = target_concept
            
            if best_score > 0.3:  # 阈值
                mapping[source_concept] = best_match
        
        return mapping
    
    def transfer_knowledge(self, source_concept, target_concept, mapping=None):
        """
        将知识从源概念迁移到目标概念
        :param source_concept: 源概念
        :param target_concept: 目标概念
        :param mapping: 可选的预计算类比映射
        :return: 是否成功迁移
        """
        if mapping is None:
            mapping = self.find_analogy([source_concept], [target_concept])
        
        if source_concept not in mapping:
            return False
        
        try:
            # 获取源概念的激活模式
            source_activation = self.cortex.get_activation_pattern(source_concept)
            
            # 获取源概念的关联
            source_associations = self.cortex.get_associations(source_concept)
            
            # 迁移激活模式 (衰减70%，表示类比推理的不确定性)
            self.cortex.set_activation_pattern(target_concept, source_activation * 0.7)
            
            # 迁移关联关系
            for assoc in source_associations:
                if assoc in mapping:
                    # 如果关联概念也有映射，迁移这个关系
                    target_assoc = mapping[assoc]
                    relation = self._infer_relation(source_concept, assoc)
                    if relation:
                        self.cortex.create_association(target_concept, target_assoc, relation)
            
            print(f"知识迁移：{source_concept} → {target_concept}")
            return True
            
        except Exception as e:
            print(f"知识迁移失败: {e}")
            return False
    
    def _infer_relation(self, concept1, concept2):
        """推断两个概念之间的关系 (简化版)"""
        # 实际应用中，这应该连接到你的知识图谱
        # 这里返回一些常见关系作为示例
        relations = ["是一种", "有", "用于", "类似", "部分于"]
        # 随机返回一个关系，实际应该从数据中学习
        import random
        return random.choice(relations) if random.random() > 0.5 else None
    
    def _structure_similarity(self, concept1, graph1, concept2, graph2):
        """计算两个概念在各自图中的结构相似度"""
        # 简化的结构相似度：比较邻居数量和关系类型分布
        neighbors1 = graph1.get(concept1, [])
        neighbors2 = graph2.get(concept2, [])
        
        # 基础分数：基于邻居数量的相似度
        count_sim = 1.0 - abs(len(neighbors1) - len(neighbors2)) / max(len(neighbors1), len(neighbors2), 1)
        
        # 关系类型相似度 (简化)
        relations1 = set(r for _, r in neighbors1)
        relations2 = set(r for _, r in neighbors2)
        
        if not relations1 or not relations2:
            rel_sim = 0.5
        else:
            rel_sim = len(relations1 & relations2) / len(relations1 | relations2)
        
        return 0.6 * count_sim + 0.4 * rel_sim