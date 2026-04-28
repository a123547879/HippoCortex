import networkx as nx
import json
import os
import datetime
from typing import List, Dict, Tuple, Optional
import logging
import re

logger = logging.getLogger("KnowledgeGraphMemory")

class KnowledgeGraphMemory:
    def __init__(self, storage_dir: str, enabled: bool = True):
        """
        :param storage_dir: 存储目录
        :param enabled: 是否启用知识图谱（默认True，可关闭以提升性能）
        """
        self.storage_dir = storage_dir
        self.enabled = enabled  # 🔥 新增：全局开关
        self.graph_path = os.path.join(storage_dir, "knowledge_graph.json")
        self.G = nx.MultiDiGraph()  # 有向多重图，支持同一实体间多个关系
        self.next_entity_id = 1
        
        # 🔥 新增：轻量级预过滤关键词库
        self.entity_keywords = {
            "人物": ["先生", "女士", "博士", "教授", "医生", "老师", "工程师", "科学家", "作家", "诗人"],
            "地点": ["市", "省", "国", "山", "河", "湖", "海", "公园", "学校", "医院", "公司"],
            "事件": ["战争", "革命", "会议", "比赛", "节日", "灾难", "发现", "发明"],
            "概念": ["主义", "理论", "定律", "定理", "原则", "方法", "技术"]
        }
        # 关系关键词
        self.relation_keywords = ["是", "属于", "位于", "发生在", "发明于", "创建于", "朋友", "敌人", "主人", "宠物"]
        
        self.load()

    def _lightweight_ner_filter(self, text: str) -> bool:
        """
        🔥 新增：轻量级预过滤
        基于关键词快速判断是否需要调用LLM做实体抽取
        :return: True=需要LLM抽取，False=跳过
        """
        if not self.enabled:
            return False
        
        # 检查是否包含实体关键词
        for category, keywords in self.entity_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return True
        
        # 检查是否包含关系关键词
        for keyword in self.relation_keywords:
            if keyword in text:
                return True
        
        # 检查是否包含身份相关内容（专门优化）
        identity_keywords = ["我是谁", "你是谁", "我叫", "你叫", "名字", "身份", "主人", "你是", "我是", "关系"]
        for keyword in identity_keywords:
            if keyword in text:
                return True
        
        return False

    def extract_entities_and_relations(self, text: str, expert: str, llm) -> Tuple[List[Dict], List[Dict]]:
        """
        抽取单个文本的实体和关系（带预过滤）
        """
        if not self.enabled:
            return [], []
        
        # 轻量级预过滤
        if not self._lightweight_ner_filter(text):
            return [], []
        
        prompt = f"""
请从下面的文本中提取实体和它们之间的关系，严格按照JSON格式输出，不要添加任何其他内容。

文本：{text}

输出格式：
{{
  "entities": [
    {{"id": "e1", "name": "实体名", "type": "实体类型", "expert": "{expert}"}}
  ],
  "relations": [
    {{"from": "e1", "to": "e2", "type": "关系类型", "confidence": 0.9}}
  ]
}}

实体类型示例：人物、AI、地点、事件、概念、物品
关系类型示例：是、属于、朋友、主人、位于、发生在
"""
        try:
            response = llm.invoke(prompt).content.strip()
            # 提取JSON部分（处理LLM可能输出的多余内容）
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("entities", []), data.get("relations", [])
        except Exception as e:
            logger.warning(f"实体关系提取失败: {e}")
        return [], []

    def batch_extract_entities_and_relations(self, texts: List[str], experts: List[str], llm) -> List[Tuple[List[Dict], List[Dict]]]:
        """
        🔥 新增：批量抽取实体和关系
        一次处理多条，减少LLM调用次数（如果LLM支持长上下文）
        """
        if not self.enabled:
            return [([], []) for _ in texts]
        
        # 先预过滤，只处理需要的
        filtered_indices = []
        filtered_texts = []
        filtered_experts = []
        for i, (text, expert) in enumerate(zip(texts, experts)):
            if self._lightweight_ner_filter(text):
                filtered_indices.append(i)
                filtered_texts.append(text)
                filtered_experts.append(expert)
        
        # 初始化结果
        results = [([], []) for _ in texts]
        
        if not filtered_texts:
            return results
        
        # 如果LLM支持长上下文，批量处理
        if len(filtered_texts) <= 5:  # 假设最多一次处理5条
            try:
                # 构建批量提示
                batch_prompt = "请从下面的多个文本中分别提取实体和关系，严格按照JSON数组格式输出。\n\n"
                for i, (text, expert) in enumerate(zip(filtered_texts, filtered_experts)):
                    batch_prompt += f"文本{i+1}（专家：{expert}）：{text}\n\n"
                
                batch_prompt += """
输出格式（JSON数组，每个元素对应一个文本的结果）：
[
  {
    "entities": [{"id": "e1", "name": "实体名", "type": "实体类型", "expert": "专家名"}],
    "relations": [{"from": "e1", "to": "e2", "type": "关系类型", "confidence": 0.9}]
  }
]
"""
                response = llm.invoke(batch_prompt).content.strip()
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    batch_results = json.loads(json_match.group())
                    for i, result in enumerate(batch_results):
                        if i < len(filtered_indices):
                            results[filtered_indices[i]] = (
                                result.get("entities", []),
                                result.get("relations", [])
                            )
                    return results
            except Exception as e:
                logger.warning(f"批量抽取失败，回退到单条抽取: {e}")
        
        # 回退到单条抽取
        for i, text, expert in zip(filtered_indices, filtered_texts, filtered_experts):
            results[i] = self.extract_entities_and_relations(text, expert, llm)
        
        return results

    def add_memory(self, text: str, expert: str, mem_id: int, llm) -> None:
        """添加一条记忆到知识图谱"""
        if not self.enabled:
            return
        
        entities, relations = self.extract_entities_and_relations(text, expert, llm)
        self._add_to_graph(entities, relations, mem_id)

    def batch_add_memories(self, texts: List[str], experts: List[str], mem_ids: List[int], llm) -> None:
        """
        🔥 新增：批量添加记忆
        """
        if not self.enabled:
            return
        
        # 批量抽取
        batch_results = self.batch_extract_entities_and_relations(texts, experts, llm)
        
        # 批量添加到图
        for (entities, relations), mem_id in zip(batch_results, mem_ids):
            self._add_to_graph(entities, relations, mem_id)

    def _add_to_graph(self, entities: List[Dict], relations: List[Dict], mem_id: int) -> None:
        """内部方法：添加实体和关系到图"""
        if not entities and not relations:
            return
        
        # 1. 添加实体（合并重复实体）
        entity_id_map = {}
        for entity in entities:
            # 检查是否已存在同名同类型实体
            existing_id = self._find_existing_entity(entity["name"], entity["type"])
            if existing_id:
                entity_id_map[entity["id"]] = existing_id
                # 更新实体属性
                self.G.nodes[existing_id]["last_accessed"] = datetime.datetime.now().isoformat()
                self.G.nodes[existing_id]["access_count"] = self.G.nodes[existing_id].get("access_count", 0) + 1
                # 关联记忆ID
                if "mem_ids" not in self.G.nodes[existing_id]:
                    self.G.nodes[existing_id]["mem_ids"] = []
                if mem_id not in self.G.nodes[existing_id]["mem_ids"]:
                    self.G.nodes[existing_id]["mem_ids"].append(mem_id)
            else:
                new_id = f"e{self.next_entity_id}"
                self.next_entity_id += 1
                entity_id_map[entity["id"]] = new_id
                self.G.add_node(
                    new_id,
                    name=entity["name"],
                    type=entity["type"],
                    expert=entity["expert"],
                    mem_ids=[mem_id],
                    created_at=datetime.datetime.now().isoformat(),
                    last_accessed=datetime.datetime.now().isoformat(),
                    access_count=1
                )
        
        # 2. 添加关系
        for rel in relations:
            from_id = entity_id_map.get(rel["from"])
            to_id = entity_id_map.get(rel["to"])
            if from_id and to_id:
                # 检查是否已存在相同关系
                existing_edges = self.G.get_edge_data(from_id, to_id, default={})
                edge_exists = any(
                    edge.get("type") == rel["type"]
                    for edge in existing_edges.values()
                )
                if not edge_exists:
                    self.G.add_edge(
                        from_id,
                        to_id,
                        type=rel["type"],
                        confidence=rel.get("confidence", 0.8),
                        created_at=datetime.datetime.now().isoformat(),
                        strength=1.0
                    )

    def _find_existing_entity(self, name: str, entity_type: str) -> Optional[str]:
        """查找已存在的同名同类型实体"""
        for node_id, attrs in self.G.nodes(data=True):
            if attrs.get("name") == name and attrs.get("type") == entity_type:
                return node_id
        return None

    def get_related_entities(self, entity_name: str, hops: int = 2) -> List[Dict]:
        """获取与指定实体相关的所有实体（多跳）"""
        if not self.enabled:
            return []
        
        entity_id = self._find_existing_entity(entity_name, None)
        if not entity_id:
            return []
        
        related = []
        for hop in range(1, hops+1):
            neighbors = nx.descendants_at_distance(self.G, entity_id, hop)
            for neighbor in neighbors:
                attrs = self.G.nodes[neighbor]
                # 获取关系类型
                edges = self.G.get_edge_data(entity_id, neighbor, default={})
                relations = [edge.get("type") for edge in edges.values()]
                related.append({
                    "name": attrs["name"],
                    "type": attrs["type"],
                    "relations": relations,
                    "hop": hop,
                    "expert": attrs["expert"]
                })
        return related

    def get_relation_score(self, query_entities: List[str], mem_entities: List[str]) -> float:
        """计算查询实体与记忆实体之间的关联度得分"""
        if not self.enabled or not query_entities or not mem_entities:
            return 0.0
        
        total_score = 0.0
        for q_entity in query_entities:
            q_id = self._find_existing_entity(q_entity, None)
            if not q_id:
                continue
            for m_entity in mem_entities:
                m_id = self._find_existing_entity(m_entity, None)
                if not m_id:
                    continue
                # 计算最短路径长度作为关联度
                try:
                    path_length = nx.shortest_path_length(self.G, q_id, m_id)
                    total_score += 1.0 / (path_length + 1)  # 路径越短，得分越高
                except nx.NetworkXNoPath:
                    continue
        
        return total_score / max(len(query_entities), 1)

    def sleep_consolidate(self):
        """睡眠时巩固知识图谱：修剪弱关联、合并重复实体"""
        if not self.enabled:
            return
        
        logger.info("🌙 知识图谱开始睡眠巩固...")
        
        # 1. 修剪弱关系（强度<0.3）
        weak_edges = []
        for u, v, key, attrs in self.G.edges(data=True, keys=True):
            if attrs.get("strength", 1.0) < 0.3:
                weak_edges.append((u, v, key))
        for u, v, key in weak_edges:
            self.G.remove_edge(u, v, key)
        logger.info(f"   修剪了 {len(weak_edges)} 条弱关系")
        
        # 2. 合并高度相似的实体（简化实现）
        logger.info("✅ 知识图谱睡眠巩固完成")

    def save(self):
        """保存知识图谱到本地"""
        if not self.enabled:
            return
        
        data = nx.node_link_data(self.G)
        data["next_entity_id"] = self.next_entity_id
        data["enabled"] = self.enabled  # 保存开关状态
        with open(self.graph_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("💾 知识图谱已保存")

    def load(self):
        """加载知识图谱"""
        if not os.path.exists(self.graph_path):
            logger.info("📦 无历史知识图谱，初始化新图谱")
            return
        try:
            with open(self.graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.next_entity_id = data.pop("next_entity_id", 1)
            self.enabled = data.pop("enabled", True)  # 加载开关状态
            self.G = nx.node_link_graph(data)
            logger.info(f"✅ 知识图谱加载完成，共 {len(self.G.nodes)} 个实体，{len(self.G.edges)} 条关系")
            if not self.enabled:
                logger.info("⚠️  知识图谱当前处于关闭状态")
        except Exception as e:
            logger.error(f"❌ 知识图谱加载失败: {e}")
            self.G = nx.MultiDiGraph()