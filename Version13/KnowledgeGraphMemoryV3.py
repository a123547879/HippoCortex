import networkx as nx
import json
import os
import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger("KnowledgeGraphMemory")

class KnowledgeGraphMemory:
    def __init__(self, storage_dir: str, enabled: bool = True):
        """
        极简版知识图谱：仅保留 【实体 ↔ 记忆】 双向映射
        已删除：关系抽取、多跳推理、语义关联、批量处理等所有冗余功能
        """
        self.storage_dir = storage_dir
        self.enabled = enabled
        self.graph_path = os.path.join(storage_dir, "knowledge_graph.json")
        
        # 仅保留有向图，存储 实体 和 记忆 节点
        self.G = nx.DiGraph()
        self.next_entity_id = 1

        # 确保存储目录存在
        os.makedirs(storage_dir, exist_ok=True)
        self.load()

    def _normalize_entity_name(self, name: str) -> str:
        """实体名称标准化（去重、过滤无效值）"""
        if not isinstance(name, str):
            return ""
        name = name.strip()
        if len(name) < 2 or len(name) > 50:
            return ""
        return name

    def _find_existing_entity(self, name: str) -> Optional[str]:
        """根据实体名称查找已存在的实体ID（唯一核心查找）"""
        norm_name = self._normalize_entity_name(name)
        if not norm_name:
            return None

        for node_id, attrs in self.G.nodes(data=True):
            if attrs.get("name", "").strip() == norm_name:
                return node_id
        return None

    # ===================== 核心功能：记忆与实体绑定 =====================
    def add_memory_with_entities(self, content: str, expert_name: str, mem_id: int, entities: List[str]):
        """
        🔥 唯一核心方法：将 记忆 与 实体 双向绑定
        :param content: 记忆内容
        :param expert_name: 专家分类
        :param mem_id: 记忆ID
        :param entities: 实体列表
        """
        if not self.enabled:
            return

        mem_node = f"mem_{mem_id}"

        # 1. 创建/更新记忆节点
        if mem_node not in self.G:
            self.G.add_node(
                mem_node,
                type="memory",
                content=content,
                expert=expert_name,
                mem_id=mem_id,
                created_at=datetime.datetime.now().isoformat()
            )
        else:
            # 更新已有记忆节点的专家信息
            self.G.nodes[mem_node]["expert"] = expert_name

        # 2. 处理实体，并与记忆建立双向映射
        valid_entities = [self._normalize_entity_name(e) for e in entities]
        valid_entities = [e for e in valid_entities if e]

        for entity_name in valid_entities:
            # 查找或创建实体
            entity_id = self._find_existing_entity(entity_name)
            if not entity_id:
                entity_id = f"e{self.next_entity_id}"
                self.next_entity_id += 1
                self.G.add_node(
                    entity_id,
                    name=entity_name,
                    expert=expert_name,
                    created_at=datetime.datetime.now().isoformat()
                )

            # 🔥 核心：双向映射（记忆 ↔ 实体）
            # 记忆 → 实体（包含）
            self.G.add_edge(mem_node, entity_id, type="contains")
            # 实体 → 记忆（被包含）
            self.G.add_edge(entity_id, mem_node, type="contained_in")

        logger.debug(f"✅ 图谱映射完成：记忆{mem_id} 关联 {len(valid_entities)} 个实体")

    # ===================== 核心查询：通过实体找记忆 =====================
    def get_memories_by_entities(self, entities: List[str]) -> List[int]:
        """
        唯一查询方法：根据实体列表，获取所有关联的记忆ID
        """
        if not self.enabled or not entities:
            return []

        mem_ids = set()
        for entity_name in entities:
            entity_id = self._find_existing_entity(entity_name)
            if not entity_id:
                continue

            # 遍历实体的邻居（所有关联的记忆节点）
            for neighbor in self.G.neighbors(entity_id):
                node_attrs = self.G.nodes[neighbor]
                if node_attrs.get("type") == "memory":
                    mem_ids.add(node_attrs.get("mem_id"))

        return list(mem_ids)

    # ===================== 辅助功能：清理无效节点 =====================
    def clean_invalid_nodes(self):
        """清理无关联、无名称的无效节点"""
        invalid_nodes = []
        for node_id, attrs in self.G.nodes(data=True):
            # 清理无名称的实体 / 无内容的记忆
            if attrs.get("type") != "memory" and not attrs.get("name"):
                invalid_nodes.append(node_id)
            elif attrs.get("type") == "memory" and not attrs.get("content"):
                invalid_nodes.append(node_id)

        for node in invalid_nodes:
            self.G.remove_node(node)
        if invalid_nodes:
            logger.debug(f"清理无效节点：{len(invalid_nodes)}")

    # ===================== 持久化：保存/加载 =====================
    def save(self):
        """保存图谱到文件"""
        if not self.enabled:
            return
        try:
            data = nx.node_link_data(self.G)
            data["next_entity_id"] = self.next_entity_id
            data["enabled"] = self.enabled
            with open(self.graph_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("💾 知识图谱（极简版）已保存")
        except Exception as e:
            logger.error(f"图谱保存失败：{e}")

    def load(self):
        """从文件加载图谱"""
        if not os.path.exists(self.graph_path):
            logger.info("📦 初始化极简知识图谱")
            return
        try:
            with open(self.graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.next_entity_id = data.pop("next_entity_id", 1)
            self.enabled = data.pop("enabled", True)
            self.G = nx.node_link_graph(data)
            self.clean_invalid_nodes()
            logger.info(f"✅ 极简图谱加载完成：节点={len(self.G.nodes)}")
        except Exception as e:
            logger.error(f"图谱加载失败：{e}")
            self.G = nx.DiGraph()

    def sleep_consolidate(self):
        """
        🔥 极简版睡眠巩固：无图结构、无弱边修剪
        仅保留接口兼容，不执行任何逻辑
        """
        if not self.enabled:
            return
        # 极简知识图谱无关系/边，无需巩固，仅打印日志
        logger.info("🌙 极简知识图谱：无关系网络，跳过睡眠巩固")

    # ===================== 已删除所有冗余方法 =====================
    # 已删除：关系抽取、多跳推理、关系评分、批量处理、睡眠巩固等