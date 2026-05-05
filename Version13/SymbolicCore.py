import torch
import random
from typing import Dict, List, Tuple
from collections import defaultdict
import jieba  # 轻量分词，用于实体提取（pip install jieba）

class ContextualReferenceLearner:
    def __init__(self):
        # 完全从零开始学习，没有任何预设
        self.reference_map = {}  # {指代: 真实实体名}
        self.conversation_history = []
        
    def learn_from_utterance(self, speaker: str, content: str):
        """
        从每一轮对话中自主学习指代关系
        speaker: "用户" 或 "AI"
        content: 说话内容
        """
        self.conversation_history.append((speaker, content))
        
        # 1. 学习自我介绍
        if speaker == "用户" and any(w in content for w in ["我叫", "我是", "我的名字是"]):
            # 提取用户名字
            for intro in ["我叫", "我是", "我的名字是"]:
                if intro in content:
                    name = content.split(intro, 1)[1].strip().split()[0]
                    # 学习：用户说的"我" = 这个名字
                    self.reference_map["我"] = name
                    self.reference_map["用户"] = name
                    print(f"🧠 自主学习：用户说的'我' = {name}")
                    break
        
        # 2. 学习AI的名字
        if speaker == "用户" and any(w in content for w in ["你叫", "你是", "你的名字是"]):
            for intro in ["你叫", "你是", "你的名字是"]:
                if intro in content:
                    name = content.split(intro, 1)[1].strip().split()[0]
                    # 学习：用户说的"你" = 这个名字
                    self.reference_map["你"] = name
                    self.reference_map["AI"] = name
                    print(f"🧠 自主学习：用户说的'你' = {name}")
                    break
        
        # 3. 学习第三方指代
        if speaker == "用户" and "他" in content or "她" in content:
            # 从最近对话中找最近提到的人名
            for hist_speaker, hist_content in reversed(self.conversation_history[:-1]):
                # 简单规则：找最近的实体
                words = list(jieba.cut(hist_content))
                for word in words:
                    if len(word) >= 2 and word not in ["你", "我", "他", "她"]:
                        # 假设这就是第三方
                        self.reference_map["他"] = word
                        self.reference_map["她"] = word
                        print(f"🧠 自主学习：'他/她' = {word}")
                        return
        
    def resolve_reference(self, word: str) -> str:
        """解析指代：把'我'/'你'/'他'替换成真实实体名"""
        return self.reference_map.get(word, word)

class SymbolicCore:
    def __init__(self, sdr_dim=2048, entity_neuron_count=25, reserved_neurons=500):
        self.sdr_dim = sdr_dim
        self.entity_neuron_count = entity_neuron_count  # 每个实体自动分配25个专属神经元
        self.next_neuron_start = reserved_neurons  # 前500维留给核心系统，后面动态分配给实体
        
        # ========== 全动态库（零预定义） ==========
        self.entities = {}  # {实体名: {"neurons": [索引], "type": "自动识别", "count": 出现次数}}
        self.attributes = set()  # 自动学习所有属性（喜欢、住在、是、职业...）
        self.triplet_index = defaultdict(list)  # {(主体, 谓词): [对象记忆]}
        
        # 实体别名映射（自动学习：邓尧=主人，小白=你）
        self.entity_aliases = defaultdict(set)
        self.reference_learner = ContextualReferenceLearner()
    
    def learn_from_dialogue(self, speaker: str, content: str):
        """从对话中学习指代关系"""
        self.reference_learner.learn_from_utterance(speaker, content)

    def _allocate_neurons(self) -> List[int]:
        """自动为新实体分配专属神经元区间（永不冲突）"""
        start = self.next_neuron_start
        end = start + self.entity_neuron_count
        if end > self.sdr_dim:
            raise Exception("实体神经元已用尽，可扩大sdr_dim或减少entity_neuron_count")
        self.next_neuron_start = end
        return list(range(start, end))

    def get_or_create_entity(self, entity_name: str, entity_type: str = "未知") -> Dict:
        """获取实体，不存在则自动创建并分配神经元"""
        entity_name = entity_name.strip()
        if not entity_name:
            return None
            
        # 先查别名
        for main_name, aliases in self.entity_aliases.items():
            if entity_name in aliases:
                entity_name = main_name
                break
                
        if entity_name not in self.entities:
            # 自动创建新实体，分配专属神经元
            self.entities[entity_name] = {
                "neurons": self._allocate_neurons(),
                "type": entity_type,
                "count": 1
            }
            print(f"🧠 自动学习新实体: {entity_name} | 分配神经元: {self.entities[entity_name]['neurons'][0]}-{self.entities[entity_name]['neurons'][-1]}")
        else:
            self.entities[entity_name]["count"] += 1
            
        return self.entities[entity_name]

    def add_alias(self, alias: str, main_entity: str):
        """自动添加实体别名（如：邓尧→主人，你→小白）"""
        alias = alias.strip()
        main_entity = main_entity.strip()
        if alias and main_entity and alias != main_entity:
            self.entity_aliases[main_entity].add(alias)
            # 反向映射
            self.entity_aliases[alias].add(main_entity)

    def add_triplet(self, subject: str, predicate: str, obj: str, mem_id: str = None):
        """添加结构化三元组（自动学习属性）"""
        subject = subject.strip()
        predicate = predicate.strip()
        obj = obj.strip()
        
        if not all([subject, predicate, obj]):
            return
            
        # 自动学习新属性
        self.attributes.add(predicate)
        
        # 自动创建主体和对象实体
        self.get_or_create_entity(subject)
        self.get_or_create_entity(obj)
        
        key = (subject, predicate)
        # 去重
        exists = any(t["object"] == obj for t in self.triplet_index[key])
        if not exists:
            self.triplet_index[key].append({
                "object": obj,
                "mem_id": mem_id
            })
            print(f"📝 自动学习三元组: ({subject}, {predicate}, {obj})")

    def get_entity_sdr(self, entity_name: str) -> torch.Tensor:
        """获取实体绑定的SDR向量（用于神经通路激活）"""
        entity = self.get_or_create_entity(entity_name)
        if not entity:
            return torch.zeros(self.sdr_dim)
        sdr = torch.zeros(self.sdr_dim)
        sdr[entity["neurons"]] = 1.0
        return sdr

    def parse_question(self, question: str) -> Dict:
        """【零硬编码】问句解析"""
        result = {
            "entities": [],
            "predicate": None,
            "intent": "事实询问"
        }
        
        # 1. 先学习当前问句
        self.learn_from_dialogue("用户", question)
        
        # 2. 分词
        words = jieba.lcut(question)
        
        # 3. 匹配实体（先解析指代）
        for word in words:
            resolved = self.reference_learner.resolve_reference(word)
            if resolved in self.entities:
                result["entities"].append(resolved)
        
        # 4. 匹配属性（完全从已学习的属性中找）
        for attr in self.attributes:
            if attr in question:
                result["predicate"] = attr
                break
        
        return result

    def symbolic_retrieve(self, parsed_question: Dict) -> List[Dict]:
        """符号通路精准检索（零硬编码）"""
        results = []
        if not parsed_question["entities"] or not parsed_question["predicate"]:
            return results
            
        for subj in parsed_question["entities"]:
            key = (subj, parsed_question["predicate"])
            if key in self.triplet_index:
                results.extend(self.triplet_index[key])
        return results
    

    def get_all_triplets(self):
        """获取所有三元组（供认知能量场调用）"""
        triplets = []
        for (subj, pred), objs in self.triplet_index.items():
            for obj_item in objs:
                triplets.append((subj, pred, obj_item["object"]))
        return triplets