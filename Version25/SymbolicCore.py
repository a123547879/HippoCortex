import torch
import random
from typing import Dict, List, Tuple
from collections import defaultdict
import jieba

class ContextualReferenceLearner:
    def __init__(self):
        self.reference_map = {}
        self.conversation_history = []

    def _is_question(self, content: str) -> bool:
        """【核心】判断是否为疑问句，疑问句绝不学习"""
        question_words = ["？", "?", "什么", "哪里", "谁", "怎么", "吗", "呢", "为什么"]
        return any(word in content for word in question_words)

    def learn_from_utterance(self, speaker: str, content: str):
        """【修复】只有非疑问句 + 明确自我介绍 才学习"""
        # 🔴 核心：疑问句直接返回，不做任何学习
        if self._is_question(content):
            return
            
        self.conversation_history.append((speaker, content))

        # 1. 仅学习用户明确自我介绍（我叫X/我是X）
        if speaker == "用户":
            intro_patterns = ["我叫", "我是", "我的名字是"]
            for intro in intro_patterns:
                if intro in content:
                    try:
                        name = content.split(intro, 1)[1].strip().split()[0]
                        if len(name) >= 2 and name not in ["什么", "谁", "不知道"]:
                            self.reference_map["我"] = name
                            self.reference_map["用户"] = name
                            print(f"🧠 自主学习：用户说的'我' = {name}")
                    except:
                        pass
                    break

        # 2. 仅学习用户明确告知AI名字（你叫X/你是X）
        if speaker == "用户":
            ai_intro_patterns = ["你叫", "你是", "你的名字是"]
            for intro in ai_intro_patterns:
                if intro in content:
                    try:
                        name = content.split(intro, 1)[1].strip().split()[0]
                        if len(name) >= 2 and name not in ["什么", "谁"]:
                            self.reference_map["你"] = name
                            self.reference_map["AI"] = name
                            print(f"🧠 自主学习：用户说的'你' = {name}")
                    except:
                        pass
                    break

    def resolve_reference(self, word: str) -> str:
        """解析指代，无匹配则返回原词"""
        return self.reference_map.get(word, word)

class SymbolicCore:
    def __init__(self, sdr_dim=2048, entity_neuron_count=25, reserved_neurons=500):
        self.sdr_dim = sdr_dim
        self.entity_neuron_count = entity_neuron_count
        self.next_neuron_start = reserved_neurons
        
        self.entities = {}
        self.attributes = set()
        self.triplet_index = defaultdict(list)
        self.entity_aliases = defaultdict(set)
        self.reference_learner = ContextualReferenceLearner()
    
    def learn_from_dialogue(self, speaker: str, content: str):
        """对外调用的学习接口"""
        self.reference_learner.learn_from_utterance(speaker, content)

    def _allocate_neurons(self) -> List[int]:
        start = self.next_neuron_start
        end = start + self.entity_neuron_count
        if end > self.sdr_dim:
            raise Exception("实体神经元已用尽")
        self.next_neuron_start = end
        return list(range(start, end))

    def get_or_create_entity(self, entity_name: str, entity_type: str = "未知") -> Dict:
        entity_name = entity_name.strip()
        if not entity_name:
            return None
            
        for main_name, aliases in self.entity_aliases.items():
            if entity_name in aliases:
                entity_name = main_name
                break
                
        if entity_name not in self.entities:
            self.entities[entity_name] = {
                "neurons": self._allocate_neurons(),
                "type": entity_type,
                "count": 1
            }
        else:
            self.entities[entity_name]["count"] += 1
            
        return self.entities[entity_name]

    def add_alias(self, alias: str, main_entity: str):
        alias = alias.strip()
        main_entity = main_entity.strip()
        if alias and main_entity and alias != main_entity:
            self.entity_aliases[main_entity].add(alias)
            self.entity_aliases[alias].add(main_entity)

    def add_triplet(self, subject: str, predicate: str, obj: str, mem_id: str = None):
        subject = subject.strip()
        predicate = predicate.strip()
        obj = obj.strip()
        
        if not all([subject, predicate, obj]):
            return
            
        self.attributes.add(predicate)
        self.get_or_create_entity(subject)
        self.get_or_create_entity(obj)
        
        key = (subject, predicate)
        exists = any(t["object"] == obj for t in self.triplet_index[key])
        if not exists:
            self.triplet_index[key].append({
                "object": obj,
                "mem_id": mem_id
            })

    def get_entity_sdr(self, entity_name: str) -> torch.Tensor:
        entity = self.get_or_create_entity(entity_name)
        if not entity:
            return torch.zeros(self.sdr_dim)
        sdr = torch.zeros(self.sdr_dim)
        sdr[entity["neurons"]] = 1.0
        return sdr

    def parse_question(self, question: str) -> Dict:
        """【修复】问句解析 不触发学习"""
        result = {
            "entities": [],
            "predicate": None,
            "intent": "事实询问"
        }
        
        words = jieba.lcut(question)
        
        # 仅解析指代，不学习
        for word in words:
            resolved = self.reference_learner.resolve_reference(word)
            if resolved in self.entities:
                result["entities"].append(resolved)
        
        # 匹配属性
        for attr in self.attributes:
            if attr in question:
                result["predicate"] = attr
                break
        
        return result

    def symbolic_retrieve(self, parsed_question: Dict) -> List[Dict]:
        results = []
        if not parsed_question["entities"] or not parsed_question["predicate"]:
            return results
            
        for subj in parsed_question["entities"]:
            key = (subj, parsed_question["predicate"])
            if key in self.triplet_index:
                results.extend(self.triplet_index[key])
        return results

    def get_all_triplets(self):
        triplets = []
        for (subj, pred), objs in self.triplet_index.items():
            for obj_item in objs:
                triplets.append((subj, pred, obj_item["object"]))
        return triplets