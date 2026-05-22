import numpy as np

class CognitiveEnergyField:
    def __init__(self):
        self.eps = 1e-8

    def neural_energy(self, routing_probs: list) -> float:
        """🔥 修复：列表转numpy数组，计算路由熵能量（越低越稳定）"""
        # 空列表直接返回0能量
        if not routing_probs:
            return 0.0
        
        # 核心修复：转numpy数组，归一化保证概率合法
        probs = np.array(routing_probs, dtype=np.float32)
        probs = probs / (np.sum(probs) + self.eps)  # 归一化
        probs = np.clip(probs, self.eps, 1.0)       # 防止log(0)
        
        # 计算熵
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)

    def symbolic_energy(self, triple_scores: list) -> float:
        """符号能量：三元组越多，认知负载越高"""
        return float(len(triple_scores)) * 0.1

    def retrieval_energy(self, sim_scores: list) -> float:
        """检索能量：相似度越低，能量越高（越混乱）"""
        if not sim_scores:
            return 0.0
        mean_sim = np.mean(np.array(sim_scores, dtype=np.float32))
        return float((1.0 - mean_sim) * 2.0)

    def rule_energy(self, rule_match: bool) -> float:
        """规则匹配能量：匹配失败=混乱，能量更高"""
        return 0.0 if rule_match else 1.5

    def synaptic_energy(self, synapse_change: float) -> float:
        """突触能量：权重变化越大，能量越高"""
        return float(synapse_change * 0.001)

    def fatigue_energy(self, fatigue_level: float) -> float:
        """疲劳能量：疲劳越高，能量越高"""
        return float(fatigue_level * 3.0)

    def wander_energy(self, is_wandering: bool) -> float:
        """走神能量：走神时大脑混乱，能量更高"""
        return 2.0 if is_wandering else 0.0

    def total_energy(self, 
                    routing_probs: list,
                    triple_scores: list,
                    sim_scores: list,
                    rule_match: bool,
                    synapse_change: float,
                    is_wandering: bool,
                    fatigue_level: float) -> tuple[float, dict]:
        """计算总认知能量（越低=越稳定专注）"""
        # 分项能量
        e_neural = self.neural_energy(routing_probs)
        e_symbol = self.symbolic_energy(triple_scores)
        e_retriev = self.retrieval_energy(sim_scores)
        e_rule = self.rule_energy(rule_match)
        e_synap = self.synaptic_energy(synapse_change)
        e_fatigue = self.fatigue_energy(fatigue_level)
        e_wander = self.wander_energy(is_wandering)

        # 总能量
        total = e_neural + e_symbol + e_retriev + e_rule + e_synap + e_fatigue + e_wander

        # 详情
        detail = {
            "总能量": round(total, 1),
            "路由能量": round(e_neural, 1),
            "符号能量": round(e_symbol, 1),
            "检索能量": round(e_retriev, 1),
            "规则能量": round(e_rule, 1),
            "突触能量": round(e_synap, 1),
            "疲劳能量": round(e_fatigue, 1),
            "走神能量": round(e_wander, 1)
        }

        return round(total, 1), detail
    
    # ================== 🔥 新增：添加这两个空方法即可 ==================
    def initialize(self):
        """空初始化方法，兼容架构调用"""
        pass

    def shutdown(self):
        """空关闭方法，兼容架构调用"""
        pass
    # =====================================================================