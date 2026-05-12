import numpy as np
from collections import deque

class DopamineSystem:
    def __init__(self, tau_reward=0.1, tau_dopamine=0.05):
        """
        初始化多巴胺系统
        :param tau_reward: 奖励期望的更新速率 (0-1)
        :param tau_dopamine: 多巴胺浓度的衰减速率
        """
        self.expected_reward = 0.0
        self.tau_reward = tau_reward
        self.tau_dopamine = tau_dopamine
        
        # 当前多巴胺浓度
        self.dopamine_level = 0.0
        
        # 历史记录用于分析
        self.reward_history = deque(maxlen=1000)
        self.rpe_history = deque(maxlen=1000)
        
    def compute_reward_prediction_error(self, actual_reward):
        """
        计算奖励预测误差 (RPE)
        :param actual_reward: 实际获得的奖励 (-1 到 1)
        :return: 奖励预测误差
        """
        rpe = actual_reward - self.expected_reward
        
        # 更新奖励期望 (指数移动平均)
        self.expected_reward = (1 - self.tau_reward) * self.expected_reward + self.tau_reward * actual_reward
        
        # 更新多巴胺浓度
        self.dopamine_level += rpe
        self.dopamine_level *= (1 - self.tau_dopamine)
        self.dopamine_level = np.clip(self.dopamine_level, -1.0, 1.0)
        
        # 记录历史
        self.reward_history.append(actual_reward)
        self.rpe_history.append(rpe)
        
        return rpe
    
    def modulate_synaptic_plasticity(self, synapse_weights, activations, rpe, learning_rate=0.01):
        """
        基于多巴胺调节突触可塑性
        :param synapse_weights: 突触权重矩阵
        :param activations: 神经元激活向量
        :param rpe: 奖励预测误差
        :param learning_rate: 基础学习率
        :return: 更新后的突触权重
        """
        # 多巴胺门控：只有当RPE不为零时才发生显著学习
        dopamine_gate = np.abs(rpe)
        
        # Hebbian学习 + 多巴胺调节
        # 正RPE：增强共激活的突触
        # 负RPE：减弱共激活的突触
        weight_update = learning_rate * dopamine_gate * rpe * np.outer(activations, activations)
        
        # 应用更新并保持权重在合理范围内
        synapse_weights += weight_update
        synapse_weights = np.clip(synapse_weights, -1.0, 1.0)
        
        return synapse_weights
    
    def get_curiosity_reward(self, information_gain):
        """
        计算基于信息增益的内部好奇心奖励
        :param information_gain: 信息增益 (0-1)
        :return: 好奇心奖励
        """
        return 0.5 * information_gain  # 好奇心奖励权重
    
    def get_prediction_reward(self, prediction_accuracy):
        """
        计算基于预测成功的内部奖励
        :param prediction_accuracy: 预测准确率 (0-1)
        :return: 预测奖励
        """
        return 0.3 * (prediction_accuracy - 0.5)  # 预测准确率超过50%才给正奖励