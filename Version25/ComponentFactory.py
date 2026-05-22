from BrainConfig import config
import os
import logging
from typing import Dict, Annotated, Optional, Any
from brain_core import BrainCore
from ILifecycle import ILifecycle
from event_system import EventBus
from DynamicExpert import DynamicExpert
from LearnableSparseEncoderV2 import LearnableSparseEncoder
from ServiceContainer import ServiceContainer
from Wrappers.PersistentCortexWrapper import PersistentCortexWrapper
from Wrappers.PerceptionLoopWrapper import PerceptionLoopWrapper
from Wrappers.LearningLoopWrapper import LearningLoopWrapper
from Wrappers.ConsolidationLoopWrapper import ConsolidationLoopWrapper
from Wrappers.HippocampusRouterWrapper import HippocampusRouterWrapper
from Wrappers.SymbolicCoreWrapper import SymbolicCoreWrapper
from Wrappers.VAEManagerWrapper import VAEManagerWrapper
from Wrappers.ThalamusWrapper import ThalamusWrapper
from Wrappers.DopamineSystemWrapper import DopamineSystemWrapper
from Wrappers.MetacognitionWrapper import MetacognitionWrapper
from Wrappers.CuriosityWrapper import CuriosityWrapper
from Wrappers.DreamingLoopWrapper import DreamingLoopWrapper
from PerceptionLoop.BookReadingService import BookReadingService
from PerceptionLoop.CrossModalBridge import CrossModalBridge
from PerceptionLoop.IntentionService import IntentionService
from PerceptionLoop.MindWanderingService import MindWanderingService
from PerceptionLoop.ThinkEngine import ThinkEngine

class ComponentFactory:
    """组件工厂：负责创建所有认知组件"""
    
    @staticmethod
    def create_experts() -> Dict[str, DynamicExpert]:
        """创建专家网络"""
        expert_names = ["身份", "概念", "空间", "抽象", "视觉"]
        experts = {}
        for name in expert_names:
            experts[name] = DynamicExpert(
                name,
                initial_dim=config.sdr_dim,
                max_dim=config.max_expert_dim,
                active_size=config.sdr_active_size
            )
        return experts
    
    @staticmethod
    def create_sdr_encoders(input_dim: int) -> Dict[str, LearnableSparseEncoder]:
        """创建专家专属稀疏编码器"""
        expert_names = ["身份", "概念", "空间", "抽象", "视觉"]
        encoders = {}
        for name in expert_names:
            encoders[name] = LearnableSparseEncoder(
                input_dim=input_dim,
                sdr_dim=config.sdr_dim,
                active_size=config.sdr_active_size,
                expert_name=name
            )
        return encoders
    
    @staticmethod
    def build_container(embedding_model, llm, kg_enabled: bool = True) -> ServiceContainer:
        container = ServiceContainer(embedding_model, llm, kg_enabled)
        
        # 创建并注册专家和编码器
        container.register("experts", ComponentFactory.create_experts())
        container.register("sdr_encoders", ComponentFactory.create_sdr_encoders(
            input_dim=container["brain_core"].config.dim
        ))
        
        # 创建并注册核心组件
        container.register("cortex", PersistentCortexWrapper())
        container.register("hippocampus_router", HippocampusRouterWrapper())
        container.register("symbolic_core", SymbolicCoreWrapper())
        container.register("vae_manager", VAEManagerWrapper())
        container.register("thalamus", ThalamusWrapper())
        
        # 创建并注册类人学习模块
        container.register("dopamine_system", DopamineSystemWrapper())
        container.register("metacognition", MetacognitionWrapper())
        container.register("curiosity", CuriosityWrapper())
        
        # 创建并注册拆分后的感知循环模块
        container.register("cross_modal_bridge", CrossModalBridge())
        container.register("think_engine", ThinkEngine())
        container.register("intention_service", IntentionService())
        container.register("book_reading_service", BookReadingService())
        container.register("mind_wandering_service", MindWanderingService())
        
        # 创建并注册感知循环包装器
        container.register("perception_loop", PerceptionLoopWrapper())
        
        # 创建并注册其他循环
        container.register("learning_loop", LearningLoopWrapper())
        container.register("dreaming_loop", DreamingLoopWrapper())
        container.register("consolidation_loop", ConsolidationLoopWrapper())
        
        return container