import os
from typing import Optional, List, Dict, Any
from PIL import Image
from brain_core import BrainCore
from Cognitive_system import CognitiveSystem
from event_system import EventBus, Event, EventType
from plugin_system import Plugin, PerceptionPlugin, ActionPlugin
import importlib
import logging

logger = logging.getLogger("BrainInterface")

class PluginManager:
    def __init__(self):
        self.core = BrainCore()
        self._perception_plugins: Dict[str, PerceptionPlugin] = {}
        self._action_plugins: Dict[str, ActionPlugin] = {}
        self._all_plugins: Dict[str, Plugin] = {}
    
    def load_plugin(self, plugin_path: str) -> bool:
        try:
            if os.path.exists(plugin_path):
                spec = importlib.util.spec_from_file_location("plugin_module", plugin_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                module = importlib.import_module(plugin_path)
            
            plugin_classes = [
                obj for name, obj in module.__dict__.items()
                if isinstance(obj, type) and issubclass(obj, Plugin) and obj != Plugin
            ]
            
            for plugin_class in plugin_classes:
                plugin = plugin_class()
                self._register_plugin(plugin)
            
            return True
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_path}: {e}")
            return False
    
    def _register_plugin(self, plugin: Plugin):
        plugin.initialize()
        plugin_id = f"{plugin.get_name()}_{plugin.get_version()}"
        self._all_plugins[plugin_id] = plugin
        
        if isinstance(plugin, PerceptionPlugin):
            self._perception_plugins[plugin.get_modality()] = plugin
        elif isinstance(plugin, ActionPlugin):
            self._action_plugins[plugin.get_action_type()] = plugin
    
    def get_perception_plugin(self, modality: str) -> Optional[PerceptionPlugin]:
        return self._perception_plugins.get(modality)
    
    def get_action_plugin(self, action_type: str) -> Optional[ActionPlugin]:
        return self._action_plugins.get(action_type)
    
    def unload_all(self):
        for plugin in self._all_plugins.values():
            plugin.shutdown()
        self._perception_plugins.clear()
        self._action_plugins.clear()
        self._all_plugins.clear()

class BrainInterface:
    def __init__(self, embedding_model, llm, kg_enabled: bool = True):
        self.core = BrainCore()
        self.cognitive_system = CognitiveSystem(embedding_model, llm, kg_enabled)
        self.plugin_manager = PluginManager()
        self.event_bus = EventBus()
    
    def start(self, storage_dir: str):
        """启动整个大脑系统"""
        logger.info("🚀 启动大脑系统...")
        self.core.start(storage_dir)
        self.cognitive_system.initialize(storage_dir)
        self.event_bus.start_processing()
        logger.info("✅ 大脑系统启动完成")
    
    def stop(self):
        """停止整个大脑系统"""
        logger.info("🛑 停止大脑系统...")
        self.event_bus.stop_processing()
        self.plugin_manager.unload_all()
        self.core.stop()
        logger.info("✅ 大脑系统已停止")
    
    def chat(self, text: str) -> Optional[str]:
        """文本对话接口"""
        result = self.cognitive_system.think(text)
        if "error" in result:
            return result["error"]
        
        # 构建回复
        response = result["thought_chain"]
        if result["core_ideas"]:
            response += "\n\n核心观点：" + "、".join(result["core_ideas"])
        return response
    
    def learn_text(self, text: str, force_expert=None):
        """学习文本记忆"""
        return self.cognitive_system.learn(text, force_expert)
    
    def batch_learn_text(self, texts: List[str]):
        """
        🔥 初始批量知识库导入：直接写入皮层，跳过海马体/丘脑
        不传专家则默认使用【概念】
        """
        return self.cognitive_system.batch_learn(texts)
    
    def process_image(self, image_path: str, description: Optional[str] = None) -> Optional[str]:
        """处理图像（编码+学习）"""
        try:
            # 使用VAE编码图像
            latent_data = self.cognitive_system.vae_manager.encode_image(image_path)
            
            # 构建视觉记忆文本
            mem_text = f"[视觉记忆-{os.path.basename(image_path)}] {description or '图像记忆'}"
            if description:
                mem_text += f" | 描述：{description}"
            
            # 学习视觉记忆
            mem_id = self.cognitive_system.learn(mem_text, force_expert="视觉")
            
            # 存储VAE潜在向量到元数据
            if mem_id and mem_id in self.cognitive_system.cortex.index.memory_store:
                self.cognitive_system.cortex.index.memory_store[mem_id]["metadata"]["vae_latent"] = latent_data
            
            return f"✅ 图像记忆已存储 | ID: {mem_id}"
        except Exception as e:
            logger.error(f"❌ 图像处理失败: {e}")
            return f"图像处理失败: {e}"
    
    def decode_image(self, mem_id: str) -> Optional[Image.Image]:
        """从记忆解码图像"""
        try:
            mem = self.cognitive_system.cortex.index.get_memory(mem_id)
            if not mem:
                return None
            
            latent_data = mem.get("metadata", {}).get("vae_latent")
            if not latent_data:
                return None
            
            return self.cognitive_system.vae_manager.decode_latent(latent_data)
        except Exception as e:
            logger.error(f"❌ 图像解码失败: {e}")
            return None
    
    def trigger_sleep(self, is_manual: bool = False) -> Dict:
        """触发睡眠流程
        :param is_manual: 是否为用户手动触发（手动睡眠巩固100%记忆）
        :return: 睡眠质量报告
        """
        logger.info("🌙 大脑接口收到睡眠请求")
        self.core.is_sleeping = True
        
        try:
            # 调用认知系统的三阶段睡眠
            sleep_report = self.cognitive_system.sleep_consolidate_all(is_manual=is_manual)
            return sleep_report
        except Exception as e:
            logger.error(f"❌ 睡眠流程执行失败: {e}", exc_info=True)
            return {"error": str(e)}
        finally:
            self.core.is_sleeping = False
    
    def load_plugins(self, plugin_paths: List[str]):
        """加载插件"""
        for path in plugin_paths:
            self.plugin_manager.load_plugin(path)
    
    def perceive_from_modality(self, modality: str) -> Optional[str]:
        """从感知插件获取输入"""
        plugin = self.plugin_manager.get_perception_plugin(modality)
        if plugin:
            data = plugin.perceive()
            if data:
                return self.chat(str(data))
        return None
    
    def execute_action(self, action_type: str, params: Dict[str, Any]) -> Any:
        """执行动作插件"""
        plugin = self.plugin_manager.get_action_plugin(action_type)
        if plugin:
            return plugin.execute(params)
        return None
    
    def get_status(self) -> Dict:
        """获取大脑状态"""
        return {
            "is_running": self.core.is_running,
            "fatigue_level": self.core.fatigue_level,
            "is_mind_wandering": self.core.is_mind_wandering,
            "intention_queue_size": len(self.cognitive_system.intention_queue)
        }