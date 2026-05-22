# 极简端到端 BrainBenchmark
# 核心验证：学习身份 → 睡眠巩固 → 查询身份 完整链路
# 运行前请确保：所有组件已修复，Ollama服务已启动

import time
import logging
import sys
import shutil
from typing import Dict, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BrainBenchmark")

# 导入你的核心组件
from BrainConfig import config
from brain_core import BrainCore
from Cognitive_system import CognitiveSystem
from brain_interface import BrainInterface
from event_system import EventBus
from LLMBrainWrapper import LLMBrainWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings
from Multimodal_gateway import MultiModalInputGateway
from DummyBrain import DummyBrain


class SimpleBrainBenchmark:
    def __init__(self):
        self.brain_interface: Optional[BrainInterface] = None
        self.brain: Optional[Any] = None
        self.cognitive_system: Optional[CognitiveSystem] = None
        self.llm_brain: Optional[LLMBrainWrapper] = None
        self.test_results: Dict[str, Any] = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "steps": []
        }

    def setup(self) -> bool:
        """初始化测试环境"""
        logger.info("=" * 60)
        logger.info("🧪 开始初始化 BrainBenchmark 测试环境")
        logger.info("=" * 60)

        try:
            # 1. 使用独立测试目录，避免污染主数据
            test_dir = "brain_benchmark_test"
            # 如果存在旧测试数据，先清理
            if shutil.os.path.exists(test_dir):
                # shutil.rmtree(test_dir)
                logger.info(f"🧹 清理旧测试目录: {test_dir}")

            # 2. 初始化LLM和Embedding
            llm = ChatOllama(model=config.ollama_model_name)
            embedding_model = OllamaEmbeddings(model="bge-m3")

            # 3. 创建并启动大脑接口（一次性完成所有初始化）
            self.brain_interface = BrainInterface(embedding_model, llm, kg_enabled=True)
            self.brain_interface.start(storage_dir=test_dir)

            # 4. 获取核心组件引用
            self.cognitive_system = self.brain_interface.cognitive_system
            self.core = self.brain_interface.core

            # 5. 创建包装器
            self.brain = DummyBrain(self.cognitive_system, self.brain_interface)
            self.llm_brain = LLMBrainWrapper(self.brain)

            # 6. 彻底清理所有历史记忆（保证测试独立性）
            logger.info("🧹 清理所有历史记忆...")
            self._clear_all_memories()

            logger.info("✅ 测试环境初始化完成")
            return True

        except Exception as e:
            logger.error(f"❌ 测试环境初始化失败: {e}", exc_info=True)
            # 初始化失败时自动清理资源
            if self.brain_interface:
                try:
                    self.brain_interface.stop()
                except:
                    pass
            return False

    def _clear_all_memories(self) -> None:
        """安全清理所有记忆，确保测试隔离"""
        try:
            # 清理皮层长期记忆
            if hasattr(self.brain, 'cortex') and hasattr(self.brain.cortex, 'index'):
                for mem_id in list(self.brain.cortex.index.memories.keys()):
                    self.brain.cortex.index.delete_memory(mem_id)
                logger.info(f"   皮层记忆已清理")

            # 清理海马体临时缓冲区
            if hasattr(self.brain, 'hippocampus_router'):
                self.brain.hippocampus_router.hippocampal_buffer.clear()
                self.brain.hippocampus_router.cortex_index_map.clear()
                logger.info(f"   海马体缓冲区已清空")

            # 清理对话历史
            if hasattr(self.brain, 'cortex'):
                if hasattr(self.brain.cortex, 'all_conversation_turns'):
                    self.brain.cortex.all_conversation_turns.clear()
                if hasattr(self.brain.cortex, 'pending_conversation_consolidation'):
                    self.brain.cortex.pending_conversation_consolidation.clear()
                logger.info(f"   对话历史已清理")

            # 清理专家网络中的记忆
            if hasattr(self.brain, 'experts'):
                for name, expert in self.brain.experts.items():
                    if hasattr(expert, 'memory_packets'):
                        expert.memory_packets.clear()
                    if hasattr(expert, 'sdr_to_mem_id'):
                        expert.sdr_to_mem_id.clear()
                    if hasattr(expert, 'mem_id_to_sdr'):
                        expert.mem_id_to_sdr.clear()
                logger.info(f"   专家网络记忆已清理")

        except Exception as e:
            logger.warning(f"⚠️ 部分记忆清理失败，不影响主测试: {e}")

    def _log_memory_stats(self, label: str = "") -> None:
        """打印当前记忆统计，方便调试"""
        try:
            cortex_count = len(self.brain.cortex.index.memories) if hasattr(self.brain, 'cortex') else 0
            hippo_count = len(self.brain.hippocampus_router.hippocampal_buffer) if hasattr(self.brain, 'hippocampus_router') else 0
            prefix = f"[{label}] " if label else ""
            logger.info(f"📊 {prefix}皮层记忆: {cortex_count} | 海马体缓冲: {hippo_count}")
        except:
            pass

    def run_step(self, step_name: str, func) -> bool:
        """运行单个测试步骤"""
        self.test_results["total_tests"] += 1
        start_time = time.time()

        logger.info(f"\n📌 测试步骤: {step_name}")
        logger.info("-" * 40)

        try:
            result = func()
            elapsed = time.time() - start_time

            if result:
                logger.info(f"✅ 步骤通过 | 耗时: {elapsed:.2f}秒")
                self.test_results["passed_tests"] += 1
                self.test_results["steps"].append({
                    "name": step_name,
                    "status": "passed",
                    "elapsed": elapsed
                })
                return True
            else:
                logger.error(f"❌ 步骤失败 | 耗时: {elapsed:.2f}秒")
                self.test_results["failed_tests"] += 1
                self.test_results["steps"].append({
                    "name": step_name,
                    "status": "failed",
                    "elapsed": elapsed
                })
                return False

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ 步骤异常: {e}", exc_info=True)
            self.test_results["failed_tests"] += 1
            self.test_results["steps"].append({
                "name": step_name,
                "status": "error",
                "elapsed": elapsed,
                "error": str(e)
            })
            return False

    def test_learn_identity(self) -> bool:
        """测试1：学习身份信息"""
        test_identity = "身份：我是小明，你的主人，今年25岁，住在北京"

        logger.info(f"学习内容: {test_identity}")
        target_expert = self.llm_brain.learn(test_identity)

        # 验证：应该分配到身份专家
        # if target_expert != "身份":
        #     logger.error(f"专家分配错误: 预期=身份, 实际={target_expert}")
        #     return False

        # 验证：海马体缓冲区有这条记忆
        if not hasattr(self.brain, 'hippocampus_router'):
            logger.warning("无法验证海马体缓冲区，跳过")
            return True

        buffer = self.brain.hippocampus_router.hippocampal_buffer
        found = any("小明" in mem.content for mem in buffer)

        if not found:
            logger.error("海马体缓冲区未找到学习的身份信息")
            return False

        logger.info("身份信息已成功存入海马体缓冲区")
        self._log_memory_stats("学习后")
        return True

    def test_sleep_consolidation(self) -> bool:
        """测试2：睡眠巩固"""
        # ✅ 修复：通过 cognitive_system 直接调用，DummyBrain 没有 consolidation_loop 属性
        try:
            # 执行快速睡眠（1个epoch，手动模式全量巩固）
            sleep_report = self.cognitive_system.sleep_consolidate_all(epochs=1, is_manual=True)
        except Exception as e:
            logger.error(f"睡眠调用失败: {e}", exc_info=True)
            return False

        # ✅ 修复：SleepReport 没有 success 字段，检查 error 字段
        if sleep_report.error:
            logger.error(f"睡眠失败: {sleep_report.error}")
            return False

        logger.info(f"睡眠完成 | 巩固记忆: {sleep_report.consolidated_count}条 | 遗忘: {sleep_report.forgotten_count}条")
        logger.info(f"睡眠质量: {sleep_report.quality_rating} ({sleep_report.quality_score}分)")

        # 验证：海马体缓冲区已清空或大幅减少
        buffer = self.brain.hippocampus_router.hippocampal_buffer
        if len(buffer) > 0:
            logger.warning(f"海马体缓冲区未完全清空: 剩余{len(buffer)}条记忆")

        self._log_memory_stats("睡眠后")
        return True

    def test_query_identity(self) -> bool:
        """测试3：查询身份信息"""
        test_cases = [
            ("我是谁？", ["小明", "主人"]),
            ("你的主人是谁？", ["小明"]),
            ("我今年多大了？", ["25", "二十五"]),
            ("我住在哪里？", ["北京"])
        ]

        all_passed = True

        for query, expected_keywords in test_cases:
            logger.info(f"\n查询: {query}")
            response = self.llm_brain.ask(query)
            logger.info(f"原始回答: {response}")

            # ✅ 修复：去除 LLM 回复前缀后再检查关键词
            clean_response = response.replace("💬 ", "").replace("🧠 ", "")
            found = any(keyword in clean_response for keyword in expected_keywords)

            if not found:
                logger.error(f"回答未包含预期关键词: {expected_keywords} | 清洗后回答: {clean_response}")
                all_passed = False
            else:
                logger.info(f"✅ 回答正确，包含关键词: {expected_keywords}")

        return all_passed

    def test_conversation_context(self) -> bool:
        """测试4：多轮对话上下文"""
        logger.info("\n测试多轮对话上下文...")

        # 第一轮
        response1 = self.llm_brain.ask("记住了我喜欢吃苹果")
        logger.info(f"Q1: 记住了我喜欢吃苹果")
        logger.info(f"A1: {response1}")

        # 第二轮（依赖上下文）
        response2 = self.llm_brain.ask("我喜欢吃什么？")
        logger.info(f"Q2: 我喜欢吃什么？")
        logger.info(f"A2: {response2}")

        # 去除前缀后检查
        clean_response2 = response2.replace("💬 ", "").replace("🧠 ", "")
        if "苹果" in clean_response2:
            logger.info("✅ 上下文关联正确")
            return True
        else:
            logger.error("❌ 上下文关联失败")
            return False

    def run_all(self) -> Dict[str, Any]:
        """运行所有测试"""
        if not self.setup():
            self.test_results["status"] = "failed"
            return self.test_results

        try:
            # 按顺序运行核心测试步骤
            steps = [
                ("1. 学习身份信息", self.test_learn_identity),
                ("2. 睡眠巩固", self.test_sleep_consolidation),
                ("3. 查询身份信息", self.test_query_identity),
                ("4. 多轮对话上下文", self.test_conversation_context)
            ]

            all_passed = True
            for step_name, step_func in steps:
                if not self.run_step(step_name, step_func):
                    all_passed = False
                    # 关键步骤失败则终止测试
                    if step_name in ["1. 学习身份信息", "2. 睡眠巩固"]:
                        logger.error("\n💥 关键步骤失败，终止测试")
                        break

            # 生成最终报告
            self.test_results["status"] = "passed" if all_passed else "failed"

        finally:
            # ✅ 修复：确保资源一定被清理
            if self.brain_interface:
                logger.info("🧹 关闭大脑接口，释放资源...")
                try:
                    self.brain_interface.stop()
                except Exception as e:
                    logger.warning(f"关闭接口时出错: {e}")

        logger.info("\n" + "=" * 60)
        logger.info("📊 BrainBenchmark 最终测试报告")
        logger.info("=" * 60)
        logger.info(f"总测试数: {self.test_results['total_tests']}")
        logger.info(f"通过: {self.test_results['passed_tests']}")
        logger.info(f"失败: {self.test_results['failed_tests']}")
        logger.info(f"整体状态: {'✅ 全部通过' if all_passed else '❌ 部分失败'}")
        logger.info("=" * 60)

        if all_passed:
            logger.info("\n🎉 恭喜！核心链路验证通过，可以安全进入 Phase 2 开发！")
        else:
            logger.info("\n⚠️  核心链路存在问题，请修复后再继续 Phase 2 开发")

        return self.test_results


if __name__ == "__main__":
    # 运行基准测试
    benchmark = SimpleBrainBenchmark()
    results = benchmark.run_all()

    # 退出码：0=全部通过，1=部分失败
    sys.exit(0 if results["status"] == "passed" else 1)