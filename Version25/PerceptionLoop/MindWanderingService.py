import datetime, torch
import random
import threading
import logging
import time
from typing import Optional

from ILifecycle import ILifecycle, IService
from ServiceContainer import ServiceContainer
from event_system import EventBus, Event, EventType
from Data_models import Intention

logger = logging.getLogger("MindWanderingService")

class MindWanderingService(ILifecycle, IService):
    def __init__(self):
        self.mind_wandering_enabled: bool = True
        self.mind_wandering_idle_threshold: int = 30
        self.last_interaction_time: datetime.datetime = datetime.datetime.now()
        self.mind_wandering_recall_prob: float = 0.5
        self.mind_wandering_assoc_prob: float = 0.3
        self._mind_wandering_running: bool = False
        self.mind_wandering_thread: Optional[threading.Thread] = None
        
        self._container = None
    
    def initialize(self, container: 'ServiceContainer') -> None:
        self._container = container
        
        EventBus().subscribe(EventType.INTERACTION_UPDATED, self._on_interaction_updated)
    
    def start(self) -> None:
        pass
    
    def stop(self) -> None:
        self._stop_mind_wandering()
    
    def save(self, storage_dir: str) -> None:
        pass
    
    def load(self, storage_dir: str) -> None:
        pass
    
    def _on_interaction_updated(self, event: Event):
        self.last_interaction_time = datetime.datetime.now()
        # if self._container["brain_core"].is_mind_wandering:
        #     self._stop_mind_wandering()
    
    def check_mind_wandering_trigger(self) -> None:
        brain_core = self._container["brain_core"]
        
        if (not self.mind_wandering_enabled 
            or brain_core.is_mind_wandering 
            or brain_core.fatigue_level >= brain_core.fatigue_sleep_threshold):
            return
        
        idle_seconds = (datetime.datetime.now() - self.last_interaction_time).total_seconds()
        if idle_seconds >= self.mind_wandering_idle_threshold:
            self._start_mind_wandering()
    
    def _start_mind_wandering(self) -> None:
        brain_core = self._container["brain_core"]
        
        if brain_core.is_mind_wandering:
            return
            
        logger.info("🌙 大脑进入走神状态...")
        brain_core.is_mind_wandering = True
        self._mind_wandering_running = True
        
        self.mind_wandering_thread = threading.Thread(target=self._mind_wandering_loop, daemon=True)
        self.mind_wandering_thread.start()
        
        EventBus().emit(Event(
            event_type=EventType.MIND_WANDER_STARTED,
            data={},
            timestamp=time.time()
        ))
    
    def _stop_mind_wandering(self) -> None:
        brain_core = self._container["brain_core"]
        
        if not brain_core.is_mind_wandering:
            return

        logger.info("⚡ 大脑安全回神！")

        self._mind_wandering_running = False
        brain_core.is_mind_wandering = False

        try:
            if self.mind_wandering_thread is not None and self.mind_wandering_thread.is_alive():
                self.mind_wandering_thread.join(timeout=1.0)
        except:
            pass

        self.mind_wandering_thread = None
        
        EventBus().emit(Event(
            event_type=EventType.MIND_WANDER_STOPPED,
            data={},
            timestamp=time.time()
        ))
    
    def _mind_wandering_loop(self) -> None:
        wander_start_time = datetime.datetime.now()
        MAX_WANDER_DURATION = 120
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 5

        while self._mind_wandering_running:
            try:
                wander_elapsed = (datetime.datetime.now() - wander_start_time).total_seconds()
                
                if wander_elapsed > MAX_WANDER_DURATION:
                    logger.info("⏰ 走神已达最大时长，自动回神")
                    self._stop_mind_wandering()
                    break

                brain_core = self._container["brain_core"]
                if brain_core.fatigue_level >= brain_core.fatigue_sleep_threshold:
                    logger.info("😴 疲劳已达睡眠阈值，停止所有走神活动")
                    brain_core.needs_sleep_request = True
                    self._stop_mind_wandering()
                    break

                hippocampus_router = self._container["hippocampus_router"].router
                if hasattr(hippocampus_router, 'last_scores') and hippocampus_router.last_scores is not None:
                    routing_probs = list(hippocampus_router.last_scores.values())
                else:
                    routing_probs = [0.2, 0.2, 0.2, 0.2, 0.2]

                triple_scores = []
                symbolic_core = self._container["symbolic_core"].symbolic_core if "symbolic_core" in self._container._services else None
                if symbolic_core and hasattr(symbolic_core, 'get_all_triplets'):
                    try:
                        triplets = symbolic_core.get_all_triplets()
                        triple_scores = [1.0 for _ in triplets]
                    except:
                        triple_scores = []

                try:
                    synapse_change = self._container["think_engine"].get_synapse_change()
                except:
                    synapse_change = 0.0

                rule_match = False

                total_energy, energy_detail = brain_core.energy_field.total_energy(
                    routing_probs=routing_probs,
                    triple_scores=triple_scores,
                    sim_scores=[],
                    rule_match=rule_match,
                    synapse_change=synapse_change * 0.3,
                    is_wandering=brain_core.is_mind_wandering,
                    fatigue_level=brain_core.fatigue_level
                )
                
                total_energy = min(total_energy, 25.0)

                base_fatigue = 0.001
                energy_fatigue_multiplier = 1.0 + max(0, (total_energy - 15) / 20)
                brain_core.fatigue_level = min(1.0, brain_core.fatigue_level + base_fatigue * energy_fatigue_multiplier)
                logger.debug(f"🧠 走神中 | 能量:{total_energy:.1f} | 疲劳:{brain_core.fatigue_level:.2f} | 已运行:{wander_elapsed:.0f}秒")

                if not brain_core.needs_sleep_request:
                    if brain_core.fatigue_level >= brain_core.fatigue_sleep_threshold * 0.95:
                        logger.info(f"😴 疲劳即将超限({brain_core.fatigue_level:.2f})，请求睡眠...")
                        brain_core.needs_sleep_request = True
                        self._stop_mind_wandering()
                        break

                base_recall = self.mind_wandering_recall_prob
                base_assoc = self.mind_wandering_assoc_prob
                
                if total_energy > 20:
                    dynamic_recall = min(0.7, base_recall * 1.3)
                    dynamic_assoc = min(0.6, base_assoc * 1.3)
                elif total_energy < 10:
                    dynamic_recall = max(0.2, base_recall * 0.6)
                    dynamic_assoc = max(0.1, base_assoc * 0.6)
                else:
                    dynamic_recall = base_recall
                    dynamic_assoc = base_assoc

                book_reading_service = self._container["book_reading_service"]
                is_reading = random.random() < book_reading_service.read_mode_prob and book_reading_service.book_reader.get_all_books()
                if is_reading:
                    book_reading_service.read_book()
                else:
                    if random.random() < dynamic_recall:
                        self._mind_wandering_memory_recall()
                    if random.random() < dynamic_assoc:
                        self._mind_wandering_association()

                if brain_core.fatigue_level < 0.8:
                    if random.random() < 0.2:
                        self._container["intention_service"].generate_intentions()

                    if random.random() < 0.35 and not self._container["intention_service"].pending_social_intention:
                        intention = self._container["intention_service"].execute_highest_priority_intention()
                else:
                    intention_service = self._container["intention_service"]
                    if not any(i.type == "physiological" for i in intention_service.intention_queue):
                        safe_priority = max(0.0, min(2.0, 
                            brain_core.fatigue_level * intention_service.intention_weights["physiological"] * 1.5
                        ))
                        intention_service.intention_queue.append(Intention(
                            type="physiological",
                            priority=safe_priority,
                            content="我有点困了，先睡一会儿哦~ 睡醒了会更聪明的！",
                            action="express_tiredness",
                            need_sleep=True
                        ))

                consecutive_errors = 0
                time.sleep(3)

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"❌ 走神过程出错({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {e}", exc_info=True)
                
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.error("💥 走神连续出错，强制停止")
                    self._mind_wandering_running = False
                    self._container["brain_core"].is_mind_wandering = False
                    break
                    
                time.sleep(3)
    
    def _mind_wandering_memory_recall(self) -> None:
        try:
            cortex = self._container["cortex"].cortex
            all_mem_ids = list(cortex.index.memories.keys())
            if not all_mem_ids:
                return
                
            weighted_mem_ids = []
            for mem_id in all_mem_ids:
                mem = cortex.index.get_memory(mem_id)
                if not mem:
                    continue
                weight = mem.importance * 2 + mem.metadata.get("recency", 0.5)
                weighted_mem_ids.extend([mem_id] * int(weight * 10))
            
            if not weighted_mem_ids:
                return
                
            target_mem_id = random.choice(weighted_mem_ids)
            mem = cortex.index.get_memory(target_mem_id)
            if mem:
                cortex.increment_access_count(target_mem_id)
                logger.info(f"💭 记忆闪回: {mem.content[:40]}...")
                
        except Exception as e:
            logger.debug(f"记忆闪回失败: {e}")
    
    def _mind_wandering_association(self) -> None:
        try:
            experts = self._container["experts"]
            expert_name = random.choice(list(experts.keys()))
            expert = experts.get(expert_name)
            if not expert or not hasattr(expert, 'memory_packets') or not expert.memory_packets:
                return
                
            random_idx = random.randint(0, len(expert.memory_packets) - 1)
            start_sdr = expert.memory_packets[random_idx].sdr
            
            with torch.no_grad():
                if hasattr(expert, 'forward'):
                    activated = expert.forward(start_sdr, steps=1, top_k=30)
                else:
                    activated = start_sdr
                
                if hasattr(expert, 'retrieve'):
                    assoc_results = expert.retrieve(activated, top_k=2)
                    if assoc_results and len(assoc_results[0]) >= 3:
                        assoc_content = assoc_results[0][2]
                        logger.info(f"🤔 联想想象: → {assoc_content[:40]}...")
                    
        except Exception as e:
            logger.debug(f"联想想象失败: {e}")