import os
import yaml
import logging
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from orchestrator.memory_chain import get_memory
from orchestrator.dynamic_router import DynamicRouter
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import asyncio
from prometheus_client import Gauge, start_http_server
import threading
from datetime import datetime

# Configure logging with performance metrics
logging.basicConfig(
    filename=os.path.expanduser("~/ooba-hybrid/logs/orchestrator.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s %(process)s"
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_LATENCY = Gauge('orchestrator_request_latency_seconds', 'Latency of request processing')
TASK_SUCCESS_RATE = Gauge('orchestrator_task_success_rate', 'Success rate of tasks', ['agent', 'task'])

@dataclass
class Agent:
    """Enhanced agent definition with telemetry and adaptive capabilities."""
    name: str
    model: str
    tasks: Dict[str, Dict[str, Any]]
    capabilities: List[str]
    integrations: List[str]
    error_handling: Dict[str, Any]
    telemetry: Dict[str, float] = None  # Runtime metrics (e.g., latency, success rate)

class LLMWrapper:
    """LLM wrapper with dynamic model selection and caching."""
    def __init__(self, model_name: str, cache_size: int = 1000):
        self.model_name = model_name
        self.client = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_retries=3
        )
        self.cache: Dict[str, str] = {}
        self.cache_size = cache_size
        self.lock = threading.Lock()

    def call(self, prompt: str) -> str:
        with self.lock:
            if prompt in self.cache:
                return self.cache[prompt]
            response = self.client.invoke(prompt)
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[prompt] = response
            return response

class MemoryWrapper:
    """Memory wrapper with asynchronous context loading."""
    def __init__(self):
        self.vectorstore = get_memory()

    async def search(self, query: str, k: int = 3) -> List[Dict]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.vectorstore.similarity_search(query, k))

class Orchestrator:
    """Advanced orchestrator with multi-agent coordination and performance optimization."""
    def __init__(self, global_settings: Dict[str, Any]):
        self.global_settings = global_settings
        self.memory = MemoryWrapper()
        self.router = DynamicRouter()
        self.agents = self.load_agents()
        self.agents_by_name = {agent.name: agent for agent in self.agents}
        self.prompt_dir = os.path.expanduser("~/ooba-hybrid/orchestrator/prompt_profiles")
        self.chain_map = self.build_chain_map()
        self.setup_global_settings()
        self.task_queue = asyncio.Queue()
        self.metrics_server = start_http_server(8000)  # Expose metrics on port 8000
        asyncio.run(self._start_monitoring())

    def setup_global_settings(self):
        performance = self.global_settings.get("performance", {})
        self.cache_strategy = performance.get("cache_strategy", "LRU")
        self.cache_size = performance.get("cache_size", "10GB")
        self.max_parallel_tasks = performance.get("max_parallel_tasks", 5)
        logger.info(f"Global settings: cache_strategy={self.cache_strategy}, cache_size={self.cache_size}, max_parallel={self.max_parallel_tasks}")

    def load_agents(self) -> List[Agent]:
        agents_path = os.path.expanduser("~/ooba-hybrid/config/agents.yaml")
        with open(agents_path) as f:
            config = yaml.safe_load(f)

        agents = []
        for agent_name, agent_config in config.get("agents", {}).items():
            if not isinstance(agent_config, dict):
                logger.error(f"Invalid config for '{agent_name}'")
                continue

            tasks_list = agent_config.get("tasks", [])
            task_dict = {task["name"]: task for task in tasks_list if isinstance(task, dict) and "name" in task}

            agent = Agent(
                name=agent_name,
                model=agent_config.get("model", "gpt-4o"),
                tasks=task_dict,
                capabilities=agent_config.get("capabilities", []),
                integrations=agent_config.get("integrations", []),
                error_handling=agent_config.get("error_handling", {"retry_attempts": 3, "fallback": "default"}),
                telemetry={"latency": 0.0, "success_rate": 1.0}
            )
            agents.append(agent)
            logger.info(f"Loaded agent: {agent.name} with {len(agent.tasks)} tasks")
        return agents

    def build_chain_map(self) -> Dict[str, Dict[str, Any]]:
        chain_map = {}
        for agent in self.agents:
            prompt_path = f"{self.prompt_dir}/{agent.name}_prompt.txt"
            prompt_text = "You are a highly capable AI assistant."

            if os.path.exists(prompt_path):
                with open(prompt_path) as f:
                    prompt_text = f.read()
            else:
                logger.warning(f"Prompt missing for '{agent.name}', using fallback.")

            prompt_template = PromptTemplate.from_template(prompt_text)
            llm = LLMWrapper(agent.model, cache_size=1000 // len(self.agents))

            chain_map[agent.name] = {
                "llm": llm,
                "prompt": prompt_template,
                "tasks": agent.tasks,
                "error_handling": agent.error_handling,
                "integrations": agent.integrations
            }
        return chain_map

    def match_task(self, query: str, tasks: Dict[str, Dict[str, Any]]) -> str:
        query_lower = query.lower()
        for task_name, task_info in tasks.items():
            keywords = task_info.get("description", "").lower().split()
            if any(kw in query_lower for kw in keywords) or task_name.lower() in query_lower:
                return task_name
        return max(tasks.items(), key=lambda x: len(x[1].get("description", "")))[0]

    async def _process_task(self, agent: Agent, task_name: str, query: str, context: str) -> List[str]:
        start_time = datetime.now()
        agent_config = self.chain_map.get(agent.name)
        if not agent_config:
            raise ValueError(f"No config for agent '{agent.name}'")

        task_params = agent_config["tasks"].get(task_name, {}).get("params", {})
        prompt_input = {"input": query, "context": context, **task_params}

        for attempt in range(agent_config["error_handling"].get("retry_attempts", 3)):
            try:
                prompt = agent_config["prompt"].format(**prompt_input)
                response = agent_config["llm"].call(prompt)
                latency = (datetime.now() - start_time).total_seconds()
                agent.telemetry["latency"] = (agent.telemetry["latency"] * 0.7 + latency * 0.3)  # Exponential moving average
                agent.telemetry["success_rate"] = min(1.0, agent.telemetry["success_rate"] + 0.1)
                TASK_SUCCESS_RATE.labels(agent.name, task_name).set(agent.telemetry["success_rate"])
                REQUEST_LATENCY.set(latency)
                logger.info(json.dumps({
                    "query": query,
                    "agent": agent.name,
                    "task": task_name,
                    "response": str(response)[:200],
                    "latency": latency
                }))
                return [str(response), f"Agent: {agent.name}", f"Task: {task_name}"]
            except Exception as e:
                if attempt == agent_config["error_handling"]["retry_attempts"] - 1:
                    response = agent_config["error_handling"].get("fallback", f"Error: {e}")
                    logger.error(f"Task failed after retries: {e}")
                continue

        return [response]

    async def route_task(self, query: str, context: str) -> tuple[Agent, str]:
        agent_name = self.router.decide_agent(query, context)
        agent = self.agents_by_name.get(agent_name)
        if not agent:
            raise ValueError(f"No agent '{agent_name}' found")
        task_name = self.match_task(query, agent.tasks)
        return agent, task_name

    async def _start_monitoring(self):
        """Asynchronous monitoring loop for agent performance."""
        while True:
            for agent in self.agents:
                logger.info(f"Agent {agent.name} - Latency: {agent.telemetry['latency']:.2f}s, Success Rate: {agent.telemetry['success_rate']:.2f}")
            await asyncio.sleep(60)

    async def process_query(self, query: str) -> List[str]:
        try:
            memory_results = await self.memory.search(query)
            context = "\n".join([doc.page_content for doc in memory_results]) if memory_results else ""

            agent, task_name = await self.route_task(query, context)
            return await self._process_task(agent, task_name, query, context)
        except Exception as e:
            logger.exception("Query processing failed")
            return [f"Error: {e}"]

# Example usage (for testing)
if __name__ == "__main__":
    config = {"global_settings": {"performance": {"cache_strategy": "LRU", "cache_size": "10GB"}}}
    orchestrator = Orchestrator(config)
    asyncio.run(orchestrator.process_query("Plan my week"))