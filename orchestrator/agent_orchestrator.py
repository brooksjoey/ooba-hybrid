import os
import yaml
import logging
import json
from dataclasses import dataclass
from typing import Dict, List, Any
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from orchestrator.memory_chain import get_memory
from orchestrator.dynamic_router import DynamicRouter
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(
    filename=os.path.expanduser("~/ooba-hybrid/logs/orchestrator.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

@dataclass
class Agent:
    name: str
    model: str
    tasks: Dict[str, Dict[str, Any]]
    capabilities: List[str]
    integrations: List[str]
    error_handling: Dict[str, Any]

class LLMWrapper:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def call(self, prompt: str) -> str:
        return self.client.invoke(prompt)

class MemoryWrapper:
    def __init__(self):
        self.vectorstore = get_memory()

    def search(self, query: str, k: int = 3):
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logging.error(f"Memory search failed: {e}")
            return []

class Orchestrator:
    def __init__(self, global_settings: Dict[str, Any]):
        self.global_settings = global_settings
        self.memory = MemoryWrapper()
        self.router = DynamicRouter()
        self.agents = self.load_agents()
        self.agents_by_name = {agent.name: agent for agent in self.agents}
        self.prompt_dir = os.path.expanduser("~/ooba-hybrid/orchestrator/prompt_profiles")
        self.chain_map = self.build_chain_map()
        self.setup_global_settings()

    def setup_global_settings(self):
        performance = self.global_settings.get("performance", {})
        self.cache_strategy = performance.get("cache_strategy", "LRU")
        self.cache_size = performance.get("cache_size", "10GB")
        logging.info(f"Global settings applied: cache_strategy={self.cache_strategy}, cache_size={self.cache_size}")

    def load_agents(self) -> List[Agent]:
        agents_path = os.path.expanduser("~/ooba-hybrid/config/agents.yaml")
        with open(agents_path) as f:
            config = yaml.safe_load(f)

        agents = []
        for agent_name, agent_config in config.get("agents", {}).items():
            if not isinstance(agent_config, dict):
                logging.error(f"Invalid agent config format for '{agent_name}'")
                continue

            tasks_list = agent_config.get("tasks", [])
            task_dict = {task["name"]: task for task in tasks_list if isinstance(task, dict) and "name" in task}

            try:
                agent = Agent(
                    name=agent_name,
                    model=agent_config.get("model", "gpt-4o"),
                    tasks=task_dict,
                    capabilities=agent_config.get("capabilities", []),
                    integrations=agent_config.get("integrations", []),
                    error_handling=agent_config.get("error_handling", {"retry_attempts": 3, "fallback": "default"})
                )
                agents.append(agent)
                logging.info(f"Loaded agent: {agent.name} with {len(agent.tasks)} tasks")
            except Exception as e:
                logging.error(f"Failed to instantiate agent '{agent_name}': {e}")
        return agents

    def build_chain_map(self) -> Dict[str, Dict[str, Any]]:
        chain_map = {}
        for agent in self.agents:
            prompt_path = f"{self.prompt_dir}/{agent.name}_prompt.txt"
            prompt_text = "You are a helpful AI."

            if os.path.exists(prompt_path):
                prompt_text = open(prompt_path).read()
            else:
                logging.warning(f"Prompt missing for agent '{agent.name}', using fallback.")

            prompt_template = PromptTemplate.from_template(prompt_text)
            llm = LLMWrapper(agent.model)

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
            if task_name.lower() in query_lower or task_info.get("description", "").lower() in query_lower:
                return task_name
        return list(tasks.keys())[0]

    def route_task(self, query: str, context: str) -> tuple[Agent, str]:
        agent_name = self.router.decide_agent(query, context)
        agent = self.agents_by_name.get(agent_name)
        if not agent:
            raise ValueError(f"No agent named '{agent_name}' found.")
        if not agent.tasks:
            raise ValueError(f"No tasks defined for agent '{agent_name}'")
        task_name = self.match_task(query, agent.tasks)
        return agent, task_name

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def process_query(self, query: str) -> List[str]:
        try:
            memory_results = self.memory.search(query)
            context = "\n".join([doc.page_content for doc in memory_results]) if memory_results else ""

            agent, task_name = self.route_task(query, context)
            agent_config = self.chain_map.get(agent.name)

            if not agent_config:
                raise Exception(f"No agent config found for '{agent.name}'")

            task_params = agent_config["tasks"].get(task_name, {}).get("params", {})
            prompt_input = {
                "input": query,
                "context": context,
                **task_params
            }

            for attempt in range(agent.error_handling.get("retry_attempts", 3)):
                try:
                    prompt = agent_config["prompt"].format(**prompt_input)
                    response = agent_config["llm"].call(prompt)
                    break
                except Exception as e:
                    if attempt == agent.error_handling["retry_attempts"] - 1:
                        response = agent.error_handling.get("fallback", "Error occurred")
                        logging.error(f"Task failed after retries: {e}")

            for integration in agent_config["integrations"]:
                logging.info(f"Integration used: {integration}")

            logging.info(json.dumps({
                "query": query,
                "agent": agent.name,
                "task": task_name,
                "response": str(response),
                "context_snippet": context[:200]
            }))

            return [str(response), f"Agent: {agent.name}", f"Task: {task_name}", f"Memory context: {context}"]

        except Exception as e:
            logging.exception("Failed to process query")
            return [f"Error: {e}"]
