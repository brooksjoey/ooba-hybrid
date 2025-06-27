import os
import yaml
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
import jsonschema
from importlib import import_module
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# Configure logging with detailed agent loading insights
logging.basicConfig(
    filename=os.path.expanduser("~/ooba-hybrid/logs/orchestrator.log"),
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Enhanced agent configuration with runtime metadata."""
    name: str
    model: str
    tasks: Dict[str, Dict[str, Any]]
    capabilities: List[str]
    integrations: List[str]
    error_handling: Dict[str, Any]
    load_time: float = 0.0  # Track loading performance
    is_active: bool = True  # Dynamic activation state

class AgentLoader:
    """Advanced agent loader with schema validation and lazy initialization."""
    
    _schema = {
        "type": "object",
        "properties": {
            "agents": {
                "type": "object",
                "patternProperties": {
                    "^[a-zA-Z0-9_-]+$": {
                        "type": "object",
                        "properties": {
                            "model": {"type": "string"},
                            "tasks": {"type": "array"},
                            "capabilities": {"type": "array", "items": {"type": "string"}},
                            "integrations": {"type": "array", "items": {"type": "string"}},
                            "error_handling": {
                                "type": "object",
                                "properties": {
                                    "retry_attempts": {"type": "integer"},
                                    "fallback": {"type": "string"}
                                }
                            }
                        },
                        "required": ["model", "tasks"]
                    }
                }
            }
        }
    }

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_schema() -> None:
        """Cache schema validation for performance."""
        with open(os.path.expanduser("~/ooba-hybrid/memory/schema.json")) as f:
            AgentLoader._schema.update(yaml.safe_load(f))
        logger.info("Schema loaded and cached")

    @staticmethod
    def _validate_config(config: Dict) -> bool:
        """Validate YAML against schema with detailed logging."""
        try:
            jsonschema.validate(instance=config, schema=AgentLoader._schema)
            logger.info("Config validated successfully")
            return True
        except jsonschema.exceptions.ValidationError as e:
            logger.error(f"Config validation failed: {e.message}")
            return False

    @staticmethod
    def _parse_tasks(tasks_list: List[Dict]) -> Dict[str, Dict[str, Any]]:
        """Parse tasks with parallel processing for efficiency."""
        with ThreadPoolExecutor() as executor:
            tasks = list(executor.map(lambda task: (task["name"], task), 
                                   [t for t in tasks_list if isinstance(t, dict) and "name" in t]))
        return dict(tasks)

    @classmethod
    def load_agents(cls, config_path: str = os.path.expanduser("~/ooba-hybrid/config/agents.yaml")) -> List[AgentConfig]:
        """Load agents with lazy initialization and performance tracking."""
        if not os.path.exists(config_path):
            logger.error(f"Agent config file not found: {config_path}")
            return []

        with open(config_path) as f:
            config = yaml.safe_load(f)

        if not cls._validate_config(config):
            return []

        agents = []
        cls._load_schema()  # Ensure schema is loaded

        import time
        start_time = time.time()
        for agent_name, agent_config in config.get("agents", {}).items():
            if not isinstance(agent_config, dict):
                logger.error(f"Invalid config for '{agent_name}': expected dict")
                continue

            tasks_list = agent_config.get("tasks", [])
            task_dict = cls._parse_tasks(tasks_list)

            try:
                agent = AgentConfig(
                    name=agent_name,
                    model=agent_config.get("model", "gpt-4o"),
                    tasks=task_dict,
                    capabilities=agent_config.get("capabilities", []),
                    integrations=agent_config.get("integrations", []),
                    error_handling=agent_config.get("error_handling", {"retry_attempts": 3, "fallback": "default"}),
                    load_time=time.time() - start_time
                )
                agents.append(agent)
                logger.info(f"Loaded agent: {agent.name} with {len(agent.tasks)} tasks in {agent.load_time:.2f}s")
            except Exception as e:
                logger.error(f"Failed to load agent '{agent_name}': {e}")

        return agents

    @classmethod
    def reload_agents(cls) -> List[AgentConfig]:
        """Reload agents with dynamic reconfiguration."""
        cls._load_schema.cache_clear()
        return cls.load_agents()

if __name__ == "__main__":
    agents = AgentLoader.load_agents()
    for agent in agents:
        print(f"Agent: {agent.name}, Tasks: {len(agent.tasks)}, Load Time: {agent.load_time:.2f}s")