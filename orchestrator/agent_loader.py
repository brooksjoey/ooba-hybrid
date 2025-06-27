import os
import yaml
import logging
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class AgentConfig:
    name: str
    model: str
    tasks: Dict[str, Dict[str, Any]]
    capabilities: List[str]
    integrations: List[str]
    error_handling: Dict[str, Any]

class AgentLoader:
    @staticmethod
    def load_agents(config_path: str) -> List[AgentConfig]:
        """Load and parse agents from YAML config file"""
        if not os.path.exists(config_path):
            logging.error(f"Agent config file not found: {config_path}")
            return []

        with open(config_path) as f:
            config = yaml.safe_load(f)

        agents = []
        for agent_name, agent_config in config.get("agents", {}).items():
            if not isinstance(agent_config, dict):
                logging.error(f"Invalid agent config for '{agent_name}': expected dict, got {type(agent_config)}")
                continue

            tasks_list = agent_config.get("tasks", [])
            task_dict = AgentLoader._parse_tasks(tasks_list)

            try:
                agent = AgentConfig(
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
                logging.error(f"Failed to load agent '{agent_name}': {e}")

        return agents

    @staticmethod
    def _parse_tasks(tasks_list: List[Dict]) -> Dict[str, Dict[str, Any]]:
        """Convert list of task dicts to dict keyed by task name"""
        return {task["name"]: task for task in tasks_list if isinstance(task, dict) and "name" in task}
