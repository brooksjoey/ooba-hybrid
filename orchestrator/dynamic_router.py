#!/usr/bin/env python3
import os
import yaml
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_openai import ChatOpenAI
import asyncio
from typing import Dict, Optional
import random
from collections import defaultdict

# Configure logging with detailed routing insights
logging.basicConfig(
    filename=os.path.expanduser("~/ooba-hybrid/logs/orchestrator.log"),
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

class DynamicRouter:
    """Sophisticated router with predictive task assignment and load balancing."""
    
    def __init__(self):
        self.config_path = os.path.expanduser("~/ooba-hybrid/config/routing.yaml")
        self.config = self.load_config()
        self.llms = self.initialize_llms()
        self.load_metrics = defaultdict(float)  # Track LLM load
        self.prediction_model = self._build_prediction_model()
        asyncio.run(self._start_load_monitor())

    def load_config(self) -> Dict:
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def initialize_llms(self) -> Dict[str, ChatOpenAI]:
        llms = {}
        for provider, details in self.config.get("fallbacks", {}).items():
            try:
                llms[provider] = ChatOpenAI(
                    model_name=details["model"],
                    api_key=os.getenv("OPENAI_API_KEY"),
                    max_retries=3,
                    request_timeout=30
                )
            except Exception as e:
                logger.error(f"Failed to init model for {provider}: {e}")
        return llms

    def _build_prediction_model(self):
        """Simple heuristic predictor (replace with ML model in production)."""
        def predict_agent(query: str, context: str) -> str:
            keywords = {
                "chat": ["yo", "hey", "what's up"],
                "summarize": ["summary", "summarize"],
                "plan": ["plan", "next steps"]
            }
            combined_text = f"{query} {context}".lower()
            for intent, keys in keywords.items():
                if any(k in combined_text for k in keys):
                    return intent + "_agent"
            return self.config.get("default_agent", "planner")
        return predict_agent

    async def _start_load_monitor(self):
        """Asynchronous load balancing monitor."""
        while True:
            total_load = sum(self.load_metrics.values())
            if total_load > 0:
                for provider in self.llms:
                    self.load_metrics[provider] /= total_load
            await asyncio.sleep(60)

    def _select_optimal_provider(self, exclude: Optional[str] = None) -> str:
        """Select provider with lowest load, excluding failed ones."""
        available_providers = [p for p in self.llms if p != exclude]
        if not available_providers:
            raise ValueError("No available providers")
        return min(available_providers, key=lambda x: self.load_metrics[x] or float('inf'))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=lambda retry_state: logger.warning(f"Retrying routing, attempt {retry_state.attempt_number}")
    )
    async def route_query(self, query: str, context: str = "") -> str:
        """Route query with predictive assignment and failover."""
        default = self.config.get("default")
        agent_name = self.prediction_model(query, context)
        
        try:
            provider = self._select_optimal_provider()
            llm = self.llms[provider]
            response = await asyncio.to_thread(llm.invoke, query + f"\nContext:\n{context}")
            self.load_metrics[provider] += 1
            logger.info(f"Routed to {provider} for {agent_name}: {str(response)[:100]}")
            return agent_name
        except Exception as e:
            logger.error(f"Primary {provider} failed: {e}")
            self.load_metrics[provider] += 10  # Penalize failed provider
            fallback_provider = self._select_optimal_provider(exclude=provider)
            if fallback_provider:
                llm = self.llms[fallback_provider]
                response = await asyncio.to_thread(llm.invoke, query + f"\nContext:\n{context}")
                self.load_metrics[fallback_provider] += 1
                logger.info(f"Fallback to {fallback_provider} for {agent_name}: {str(response)[:100]}")
                return agent_name
            raise Exception("All routing attempts failed")

    def decide_agent(self, query: str, context: str = "") -> str:
        """Synchronous wrapper for agent decision."""
        return asyncio.run(self.route_query(query, context))

if __name__ == "__main__":
    router = DynamicRouter()
    agent = router.decide_agent("Yo, what's good?", "Some context here")
    print(f"Decided agent: {agent}")