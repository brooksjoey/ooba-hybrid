#!/usr/bin/env python3
import os
import yaml
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_openai import ChatOpenAI  # Updated import

logging.basicConfig(
    filename=os.path.expanduser("~/ooba-hybrid/logs/orchestrator.log"),
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s"
)

class DynamicRouter:
    def __init__(self):
        self.config_path = os.path.expanduser("~/ooba-hybrid/config/routing.yaml")
        self.config = self.load_config()
        self.llms = self.initialize_llms()

    def load_config(self):
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def initialize_llms(self):
        llms = {}
        for provider, details in self.config.get("fallbacks", {}).items():
            try:
                llms[provider] = ChatOpenAI(model_name=details["model"])
            except Exception as e:
                logging.error(f"Failed to initialize model for {provider}: {e}")
        return llms

    def decide_agent(self, query, context=""):
        query_lower = query.lower()

        if any(word in query_lower for word in ["yo", "hey", "hello", "what's up", "you there", "how's it going"]):
            return "chat_agent"
        if "summary" in query_lower or "summarize" in query_lower:
            return "summarizer"
        if "plan" in query_lower or "next steps" in query_lower:
            return "planner"
        return self.config.get("default_agent", "planner")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def route_query(self, query, context):
        default = self.config.get("default")
        llm = self.llms.get(default)

        try:
            response = llm.invoke(query + f"\nContext:\n{context}")
            logging.info(f"Routed to {default}: {str(response)[:100]}")
            return response
        except Exception as e:
            logging.error(f"Primary model {default} failed: {e}")
            for fallback in sorted(
                self.config.get("fallbacks", {}),
                key=lambda x: self.config["fallbacks"][x].get("priority", 0)
            ):
                if fallback == default:
                    continue
                try:
                    response = self.llms[fallback].invoke(query + f"\nContext:\n{context}")
                    logging.info(f"Fallback to {fallback}: {str(response)[:100]}")
                    return response
                except Exception as e2:
                    logging.error(f"Fallback failed for {fallback}: {e2}")
            raise Exception("❌ All routing fallbacks failed.")
