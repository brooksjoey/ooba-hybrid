#!/usr/bin/env python3
import os
import logging
import dotenv
import yaml
from orchestrator.agent_orchestrator import Orchestrator

# Load environment variables
dotenv.load_dotenv(dotenv_path=os.path.expanduser("~/ooba-hybrid/config/apis.env"))

# Configure logging
logging.basicConfig(
    filename=os.path.expanduser("~/ooba-hybrid/logs/orchestrator.log"),
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s"
)

def run_diagnostics():
    print("\n[Agent Diagnostics]")
    try:
        routing_config_path = os.path.expanduser("~/ooba-hybrid/config/routing.yaml")
        with open(routing_config_path) as f:
            config = yaml.safe_load(f)

        for name, details in config.get("fallbacks", {}).items():
            print(f"- {name}: ", end="")
            try:
                from langchain_community.chat_models import ChatOpenAI
                model = ChatOpenAI(model_name=details["model"])
                _ = model("ping")
                print("✅")
            except Exception:
                print("❌")
    except Exception as e:
        logging.exception("Diagnostics check failed.")
        print("Diagnostics error. Check logs.")

def main():
    # Load global settings from agents.yaml
    config_path = os.path.expanduser("~/ooba-hybrid/config/agents.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    global_settings = config.get("global_settings", {})

    # Pass to orchestrator
    orchestrator = Orchestrator(global_settings)

    while True:
        try:
            query = input("Query (or 'exit' to quit, 'diagnose' to run diagnostics): ").strip()

            if query.lower() == "exit":
                break

            if query.lower() == "diagnose":
                run_diagnostics()
                continue

            result = orchestrator.process_query(query)

            print("\n[Results]")
            for item in result:
                print(f"- {item}")

        except Exception as e:
            logging.exception("Fatal error in orchestrator loop")
            print("⚠️ An error occurred. See logs for details.")

if __name__ == "__main__":
    main()
