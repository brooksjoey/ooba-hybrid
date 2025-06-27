#!/usr/bin/env python3
import os
import sys
import logging
import yaml
import traceback
from orchestrator.agent_orchestrator import Orchestrator
import asyncio
from typing import Dict, Any
import signal
import psutil
from datetime import datetime
from prometheus_client import Counter, Histogram, start_http_server

# Configure logging with system diagnostics
LOG_PATH = os.path.expanduser("~/ooba-hybrid/logs/orchestrator.log")
AGENTS_CONFIG = os.path.expanduser("~/ooba-hybrid/config/agents.yaml")
ROUTING_CONFIG = os.path.expanduser("~/ooba-hybrid/config/routing.yaml")

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s %(process)s"
)
logger = logging.getLogger(__name__)

# Prometheus metrics
QUERY_COUNT = Counter('orchestrator_query_count', 'Total queries processed')
ERROR_COUNT = Counter('orchestrator_error_count', 'Total errors encountered')
LATENCY_HISTOGRAM = Histogram('orchestrator_latency_seconds', 'Latency of query processing')

def load_config(path: str) -> Dict[str, Any]:
    """Load configuration with validation."""
    if not os.path.exists(path):
        logger.critical(f"Missing config: {path}")
        print(f"🔴 Critical: Missing config file {path}")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)

def diagnostics():
    """Advanced system and model diagnostics."""
    print("\n[Diagnostics]")
    try:
        routing = load_config(ROUTING_CONFIG)
        from langchain_openai import ChatOpenAI
        for name, entry in routing.get("fallbacks", {}).items():
            model_name = entry.get("model")
            print(f"- {name}: ", end="")
            try:
                client = ChatOpenAI(model=model_name)
                response = client.invoke("ping")
                print(f"✅ (Latency: {client.get_latency(response):.2f}s)")
                logger.info(f"Model {name} responsive")
            except Exception as e:
                print("❌")
                logger.error(f"Model {name} failed: {e}")
        print(f"- System Load: {psutil.cpu_percent()}% CPU, {psutil.virtual_memory().percent}% RAM")
    except Exception:
        logger.exception("Diagnostics failed")
        print("⚠️ Diagnostics error. See log.")

async def safe_input(prompt_text: str) -> str:
    """Safe input handling with signal interruption."""
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, lambda: input(prompt_text).strip())
    except (EOFError, KeyboardInterrupt):
        print("\n🛑 Interrupted or EOF. Terminating.")
        logger.info("REPL interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("Input failure")
        print("⚠️ Input error. See logs.")
        return ""

async def run_repl(orchestrator: Orchestrator):
    """Resilient REPL with distributed task execution."""
    print(f"[+] Orchestrator launched at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CDT")
    QUERY_COUNT.inc()

    while True:
        query = await safe_input("Query: ")
        if not query:
            continue

        if query.lower() == "exit":
            break
        elif query.lower() == "diagnose":
            diagnostics()
            continue

        start_time = datetime.now()
        try:
            result = await orchestrator.process_query(query)
            QUERY_COUNT.inc()
            latency = (datetime.now() - start_time).total_seconds()
            LATENCY_HISTOGRAM.observe(latency)
            print("\n[Results]")
            for line in result:
                print(f"- {line}")
            logger.info(f"Processed query '{query}' in {latency:.2f}s")
        except Exception as e:
            ERROR_COUNT.inc()
            logger.exception(f"Unhandled error in process_query: {e}")
            print(f"🔴 Fatal error: {e}. Check logs for details.")

def signal_handler(sig: int, frame: Any):
    """Graceful shutdown on signal."""
    logger.info(f"Received signal {sig}, shutting down...")
    sys.exit(0)

def main():
    """Main entry point with health monitoring and distributed setup."""
    print("[+] Starting orchestrator...")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        config = load_config(AGENTS_CONFIG)
        orchestrator = Orchestrator(config.get("global_settings", {}))
        start_http_server(8001)  # Expose metrics on port 8001
        asyncio.run(run_repl(orchestrator))
    except Exception:
        logger.critical("Startup failure:\n" + traceback.format_exc())
        print("🔴 Critical orchestrator error. See logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()