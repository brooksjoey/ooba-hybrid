# 🧠 ooba-hybrid  

An orchestrated multi-agent memory stack with real-time routing, self-healing, and intelligent fallback. Designed for high-agency, low-friction workflows — optimized for terminal use, iPad ops, and voice control.  

---  

## 🗂 Project Structure  

```
~/ooba-hybrid/
├── config/                 # Environment config, API keys, routes
│   ├── apis.env
│   ├── routing.yaml
│   └── settings.json
├── logs/                  # Runtime logs
│   └── orchestrator.log
├── memory/                # Ingested and processed memory
│   ├── db/
│   ├── ingest/
│   └── schema.json
├── orchestrator/          # Agent core, router logic, chain memory
│   ├── agent_orchestrator.py
│   ├── agents.yaml
│   ├── dynamic_router.py
│   ├── memory_chain.py
│   ├── orchestrator.py
│   └── prompt_profiles/
│       ├── planner_prompt.txt
│       └── summarizer_prompt.txt
├── scripts/               # All automation scripts
│   ├── Bootstrap/
│   │   ├── init_stack.sh
│   │   ├── install_memory_stack.sh
│   │   └── install-deps.sh
│   ├── Startup-Reboot/
│   │   ├── run_orchestrator.sh
│   │   ├── run-ooba.sh
│   │   └── start-on-reboot.sh
│   └── Live-Ops/
│       └── ingest_context.py
├── webui/                 # venv lives here
├── .env                   # Local environment variables (if used)
├── requirements.txt       # Python dependencies
└── README.md              # You're reading it
```  

---  

## ⚙️ Setup Instructions  

### 1. Bootstrap the stack  
```bash
cd ~/ooba-hybrid
./scripts/Bootstrap/init_stack.sh
```

### 2. Install Python deps  
```bash
./scripts/Bootstrap/install_memory_stack.sh
```

### 3. Add API keys  
```bash
nano ~/ooba-hybrid/config/apis.env
```

Example:  
```
OPENAI_API_KEY=sk-your-openai-key
GROQ_API_KEY=gsk-your-groq-key
OPENROUTER_API_KEY=your-openrouter-key
```

### 4. Optional: Ingest memory  
```bash
echo "Discussed quarterly goals with investors." > ~/ooba-hybrid/memory/ingest/sample.txt
./scripts/Live-Ops/ingest_context.py
```

### 5. Run the orchestrator  
```bash
./scripts/Startup-Reboot/run_orchestrator.sh
```

---  

## 🧠 System Capabilities  

• **Agent Orchestration**  
Routes tasks to the best-fit LLM via agents.yaml. Supports OpenAI, Groq, and OpenRouter (modular).  

• **Memory-Enhanced Reasoning**  
Integrates with ChromaDB for long-term memory and context-aware tasks.  

• **Resilience + Logging**  
Retry logic via tenacity, detailed logs in logs/*.log.  

• **Headless and Scriptable**  
Built for command-line flow (Termius, iPad, VPS). No GPU needed.  

---  

## 🧪 Test It Out  

Once running, the orchestrator accepts plain-text prompts:  

```
Plan a client meeting  
Summarize investor goals  
exit
```

---  

## 🔮 Next Moves  

• Add New Agents - Extend orchestrator/agents.yaml with new personas (e.g., researcher, strategist)  
• WebUI Interface - Add a Flask/FastAPI frontend inside webui/  
• Auto Snapshotting - Use snapshot_ooba.sh to capture exact state after changes  
• Voice Mode (iOS) - Pipe voice into prompts using Groq's Whisper/FasterWhisper + iOS shortcut  

---  

## 🧊 Snapshot This Build  

```bash
bash ~/snapshot_ooba.sh
```  

Creates:  
`/root/ooba-hybrid-snapshot-YYYYMMDD-HHMM.tar.gz`  

---  

🧠 Built for Builders  

This system runs lean, automates hard, and makes you faster without fluff.  
You're not debugging a toy — you're commanding a machine.  

---  

Let me know if you want a `restore_ooba.sh` script that auto-rebuilds from the snapshot file. I can drop that next.
