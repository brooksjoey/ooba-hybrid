#!/bin/bash
cd ~/ooba-hybrid/webui
source venv/bin/activate
python3 server.py --listen --listen-host 0.0.0.0 --listen-port 7860 --trust-remote-code
