#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python vibecoding_controller.py "$@"
