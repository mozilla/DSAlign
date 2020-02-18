#!/usr/bin/env bash
approot=$(cd "$(dirname "$(dirname "$0")")" && pwd)
source "$approot/venv/bin/activate"
python "$approot/align/catalog_tool.py" "$@"
stty sane
