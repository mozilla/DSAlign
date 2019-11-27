#!/usr/bin/env bash
approot="$(dirname "$(dirname "$(readlink -fm "$0")")")"
source "$approot/venv/bin/activate"
python "$approot/align/export.py" "$@"
stty sane
