#!/usr/bin/env bash
export APP_ROOT=`python -c "import os, sys; print os.path.dirname(os.path.dirname(os.path.realpath(\"$0\")))"`
source "$APP_ROOT/venv/bin/activate"
python "$APP_ROOT/align/meta.py" "$@"
stty sane
