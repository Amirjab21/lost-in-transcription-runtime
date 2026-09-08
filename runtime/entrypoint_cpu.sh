#!/usr/bin/env bash

set -euo pipefail

cd /code_execution
unzip -q ./submission/submission.zip -d ./src
python ./src/main.py

if [ ! -f ./submission/submission.csv ]; then
    echo "ERROR: main.py did not produce submission/submission.csv" >&2
    exit 1
fi
