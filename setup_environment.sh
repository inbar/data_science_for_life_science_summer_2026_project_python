#!/usr/bin/env bash

# For this setup to take effect, run
# $ source setup_environment.sh


PWD="$(pwd)"

# Append /src to your current PYTHONPATH
# This makes all modules under /src available for imprting in
# standalone scripts
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Log level
# Options: DEBUG, INFO, WARN, CRITICAL
export LOG_LEVEL=INFO