#!/usr/bin/env bash

# For this setup to take effect, run
# $ source setup_environment.sh


PWD="$(pwd)"
USER="$(whoami)"

# Append /src to your current PYTHONPATH
# This makes all modules under /src available for imprting in
# standalone scripts
export PYTHONPATH="${PWD}:${PYTHONPATH}"

case "$HOSTNAME" in
    *allegro*)
        export PROJECT_HOME_ROOT="/data/scratch/${USER}"
        ;;
    *)
        export PROJECT_HOME_ROOT="/Users/${USER}"
        ;;
esac



# Log level
# Options: DEBUG, INFO, WARN, CRITICAL
export LOG_LEVEL=DEBUG


