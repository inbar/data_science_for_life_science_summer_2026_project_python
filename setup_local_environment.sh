#!/usr/bin/env bash

# For this setup to take effect, run
# $ source setup_environment.sh


PWD="$(pwd)"
USER="$(whoami)"

# This makes all modules under /src available for imprting in standalone scripts
export PYTHONPATH="${PWD}:${PYTHONPATH}"

export PROJECT_HOME_ROOT="/Users/${USER}"

# Log level
# Options: DEBUG, INFO, WARN, CRITICAL
export LOG_LEVEL=DEBUG

conda activate data_science_in_life_sciences_project_2026_group_1

# Slurm
export SLURM_TIME_FORMAT=relative