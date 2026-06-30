#!/usr/bin/env bash

PWD="$(pwd)"

# Append /src to your current PYTHONPATH
# This makes all modules under /src available for imprting in
# standalone scripts
export PYTHONPATH="${PWD}:${PYTHONPATH}"