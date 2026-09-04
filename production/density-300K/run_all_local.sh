#!/bin/bash
# Run the whole sweep on this machine, N at a time.  The cluster path is
# submit.slurm; this exists so the same configs can be smoke-tested locally.
#
#     bash production/density-300K/run_all_local.sh 3
#
# Three runs at ~26 h each: this is an overnight-and-then-some job locally, and
# `run_one.py <index> --steps 200` is the thing to run first.
set -euo pipefail
JOBS="${1:-3}"
seq 0 2 | xargs -P "$JOBS" -I{} \
    uv run python production/density-300K/run_one.py {}
