#!/bin/bash
# Run the whole sweep on this machine, N at a time.  The cluster path is
# submit.slurm; this exists so the same configs can be smoke-tested locally.
#
#     bash production/temperature-hc/run_all_local.sh 8
set -euo pipefail
JOBS="${1:-4}"
seq 0 24 | xargs -P "$JOBS" -I{} \
    uv run python production/temperature-hc/run_one.py {}
