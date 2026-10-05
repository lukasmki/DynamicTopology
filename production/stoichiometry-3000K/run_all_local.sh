#!/bin/bash
# Run the whole sweep on this machine, N at a time.  The cluster path is
# submit.slurm; this exists so the same configs can be smoke-tested locally.
#
#     bash production/stoichiometry-3000K/run_all_local.sh 8
set -euo pipefail
# N runs on one machine: one BLAS thread each, or every run starts a thread per
# CPU and they fight over the cores -- four 192-atom water runs sharing 32 cores
# took 4.2 s per force call each unpinned and 0.11 pinned.  `scripts/nvt.py` and `npt.py` default to this too; it is stated here so
# the sweep does not depend on that.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
JOBS="${1:-4}"
seq 0 24 | xargs -P "$JOBS" -I{} \
    uv run python production/stoichiometry-3000K/run_one.py {}
