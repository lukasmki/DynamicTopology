#!/bin/bash
#SBATCH --job-name=water-density
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=4G
#SBATCH --output=production/density-300K/output/slurm-%a.out
#
# The only scheduler-specific file in this directory.  Everything it needs to
# know is the array index, so porting to another scheduler means rewriting this
# header and nothing else.
#
# One core per task: the force call is single-threaded, and the three runs are
# independent, so parallelism comes from the array rather than from within a run.
#
# **A run fits this wall, but not with much room.**  Measured by
# `make_inputs.py` on the reference machine: 1.89 s per force call at 192 atoms,
# so 50000 steps (10000 NVT + 40000 NPT) is about 26 hours.  That is half the
# wall, which is the margin a slower node needs.  Re-run `make_inputs.py` on the
# cluster before submitting rather than trusting this number -- the force call
# here is an order of magnitude more expensive than the H2/O2 sweep's, because
# water's hydrogen-bond network puts far more pairs inside the bimolecular
# cutoff.
#
#     sbatch --array=<index> submit.slurm --restart
#
# `--restart` resumes whichever stage was in progress and does not redo a
# completed NVT stage.
#
# Submit from the repository root:  sbatch production/density-300K/submit.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

srun uv run python production/density-300K/run_one.py \
    "${SLURM_ARRAY_TASK_ID}" "$@"
