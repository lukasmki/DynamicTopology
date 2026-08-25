#!/bin/bash
#SBATCH --job-name=h2o2-stoich
#SBATCH --array=0-24
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=4G
#SBATCH --output=production/stoichiometry-3000K/output/slurm-%a.out
#
# The only scheduler-specific file in this directory.  Everything it needs to
# know is the array index, so porting to another scheduler means rewriting this
# header and nothing else.
#
# One core per task: the force call is single-threaded, and the 25 runs are
# independent, so parallelism comes from the array rather than from within a run.
# --time is set generously -- at ~290 ms/step a 400k-step run is ~32 h -- and a
# job that hits the wall can be continued in place:
#
#     sbatch --array=<index> submit.slurm --restart
#
# Submit from the repository root:  sbatch production/stoichiometry-3000K/submit.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

srun uv run python production/stoichiometry-3000K/run_one.py \
    "${SLURM_ARRAY_TASK_ID}" "$@"
