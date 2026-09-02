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
#
# **A run fits this wall.**  At 0.5 fs (see sweep.toml for how that step was
# measured) a 100 ps run is 2e5 steps at ~202 ms/step, about 11 hours, so a
# single job finishes inside 48 hours with four times the margin.
#
# It did not always.  At 0.05 fs the same trajectory was 2e6 steps and ~10 days,
# and every run needed roughly six continuations.  That path still works and is
# still worth knowing about for a job interrupted for some other reason:
#
#     sbatch --array=<index> submit.slurm --restart
#
# `run_one.py --restart` picks up from the run's own trajectory, so a
# continuation is idempotent and a chain can be resumed after any interruption.
#
# One caveat if you do restart: `--restart` re-reads the last frame from disk,
# and reading a frame goes through bond perception where MD does not.  The
# frames carry their connectivity so the chemistry is preserved, but this is the
# one place in the pipeline where the perception threshold is in the loop.
#
# Submit from the repository root:  sbatch production/stoichiometry-3000K/submit.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

srun uv run python production/stoichiometry-3000K/run_one.py \
    "${SLURM_ARRAY_TASK_ID}" "$@"
