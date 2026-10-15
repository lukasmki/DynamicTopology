#!/bin/bash
#SBATCH --job-name=h2o2-temp
#SBATCH --account=m2834
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --array=0-24
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --output=production/temperature-hc/output/slurm-%a.out
#
# The only scheduler-specific file in this directory.  Everything it needs to
# know is the array index, so porting to another scheduler means rewriting this
# header and nothing else.
#
# One core per task: the force call is single-threaded, and the 25 runs are
# independent, so parallelism comes from the array rather than from within a run.
#
# A run is 2e5 steps on the same 198-atom box at every temperature; see
# README.md for the measured cost.
#
#     sbatch --array=<index> production/temperature-hc/submit.slurm --restart
#
# continues a run from its own trajectory for the steps it has left -- the
# path for a job the wall cut short, which at ~0.75 s/step is a real
# possibility: see README.md.  Restarting a finished run does nothing.
#
# Submit from the repository root:  sbatch production/temperature-hc/submit.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

source .venv/bin/activate
python production/temperature-hc/run_one.py \
    "${SLURM_ARRAY_TASK_ID}" "$@"
