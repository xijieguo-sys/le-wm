#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J lewm_eval
#SBATCH --output=slurm_logs/lewm_eval_%j.out
#SBATCH --error=slurm_logs/lewm_eval_%j.err

# Flat-LeWM Push-T eval (single-level CEM, no hierarchy).
# Use this as the regression baseline before / alongside hierarchical eval.
#
# Submit with:
#   sbatch slurm_eval_lewm.sh
#
# Optional Hydra overrides (e.g. different goal offset / checkpoint):
#   sbatch slurm_eval_lewm.sh policy=my_run/lewm_epoch_50 eval.goal_offset_steps=50

set -euo pipefail

# Project paths
PROJECT_DIR=/oscar/home/xguo84/final_project/repos/le-wm
VENV_PYTHON=/users/xguo84/final_project/repos/le-wm/.venv/bin/python

# Default low-level checkpoint -- overridable by passing policy=<...> as a
# trailing arg (Hydra uses the last value when a key is set twice).
DEFAULT_POLICY=lewm_epoch_10

# Make sure log dir exists
mkdir -p "${PROJECT_DIR}/slurm_logs"

cd "${PROJECT_DIR}"

# SLURM_JOB_NAME=bash works around Lightning's strict --ntasks validation
# (lightning/fabric/plugins/environments/slurm.py:_validate_srun_variables).

SLURM_JOB_NAME=bash "${VENV_PYTHON}" eval.py --config-name pusht \
    policy="${DEFAULT_POLICY}" \
    "$@"
