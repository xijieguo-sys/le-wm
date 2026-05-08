#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 10:00:00
#SBATCH -J hwm_eval
#SBATCH --output=slurm_logs/hwm_eval_%j.out
#SBATCH --error=slurm_logs/hwm_eval_%j.err

# Submit with:
#   sbatch slurm_eval_hwm.sh policy_high=<run-dir>/hwm_epoch_<N>
#
# Optional Hydra overrides:
#   sbatch slurm_eval_hwm.sh policy_high=hwm_run1/hwm_epoch_100 \
#       eval.goal_offset_steps=50 eval.eval_budget=100 \
#       plan_config.horizon=10 plan_config.receding_horizon=10
#
# Override the low-level checkpoint (default lewm_epoch_10):
#   sbatch slurm_eval_hwm.sh policy_high=... policy=my_lewm_run/lewm_epoch_50

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

CONFIG_NAME="${CONFIG_NAME:-pusht_hwm}"

SLURM_JOB_NAME=bash "${VENV_PYTHON}" eval.py --config-name "${CONFIG_NAME}" \
    policy="${DEFAULT_POLICY}" \
    "$@"
