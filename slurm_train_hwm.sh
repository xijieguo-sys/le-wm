#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J hwm_train
#SBATCH --output=slurm_logs/hwm_%j.out
#SBATCH --error=slurm_logs/hwm_%j.err

# Submit with:
#   sbatch slurm_train_hwm.sh
#
# Optional Hydra overrides can be appended on the command line, e.g.:
#   sbatch slurm_train_hwm.sh trainer.max_epochs=50 wandb.enabled=false

set -euo pipefail

# Project paths
PROJECT_DIR=/oscar/home/xguo84/final_project/repos/le-wm
VENV_PYTHON=/users/xguo84/final_project/repos/le-wm/.venv/bin/python
LOW_LEVEL_CKPT=/oscar/scratch/xguo84/stable-wm/lewm_epoch_10_object.ckpt

# Make sure log dir exists
mkdir -p "${PROJECT_DIR}/slurm_logs"

cd "${PROJECT_DIR}"

# SLURM_JOB_NAME=bash works around Lightning's strict --ntasks validation
# (lightning/fabric/plugins/environments/slurm.py:_validate_srun_variables).
# The override is *inside* the python invocation only; sbatch still names the
# job "hwm_train" for its own bookkeeping.


RUN_ID="hwm_${SLURM_JOB_ID}"

SLURM_JOB_NAME=bash "${VENV_PYTHON}" train_highlevel.py \
    low_level_ckpt="${LOW_LEVEL_CKPT}" \
    wandb.config.id="${RUN_ID}" \
    wandb.config.name="${RUN_ID}" \
    "$@"