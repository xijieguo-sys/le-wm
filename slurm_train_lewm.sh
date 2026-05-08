#!/bin/bash

# ============================================================
# LeWM training job on Oscar (Brown CCV)
#
# Usage:
#   sbatch train.sh                 # defaults to data=pusht
#   sbatch train.sh pusht           # same as above
#   sbatch train.sh cube            # OGB cube (needs lewm-cube/*.h5 or STABLEWM_HOME layout below)
#   sbatch train.sh pusht trainer.max_epochs=5 loader.batch_size=64
#                                   # extra args are forwarded to Hydra
#
# Monitor:
#   myq                             # job status
#   tail -f slurm-<jobid>.out       # stdout
#   tail -f slurm-<jobid>.err       # stderr
# ============================================================

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1      # must equal number of GPUs (Lightning/DDP)
#SBATCH --cpus-per-task=8        # DataLoader workers + room to spare
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J lewm_train
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# First positional arg is the Hydra data config (default: pusht).
# Any remaining args are forwarded to Hydra as overrides.
DATA=${1:-pusht}
shift || true
EXTRA_ARGS="$@"

echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Data:      $DATA"
echo "Extra:     $EXTRA_ARGS"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "============================================"

cd "$SLURM_SUBMIT_DIR"

# --- environment -------------------------------------------------
# Activate your venv (change the path if yours differs).
source /oscar/scratch/$USER/.venv/bin/activate

# Fail fast if the environment is not set up correctly.
# NOTE: this venv is managed by `uv` (no pip). Use `uv pip ...` for installs.
echo "python: $(which python)  ($(python -V 2>&1))"
python -c "import sys; print('sys.executable =', sys.executable)"
python -c "import datasets, os; print('datasets', datasets.__version__, 'at', os.path.dirname(datasets.__file__))"
python -c "from datasets import config as _; print('datasets.config OK')" \
    || { echo 'ERROR: HuggingFace datasets broken. Run (inside the venv):'; \
         echo '  uv pip uninstall datasets'; \
         echo '  uv pip install --reinstall "datasets>=2.18.0"'; \
         exit 1; }
python -c "import stable_pretraining; import torch; \
    print('stable_pretraining OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1

# $STABLEWM_HOME is where train.py looks for datasets (<name>.h5) and where
# checkpoints get written. Dataset files must live at $STABLEWM_HOME/<name>.h5
# (see README.md). Either put/symlink the .h5 into this dir, or point
# STABLEWM_HOME at the dir that already contains it.
#
# Respect an externally provided value (export STABLEWM_HOME=... before sbatch,
# or `sbatch --export=ALL,STABLEWM_HOME=...`); only fall back to a default.
: "${STABLEWM_HOME:=/oscar/scratch/$USER/stablewm_home}"
export STABLEWM_HOME
mkdir -p "$STABLEWM_HOME"
echo "STABLEWM_HOME = $STABLEWM_HOME"

# Sanity check: only pusht training needs pusht_expert_train.h5 (other data= configs
# use their own HDF5 names; see config/train/data/*.yaml).
if [ "$DATA" = "pusht" ]; then
    DATA_FILE="$STABLEWM_HOME/pusht_expert_train.h5"
    REPO_PUSHT="$SLURM_SUBMIT_DIR/lewm-pusht/pusht_expert_train.h5"
    if [ ! -e "$DATA_FILE" ] && [ -e "$REPO_PUSHT" ]; then
        echo "Linking pusht HDF5 from repo into STABLEWM_HOME..."
        ln -sf "$REPO_PUSHT" "$DATA_FILE"
    fi
    if [ ! -e "$DATA_FILE" ]; then
        echo "ERROR: missing $DATA_FILE"
        echo "Fix by symlinking or moving it, e.g.:"
        echo "  ln -sf /oscar/scratch/$USER/le-wm-me/lewm-pusht/pusht_expert_train.h5 \\"
        echo "         $DATA_FILE"
        exit 1
    fi
    echo "data: $DATA_FILE ($(du -h "$DATA_FILE" | cut -f1))"
fi

# Cube / OGB: HDF5Dataset resolves name ogbench/cube_single_expert to
#   $STABLEWM_HOME/ogbench/cube_single_expert.h5
if [ "$DATA" = "cube" ] || [ "$DATA" = "ogb" ]; then
    CUBE_DST="$STABLEWM_HOME/ogbench/cube_single_expert.h5"
    mkdir -p "$STABLEWM_HOME/ogbench"
    if [ ! -e "$CUBE_DST" ]; then
        REPO_CUBE_DIR="$SLURM_SUBMIT_DIR/lewm-cube"
        for rel in "ogbench/cube_single_expert.h5" "cube_single_expert.h5"; do
            src="$REPO_CUBE_DIR/$rel"
            if [ -e "$src" ]; then
                abs_src="$(cd "$(dirname "$src")" && pwd)/$(basename "$src")"
                echo "Linking cube HDF5 into STABLEWM_HOME (expected path for data=ogb)..."
                ln -sf "$abs_src" "$CUBE_DST"
                break
            fi
        done
    fi
    if [ ! -e "$CUBE_DST" ]; then
        echo "ERROR: missing $CUBE_DST"
        echo "stable-worldmodel loads dataset.name ogbench/cube_single_expert as that path."
        echo "Unpack under your repo, e.g.:"
        echo "  cd $SLURM_SUBMIT_DIR/lewm-cube && tar -xvf cube_single_expert.tar"
        echo "  # or: tar --zstd -xvf cube_single_expert.tar.zst"
        echo "Then either keep the .h5 as lewm-cube/cube_single_expert.h5 or lewm-cube/ogbench/cube_single_expert.h5"
        echo "and re-submit, or symlink manually to $CUBE_DST"
        exit 1
    fi
    echo "data: $CUBE_DST ($(du -h "$CUBE_DST" | cut -f1))"
fi

# DataLoader stability.
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# --- wandb ---
# The default config in config/train/lewm.yaml uses entity=lewm, project=lewm
# (the upstream authors' team) which your account cannot write to.
# Pick ONE of the options below:
#   (a) Set WANDB_MODE=offline to log locally only (default here: safe & no 401).
#   (b) Change entity/project in config/train/lewm.yaml to your own team/project,
#       then `wandb login` once and comment the line below.
#   (c) Override per-run on sbatch:
#         sbatch train.sh pusht wandb.config.entity=YOUR wandb.config.project=YOUR
export WANDB_MODE=${WANDB_MODE:-online}
echo "WANDB_MODE = $WANDB_MODE"

# --- train -------------------------------------------------------
srun python train.py data="$DATA" $EXTRA_ARGS

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"