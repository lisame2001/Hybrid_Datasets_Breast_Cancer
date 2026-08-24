#!/bin/bash
#SBATCH --job-name=full_ft_paper
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:30:00
#SBATCH --array=0-39
#SBATCH --output=slurm_logs/full_ft_paper_%A_%a.out
#SBATCH --error=slurm_logs/full_ft_paper_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root.}"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hybrid}"
DATASET_PATH="${DATASET_PATH:-dataset16062024}"

test -f "$CONDA_SH"
source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"

CONDITIONS=(
  real_only
  hybrid_025
  hybrid_050
  hybrid_100
  hybrid_200
  hybrid_250
  synthetic_only_matched
  synthetic_only_maximum
)
SEEDS=(42 43 44 45 46)

CONDITION_INDEX=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
EXPERIMENT="${CONDITIONS[$CONDITION_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
CONFIG="gan_compare/configs/swin/hybrid_experiments/full_ft/${EXPERIMENT}_seed${SEED}.yaml"

test -f "$CONFIG"
test -d "$DATASET_PATH"

echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Experiment: $EXPERIMENT"
echo "Seed: $SEED"
echo "Config: $CONFIG"
echo "Dataset: $DATASET_PATH"
echo "Node: $(hostname)"
echo "Python: $(command -v python)"
echo "Start: $(date --iso-8601=seconds)"

nvidia-smi

python -m gan_compare.scripts.train_test_classifier \
  --config_path "$CONFIG" \
  --dataset_path "$DATASET_PATH" \
  --device cuda \
  --seed "$SEED"

echo "End: $(date --iso-8601=seconds)"
