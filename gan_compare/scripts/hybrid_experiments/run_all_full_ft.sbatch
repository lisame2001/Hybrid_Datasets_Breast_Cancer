#!/bin/bash
#SBATCH --job-name=all_full_ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:30:00
#SBATCH --array=0-21
#SBATCH --output=slurm_logs/all_full_ft_%A_%a.out
#SBATCH --error=slurm_logs/all_full_ft_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hybrid

TASKS=(
  "real_only 43"
  "real_only 44"
  "real_only 45"
  "real_only 46"

  "hybrid_025 42"
  "hybrid_025 43"
  "hybrid_025 44"
  "hybrid_025 45"
  "hybrid_025 46"

  "hybrid_050 43"
  "hybrid_050 44"
  "hybrid_050 45"
  "hybrid_050 46"

  "hybrid_100 43"
  "hybrid_100 44"
  "hybrid_100 45"
  "hybrid_100 46"

  "hybrid_200 42"
  "hybrid_200 43"
  "hybrid_200 44"
  "hybrid_200 45"
  "hybrid_200 46"
)

read -r EXPERIMENT SEED <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

CONFIG="gan_compare/configs/swin/hybrid_experiments/full_ft/${EXPERIMENT}_seed${SEED}.yaml"

echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Experiment: $EXPERIMENT"
echo "Seed: $SEED"
echo "Config: $CONFIG"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Start: $(date)"

nvidia-smi

python -m gan_compare.scripts.train_test_classifier \
  --config_path "$CONFIG" \
  --dataset_path dataset16062024 \
  --device cuda \
  --seed "$SEED"

echo "Ende: $(date)"
