#!/bin/bash
#SBATCH --job-name=hybrid_250_ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:30:00
#SBATCH --array=0-4
#SBATCH --output=slurm_logs/hybrid_250_ft_%A_%a.out
#SBATCH --error=slurm_logs/hybrid_250_ft_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hybrid

SEEDS=(42 43 44 45 46)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
EXPERIMENT="hybrid_250"

CONFIG="gan_compare/configs/swin/hybrid_experiments/full_ft/${EXPERIMENT}_seed${SEED}.yaml"

echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Experiment: $EXPERIMENT"
echo "Seed: $SEED"
echo "Config: $CONFIG"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Start: $(date)"

test -f "$CONFIG"

nvidia-smi

python -m gan_compare.scripts.train_test_classifier \
  --config_path "$CONFIG" \
  --dataset_path dataset16062024 \
  --device cuda \
  --seed "$SEED"

echo "Ende: $(date)"
