#!/bin/bash
#SBATCH --job-name=hybrid_exp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-6
#SBATCH --output=slurm_logs/hybrid_%A_%a.out
#SBATCH --error=slurm_logs/hybrid_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hybrid

echo "Python: $(which python)"

EXPERIMENTS=(
  real_only
  synthetic_only_matched
  synthetic_only_maximum
  hybrid_025
  hybrid_050
  hybrid_100
  hybrid_200
)

EXPERIMENT="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
CONFIG="gan_compare/configs/swin/hybrid_experiments/${EXPERIMENT}.yaml"

echo "Experiment: $EXPERIMENT"
echo "Config: $CONFIG"
echo "Node: $(hostname)"
echo "Start: $(date)"

python -m gan_compare.scripts.train_test_classifier \
  --config_path "$CONFIG" \
  --dataset_path dataset16062024 \
  --device cuda \
  --seed 42

echo "Ende: $(date)"