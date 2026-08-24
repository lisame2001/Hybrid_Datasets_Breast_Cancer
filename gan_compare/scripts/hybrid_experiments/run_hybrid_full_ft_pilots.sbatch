#!/bin/bash
#SBATCH --job-name=hybrid_ft_pilots
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm_logs/hybrid_ft_pilot_%A_%a.out
#SBATCH --error=slurm_logs/hybrid_ft_pilot_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hybrid

EXPERIMENTS=(
  hybrid_050
  hybrid_100
)

EXPERIMENT="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
CONFIG="gan_compare/configs/swin/hybrid_experiments/full_ft_pilots/${EXPERIMENT}_full_ft_seed42.yaml"

echo "Experiment: $EXPERIMENT"
echo "Config: $CONFIG"
echo "Python: $(which python)"
echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi

python -m gan_compare.scripts.train_test_classifier \
  --config_path "$CONFIG" \
  --dataset_path dataset16062024 \
  --device cuda \
  --seed 42

echo "Ende: $(date)"
