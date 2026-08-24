#!/bin/bash
#SBATCH --job-name=fullft_pilot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/fullft_pilot_%j.out
#SBATCH --error=slurm_logs/fullft_pilot_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hybrid

CONFIG="gan_compare/configs/swin/hybrid_experiments/full_ft_pilots/real_only_full_ft_seed42.yaml"

echo "Full fine-tuning pilot"
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
