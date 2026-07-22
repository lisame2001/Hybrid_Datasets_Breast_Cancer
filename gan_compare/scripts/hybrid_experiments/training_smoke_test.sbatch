#!/bin/bash
#SBATCH --job-name=hybrid_train_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/training_smoke_test_%j.out
#SBATCH --error=slurm_logs/training_smoke_test_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Python: $(which python)"

python -m gan_compare.scripts.train_test_classifier \
  --config_path gan_compare/configs/swin/hybrid_experiments/smoke_test_real_only.yaml \
  --dataset_path dataset16062024 \
  --train_sampling_proportion 0.05 \
  --device cuda \
  --seed 42

echo "Ende: $(date)"
