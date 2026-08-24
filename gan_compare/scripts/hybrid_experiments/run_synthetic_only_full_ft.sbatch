#!/bin/bash
#SBATCH --job-name=synth_only_ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:30:00
#SBATCH --array=0-9
#SBATCH --output=slurm_logs/synth_only_ft_%A_%a.out
#SBATCH --error=slurm_logs/synth_only_ft_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate hybrid

TASKS=(
"synthetic_only_matched 42"
"synthetic_only_matched 43"
"synthetic_only_matched 44"
"synthetic_only_matched 45"
"synthetic_only_matched 46"
"synthetic_only_maximum 42"
"synthetic_only_maximum 43"
"synthetic_only_maximum 44"
"synthetic_only_maximum 45"
"synthetic_only_maximum 46"
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
