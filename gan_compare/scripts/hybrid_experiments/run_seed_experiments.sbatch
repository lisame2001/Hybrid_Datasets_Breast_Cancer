#!/bin/bash
#SBATCH --job-name=hybrid_seeds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=slurm_logs/seeds_%A_%a.out
#SBATCH --error=slurm_logs/seeds_%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

EXPERIMENTS=(
  real_only
  hybrid_050
  hybrid_100
)

SEEDS=(
  43
  44
  45
  46
)

NUM_SEEDS=${#SEEDS[@]}
EXPERIMENT_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

EXPERIMENT="${EXPERIMENTS[$EXPERIMENT_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

SOURCE_CONFIG="gan_compare/configs/swin/hybrid_experiments/${EXPERIMENT}.yaml"
CONFIG_DIR="slurm_logs/seed_configs"
CONFIG="${CONFIG_DIR}/${EXPERIMENT}_seed${SEED}.yaml"

LOGFILE="extension/hybrid_experiments/${EXPERIMENT}/seed_${SEED}/${EXPERIMENT}_seed${SEED}.txt"
MODEL_DIR="extension/hybrid_experiments/${EXPERIMENT}/seed_${SEED}/model_checkpoint/"

mkdir -p "$CONFIG_DIR"
mkdir -p "$MODEL_DIR"

sed \
  -e "s|^logfile_path:.*|logfile_path: ${LOGFILE}|" \
  -e "s|^output_model_dir:.*|output_model_dir: ${MODEL_DIR}|" \
  "$SOURCE_CONFIG" > "$CONFIG"

echo "Experiment: $EXPERIMENT"
echo "Seed: $SEED"
echo "Config: $CONFIG"
echo "Node: $(hostname)"
echo "Start: $(date)"

python -m gan_compare.scripts.train_test_classifier \
  --config_path "$CONFIG" \
  --dataset_path dataset16062024 \
  --device cuda \
  --seed "$SEED"

echo "Ende: $(date)"