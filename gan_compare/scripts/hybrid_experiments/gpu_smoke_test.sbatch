#!/bin/bash
#SBATCH --job-name=hybrid_gpu_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/gpu_smoke_test_%j.out
#SBATCH --error=slurm_logs/gpu_smoke_test_%j.err

source /home/student/s/sasschaefer/miniconda3/etc/profile.d/conda.sh
conda activate hybrid

echo "Node: $(hostname)"
echo "Python: $(command -v python)"
python --version
nvidia-smi

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch erkennt innerhalb des GPU-Jobs keine GPU.")

print("GPU:", torch.cuda.get_device_name(0))
PY
