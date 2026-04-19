#!/bin/bash
#SBATCH --job-name=pat-urban
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
cd /home/muqsitamir/repos/Part-Aware-Transformer

mkdir -p logs
mkdir -p models tb_log

source /home/muqsitamir/miniconda3/etc/profile.d/conda.sh
conda activate pat

python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

python train.py --config_file "config/UrbanElementsReID_train.yml"
