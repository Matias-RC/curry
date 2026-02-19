#!/bin/bash
#SBATCH --job-name=soko_train
#SBATCH --partition=all
#SBATCH --nodelist=scylla
#SBATCH --gres=gpu:2080_ti:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-user=matias.rodriguez@cenia.cl
#SBATCH --mail-type=BEGIN,END,FAIL

source /home/matias_rodriguez/miniconda3/etc/profile.d/conda.sh
conda activate /home/matias_rodriguez/curry/.venv

mkdir -p models_vec tensorboard logs

python -u train5.py