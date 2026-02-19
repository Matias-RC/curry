#!/bin/bash
#SBATCH --job-name=soko_multi
#SBATCH --partition=all
#SBATCH --nodelist=scylla
#SBATCH --gres=gpu:4      
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10       
#SBATCH --mem=40G                 
#SBATCH --time=24:00:00
#SBATCH --output=logs/multi_%j.out
#SBATCH --error=logs/multi_%j.err


source /home/matias_rodriguez/miniconda3/etc/profile.d/conda.sh
conda activate /mnt-homes/dccnas/CristianBuc/matias_rodriguez/curry/.venv

mkdir -p models_vec tensorboard logs


CUDA_VISIBLE_DEVICES=0 python train5.py --exp_name base_32 --heads 4 --layers 3 &
CUDA_VISIBLE_DEVICES=1 python train5.py --exp_name deep_att --layers 6 --hidden 128 &
CUDA_VISIBLE_DEVICES=2 python train5.py --exp_name fast_lr --lr 7e-4 --batch_size 128 &
CUDA_VISIBLE_DEVICES=3 python train5.py --exp_name wide_att --heads 8 --hidden 256 &

wait