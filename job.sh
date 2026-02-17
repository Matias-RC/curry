#!/bin/bash
#SBATCH --job-name=sokoban_rl
#SBATCH --partition=all
#SBATCH --nodelist=scylla
#SBATCH --gres=gpu:4      
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10  
#SBATCH --mem=32G                 
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-user=matias.rodriguez@cenia.cl
#SBATCH --mail-type=BEGIN,END,FAIL

# Usamos la ruta absoluta que sale en tu prompt para evitar errores
source /home/matias_rodriguez/miniconda3/etc/profile.d/conda.sh
conda activate /mnt-homes/dccnas/CristianBuc/matias_rodriguez/curry/.venv

mkdir -p models_vec tensorboard logs

python train.py --exp_name base_32 --heads 4 --layers 3 &
python train.py --exp_name deep_att --layers 6 --hidden 128 &
python train.py --exp_name fast_lr --lr 7e-4 --batch_size 128 &
python train.py --exp_name wide_att --heads 8 --hidden 256 &