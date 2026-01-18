#!/bin/bash

#SBATCH --job-name=maple_exp_1
#SBATCH --output=logs/exp_1.out
#SBATCH --error=logs/exp_1.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition=ialab
#SBATCH --nodelist=scylla
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=felipeurrutia65@gmail.com
#SBATCH --time=0-01:00:00

pwd; date; hostname

echo "Starting experiment 1"

# Run in parallel on 2 GPUs (GPUT:1 and GPU:2, using CUDA_VISIBLE_DEVICES) train.py --num_supervision_steps 1 and 2

CUDA_VISIBLE_DEVICES=3 python3 train.py \
    --num_supervision_steps 1 \
    --batch_size_train 8 \
    --learning_rate 0.00005 \
    --num_epochs 100 \
    --where_to_save s3://rl6-reinforcement-learning-01 \
    --output_dir behavioral-cloning/exp_10 \
    --metrics_save_epoch_rate 10 \
    --verbose 1 &

CUDA_VISIBLE_DEVICES=4 python3 train.py \
    --num_supervision_steps 2 \
    --batch_size_train 8 \
    --learning_rate 0.00005 \
    --num_epochs 100 \
    --where_to_save s3://rl6-reinforcement-learning-01 \
    --output_dir behavioral-cloning/exp_11 \
    --metrics_save_epoch_rate 10 \
    --verbose 1 

date
echo "Experiment 1 completed"