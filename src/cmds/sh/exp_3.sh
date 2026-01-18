#!/bin/bash

#SBATCH --job-name=maple_exp_3
#SBATCH --output=logs/exp_3.out
#SBATCH --error=logs/exp_3.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3
#SBATCH --partition=ialab
#SBATCH --nodelist=scylla
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=felipeurrutia65@gmail.com
#SBATCH --time=0-01:30:00

pwd; date; hostname

echo "Starting experiment 3"

# Run in parallel on 3 GPUs (GPUT:1 and GPU:2 and 3, using CUDA_VISIBLE_DEVICES) train.py --num_supervision_steps 1 and 2

CUDA_VISIBLE_DEVICES=3 python3 train.py \
    --num_supervision_steps 1 \
    --batch_size_train 16 \
    --learning_rate 0.0001 \
    --num_epochs 100 \
    --where_to_save s3://rl6-reinforcement-learning-01 \
    --output_dir behavioral-cloning/exp_30 \
    --metrics_save_epoch_rate 5 \
    --max_num_levels 16384 \
    --train_fraction 0.9 \
    --filter_by 'shortest_first' \
    --difficulty 'unfiltered' \
    --subset_name 'train' \
    --consolidator_num_encoder_layers 0 \
    --verbose 1 &

CUDA_VISIBLE_DEVICES=4 python3 train.py \
    --num_supervision_steps 2 \
    --batch_size_train 16 \
    --learning_rate 0.0001 \
    --num_epochs 100 \
    --where_to_save s3://rl6-reinforcement-learning-01 \
    --output_dir behavioral-cloning/exp_31 \
    --metrics_save_epoch_rate 5 \
    --max_num_levels 16384 \
    --train_fraction 0.9 \
    --filter_by 'shortest_first' \
    --difficulty 'unfiltered' \
    --subset_name 'train' \
    --consolidator_num_encoder_layers 0 \
    --verbose 1 &

CUDA_VISIBLE_DEVICES=5 python3 train.py \
    --num_supervision_steps 3 \
    --batch_size_train 16 \
    --learning_rate 0.0001 \
    --num_epochs 100 \
    --where_to_save s3://rl6-reinforcement-learning-01 \
    --output_dir behavioral-cloning/exp_32 \
    --metrics_save_epoch_rate 5 \
    --max_num_levels 16384 \
    --train_fraction 0.9 \
    --filter_by 'shortest_first' \
    --difficulty 'unfiltered' \
    --subset_name 'train' \
    --consolidator_num_encoder_layers 0 \
    --verbose 1 

date
echo "Experiment 2 completed"