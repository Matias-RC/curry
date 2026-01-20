#!/bin/bash

#SBATCH --job-name=maple_exp_4
#SBATCH --output=logs/exp_4.out
#SBATCH --error=logs/exp_4.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --partition=ialab
#SBATCH --nodelist=scylla
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=felipeurrutia@gmail.com
#SBATCH --time=0-23:59:59

pwd; date; hostname

echo "Starting experiment 4"

# Run in parallel on 3 GPUs (GPUT:1 and GPU:2, using CUDA_VISIBLE_DEVICES) train.py --num_supervision_steps 1 and 2
CUDA_VISIBLE_DEVICES=1 python3 train.py \
    --num_supervision_steps_train 1 \
    --num_supervision_steps_eval 10 \
    --batch_size_train 16 \
    --learning_rate 0.0001 \
    --num_epochs 100 \
    --where_to_save s3://rl6-reinforcement-learning-01 \
    --output_dir behavioral-cloning/exp_40 \
    --metrics_save_epoch_rate 5 \
    --max_num_levels 16384 \
    --train_fraction 0.9 \
    --order_by 'shortest_first' \
    --difficulty 'unfiltered' \
    --split 'train' \
    --memory_size 8 \
    --verbose 1 &

CUDA_VISIBLE_DEVICES=2 python3 train.py \
    --num_supervision_steps_train 2 \
    --num_supervision_steps_eval 10 \
    --batch_size_train 16 \
    --learning_rate 0.0001 \
    --num_epochs 100 \
    --where_to_save s3://rl6-reinforcement-learning-01 \
    --output_dir behavioral-cloning/exp_41 \
    --metrics_save_epoch_rate 5 \
    --max_num_levels 16384 \
    --train_fraction 0.9 \
    --order_by 'shortest_first' \
    --difficulty 'unfiltered' \
    --split 'train' \
    --memory_size 8 \
    --verbose 1 &

CUDA_VISIBLE_DEVICES=3 python3 train.py \
    --num_supervision_steps_train 1 \
    --num_supervision_steps_eval 10 \
    --batch_size_train 16 \
    --learning_rate 0.0001 \
    --num_epochs 100 \
    --where_to_save s3://rl6-reinforcement-learning-01 \
    --output_dir behavioral-cloning/exp_42 \
    --metrics_save_epoch_rate 5 \
    --max_num_levels 16384 \
    --train_fraction 0.9 \
    --order_by 'shortest_first' \
    --difficulty 'unfiltered' \
    --split 'train' \
    --verbose 1 &

CUDA_VISIBLE_DEVICES=4 python3 train.py \
    --num_supervision_steps_train 2 \
    --num_supervision_steps_eval 10 \
    --batch_size_train 16 \
    --learning_rate 0.0001 \
    --num_epochs 100 \
    --where_to_save s3://rl6-reinforcement-learning-01 \
    --output_dir behavioral-cloning/exp_43 \
    --metrics_save_epoch_rate 5 \
    --max_num_levels 16384 \
    --train_fraction 0.9 \
    --order_by 'shortest_first' \
    --difficulty 'unfiltered' \
    --split 'train' \
    --verbose 1 

date
echo "Experiment 4 completed"