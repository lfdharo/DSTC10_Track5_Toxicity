#!/bin/bash

#SBATCH --job-name=run
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -w ttnusa3
#SBATCH -p new

/home/chen/anaconda3/envs/transformers-3.4/bin/python3 run_generation.py \
    --model_type gpt2 \
    --model_name_or_path gpt2_medium_output \
	--prompt dailydialog_train_context.txt\
    --num_return_sequences 1 \
    --length 128 \
    --temperature 1.0 \
    --stop_token "<|endoftext|>" > output.log 2>&1 & 

    
