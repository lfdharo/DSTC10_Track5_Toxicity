#!/bin/bash                                                                                                                                                                                                    
#SBATCH --job-name=finetune
#SBATCH -n 1
#SBATCH -p new
#SBATCH --gres=gpu:1
#SBATCH --output=finetune.log

python3 run_lm_finetuning.py \
    --num_train_epochs=1.0 \
    --model_type=gpt2 \
    --model_name_or_path gpt2 \
    --do_train \
    --do_eval \
    --train_data_file ../data/train.txt \
    --eval_data_file ../data/dev.txt \
    --output_dir gpt2_output \
    --line_by_line \
    --per_gpu_train_batch_size=4 \
    --per_gpu_eval_batch_size=4 \
    --overwrite_output_dir \
    --logging_steps=5000 \
    --save_steps=5000


