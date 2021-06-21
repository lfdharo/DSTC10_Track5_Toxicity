#!/bin/bash

#SBATCH --job-name=run
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p new

PROMPT="[CLS]some stupid vlogger no one cares. I just like to say I don't care about stupid shit personally. Now someone tell me how I must actually care because I'm writing a comment about how I donut care and somehow saying you don't care about something means you actually do care, even though you just want to talk about how you don't care.[SEP]"

python3 run_generation.py \
    --model_type gpt2 \
    --model_name_or_path gpt2_output \
	--prompt "${PROMPT}" \
    --num_return_sequences 3 \
    --length 200 \
    --temperature 1.0 \
    --stop_token "<|endoftext|>"

    
