# GPT-2 Baseline

## System Requirements

1. python 3.x
2. transformers
3. pytorch=1.6.0

## Instructions to Run the Baseline

A zip file with a pretrained model can be found in the following link: https://drive.google.com/file/d/1At9UJ9VOF1F2KCafLFYQqe_avoesFI2Y/view?usp=sharing. However, instructions are given below to train and fine-tune your own baseline generator.


### Input data format for fine-tuning gpt-2

Prepare your training data in the following format:

```
[CLS]your prompt text[SEP]the corresponding response<|endoftext|>
```

### 1. Fine-tune gpt-2 model
```
#!/bin/bash                                                                                                                                                                                                    
data_dir=/path/to/pretraining/dialogue/dataset
python3 run_lm_finetuning.py \
    --num_train_epochs=1.0 \
    --model_type=gpt2 \
    --model_name_or_path gpt2 \
    --do_train \
    --do_eval \
    --train_data_file ../data/train.txt \
    --eval_data_file ../data/dev.txt \
    --output_dir baseline_dstc10.2 \
    --line_by_line \
    --per_gpu_train_batch_size=4 \
    --per_gpu_eval_batch_size=4 \
    --overwrite_output_dir \
    --logging_steps=5000 \
    --save_steps=5000
```

### 2. To generate response based on prompt
```
#!/bin/bash                                                                                                                                                                                                    
PROMPT="[CLS]your prompt text[SEP]"

python3 run_generation.py \
    --model_type gpt2 \
    --model_name_or_path baseline_dstc10.2 \
    --prompt "${PROMPT}" \
    --num_return_sequences 3 \
    --length 200 \
    --temperature 1.0 \
    --stop_token "<|endoftext|>"
```
