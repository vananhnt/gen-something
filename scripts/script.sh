#!/bin/bash

#python run.py --do_train --load_model_path ./model/checkpoint-best-ppl/pytorch_model.bin

lr=5e-5
batch_size=64
beam_size=10
source_length=256
target_length=128
data_dir=../data/code2nl/CodeSearchNet
output_dir=model
eval_steps=100 #400 for ruby, 600 for javascript, 1000 for others
train_steps=100 #20000 for ruby, 30000 for javascript, 50000 for others
pretrained_model=microsoft/codebert-base #Roberta: roberta-base

python run.py --do_train --model_type roberta --model_name_or_path $pretrained_model --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --train_steps $train_steps --eval_steps $eval_steps 