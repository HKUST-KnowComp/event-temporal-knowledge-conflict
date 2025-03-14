#!/bin/bash

task="torque"
model="roberta-large"

model_dir=

suffix="_other.json"
CUDA_VISIBLE_DEVICES=2 python code/eval_end_to_end.py \
--task_name ${task} \
--do_lower_case \
--model ${model} \
--file_suffix ${suffix} \
--data_dir data/dataset_bias/ \
--model_dir ${model_dir}/  \
--max_seq_length 178 \
--eval_batch_size 12
