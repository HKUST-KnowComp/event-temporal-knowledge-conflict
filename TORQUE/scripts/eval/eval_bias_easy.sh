#!/bin/bash

task="torque"
model="roberta-large"
suffix="_ht_bias_easy.json"
model_dir=

CUDA_VISIBLE_DEVICES=2 python code/eval_end_to_end.py \
--task_name ${task} \
--do_lower_case \
--model ${model} \
--file_suffix ${suffix} \
--data_dir data/ \
--model_dir ${model_dir}/  \
--max_seq_length 178 \
--eval_batch_size 12

suffix="_tail_bias_easy.json"
CUDA_VISIBLE_DEVICES=2 python code/eval_end_to_end.py \
--task_name ${task} \
--do_lower_case \
--model ${model} \
--file_suffix ${suffix} \
--data_dir data/ \
--model_dir ${model_dir}/  \
--max_seq_length 178 \
--eval_batch_size 12


suffix="_narrative_bias_easy.json"
CUDA_VISIBLE_DEVICES=2 python code/eval_end_to_end.py \
--task_name ${task} \
--do_lower_case \
--model ${model} \
--file_suffix ${suffix} \
--data_dir data/ \
--model_dir ${model_dir}/  \
--max_seq_length 178 \
--eval_batch_size 12

suffix="_tense_bias_easy.json"
CUDA_VISIBLE_DEVICES=2 python code/eval_end_to_end.py \
--task_name ${task} \
--do_lower_case \
--model ${model} \
--file_suffix ${suffix} \
--data_dir data/ \
--model_dir ${model_dir}/  \
--max_seq_length 178 \
--eval_batch_size 12