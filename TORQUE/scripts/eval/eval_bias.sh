#!/bin/bash

task="torque"
model="roberta-large"
# suffix="_ht_bias.json"
# suffix="_dependency_bias_align.json"
model_dir="output/end_to_end_model_roberta-large_batch_6_lr_1e-5_epochs_10_seed_7_1.0"

data_dir="data/dataset_bias_new/"

suffix="_erp_bias.json"
CUDA_VISIBLE_DEVICES=1 python code/eval_end_to_end.py \
--task_name ${task} \
--do_lower_case \
--model ${model} \
--file_suffix ${suffix} \
--data_dir ${data_dir} \
--model_dir ${model_dir}/  \
--max_seq_length 178 \
--eval_batch_size 12

suffix="_warmup_answer_bias.json"
CUDA_VISIBLE_DEVICES=4 python code/eval_end_to_end.py \
--task_name ${task} \
--do_lower_case \
--model ${model} \
--file_suffix ${suffix} \
--data_dir ${data} \
--model_dir ${model_dir}/  \
--max_seq_length 178 \
--eval_batch_size 12

suffix="_narrative_bias.json"
CUDA_VISIBLE_DEVICES=2 python code/eval_end_to_end.py \
--task_name ${task} \
--do_lower_case \
--model ${model} \
--file_suffix ${suffix} \
--data_dir ${data} \
--model_dir ${model_dir}/  \
--max_seq_length 178 \
--eval_batch_size 12

suffix="_tense_bias.json"
CUDA_VISIBLE_DEVICES=2 python code/eval_end_to_end.py \
--task_name ${task} \
--do_lower_case \
--model ${model} \
--file_suffix ${suffix} \
--data_dir ${data} \
--model_dir ${model_dir}/  \
--max_seq_length 178 \
--eval_batch_size 12


suffix="_warmup_tense_bias.json"
CUDA_VISIBLE_DEVICES=4 python code/eval_end_to_end.py \
--task_name ${task} \
--do_lower_case \
--model ${model} \
--file_suffix ${suffix} \
--data_dir ${data} \
--model_dir ${model_dir}/  \
--max_seq_length 178 \
--eval_batch_size 12

suffix="_dependency_bias.json"
CUDA_VISIBLE_DEVICES=4 python code/eval_end_to_end.py \
--task_name ${task} \
--do_lower_case \
--model ${model} \
--file_suffix ${suffix} \
--data_dir ${data} \
--model_dir ${model_dir}/  \
--max_seq_length 178 \
--eval_batch_size 12