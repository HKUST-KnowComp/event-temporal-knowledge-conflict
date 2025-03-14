#!/bin/bash

task="torque"
model="roberta-large"
# suffix="_ht_bias.json"
# suffix="_dependency_bias_align.json"
device=3
for baseline_name in poe # poe_mixin_h1e-2 poe_mixin
do
model_dir="/home/zwanggy/large_files/2023/temp_rel_bias/${baseline_name}_baseline"

data_dir="data/"
suffix="_end2end_final.json"
CUDA_VISIBLE_DEVICES=${device} python code/eval_end_to_end.py \
--task_name ${task} \
--do_lower_case \
--model ${model} \
--file_suffix ${suffix} \
--data_dir ${data_dir} \
--model_dir ${model_dir}/  \
--max_seq_length 178 \
--eval_batch_size 12

data_dir="data/dataset_bias_new/"
suffix="_erp_bias.json"
CUDA_VISIBLE_DEVICES=${device} python code/eval_end_to_end.py \
--task_name ${task} \
--do_lower_case \
--model ${model} \
--file_suffix ${suffix} \
--data_dir ${data_dir} \
--model_dir ${model_dir}/  \
--max_seq_length 178 \
--eval_batch_size 12

suffix="_warmup_answer_bias.json"
CUDA_VISIBLE_DEVICES=${device} python code/eval_end_to_end.py \
--task_name ${task} \
--do_lower_case \
--model ${model} \
--file_suffix ${suffix} \
--data_dir ${data_dir} \
--model_dir ${model_dir}/  \
--max_seq_length 178 \
--eval_batch_size 12

suffix="_narrative_bias.json"
CUDA_VISIBLE_DEVICES=${device} python code/eval_end_to_end.py \
--task_name ${task} \
--do_lower_case \
--model ${model} \
--file_suffix ${suffix} \
--data_dir ${data_dir} \
--model_dir ${model_dir}/  \
--max_seq_length 178 \
--eval_batch_size 12

suffix="_tense_relation_bias.json"
CUDA_VISIBLE_DEVICES=${device} python code/eval_end_to_end.py \
--task_name ${task} \
--do_lower_case \
--model ${model} \
--file_suffix ${suffix} \
--data_dir ${data_dir} \
--model_dir ${model_dir}/  \
--max_seq_length 178 \
--eval_batch_size 12


suffix="_warmup_tense_bias.json"
CUDA_VISIBLE_DEVICES=${device} python code/eval_end_to_end.py \
--task_name ${task} \
--do_lower_case \
--model ${model} \
--file_suffix ${suffix} \
--data_dir ${data_dir} \
--model_dir ${model_dir}/  \
--max_seq_length 178 \
--eval_batch_size 12

suffix="_dependency_bias.json"
CUDA_VISIBLE_DEVICES=${device} python code/eval_end_to_end.py \
--task_name ${task} \
--do_lower_case \
--model ${model} \
--file_suffix ${suffix} \
--data_dir ${data_dir} \
--model_dir ${model_dir}/  \
--max_seq_length 178 \
--eval_batch_size 12
done