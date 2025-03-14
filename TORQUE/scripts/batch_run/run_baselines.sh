#!/bin/bash
gpu=1
task="torque"
batchsizes=(6)
ratio=1.0
epoch=10
mlp_hid_size=64
model="roberta-large"
ga=1
prefix="end_to_end_model"
suffix="_end2end_final.json"
for s in "${batchsizes[@]}"
do
    learningrates=(1e-5)

    for l in "${learningrates[@]}"
    do
        seeds=( 7 24 123 )
        for seed in "${seeds[@]}"
        do
            CUDA_VISIBLE_DEVICES=$gpu python code/run_end_to_end.py \
            --task_name "${task}" \
            --save_model \
            --do_train \
            --do_eval \
            --do_lower_case \
            --mlp_hid_size ${mlp_hid_size} \
            --model ${model} \
            --data_dir data/ \
            --file_suffix ${suffix} \
            --train_ratio ${ratio} \
            --max_seq_length 178 \
            --train_batch_size ${s} \
            --learning_rate ${l} \
            --seed ${seed} \
            --num_train_epochs ${epoch}  \
            --gradient_accumulation_steps ${ga} \
            --output_dir xxx
        done
    done
done
