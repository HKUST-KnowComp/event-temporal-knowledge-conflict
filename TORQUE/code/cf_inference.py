import sys
sys.path.append('code')
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import pickle
import numpy as np
import json
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from collections import Counter
from transformers import *
from models import MultitaskClassifier, MultitaskClassifierRoberta
from utils import *
from optimization import *
from collections import defaultdict
import sys
from pathlib import Path
import wandb
device = torch.device("cuda")

baseline_path = "output/end_to_end_model_roberta-large_batch_6_lr_1e-5_epochs_10_seed_7_1.0/pytorch_model.bin"
event_only_path = "output/counterfactual/event_only/pytorch_model.bin"
nothing_path = "output/counterfactual/nothing/pytorch_model.bin"
eval_bias_data = load_data("data/dataset_bias_new/", "individual_dev_", "dependency_bias.json")


# lambda1, lambda2 = -0.8, 0.5
lambda1, lambda2 = -0.8, -0.1


model_state_dict = torch.load(baseline_path)

tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)
model = MultitaskClassifierRoberta.from_pretrained('roberta-large', 
                        state_dict=model_state_dict, mlp_hid=64)
model.to(device)

eval_data = load_data("data/", "individual_dev_", "end2end_final.json")
eval_data = {key:eval_data[key] for key in eval_bias_data}
tokenizer = AutoTokenizer.from_pretrained('roberta-large')
eval_features = convert_to_features_roberta(eval_data, tokenizer, max_length=178,
                evaluation=True, instance=False, end_to_end=True)
eval_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
eval_input_mask = torch.tensor(select_field(eval_features, 'mask_ids'), dtype=torch.long)
eval_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
eval_offsets = select_field(eval_features, 'offset')
eval_labels  = select_field(eval_features, 'label')
eval_key_indices = torch.tensor(list(range(len(eval_labels))), dtype=torch.long)

# collect unique question ids for EM calculation
question_ids = select_field(eval_features, 'question_id')
question_ids = [q for i, q in enumerate(question_ids) for x in range(len(eval_labels[i]))]
# collect unique question culster for EM-cluster calculation                          
question_cluster = select_field(eval_features, 'question_cluster')
question_cluster_size = select_field(eval_features, 'cluster_size')
eval_idv_answers = select_field(eval_features, 'individual_answers')
eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_key_indices)
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=32)

factual_probs = []
softmax = torch.nn.functional.softmax

for input_ids, input_masks, segment_ids, instance_indices in tqdm(eval_dataloader, desc="Evaluating"):
    input_ids = input_ids.to(device)
    input_masks = input_masks.to(device)
    segment_ids = segment_ids.to(device)
    
    offsets, labels, lengths = flatten_answers([(eval_labels[i], eval_offsets[i])
                                                for i in instance_indices.tolist()])
    labels = torch.tensor(labels).to(device)
    with torch.no_grad():
        logits, tmp_eval_loss = model(input_ids, offsets, lengths, token_type_ids=segment_ids,
                                      attention_mask=input_masks, labels=labels)

    logits = logits.detach().cpu()
    factual_prob = softmax(torch.tensor(logits), dim=1)
    factual_probs.append(factual_prob)

model_state_dict = torch.load(event_only_path)

tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)
model = MultitaskClassifierRoberta.from_pretrained('roberta-large', 
                        state_dict=model_state_dict, mlp_hid=64)
model.to(device)
eval_data = load_data("data/", "individual_dev_", "end2end_final_event_only.json")
eval_data = {key:eval_data[key] for key in eval_bias_data}
tokenizer = AutoTokenizer.from_pretrained('roberta-large')
eval_features = convert_to_features_roberta(eval_data, tokenizer, max_length=178,
                evaluation=True, instance=False, end_to_end=True)

eval_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
eval_input_mask = torch.tensor(select_field(eval_features, 'mask_ids'), dtype=torch.long)
eval_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
eval_offsets = select_field(eval_features, 'offset')
eval_labels  = select_field(eval_features, 'label')
eval_key_indices = torch.tensor(list(range(len(eval_labels))), dtype=torch.long)

# collect unique question ids for EM calculation
question_ids = select_field(eval_features, 'question_id')
question_ids = [q for i, q in enumerate(question_ids) for x in range(len(eval_labels[i]))]
# collect unique question culster for EM-cluster calculation                          
question_cluster = select_field(eval_features, 'question_cluster')
question_cluster_size = select_field(eval_features, 'cluster_size')
eval_idv_answers = select_field(eval_features, 'individual_answers')
eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_key_indices)
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=32)


event_only_probs = []
softmax = torch.nn.functional.softmax

for input_ids, input_masks, segment_ids, instance_indices in tqdm(eval_dataloader, desc="Evaluating"):
    input_ids = input_ids.to(device)
    input_masks = input_masks.to(device)
    segment_ids = segment_ids.to(device)
    
    offsets, labels, lengths = flatten_answers([(eval_labels[i], eval_offsets[i])
                                                for i in instance_indices.tolist()])
    labels = torch.tensor(labels).to(device)
    with torch.no_grad():
        logits, tmp_eval_loss = model(input_ids, offsets, lengths, token_type_ids=segment_ids,
                                      attention_mask=input_masks, labels=labels)

    logits = logits.detach().cpu()
    event_only_prob = softmax(torch.tensor(logits), dim=1)
    event_only_probs.append(event_only_prob)

model_state_dict = torch.load(nothing_path)

tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)
model = MultitaskClassifierRoberta.from_pretrained('roberta-large', 
                        state_dict=model_state_dict, mlp_hid=64)
model.to(device)
eval_data = load_data("data/", "individual_dev_", "end2end_final_nothing.json")
eval_data = {key:eval_data[key] for key in eval_bias_data}
tokenizer = AutoTokenizer.from_pretrained('roberta-large')
eval_features = convert_to_features_roberta(eval_data, tokenizer, max_length=178,
                evaluation=True, instance=False, end_to_end=True)

eval_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
eval_input_mask = torch.tensor(select_field(eval_features, 'mask_ids'), dtype=torch.long)
eval_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
eval_offsets = select_field(eval_features, 'offset')
eval_labels  = select_field(eval_features, 'label')
eval_key_indices = torch.tensor(list(range(len(eval_labels))), dtype=torch.long)

# collect unique question ids for EM calculation
question_ids = select_field(eval_features, 'question_id')
question_ids = [q for i, q in enumerate(question_ids) for x in range(len(eval_labels[i]))]
# collect unique question culster for EM-cluster calculation                          
question_cluster = select_field(eval_features, 'question_cluster')
question_cluster_size = select_field(eval_features, 'cluster_size')
eval_idv_answers = select_field(eval_features, 'individual_answers')
eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_key_indices)
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=32)



nothing_probs = []
softmax = torch.nn.functional.softmax

for input_ids, input_masks, segment_ids, instance_indices in tqdm(eval_dataloader, desc="Evaluating"):
    input_ids = input_ids.to(device)
    input_masks = input_masks.to(device)
    segment_ids = segment_ids.to(device)
    
    offsets, labels, lengths = flatten_answers([(eval_labels[i], eval_offsets[i])
                                                for i in instance_indices.tolist()])
    labels = torch.tensor(labels).to(device)
    with torch.no_grad():
        logits, tmp_eval_loss = model(input_ids, offsets, lengths, token_type_ids=segment_ids,
                                      attention_mask=input_masks, labels=labels)

    logits = logits.detach().cpu()
    nothing_prob = softmax(torch.tensor(logits), dim=1)
    nothing_probs.append(nothing_prob)

label_map = {0: 'Negative', 1: 'Positive'}
eval_loss, eval_accuracy, best_eval_f1, nb_eval_examples, nb_eval_steps = 0.0, 0.0, 0.0, 0, 0
all_preds, all_golds, max_f1s, macro_f1s = [], [], [], []
f1_dist = defaultdict(list)
em_counter = 0
em_cluster_agg, em_cluster_relaxed, f1_cluster_80 = {}, {}, {}

for i, (input_ids, input_masks, segment_ids, instance_indices) in enumerate(eval_dataloader):

    offsets, labels, lengths = flatten_answers([(eval_labels[i], eval_offsets[i])
                                                for i in instance_indices.tolist()])
    all_golds.extend(labels)
    labels = torch.tensor(labels).to(device)

    ###
#     logits = logits.detach().cpu().numpy()
    p_factual = factual_probs[i].cpu()

    batch_preds = np.argmax(p_factual.numpy(), axis=1)
    ###
    labels = labels.to('cpu').numpy()

    nb_eval_examples += labels.shape[0]
    nb_eval_steps += 1


    bi = 0
    for l, idx in enumerate(instance_indices):
        pred = [batch_preds[bi + li] for li in range(lengths[l])]
        pred_names = [label_map[p] for p in pred]
        gold_names = [label_map[labels[bi + li]] for li in range(lengths[l])]
        is_em = (pred_names == gold_names)
        if sum([labels[bi + li] for li in range(lengths[l])]) == 0 and sum(pred) == 0:
            macro_f1s.append(1.0)
        else:
            macro_f1s.append(cal_f1(pred_names, gold_names, {v:k for k,v in label_map.items()}))

        max_f1, instance_matched = 0, 0
        for gold in eval_idv_answers[idx]:
            label_names = [label_map[l] for l in gold]
            if pred_names == label_names: instance_matched = 1
            if sum(gold) == 0 and sum(pred) == 0:
                f1 = 1.0
            else:
                f1 = cal_f1(pred_names, label_names, {v:k for k,v in label_map.items()})
            if f1 >= max_f1:
                max_f1 = f1
                key = len(gold)

        if question_cluster_size[idx] > 1:
            if question_cluster[idx] not in em_cluster_agg:
                em_cluster_agg[question_cluster[idx]] = 1
            if is_em == 0: em_cluster_agg[question_cluster[idx]] = 0

            if question_cluster[idx] not in em_cluster_relaxed:
                em_cluster_relaxed[question_cluster[idx]] = 1
            if instance_matched == 0: em_cluster_relaxed[question_cluster[idx]] = 0

            if question_cluster[idx] not in f1_cluster_80:
                f1_cluster_80[question_cluster[idx]] = 1
            if max_f1 < 0.8: f1_cluster_80[question_cluster[idx]] = 0

        bi += lengths[l]
        max_f1s.append(max_f1)
        em_counter += instance_matched
        f1_dist[key].append(max_f1)

    all_preds.extend(batch_preds)

assert len(em_cluster_relaxed) == len(em_cluster_agg)
assert len(f1_cluster_80) == len(em_cluster_agg) 

em_cluster_relaxed_res = sum(em_cluster_relaxed.values()) / len(em_cluster_relaxed)
em_cluster_agg_res = sum(em_cluster_agg.values()) / len(em_cluster_agg)
f1_cluster_80_res = sum(f1_cluster_80.values()) / len(f1_cluster_80)

label_names = [label_map[l] for l in all_golds]
pred_names = [label_map[p] for p in all_preds]

# question_id is also flattened
em = exact_match(question_ids, label_names, pred_names)
eval_pos_f1 = cal_f1(pred_names, label_names, {v:k for k,v in label_map.items()})

print("the current eval positive class Micro F1 (Agg) is: %.4f" % eval_pos_f1)
print("the current eval positive class Macro F1 (Relaxed) is: %.4f" % np.mean(max_f1s))
print("the current eval positive class Macro F1 (Agg) is: %.4f" % np.mean(macro_f1s))

print("the current eval exact match (Agg) ratio is: %.4f" % em)
print("the current eval exact match ratio (Relaxed) is: %.4f" % (em_counter / len(eval_features)))

print("%d Clusters" % len(em_cluster_relaxed))
print("the current eval clustered EM (Agg) is: %.4f" % (em_cluster_agg_res))
print("the current eval clustered EM (Relaxed) is: %.4f" % (em_cluster_relaxed_res))
print("the current eval clusrered F1 (max>=0.8) is: %.4f" % (f1_cluster_80_res))



label_map = {0: 'Negative', 1: 'Positive'}
eval_loss, eval_accuracy, best_eval_f1, nb_eval_examples, nb_eval_steps = 0.0, 0.0, 0.0, 0, 0
all_preds, all_golds, max_f1s, macro_f1s = [], [], [], []
f1_dist = defaultdict(list)
em_counter = 0
em_cluster_agg, em_cluster_relaxed, f1_cluster_80 = {}, {}, {}

for i, (input_ids, input_masks, segment_ids, instance_indices) in enumerate(eval_dataloader):

    offsets, labels, lengths = flatten_answers([(eval_labels[i], eval_offsets[i])
                                                for i in instance_indices.tolist()])
    all_golds.extend(labels)
    labels = torch.tensor(labels).to(device)

    ###
#     logits = logits.detach().cpu().numpy()
    p_factual = factual_probs[i].cpu()
    p_event_only = event_only_probs[i].cpu()
    p_nothing = nothing_probs[i].cpu()

    prob = p_factual - lambda1*p_event_only - lambda2*p_nothing
    batch_preds = np.argmax(prob.numpy(), axis=1)
    ###
    labels = labels.to('cpu').numpy()

    nb_eval_examples += labels.shape[0]
    nb_eval_steps += 1


    bi = 0
    for l, idx in enumerate(instance_indices):
        pred = [batch_preds[bi + li] for li in range(lengths[l])]
        pred_names = [label_map[p] for p in pred]
        gold_names = [label_map[labels[bi + li]] for li in range(lengths[l])]
        is_em = (pred_names == gold_names)
        if sum([labels[bi + li] for li in range(lengths[l])]) == 0 and sum(pred) == 0:
            macro_f1s.append(1.0)
        else:
            macro_f1s.append(cal_f1(pred_names, gold_names, {v:k for k,v in label_map.items()}))

        max_f1, instance_matched = 0, 0
        for gold in eval_idv_answers[idx]:
            label_names = [label_map[l] for l in gold]
            if pred_names == label_names: instance_matched = 1
            if sum(gold) == 0 and sum(pred) == 0:
                f1 = 1.0
            else:
                f1 = cal_f1(pred_names, label_names, {v:k for k,v in label_map.items()})
            if f1 >= max_f1:
                max_f1 = f1
                key = len(gold)

        if question_cluster_size[idx] > 1:
            if question_cluster[idx] not in em_cluster_agg:
                em_cluster_agg[question_cluster[idx]] = 1
            if is_em == 0: em_cluster_agg[question_cluster[idx]] = 0

            if question_cluster[idx] not in em_cluster_relaxed:
                em_cluster_relaxed[question_cluster[idx]] = 1
            if instance_matched == 0: em_cluster_relaxed[question_cluster[idx]] = 0

            if question_cluster[idx] not in f1_cluster_80:
                f1_cluster_80[question_cluster[idx]] = 1
            if max_f1 < 0.8: f1_cluster_80[question_cluster[idx]] = 0

        bi += lengths[l]
        max_f1s.append(max_f1)
        em_counter += instance_matched
        f1_dist[key].append(max_f1)

    all_preds.extend(batch_preds)

assert len(em_cluster_relaxed) == len(em_cluster_agg)
assert len(f1_cluster_80) == len(em_cluster_agg) 

em_cluster_relaxed_res = sum(em_cluster_relaxed.values()) / len(em_cluster_relaxed)
em_cluster_agg_res = sum(em_cluster_agg.values()) / len(em_cluster_agg)
f1_cluster_80_res = sum(f1_cluster_80.values()) / len(f1_cluster_80)

label_names = [label_map[l] for l in all_golds]
pred_names = [label_map[p] for p in all_preds]

# question_id is also flattened
em = exact_match(question_ids, label_names, pred_names)
eval_pos_f1 = cal_f1(pred_names, label_names, {v:k for k,v in label_map.items()})

print("the current eval positive class Micro F1 (Agg) is: %.4f" % eval_pos_f1)
print("the current eval positive class Macro F1 (Relaxed) is: %.4f" % np.mean(max_f1s))
print("the current eval positive class Macro F1 (Agg) is: %.4f" % np.mean(macro_f1s))

print("the current eval exact match (Agg) ratio is: %.4f" % em)
print("the current eval exact match ratio (Relaxed) is: %.4f" % (em_counter / len(eval_features)))

print("%d Clusters" % len(em_cluster_relaxed))
print("the current eval clustered EM (Agg) is: %.4f" % (em_cluster_agg_res))
print("the current eval clustered EM (Relaxed) is: %.4f" % (em_cluster_relaxed_res))
print("the current eval clusrered F1 (max>=0.8) is: %.4f" % (f1_cluster_80_res))