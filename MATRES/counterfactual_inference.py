import tqdm
import wandb
import argparse
import time
import datetime
from datetime import datetime 
import random
import os
from os import listdir
from os.path import isfile, join
from torch.utils.data import DataLoader
from util import format_time, count_parameters, set_seed, collate_fn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from exp import *
import numpy as np
import json
import sys
from synonyms import *
import pickle
# from timeline_construct import *
# from ts import func, ModelWithTemperature
from functools import partial
from model import transformers_mlp_cons

params = {'transformers_model': 'google/bigbird-roberta-large',
          'dataset': 'MATRES',   # 'HiEve', 'IC', 'MATRES' 
          'testdata': 'none', # 
          'block_size': 64,
          'add_loss': 0, 
          'batch_size': 64,    # 6 works on 48G gpu. In the paper: 20 
          'accum_iter':1,
          'epochs': 1,
          'learning_rate': 1e-5,    # subject to change
          'seed': 7,
          'debug': 0,
          'rst_file_name': 'init_test.rst',    # subject to change
          'mask_in_input_ids': 0,
          'mask_in_input_mask': 0,
          'marker': 'abc', 
          'tense_acron': 0, # 1 (acronym of tense) or 0 (original tense)
          't_marker': 1, # 2 (trigger enclosed by special tokens) or 1 (tense enclosed by **)
          'td': 0, # 0 (no tense detection) or 1 (tense detection, add tense info)
          'dpn': 0, # 1 if use DPN; else 0
          'lambda_1': -10, # lower bound * 10
          'lambda_2': 11, # upper bound * 10
          'f1_metric': 'micro_f1', 
}
if params['transformers_model'][-5:] == "large":
    params['emb_size'] = 1024
elif params['transformers_model'][-4:] == "base":
    params['emb_size'] = 768
    
tokenizer = AutoTokenizer.from_pretrained(params['transformers_model'])   

model = AutoModel.from_pretrained(params['transformers_model'])

cuda = torch.device('cuda')
model = model.to(cuda)

params['model'] = model
params['cuda'] = cuda
OnePassModel = transformers_mlp_cons(params)
OnePassModel.to(cuda)
OnePassModel.zero_grad()
print("# of parameters:", count_parameters(OnePassModel))

OnePassModel.load_state_dict(torch.load("output/counterfactual/event_only/roberta-large-e40-linear/model.pt"))

# loader

eval_feature = json.load(open("data/valid_features_matres_event_only.json"))
eval_dataloader = DataLoader(eval_feature, batch_size=64, shuffle=False, 
            collate_fn=partial(collate_fn, mask_in_input_ids=0, mask_in_input_mask=0), drop_last=False)
from tqdm import tqdm
all_probs_event_only = torch.tensor([])

for batch in tqdm(eval_dataloader):
    with torch.no_grad():
        logits, loss = OnePassModel(
            batch[0].to(cuda), batch[1].to(cuda), batch[2], batch[3], batch[4], batch[5]
        )
        all_probs_event_only = torch.concat( (all_probs_event_only, torch.nn.functional.softmax(logits).detach().cpu()) )

OnePassModel.load_state_dict(torch.load("path"))

from tqdm import tqdm

eval_feature = json.load(open("data/valid_features_matres.json"))
eval_dataloader = DataLoader(eval_feature, batch_size=64, shuffle=False, 
            collate_fn=partial(collate_fn, mask_in_input_ids=0, mask_in_input_mask=0), drop_last=False)
all_probs_factual = torch.tensor([])

for batch in tqdm(eval_dataloader):
    with torch.no_grad():
        logits, loss = OnePassModel(
            batch[0].to(cuda), batch[1].to(cuda), batch[2], batch[3], batch[4], batch[5]
        )
        all_probs_factual = torch.concat( (all_probs_factual, torch.nn.functional.softmax(logits).detach().cpu()) )

from sklearn.metrics import f1_score, accuracy_score, classification_report
lambda1, lambda2 = [-0.1, 0.3]
new_probs = all_probs_factual - lambda1 * all_probs_event_only - lambda2 * torch.tensor(class_prior)
preds = torch.argmax(new_probs, dim=1)
f1_score(all_labels, preds, average='micro'), f1_score(all_labels, preds, average='macro')

