import tqdm
import argparse
import time
import random
import os
from os import listdir
from torch.utils.data import DataLoader
from util import format_time, count_parameters, set_seed, collate_fn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from exp import *
import numpy as np
import json
import sys
from synonyms import *
import pickle
from functools import partial
from model import transformers_mlp_cons

def main():
    parser = argparse.ArgumentParser()
    ## Required parameters                                                                                         
    parser.add_argument("--transformers_model",
                        default='google/bigbird-roberta-large', type=str,
                        help="Backbone transformers model.")
    parser.add_argument("--dataset",
                        default='MATRES', type=str, help="Dataset",
                        choices=['HiEve', 'IC', 'MATRES' ])
    parser.add_argument("--train_from_path", type=str,
                        default="", help="Path to resume training from.")
    parser.add_argument("--load_path", type=str,
                        help="Path to load saved model")

    parser.add_argument("--eval_data_dir",
                        nargs='+',default=None,required=False,
                        help="The input data dir of eval files. A list")
    parser.add_argument("--eval_data_name",
                        nargs='+',default=None,required=False,
                        help="Names. for display only")
    parser.add_argument("--use_tense", default=0, type=int, help="Whether to use tense info in the model.")

    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()

    params = {'transformers_model': args.transformers_model,
          'dataset': 'MATRES',   # 'HiEve', 'IC', 'MATRES' 
          'block_size': 64,
          'add_loss': 0, 
          'seed': args.seed,
          'debug': 0,
          'rst_file_name': 'init_test.rst',    # subject to change
          'marker': 'abc', 
          'tense_acron': 0, # 1 (acronym of tense) or 0 (original tense)
          't_marker': 1, # 2 (trigger enclosed by special tokens) or 1 (tense enclosed by **)
          'td': args.use_tense, # 0 (no tense detection) or 1 (tense detection, add tense info)
          'dpn': 0, # 1 if use DPN; else 0
          'lambda_1': -10, # lower bound * 10
          'lambda_2': 11, # upper bound * 10
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

    OnePassModel.load_state_dict(torch.load(args.load_path))

    # load dataset

    
    mem_exp_test = exp(cuda, OnePassModel, 0, 0, 
                       None, None, None, 
                       params['dataset'], "", None, params['dpn'],model_name="eval"
                       )

    eval_features = [json.load(open(eval_dir)) for eval_dir in args.eval_data_dir]
    eval_dataloaders = [DataLoader(eval_feature, batch_size=64, shuffle=False, 
                collate_fn=partial(collate_fn, mask_in_input_ids=0, mask_in_input_mask=0), drop_last=False)\
                      for eval_feature in eval_features]

    if not args.eval_data_name:
        args.eval_data_name = [str(i) for i in range(len(eval_dataloaders))]
    additional_eval_loader_dict = dict([(name, loader) for name, loader in zip(args.eval_data_name, eval_dataloaders)])

    for name, eval_loader in additional_eval_loader_dict.items():
        micro_f1, macro_f1 = mem_exp_test.evaluate_simple(eval_loader)
        print(name, micro_f1, macro_f1)
if __name__ == "__main__":
    main()
