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


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--transformers_model",
                        default='google/bigbird-roberta-large', type=str,
                        help="Backbone transformers model.")
    parser.add_argument("--dataset",
                        default='MATRES', type=str, help="Dataset",
                        choices=['HiEve', 'IC', 'MATRES'])
    parser.add_argument("--train_from_path", type=str,
                        default="", help="Path to resume training from.")
    parser.add_argument("--best_path", type=str,
                        help="Path to save model")
    parser.add_argument("--train_json_path", type=str,
                        default="./data/train_features_matres_with_tense.json")
    parser.add_argument("--valid_json_path", type=str,
                        default="./data/valid_features_matres_with_tense.json")
    parser.add_argument("--test_json_path", type=str,
                        default="./data/test_features_matres_with_tense.json")
    parser.add_argument("--eval_data_dir",
                        nargs='+', default=None, required=False,
                        help="The input data dir of eval files. A list")
    parser.add_argument("--eval_data_name",
                        nargs='+', default=None, required=False,
                        help="Names. for display only")

    parser.add_argument("--testdata",
                        default='None', type=str, help="Test dataset. None for training mode",
                        choices=['MATRES', 'MATRES_nd', 'TDD', 'PRED', 'None'])
    parser.add_argument("--batch_size", default=10, type=int, )
    parser.add_argument("--accum_iter", default=2, type=int, help="real training bs = batch_size * accum_iter")
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--scheduler", type=str, default="constant")

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--learning_rate", default=5e-6, type=float)
    parser.add_argument("--use_tense", default=0, type=int, help="Whether to use tense info in the model.")
    parser.add_argument("--f1_metric", default='micro_f1',
                        type=str, help="eval metric.")
    parser.add_argument("--debug", action="store_true",
                        help="debug mode, wandb will be disabled")

    args = parser.parse_args()

    wandb_mode = "disabled" if args.debug else "online"
    wandb.init(project="MATRES",
               config=args, mode=wandb_mode)

    # datetime object containing current date and time
    now = datetime.datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
    print("date and time =", dt_string)

    # label_dict={"SuperSub": 0, "SubSuper": 1, "Coref": 2, "NoRel": 3}
    num_dict = {0: "before", 1: "after", 2: "equal", 3: "vague"}

    mask_in_input_ids = 0  # note that [MASK] is actually learned through pre-training
    mask_in_input_mask = 0  # when input is masked through attention, it would be replaced with [PAD]
    acronym = 0  # using acronym for tense (e.g., pastsimp): 1; else (e.g., past simple): 0
    t_marker = 1

    #############################
    ### Setting up parameters ###
    #############################
    best_PATH = args.best_path
    os.makedirs(os.path.dirname(best_PATH), exist_ok=True)
    f1_metric = args.f1_metric  # 'micro'
    params = {'transformers_model': args.transformers_model,
              'dataset': args.dataset,  # 'HiEve', 'IC', 'MATRES'
              'testdata': args.testdata,  #
              'block_size': 64,
              'add_loss': 0,
              'batch_size': args.batch_size,  # 6 works on 48G gpu. In the paper: 20
              'accum_iter': args.accum_iter,
              'epochs': args.epochs,
              'learning_rate': args.learning_rate,  # subject to change
              'seed': args.seed,
              'debug': args.debug,
              'rst_file_name': 'init_test.rst',  # subject to change
              'mask_in_input_ids': mask_in_input_ids,
              'mask_in_input_mask': mask_in_input_mask,
              'marker': 'abc',
              'tense_acron': 0,  # 1 (acronym of tense) or 0 (original tense)
              't_marker': 1,  # 2 (trigger enclosed by special tokens) or 1 (tense enclosed by **)
              'td': args.use_tense,  # 0 (no tense detection) or 1 (tense detection, add tense info)
              'dpn': 0,  # 1 if use DPN; else 0
              'lambda_1': -10,  # lower bound * 10
              'lambda_2': 11,  # upper bound * 10
              'f1_metric': f1_metric,
              }
    # $acr $tmarker $td $dpn $mask $lambda_1 $lambda_2

    if params['testdata'] == 'MATRES_nd':
        params['nd'] = True
    else:
        params['nd'] = False

    if params['transformers_model'][-5:] == "large":
        params['emb_size'] = 1024
    elif params['transformers_model'][-4:] == "base":
        params['emb_size'] = 768
    else:
        print("emb_size is neither 1024 nor 768? ...")

    set_seed(params['seed'])
    rst_file_name = params['rst_file_name']

    model_name = rst_file_name.replace(".rst", "")
    with open("config/" + rst_file_name.replace("rst", "json"), 'w') as config_file:
        json.dump(params, config_file)

    cuda = torch.device('cuda')
    params['cuda'] = cuda  # not included in config file

    #######################
    ### Data processing ###
    #######################

    print("Processing " + params['dataset'] + " dataset...")
    t0 = time.time()
    if params['dataset'] == "IC":
        dir_name = "./IC/IC_Processed/"
        # max_sent_len = 193
    elif params['dataset'] == "HiEve":
        dir_name = "./hievents_v2/processed/"
        # max_sent_len = 155
    elif params['dataset'] == "MATRES":
        dir_name = ""
    else:
        print("Not supporting this dataset yet!")

    tokenizer = AutoTokenizer.from_pretrained(params['transformers_model'])
    if acronym:
        special_tokens_dict = {'additional_special_tokens':
                                   [' [futuperfsimp]', ' [futucont]', ' [futuperfcont]', ' [futusimp]', ' [pastcont]',
                                    ' [pastperfcont]', ' [pastperfsimp]', ' [pastsimp]', ' [prescont]',
                                    ' [presperfcont]', ' [presperfsimp]', ' [pressimp]', ' [futuperfsimppass]',
                                    ' [futucontpass]', ' [futuperfcontpass]', ' [futusimppass]', ' [pastcontpass]',
                                    ' [pastperfcontpass]', ' [pastperfsimppass]', ' [pastsimppass]', ' [prescontpass]',
                                    ' [presperfcontpass]', ' [presperfsimppass]', ' [pressimppass]', ' [none]'
                                    ]}
        spec_toke_list = []
        for t in special_tokens_dict['additional_special_tokens']:
            spec_toke_list.append(" [/" + t[2:])
        special_tokens_dict['additional_special_tokens'] += spec_toke_list
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model = AutoModel.from_pretrained(params['transformers_model'])
        model.resize_token_embeddings(len(tokenizer))
    else:
        model = AutoModel.from_pretrained(params['transformers_model'])

    params['model'] = model
    debug = params['debug']
    if debug:
        params['epochs'] = 1

    model = model.to(cuda)

    features_train = json.load(open(args.train_json_path))
    features_valid = json.load(open(args.valid_json_path))
    features_test = json.load(open(args.test_json_path))

    remove_list = ["being", "doing", "having", "'ve", "'re", "did", "'s", "are", "is", "am", "was", "were", "been",
                   "had", "said", "be", "have", "can", "could", "may", "might", "must", "ought", "shall", "will",
                   "would", "say", "nee", "need", "do", "happen", "occur"]

    # if debug:
    #     train_dataloader = DataLoader(features_train, batch_size=params['batch_size'], shuffle=True,
    #                                     collate_fn=partial(collate_fn, mask_in_input_ids=mask_in_input_ids, mask_in_input_mask=mask_in_input_mask),
    #                                      drop_last=True)
    #     valid_dataloader = test_dataloader = train_dataloader
    #     for step, batch in enumerate(train_dataloader):
    #         print(batch)
    if params['testdata'] == 'TDD':
        features_valid, labels_valid, labels_full_valid = TDD_processor('man-dev')
        features_test, labels_test, labels_full_test = TDD_processor('man-test')
        valid_dataloader = DataLoader(features_valid, batch_size=params['batch_size'], shuffle=False,
                                      collate_fn=partial(collate_fn, mask_in_input_ids=mask_in_input_ids,
                                                         mask_in_input_mask=mask_in_input_mask), drop_last=False)
        test_dataloader = DataLoader(features_test, batch_size=params['batch_size'], shuffle=False,
                                     collate_fn=partial(collate_fn, mask_in_input_ids=mask_in_input_ids,
                                                        mask_in_input_mask=mask_in_input_mask), drop_last=False)
    elif params['testdata'] == 'PRED':
        test_dataloader = DataLoader(features_test, batch_size=params['batch_size'], shuffle=False,
                                     collate_fn=partial(collate_fn, mask_in_input_ids=mask_in_input_ids,
                                                        mask_in_input_mask=mask_in_input_mask), drop_last=False)
    else:
        train_dataloader = DataLoader(features_train, batch_size=params['batch_size'], shuffle=True,
                                      collate_fn=partial(collate_fn, mask_in_input_ids=mask_in_input_ids,
                                                         mask_in_input_mask=mask_in_input_mask), drop_last=True)
        valid_dataloader = DataLoader(features_valid, batch_size=params['batch_size'], shuffle=False,
                                      collate_fn=partial(collate_fn, mask_in_input_ids=mask_in_input_ids,
                                                         mask_in_input_mask=mask_in_input_mask), drop_last=False)
        test_dataloader = DataLoader(features_test, batch_size=params['batch_size'], shuffle=False,
                                     collate_fn=partial(collate_fn, mask_in_input_ids=mask_in_input_ids,
                                                        mask_in_input_mask=mask_in_input_mask), drop_last=False)
    print("  Data processing took: {:}".format(format_time(time.time() - t0)))

    from model import transformers_mlp_cons

    print(f'current device: {torch.cuda.current_device()}')

    OnePassModel = transformers_mlp_cons(params)
    OnePassModel.to(cuda)
    OnePassModel.zero_grad()
    print("# of parameters:", count_parameters(OnePassModel))

    if args.train_from_path:
        print("Loading model from " + args.train_from_path)
        OnePassModel.load_state_dict(torch.load(args.train_from_path))

    # Training and prediction

    mem_exp_test = exp(cuda, OnePassModel, params['epochs'], params['learning_rate'],
                       train_dataloader, valid_dataloader, test_dataloader,
                       params['dataset'], best_PATH, None, params['dpn'], warmup_ratio=args.warmup_ratio,
                       scheduler_type=args.scheduler,
                       model_name=model_name, relation_stats=None, lambdas=None, accum_iter=params["accum_iter"],
                       f1_metric=params["f1_metric"])

    args.eval_data_dir = [] if args.eval_data_dir is None else args.eval_data_dir
    args.eval_data_name = [] if args.eval_data_name is None else args.eval_data_name

    eval_features = [json.load(open(eval_dir)) for eval_dir in args.eval_data_dir]
    eval_dataloaders = [DataLoader(eval_feature, batch_size=params['batch_size'], shuffle=False,
                                   collate_fn=partial(collate_fn, mask_in_input_ids=mask_in_input_ids,
                                                      mask_in_input_mask=mask_in_input_mask), drop_last=False) \
                        for eval_feature in eval_features]

    mem_exp_test.train(dict([(name, loader) for name, loader in zip(args.eval_data_name, eval_dataloaders)]))

    mem_exp_test.evaluate(mem_exp_test.dataset)


if __name__ == "__main__":
    main()