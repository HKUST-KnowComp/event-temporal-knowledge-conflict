import argparse
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from exp import *
import json
from functools import partial
import logging, re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
set_global_logging_level(logging.ERROR)


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters                                                                                         
    parser.add_argument("--transformers_model",
                        default='google/bigbird-roberta-large', type=str,
                        help="Backbone transformers model.")
    parser.add_argument("--dataset",
                        default='MATRES', type=str, help="Dataset",
                        choices=['HiEve', 'IC', 'MATRES' ])
    parser.add_argument("--best_path", type=str,
                        help="Path to save model")
    parser.add_argument("--eval_path", type=str,
                        default="./data/test_features_matres.json")

    parser.add_argument("--batch_size", default=5, type=int, )

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--use_tense", default=0, type=int, help="Whether to use tense info in the model.")

    args = parser.parse_args()

    # datetime object containing current date and time
    now = datetime.datetime.now()
    # dd/mm/YY H:M:S
    # dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
    # print("date and time =", dt_string)

    #label_dict={"SuperSub": 0, "SubSuper": 1, "Coref": 2, "NoRel": 3}
    num_dict = {0: "before", 1: "after", 2: "equal", 3: "vague"}

    mask_in_input_ids = 0 # note that [MASK] is actually learned through pre-training
    mask_in_input_mask = 0 # when input is masked through attention, it would be replaced with [PAD]
    acronym = 0 # using acronym for tense (e.g., pastsimp): 1; else (e.g., past simple): 0
    t_marker = 1


    #############################
    ### Setting up parameters ###
    #############################
    best_PATH = args.best_path

    f1_metric = 'micro'
    params = {'transformers_model': 'google/bigbird-roberta-large',
              'dataset': 'MATRES' ,   # 'HiEve', 'IC', 'MATRES' 
              'testdata': 'None', # 
              'block_size': 64,
              'add_loss': 0, 
              'batch_size': 5,    # 6 works on 48G gpu. In the paper: 20 
              'accum_iter':1,
              'epochs': 0,
              'learning_rate': 0,    # subject to change
              'seed': 0,
              'debug': 0,
              'rst_file_name': 'init_test.rst',    # subject to change
              'mask_in_input_ids': mask_in_input_ids,
              'mask_in_input_mask': mask_in_input_mask,
              'marker': 'abc', 
              'tense_acron': 0, # 1 (acronym of tense) or 0 (original tense)
              't_marker': 1, # 2 (trigger enclosed by special tokens) or 1 (tense enclosed by **)
              'td': 0, # 0 (no tense detection) or 1 (tense detection, add tense info)
              'dpn': 0, # 1 if use DPN; else 0
              'lambda_1': -10, # lower bound * 10
              'lambda_2': 11, # upper bound * 10
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

    cuda = torch.device('cuda')
    params['cuda'] = cuda # not included in config file

    #######################
    ### Data processing ###
    #######################

    # print("Processing " + params['dataset'] + " dataset...")
    t0 = time.time()
    if params['dataset'] == "IC":
        dir_name = "./IC/IC_Processed/"
        #max_sent_len = 193
    elif params['dataset'] == "HiEve":
        dir_name = "./hievents_v2/processed/"
        #max_sent_len = 155
    elif params['dataset'] == "MATRES":
        dir_name = ""
    else:
        print("Not supporting this dataset yet!")

    tokenizer = AutoTokenizer.from_pretrained(params['transformers_model'])   
    if acronym:
        special_tokens_dict = {'additional_special_tokens': 
                               [' [futuperfsimp]',' [futucont]',' [futuperfcont]',' [futusimp]', ' [pastcont]', ' [pastperfcont]', ' [pastperfsimp]', ' [pastsimp]', ' [prescont]', ' [presperfcont]', ' [presperfsimp]', ' [pressimp]', ' [futuperfsimppass]',' [futucontpass]',' [futuperfcontpass]',' [futusimppass]', ' [pastcontpass]', ' [pastperfcontpass]', ' [pastperfsimppass]', ' [pastsimppass]', ' [prescontpass]', ' [presperfcontpass]', ' [presperfsimppass]', ' [pressimppass]', ' [none]'
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

    from model import transformers_mlp_cons

    # print(f'current device: { torch.cuda.current_device()}')

    OnePassModel = transformers_mlp_cons(params)
    OnePassModel.to(cuda)
    OnePassModel.zero_grad()
    # print("# of parameters:", count_parameters(OnePassModel))
    #
    # print("loading checkpoint from:", best_PATH)
    state_dict = torch.load(best_PATH)
    OnePassModel.load_state_dict(state_dict)
    OnePassModel.eval()
    
    ## load data
    features_eval = json.load(open(args.eval_path))

    remove_list = ["being", "doing", "having", "'ve", "'re", "did", "'s", 
                   "are", "is", "am", "was", "were", "been", "had", "said", 
                   "be", "have", "can", "could", "may", "might", "must", "ought", "shall", "will", "would", "say", "nee", "need", "do", "happen", "occur"]

    eval_dataloader = DataLoader(features_eval, batch_size=5, shuffle=False, 
                                  collate_fn=partial(collate_fn, mask_in_input_ids=mask_in_input_ids, mask_in_input_mask=mask_in_input_mask), 
                                  drop_last=False)

    ## eval
    mem_exp_test = exp(cuda, OnePassModel, params['epochs'], params['learning_rate'], 
                       None, None, None, 
                       params['dataset'], best_PATH, None, params['dpn'], 
                       model_name=model_name, relation_stats=None, lambdas=None)

    micro_f1, macro_f1 = mem_exp_test.evaluate_simple(eval_dataloader)

    micro_f1, macro_f1 = micro_f1 * 100, macro_f1 * 100
    # print(f"--------------------------------------")
    print("&{:.1f}".format(micro_f1) + "&" + "{:.1f}".format(macro_f1))


if __name__ == "__main__":
    main()