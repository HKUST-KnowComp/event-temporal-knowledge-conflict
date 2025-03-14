import argparse
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from exp import *
import json
from functools import partial
from sklearn import svm, linear_model


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
    parser.add_argument("--output_path", type=str,
                        default="", help="Path to output filtered training data.")
    parser.add_argument("--best_path", type=str,
                        help="Path to save model")
    parser.add_argument("--train_json_path", type=str,
                        default="./data/train_features_matres_with_tense.json")
    parser.add_argument("--eval_data_dir",
                        nargs='+', default=None, required=False,
                        help="The input data dir of eval files. A list")
    parser.add_argument("--eval_data_name",
                        nargs='+', default=None, required=False,
                        help="Names. for display only")
    parser.add_argument("--testdata",
                        default='None', type=str, help="Test dataset. None for training mode",
                        choices=['MATRES', 'MATRES_nd', 'TDD', 'PRED', 'None'])
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--use_tense", default=0, type=int, help="Whether to use tense info in the model.")
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
    params = {'transformers_model': args.transformers_model,
              'dataset': args.dataset,  # 'HiEve', 'IC', 'MATRES'
              'testdata': args.testdata,  #
              'block_size': 64,
              'add_loss': 0,
              'batch_size': args.batch_size,  # 6 works on 48G gpu. In the paper: 20
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
    model = AutoModel.from_pretrained(params['transformers_model'])

    params['model'] = model
    debug = params['debug']

    features_train = json.load(open(args.train_json_path))
    dataloader = DataLoader(features_train, batch_size=params['batch_size'], shuffle=False,
                                  collate_fn=partial(collate_fn, mask_in_input_ids=mask_in_input_ids,
                                                     mask_in_input_mask=mask_in_input_mask), drop_last=False)

    model = model.to(cuda)
    print("  Data processing took: {:}".format(format_time(time.time() - t0)))

    from model import AFLiteEncoder

    print(f'current device: {torch.cuda.current_device()}')

    OnePassModel = AFLiteEncoder(params)
    OnePassModel.to(cuda)
    OnePassModel.zero_grad()
    OnePassModel.eval()
    print("# of parameters:", count_parameters(OnePassModel))

    # Training and prediction
    embedding_list, label_list = [], []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader)):
            # Progress update every 40 batches.
            embeddings = OnePassModel(batch[0].to(cuda), batch[1].to(cuda), batch[2], batch[3], batch[4],
                                      batch[5])
            embedding_list.append(embeddings.cpu())
            label_list.extend([i[0] for i in batch[5]])
    embedding_list = torch.cat(embedding_list, dim=0).numpy()
    label_list = np.array(label_list)
    sampled_count_list, right_count_list = [0] * len(label_list), [0] * len(label_list)

    shuffled_idx_list = list(range(len(label_list)))
    random.shuffle(shuffled_idx_list)
    chunk_size = len(shuffled_idx_list) // 20
    for i in tqdm(range(20), "linear model"):
        train_idx = shuffled_idx_list[chunk_size * i: chunk_size * (i + 1)]
        eval_idx = shuffled_idx_list[: chunk_size * i] + shuffled_idx_list[chunk_size * (i + 1):]
        train_embedding, train_label = embedding_list[train_idx], label_list[train_idx]
        eval_embedding, eval_label = embedding_list[eval_idx], label_list[eval_idx]
        lin = linear_model.SGDClassifier(max_iter=10000,
                                         tol=1e-5)
        lin.fit(train_embedding, train_label)
        eval_predicted = lin.predict(eval_embedding)

        for idx, gt, pred in zip(eval_idx, eval_label, eval_predicted):
            sampled_count_list[idx] += 1
            right_count_list[idx] += gt == pred

    prob_list = [r_c / s_c if s_c else 0 for r_c, s_c in zip(right_count_list, sampled_count_list)]

    new_data_list = []
    for p, d in zip(prob_list, features_train):
        if p < 0.90:
            new_data_list.append(d)

    print(len(new_data_list))
    print("filtered: ", len(features_train) - len(new_data_list))
    print(len(prob_list))

    filename = args.output_path
    with open(filename, "w") as fout:
        fout.write(json.dumps(new_data_list))

if __name__ == "__main__":
    main()
