# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import random
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import RobertaTokenizer
from models import AFLiteRoberta
from utils import *
from optimization import *
from pathlib import Path
import wandb
from sklearn import svm, linear_model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PYTORCH_PRETRAINED_ROBERTA_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_ROBERTA_CACHE',
                                                  Path.home() / '.pytorch_pretrained_roberta'))


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters                                                                                         
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .json files (or other data files) for the task.")
    parser.add_argument("--load_from_path",
                        default=None,
                        type=str,
                        required=False,
                        help="path of checkpoint to resume training from.")
    parser.add_argument("--eval_data_dir",
                        nargs='+',
                        default=None,
                        required=True,
                        help="The input data dir of eval files. A list")
    parser.add_argument("--eval_suffix",
                        nargs='+',
                        default="_end2end_final.json",
                        required=False,
                        help="The input data dir. Should contain the .json files (or other data files) for the task.")
    parser.add_argument("--model", default=None, type=str, required=True,
                        help="pre-trained model selected in the list: roberta-base, "
                             "roberta-large, bert-base, bert-large. ")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--file_suffix",
                        default=None,
                        type=str,
                        required=True,
                        help="unique identifier for data file")
    ## Other parameters                                                                                            
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument('--cuda', type=str, default="", help="cuda index")

    # wandb related:
    parser.add_argument('--project_name', type=str, default="baseline", help="name of the project")
    parser.add_argument('--run_name', type=str, default="baseline", help="special name of the run")

    args = parser.parse_args()

    now = datetime.now()

    wandb.init(project=args.project_name,
               name=args.run_name + "_" + now.strftime("%m-%d-%H:%M"),
               config=args, mode="disabled")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # fix all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()
    logger.info("current task is " + str(task_name))
    if args.load_from_path is not None:
        if os.path.isfile(args.load_from_path):
            model_state_dict = torch.load(args.load_from_path)
        elif os.path.isdir(args.load_from_path):
            model_state_dict = torch.load(args.load_from_path + "pytorch_model.bin")
        else:
            raise ValueError

    # construct model
    tokenizer = RobertaTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
    cache_dir = PYTORCH_PRETRAINED_ROBERTA_CACHE / 'distributed_{}'.format(args.local_rank)
    if args.load_from_path is None:
        model = AFLiteRoberta.from_pretrained(args.model, cache_dir=cache_dir)
    else:
        model = AFLiteRoberta.from_pretrained(args.model, state_dict=model_state_dict,
                                              cache_dir=cache_dir)

    model.to(device)
    all_data_dict = load_data(args.data_dir, "train", args.file_suffix)
    if 'roberta' in args.model:
        train_features = convert_to_features_roberta(all_data_dict, tokenizer, instance=False,
                                                     max_length=args.max_seq_length, end_to_end=True)
    else:
        train_features = convert_to_features(all_data_dict, tokenizer, instance=False,
                                             max_length=args.max_seq_length, end_to_end=True)

    all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(train_features, 'mask_ids'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)

    all_offsets = select_field(train_features, 'offset')
    all_labels = select_field(train_features, 'label')

    all_key_indices = torch.tensor(list(range(len(all_labels))), dtype=torch.long)
    logger.info("id_size: {} mask_size: {}, instance_key_size: {}, segment_size: {}".format(
        all_input_ids.size(), all_input_mask.size(), all_key_indices.size(), all_segment_ids.size()))

    all_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_key_indices)

    # free memory
    del train_features
    del all_input_ids
    del all_input_mask
    del all_segment_ids

    encoder_sampler = SequentialSampler(all_data)

    dataloader = DataLoader(all_data, sampler=encoder_sampler, batch_size=args.batch_size)
    model.eval()

    label_list, embedding_list = [], []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc="Encoding")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_masks, segment_ids, instance_indices = batch
            offsets, labels, lengths = flatten_answers([(all_labels[i], all_offsets[i])
                                                        for i in instance_indices.cpu().tolist()])
            labels = torch.tensor(labels).to(device)
            embeddings = model(input_ids, offsets, lengths, attention_mask=input_masks,
                               token_type_ids=segment_ids)
            label_list.append(labels.cpu())
            embedding_list.append(embeddings.cpu())
        label_list = torch.cat(label_list, dim=0).numpy()
        embedding_list = torch.cat(embedding_list, dim=0).numpy()

    right_count_list, sampled_count_list = [0] * len(label_list), [0] * len(label_list)
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
    length_list = [len(i) for i in all_offsets]

    start_idx, new_data_dict = 0, {}
    for l, (key, value) in zip(length_list, all_data_dict.items()):
        cur_prob = prob_list[start_idx: start_idx + l]
        min_prob = min(cur_prob)
        if min_prob < 0.90:
            new_data_dict[key] = value
        start_idx += l

    print(len(new_data_dict))
    print("filtered: ", len(all_data_dict) - len(new_data_dict))
    print(len(prob_list))

    filename = os.path.join(args.data_dir, "aflite_train_" + args.file_suffix)
    with open(filename, "w") as fout:
        fout.write(json.dumps(new_data_dict))


if __name__ == "__main__":
    main()
