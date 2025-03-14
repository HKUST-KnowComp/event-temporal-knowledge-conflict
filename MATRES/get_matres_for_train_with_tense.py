from matres_reader_with_tense import *

import tqdm
import time
import datetime
from datetime import datetime 
import random
from os import listdir
from os.path import isfile, join
from util import *
from transformers import AutoTokenizer, AutoModel
import numpy as np
import json
import sys
from synonyms import *
# datetime object containing current date and time
now = datetime.datetime.now()
# dd/mm/YY H:M:S
dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
print("date and time =", dt_string)

#label_dict={"SuperSub": 0, "SubSuper": 1, "Coref": 2, "NoRel": 3}
num_dict = {0: "before", 1: "after", 2: "equal", 3: "vague"}
#def label_to_num(label):
#    return label_dict[label]
def num_to_label(num):
    return num_dict[num]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def tml_reader(dir_name, file_name, tokenizer):
    my_dict = {}
    my_dict["event_dict"] = {}
    my_dict["eiid_dict"] = {}
    my_dict["doc_id"] = file_name.replace(".tml", "") 
    # e.g., file_name = "ABC19980108.1830.0711.tml"
    # dir_name = '/shared/why16gzl/logic_driven/EMNLP-2020/MATRES/TBAQ-cleaned/TimeBank/'
    tree = ET.parse(dir_name + file_name)
    root = tree.getroot()
    MY_STRING = str(ET.tostring(root))
    # ================================================
    # Load the lines involving event information first
    # ================================================
    event_id_why = 0
    for makeinstance in root.findall('MAKEINSTANCE'):
        instance_str = str(ET.tostring(makeinstance)).split(" ")
        try:
            assert instance_str[3].split("=")[0] == "eventID"
            assert instance_str[2].split("=")[0] == "eiid"
            eiid = int(instance_str[2].split("=")[1].replace("\"", "")[2:])
            eID = instance_str[3].split("=")[1].replace("\"", "")
        except:
            for i in instance_str:
                if i.split("=")[0] == "eventID":
                    eID = i.split("=")[1].replace("\"", "")
                if i.split("=")[0] == "eiid":
                    eiid = int(i.split("=")[1].replace("\"", "")[2:])
        # Not all document in the dataset contributes relation pairs in MATRES
        # Not all events in a document constitute relation pairs in MATRES
        
        if my_dict["doc_id"] in eiid_to_event_trigger.keys():
            if eiid in eiid_to_event_trigger[my_dict["doc_id"]].keys():
                event_id_why += 1
                my_dict["event_dict"][eID] = {"eiid": eiid, "mention": eiid_to_event_trigger[my_dict["doc_id"]][eiid], "event_id_why": event_id_why}
                my_dict["eiid_dict"][eiid] = {"eID": eID}
        
    # ==================================
    #              Load Text
    # ==================================
    start = MY_STRING.find("<TEXT>") + 6
    end = MY_STRING.find("</TEXT>")
    MY_TEXT = MY_STRING[start:end]
    while MY_TEXT[0] == " ":
        MY_TEXT = MY_TEXT[1:]
    MY_TEXT = MY_TEXT.replace("\\n", " ")
    MY_TEXT = MY_TEXT.replace("\\'", "\'")
    MY_TEXT = MY_TEXT.replace("  ", " ")
    MY_TEXT = MY_TEXT.replace(" ...", "...")
    
    # ========================================================
    #    Load position of events, in the meantime replacing 
    #    "<EVENT eid="e1" class="OCCURRENCE">turning</EVENT>"
    #    with "turning"
    # ========================================================
    while MY_TEXT.find("<") != -1:
        start = MY_TEXT.find("<")
        end = MY_TEXT.find(">")
        if MY_TEXT[start + 1] == "E":
            event_description = MY_TEXT[start:end].split(" ")
            desp_idx = 1
            if not event_description[desp_idx].split("=")[0] == "eid":
                desp_idx = 2
            eID = (event_description[desp_idx].split("="))[1].replace("\"", "")
            MY_TEXT = MY_TEXT[:start] + MY_TEXT[(end + 1):]
            if eID in my_dict["event_dict"].keys():
                my_dict["event_dict"][eID]["start_char"] = start # loading position of events
        else:
            MY_TEXT = MY_TEXT[:start] + MY_TEXT[(end + 1):]
    
    # =====================================
    # Enter the routine for text processing
    # =====================================
    
    my_dict["doc_content"] = MY_TEXT
    my_dict["sentences"] = []
    my_dict["relation_dict"] = {}
    sent_tokenized_text = sent_tokenize(my_dict["doc_content"])
    #sent_tokenized_text = []
    #tense_res = tense_getter(MY_TEXT)
    #for sentence in tense_res['sentences']:
    #    sent_tokenized_text.append(sentence[0])
        
    sent_span = tokenized_to_origin_span(my_dict["doc_content"], sent_tokenized_text)
    end_pos = [1]
    for count_sent, sent in enumerate(sent_tokenized_text):
        sent_dict = {}
        sent_dict["sent_id"] = count_sent
        sent_dict["content"] = sent
        sent_dict["sent_start_char"] = sent_span[count_sent][0]
        sent_dict["sent_end_char"] = sent_span[count_sent][1]
        sent_dict["tense_list"] = tense_getter(sent)
        
        spacy_token = nlp(sent_dict["content"])
        sent_dict["tokens"] = []
        sent_dict["pos"] = []
        # spaCy-tokenized tokens & Part-Of-Speech Tagging
        for token in spacy_token:
            sent_dict["tokens"].append(token.text)
            sent_dict["pos"].append(token.pos_)
        sent_dict["token_span_SENT"] = tokenized_to_origin_span(sent, sent_dict["tokens"])
        sent_dict["token_span_DOC"] = span_SENT_to_DOC(sent_dict["token_span_SENT"], sent_dict["sent_start_char"])

        # huggingface tokenizer
        sent_dict["_subword_to_ID"], sent_dict["_subwords"], \
        sent_dict["_subword_span_SENT"], sent_dict["_subword_map"] = \
        transformers_list(sent_dict["content"], tokenizer, sent_dict["tokens"], sent_dict["token_span_SENT"])
            
        if count_sent == 0:
            end_pos.append(len(sent_dict["_subword_to_ID"]))
        else:
            end_pos.append(end_pos[-1] + len(sent_dict["_subword_to_ID"]) - 1)
            
        sent_dict["_subword_span_DOC"] = \
        span_SENT_to_DOC(sent_dict["_subword_span_SENT"], sent_dict["sent_start_char"])
        
        sent_dict["_subword_pos"] = []
        for token_id in sent_dict["_subword_map"]:
            if token_id == -1 or token_id is None:
                sent_dict["_subword_pos"].append("None")
            else:
                sent_dict["_subword_pos"].append(sent_dict["pos"][token_id])
        
        my_dict["sentences"].append(sent_dict)
        count_sent += 1
        
    my_dict['end_pos'] = end_pos
    # Add sent_id as an attribute of event
    for event_id, event_dict in my_dict["event_dict"].items():
        my_dict["event_dict"][event_id]["sent_id"] = sent_id = \
        sent_id_lookup(my_dict, event_dict["start_char"])
        my_dict["event_dict"][event_id]["token_id"] = \
        id_lookup(my_dict["sentences"][sent_id]["token_span_DOC"], event_dict["start_char"])
        my_dict["event_dict"][event_id]["_subword_id"] = \
        id_lookup(my_dict["sentences"][sent_id]["_subword_span_DOC"], event_dict["start_char"]) + 1 
        # updated on Mar 20, 2021
        #my_dict["event_dict"][event_id]["tense"] = tense_finder(tense_res['sentences'][sent_id][1], event_dict["start_char"] - my_dict['sentences'][sent_id]["sent_start_char"])
        # updated on Feb 21, 2021
        my_dict["event_dict"][event_id]["tense"] = tense_finder(my_dict['sentences'][sent_id]["tense_list"], event_dict["start_char"] - my_dict['sentences'][sent_id]["sent_start_char"])
        # updated on Oct 24, 2022, because of change of tense identification service
        
    return my_dict

def add_tense_info(x_sent, tense, start, mention, special_1, special_2):
    # x:
    # special_1: 2589
    # special_2: 1736
    
    # y:
    # special_1: 1404
    # special_2: 5400
    orig_len = len(x_sent)        
    if tense:
        if acronym:
            tense_marker = tokenizer.encode(" " + tense[acronym])[1:-1]
        else:
            tense_marker = tokenizer.encode(tense[acronym])[1:-1]
    else:
        if acronym:
            tense_marker = tokenizer.encode(" [none]")[1:-1]
        else:
            tense_marker = tokenizer.encode("None")[1:-1]
    subword_len = len(tokenizer.encode(mention)) - 2
    if t_marker == 2:
        # trigger enclosed by special tense tokens
        assert acronym == 1
        x_sent = x_sent[0:start] + tense_marker + x_sent[start:start+subword_len] + tokenizer.encode(" [/" + tokenizer.decode(tense_marker)[2:])[1:-1] + x_sent[start+subword_len:]
        new_start = start + len(tense_marker)
    elif t_marker == 1:
        # tense enclosed by * *
        x_sent = x_sent[0:start] + [special_1, special_2] + tense_marker + [special_2] + x_sent[start:start+subword_len] + [special_1] + x_sent[start+subword_len:]
        new_start = start + len([special_1, special_2] + tense_marker + [special_2])
    new_end = new_start + subword_len
    offset = len(x_sent) - orig_len
    return x_sent, offset, new_start, new_end

######MAIN#######

mypath_TB = './MATRES_old/TBAQ-cleaned/TimeBank/' # after correction
onlyfiles_TB = [f for f in listdir(mypath_TB) if isfile(join(mypath_TB, f))]
mypath_AQ = './MATRES_old/TBAQ-cleaned/AQUAINT/' 
onlyfiles_AQ = [f for f in listdir(mypath_AQ) if isfile(join(mypath_AQ, f))]
mypath_PL = './MATRES_old/te3-platinum/'
onlyfiles_PL = [f for f in listdir(mypath_PL) if isfile(join(mypath_PL, f))]

# ========================================
#        MATRES: read relation file
# ========================================
# MATRES has separate text files and relation files
# We first read relation files


MATRES_timebank = './MATRES_old/timebank.txt'
MATRES_aquaint = './MATRES_old/aquaint.txt'
MATRES_platinum = './MATRES_old/platinum.txt'
temp_label_map = {"BEFORE": 0, "AFTER": 1, "EQUAL": 2, "VAGUE": 3}
eiid_to_event_trigger = {}
eiid_pair_to_label = {}   

# =========================
#       MATRES Reader
# =========================
def MATRES_READER(matres_file, eiid_to_event_trigger, eiid_pair_to_label):
    with open(matres_file, "r") as f_matres:
        content = f_matres.read().split("\n")
#         assert len(content[-1].split("\t")) == 1
#         content = content[:-1]
        for rel in content:
            rel = rel.split("\t")
            fname = rel[0]
            trigger1 = rel[1]
            trigger2 = rel[2]
            eiid1 = int(rel[3])
            eiid2 = int(rel[4])
            tempRel = temp_label_map[rel[5]]

            if fname not in eiid_to_event_trigger:
                eiid_to_event_trigger[fname] = {}
                eiid_pair_to_label[fname] = {}
            eiid_pair_to_label[fname][(eiid1, eiid2)] = tempRel
            if eiid1 not in eiid_to_event_trigger[fname].keys():
                eiid_to_event_trigger[fname][eiid1] = trigger1
            if eiid2 not in eiid_to_event_trigger[fname].keys():
                eiid_to_event_trigger[fname][eiid2] = trigger2

MATRES_READER(MATRES_timebank, eiid_to_event_trigger, eiid_pair_to_label)
MATRES_READER(MATRES_aquaint, eiid_to_event_trigger, eiid_pair_to_label)
MATRES_READER(MATRES_platinum, eiid_to_event_trigger, eiid_pair_to_label)


mask_in_input_ids = 0 # note that [MASK] is actually learned through pre-training
mask_in_input_mask = 0 # when input is masked through attention, it would be replaced with [PAD]
acronym = 0 # using acronym for tense (e.g., pastsimp): 1; else (e.g., past simple): 0
t_marker = 1


f1_metric = 'micro'
params = {'transformers_model': 'google/bigbird-roberta-large',
          'dataset': 'MATRES',   # 'HiEve', 'IC', 'MATRES' 
          'testdata': 'None', # MATRES / MATRES_nd / TDD / PRED / None; None means training mode
          'block_size': 64,
          'add_loss': 0, 
          'batch_size': 1,    # 6 works on 48G gpu
          'epochs': 40,
          'learning_rate': 5e-6,    # subject to change
          'seed': 0,
          'gpu_id': '11453',    # subject to change
          'debug': 0,
#           'rst_file_name': '0511pm-lr5e-6-b20-gpu9942-loss0-dataMATRES-accum1-marker@**@-pair1-acr0-tmarker1-td1-dpn1-mask0.rst',    # subject to change
          'mask_in_input_ids': mask_in_input_ids,
          'mask_in_input_mask': mask_in_input_mask,
          'marker': 'abc', 
          'tense_acron': 0, # 1 (acronym of tense) or 0 (original tense)
          't_marker': 1, # 2 (trigger enclosed by special tokens) or 1 (tense enclosed by **)
          'td': 1, # 0 (no tense detection) or 1 (tense detection, add tense info)
          'dpn': 1, # 1 if use DPN; else 0
          'lambda_1': -10, # lower bound * 10
          'lambda_2': 11, # upper bound * 10
          'f1_metric': f1_metric, 
         }

if params['testdata'] == 'MATRES_nd':
    params['nd'] = True
else:
    params['nd'] = False
    
###########
# NO MASK #
###########

if params['transformers_model'][-5:] == "large":
    params['emb_size'] = 1024
elif params['transformers_model'][-4:] == "base":
    params['emb_size'] = 768
else:
    print("emb_size is neither 1024 nor 768? ...")
    
set_seed(params['seed'])

print("Processing " + params['dataset'] + " dataset...")
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

debug = params['debug']
if debug:
    params['epochs'] = 1
    
doc_id = -1
features_train = []
features_valid = []
features_test = []
t0 = time.time()
relation_stats = {0: 0, 1: 0, 2: 0, 3: 0}
t_marker = params['t_marker']
# 2: will [futusimp] begin [/futusimp]
# 1: will @ * Future Simple * begin @ 

max_len = 0
sent_num = 0
pair_num = 0
test_labels = []
context_len = {}
timeline_input = []
for fname in tqdm.tqdm(eiid_pair_to_label.keys()):
    file_name = fname + ".tml"
    if file_name in onlyfiles_TB:
        dir_name = mypath_TB
    elif file_name in onlyfiles_AQ:
        dir_name = mypath_AQ
    elif file_name in onlyfiles_PL:
        dir_name = mypath_PL
    else:
        continue

    my_dict = tml_reader(dir_name, file_name, tokenizer) 
    
    for (eiid1, eiid2) in eiid_pair_to_label[fname].keys():
        pair_num += 1
        event_pos = []
        event_pos_end = []
        relations = []
        TokenIDs = [65]
        if eiid1 == 0:
            print('error', fname, eiid1)
            continue
        x = my_dict["eiid_dict"][eiid1]["eID"] # eID
        y = my_dict["eiid_dict"][eiid2]["eID"]
        x_sent_id = my_dict["event_dict"][x]["sent_id"]
        y_sent_id = my_dict["event_dict"][y]["sent_id"]
        reverse = False
        if x_sent_id > y_sent_id:
            reverse = True
            x = my_dict["eiid_dict"][eiid2]["eID"]
            y = my_dict["eiid_dict"][eiid1]["eID"]
            x_sent_id = my_dict["event_dict"][x]["sent_id"]
            y_sent_id = my_dict["event_dict"][y]["sent_id"]
        elif x_sent_id == y_sent_id:
            x_position = my_dict["event_dict"][x]["_subword_id"]
            y_position = my_dict["event_dict"][y]["_subword_id"]
            if x_position > y_position:
                reverse = True
                x = my_dict["eiid_dict"][eiid2]["eID"]
                y = my_dict["eiid_dict"][eiid1]["eID"]
        x_sent = my_dict["sentences"][x_sent_id]["_subword_to_ID"]
        y_sent = my_dict["sentences"][y_sent_id]["_subword_to_ID"]
        # This guarantees that trigger x is always before trigger y in narrative order

        context_start_sent_id = max(x_sent_id-1, 0)
        context_end_sent_id = min(y_sent_id+2, len(my_dict["sentences"]))
        c_len = context_end_sent_id - context_start_sent_id
        if c_len in context_len.keys():
            context_len[c_len] += 1
        else:
            context_len[c_len] = 1
        sent_num += c_len
        
        if params['td'] == 1:
            x_sent, offset_x, new_start_x, new_end_x = add_tense_info(x_sent, my_dict["event_dict"][x]['tense'], my_dict['event_dict'][x]['_subword_id'], my_dict["event_dict"][x]['mention'], 2589, 1736)
        else:
            x_sent, offset_x, new_start_x, new_end_x = x_sent, 0, my_dict['event_dict'][x]['_subword_id'], my_dict['event_dict'][x]['_subword_id'] + len(tokenizer.encode(my_dict["event_dict"][x]['mention'])) - 2
            
        if x_sent_id != y_sent_id:
            if params['td'] == 1:
                y_sent, offset_y, new_start_y, new_end_y = add_tense_info(y_sent, my_dict["event_dict"][y]['tense'], my_dict['event_dict'][y]['_subword_id'], my_dict["event_dict"][y]['mention'], 1404, 5400)
            else:
                y_sent, offset_y, new_start_y, new_end_y = y_sent, 0, my_dict['event_dict'][y]['_subword_id'], my_dict['event_dict'][y]['_subword_id'] + len(tokenizer.encode(my_dict["event_dict"][y]['mention'])) - 2
            for sid in range(context_start_sent_id, context_end_sent_id):
                if sid == x_sent_id:
                    event_pos.append(new_start_x + len(TokenIDs) - 1)
                    event_pos_end.append(new_end_x + len(TokenIDs) - 1)
                    TokenIDs += x_sent[1:]
                elif sid == y_sent_id:
                    event_pos.append(new_start_y + len(TokenIDs) - 1)
                    event_pos_end.append(new_end_y + len(TokenIDs) - 1)
                    TokenIDs += y_sent[1:]
                else:
                    TokenIDs += my_dict["sentences"][sid]["_subword_to_ID"][1:]
        else:
            if params['td'] == 1:
                y_sent, offset_y, new_start_y, new_end_y = add_tense_info(x_sent, my_dict["event_dict"][y]['tense'], my_dict['event_dict'][y]['_subword_id'] + offset_x, my_dict["event_dict"][y]['mention'], 1404, 5400)
            else:
                y_sent, offset_y, new_start_y, new_end_y = y_sent, 0, my_dict['event_dict'][y]['_subword_id'], my_dict['event_dict'][y]['_subword_id'] + len(tokenizer.encode(my_dict["event_dict"][y]['mention'])) - 2
            for sid in range(context_start_sent_id, context_end_sent_id):
                if sid == y_sent_id:
                    event_pos.append(new_start_x + len(TokenIDs) - 1)
                    event_pos_end.append(new_end_x + len(TokenIDs) - 1)
                    event_pos.append(new_start_y + len(TokenIDs) - 1)
                    event_pos_end.append(new_end_y + len(TokenIDs) - 1)
                    TokenIDs += y_sent[1:]
                else:
                    TokenIDs += my_dict["sentences"][sid]["_subword_to_ID"][1:]
                    
        if reverse:
            event_pos = reverse_num(event_pos)
            event_pos_end = reverse_num(event_pos_end)
            
        xy = eiid_pair_to_label[fname][(eiid1, eiid2)]
        
        relations.append(xy)
        relation_stats[xy] += 1
        if len(TokenIDs) > max_len:
            max_len = len(TokenIDs)
        
        if debug or pair_num < 5:
            print("first event of the pair:", tokenizer.decode(TokenIDs[event_pos[0]:event_pos_end[0]]))
            print("second event of the pair:", tokenizer.decode(TokenIDs[event_pos[1]:event_pos_end[1]]))
            print("TokenIDs:", tokenizer.decode(TokenIDs))
        
        if params['nd']:
            syn_0 = replace_with_syn(tokenizer.decode(TokenIDs[event_pos[0]:event_pos_end[0]]))
            syn_1 = replace_with_syn(tokenizer.decode(TokenIDs[event_pos[1]:event_pos_end[1]]))
            if len(syn_0) > 0:
                TokenIDs = TokenIDs[0:event_pos[0]] + tokenizer.encode(syn_0[0])[1:-1] + TokenIDs[event_pos_end[0]:]
                prev = event_pos_end[0]
                event_pos_end[0] = event_pos[0] + len(tokenizer.encode(syn_0[0])[1:-1])
                if prev != event_pos_end[0]:
                    offset = event_pos_end[0] - prev
                    event_pos[1] += offset
                    event_pos_end[1] += offset
            if len(syn_1) > 0:
                TokenIDs = TokenIDs[0:event_pos[1]] + tokenizer.encode(syn_1[0])[1:-1] + TokenIDs[event_pos_end[1]:]
                prev = event_pos_end[1]
                event_pos_end[1] = event_pos[1] + len(tokenizer.encode(syn_1[0])[1:-1])
            #assert 1 == 0
        feature = {'input_ids': TokenIDs,
                   'event_pos': event_pos,
                   'event_pos_end': event_pos_end,
                   'event_pair': [[1, 2]],
                   'labels': relations,
                  }
        if file_name in onlyfiles_TB:
            features_train.append(feature)
        elif file_name in onlyfiles_AQ:
            features_valid.append(feature)
        elif file_name in onlyfiles_PL:
            features_test.append(feature)
            test_labels.append(xy)
            timeline_input.append([fname, x, y, xy])
    if debug:
        break
        
elapsed = format_time(time.time() - t0)
print("MATRES Preprocessing took {:}".format(elapsed)) 
print("Temporal Relation Stats:", relation_stats)
print("Total num of pairs:", pair_num)
print("Max length of context:", max_len)
print("Avg num of sentences that context contains:", sent_num/pair_num)
print("Context length stats(unit: sentence): ", context_len)
print("MATRES train valid test pair num:", len(features_train), len(features_valid), len(features_test))
#with open("MATRES_test_timeline_input.json", 'w') as f:
#    json.dump(timeline_input, f)
#    assert 0 == 1
    
#output_file = open('test_labels.txt', 'w')
#for label in test_labels:
#    output_file.write(str(label) + '\n')
#output_file.close()
#if debug:
#    assert 0 == 1
