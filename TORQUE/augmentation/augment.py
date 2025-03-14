# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

from eda import *
import json
import pandas as pd

#nlpaug import
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action

#arguments to be parsed from command line
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
ap.add_argument("--output", required=False, type=str, help="output file of unaugmented data")
ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha_sr", required=False, type=float, help="percent of words in each sentence to be replaced by synonyms")
ap.add_argument("--alpha_ri", required=False, type=float, help="percent of words in each sentence to be inserted")
ap.add_argument("--alpha_rs", required=False, type=float, help="percent of words in each sentence to be swapped")
ap.add_argument("--alpha_rd", required=False, type=float, help="percent of words in each sentence to be deleted")
ap.add_argument("--aug",required = True,type = str, help = "the type of augmentation")
#contextual augment arguments
ap.add_argument("--context_model",required = False,type = str,help = "the model used in contextual augmentation(bert or multilingual bert)")
ap.add_argument("--context_aug_p",required = False,type = float,help = "Should be a number between 0 and 1")
ap.add_argument("--context_aug_action",required = False,type = str,help = "augment mode, substitute or insert")
ap.add_argument("--augment_times",required = False,type = int,help = "how many times we do augmentation for one instance")





args = ap.parse_args()

#the output file
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join
    output = join(dirname(args.input), 'eda_' + basename(args.input))

#number of augmented sentences to generate per original sentence
num_aug = 9 #default
if args.num_aug:
    num_aug = args.num_aug

#how much to replace each word by synonyms
alpha_sr = 0.1#default
if args.alpha_sr is not None:
    alpha_sr = args.alpha_sr

#how much to insert new words that are synonyms
alpha_ri = 0.1#default
if args.alpha_ri is not None:
    alpha_ri = args.alpha_ri

#how much to swap words
alpha_rs = 0.1#default
if args.alpha_rs is not None:
    alpha_rs = args.alpha_rs

#how much to delete words
alpha_rd = 0.1#default
if args.alpha_rd is not None:
    alpha_rd = args.alpha_rd

if alpha_sr == alpha_ri == alpha_rs == alpha_rd == 0:
     ap.error('At least one alpha should be greater than zero')


def check_case(sentence):
    sentence = sentence.replace('personxs','PersonX\'s')
    sentence = sentence.replace('personx','PersonX')
    sentence = sentence.replace('personys','PersonY\'s')
    sentence = sentence.replace('persony','PersonY')
    return sentence

ignore_list = ["[","]","[]","[\"none\"]"]
def gen(tail_list,alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):
    if len(tail_list) == 0 or tail_list in ignore_list: # if the list is emepty
        return tail_list
    augmented = {}
    tail_list = tail_list.replace("[","")
    tail_list = tail_list.replace("]","")
    tail_list = tail_list.replace("\"","")
    tail_list = tail_list.replace("\"","")
    tail_list = tail_list.split(",")
    for tails in tail_list:
        augmented[tails] = []
        if tails == "none" or tails == '':
            continue
        aug_tails =  eda(tails,alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=num_aug) #calculate the augmented data
        augmented[tails] += aug_tails
        augmented[tails] = [check_case(sent) for sent in list(set(augmented[tails]))]
    # augmented = list(set(augmented))
    return json.dumps(augmented)

def gen_contextual(tail_list,model_path,action,augment_times):
    aug = naw.ContextualWordEmbsAug(model_path=model_path, action = action)
    if len(tail_list) == 0 or tail_list in ignore_list: # if the list is emepty
        return tail_list
    augmented = {}
    tail_list = tail_list.replace("[","")
    tail_list = tail_list.replace("]","")
    tail_list = tail_list.replace("\"","")
    tail_list = tail_list.replace("\"","")
    tail_list = tail_list.split(",")
    for tails in tail_list:
        augmented[tails] = []
        if tails == "none" or tails == '':
            continue    
        aug_tails = []
        for time in range(augment_times):
            aug_tails.append(aug.augment(tails))
        augmented[tails] += aug_tails
    augmented[tails] = [aug[0] for aug in augmented[tails]]
    augmented[tails] = [check_case(sent) for sent in list(set(augmented[tails]))]
    return json.dumps(augmented) 

wordembaug_dict = {'word2vec': '../model/GoogleNews-vectors-negative300.bin','glove' : '../model/glove.840B.300d.txt'}
def gen_wordembaug(tail_list,model_type,action,augment_times):
    aug = naw.WordEmbsAug(
    model_type=model_type, model_path=wordembaug_dict[args.aug],
    action=action)
    if len(tail_list) == 0 or tail_list in ignore_list: # if the list is emepty
        return tail_list
    augmented = {}
    tail_list = tail_list.replace("[","")
    tail_list = tail_list.replace("]","")
    tail_list = tail_list.replace("\"","")
    tail_list = tail_list.replace("\"","")
    tail_list = tail_list.split(",")
    for tails in tail_list:
        augmented[tails] = []
        if tails == "none" or tails == '':
            continue    
        aug_tails = []
        for time in range(augment_times):
            aug_tails.append(aug.augment(tails))
        augmented[tails] += aug_tails
    augmented[tails] = [aug[0] for aug in augmented[tails]]
    augmented[tails] = [check_case(sent) for sent in list(set(augmented[tails]))]
    return json.dumps(augmented) 

def gen_synonym(tail_list,augment_times):
    aug = naw.SynonymAug(aug_src = 'wordnet')
    if len(tail_list) == 0 or tail_list in ignore_list: # if the list is emepty
        return tail_list
    augmented = {}
    tail_list = tail_list.replace("[","")
    tail_list = tail_list.replace("]","")
    tail_list = tail_list.replace("\"","")
    tail_list = tail_list.replace("\"","")
    tail_list = tail_list.split(",")
    for tails in tail_list:
        augmented[tails] = []
        if tails == "none" or tails == '':
            continue    
        aug_tails = []
        for time in range(augment_times):
            aug_tails.append(aug.augment(tails))
        augmented[tails] += aug_tails
    augmented[tails] = [aug[0] for aug in augmented[tails]]
    augmented[tails] = [check_case(sent) for sent in list(set(augmented[tails]))]
    return json.dumps(augmented) 

#generate more data with standard augmentation
def gen_eda(train_orig, output_file, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):

    writer = open(output_file, 'w')
    lines = open(train_orig, 'r').readlines()
    data = pd.read_csv(train_orig)
    augmented = data
    augmented['augment_head'] = augmented['instance'].apply(lambda x:gen(x,alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug))
    augmented.to_csv(args.output,index  = False)
    writer.close()
    print('finished')

def gen_contextual_word_embedding(train_orig,output_file,model_path, action, augment_times):
    writer = open(output_file, 'w')
    lines = open(train_orig, 'r').readlines()
    data = pd.read_csv(train_orig)
    augmented = data
    augmented['augment_head'] = augmented['instance'].apply(lambda x:gen_contextual(x,model_path,action,augment_times))
    augmented.to_csv(args.output,index  = False)
    writer.close()
    print('finished')

def gen_word_embedding_aug(train_orig,output_file,model_type,action,augment_times):
    writer = open(output_file, 'w')
    lines = open(train_orig, 'r').readlines()
    data = pd.read_csv(train_orig)
    augmented = data
    augmented['augment_head'] = augmented['instance'].apply(lambda x:gen_wordembaug(x,model_type,action,augment_times))
    augmented.to_csv(args.output,index  = False)
    writer.close()
    print('finished')

def gen_synonym_aug(train_orig,output_file,augment_times):
    writer = open(output_file, 'w')
    lines = open(train_orig, 'r').readlines()
    data = pd.read_csv(train_orig)
    augmented = data
    augmented['augment_head'] = augmented['instance'].apply(lambda x:gen_synonym(x,augment_times))
    augmented.to_csv(args.output,index  = False)
    writer.close()
    print('finished')
def augmentation(args):
    if args.aug == 'eda':
        gen_eda(args.input, args.output, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug=num_aug)
    elif args.aug == 'contextualwordembedding':
        gen_contextual_word_embedding(args.input,args.output,args.context_model,args.context_aug_action,args.augment_times)
    elif args.aug == 'word2vec' or args.aug == 'glove':
        gen_word_embedding_aug(args.input,args.output,args.aug,"substitute",args.augment_times)
    elif args.aug == 'synonym':
        gen_synonym_aug(args.input,args.output,args.augment_times)
    return
#main function
if __name__ == "__main__":

    #generate augmented sentences and output into a new file
    augmentation(args)
    exit(0)