import os
import openai
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report

openai.api_key = os.getenv("OPENAI_API_KEY")

import json
import numpy as np

np.random.seed(0)
features_valid = json.load(open("../data/valid_subset_text.json"))
features_valid_erp = json.load(open("../dataset_bias/valid_subset_text_erp.json"))
features_valid_tense_all = json.load(open("../dataset_bias/valid_subset_text_tense_bias_vague.json"))

# features_valid_tense_vague = [item for item in features_valid_tense_all if item['labels'][0] == 3]
# features_valid_dep = json.load(open("../dataset_bias/valid_text_features_matres_dep_bias.json"))

examplar_0 = "".join(open("prompts/chan_prompt_icl_examplar_1_shot_0.txt").readlines())

oneshot_results_0 = []
for i in tqdm(range(len(features_valid))):
    item = features_valid[i]
    context = item['text'].replace("[CLS]", "").replace("[SEP]", "").strip()
    e1 = item['e1']
    e2 = item['e2']
    
    prompt = f"Determine the temporal order from \"{e1}\" to \"{e2}\" in the following sentence: \"{context}\". Only answer one word from AFTER, BEFORE, EQUAL, VAGUE. Answer:"
#     print(prompt)
    while True:
        try:
            oneshot_results_0.append(openai.Completion.create(
                        model="text-davinci-003",
                        prompt=examplar_0 + "\n\n" + prompt,
                        max_tokens=20,
                        temperature=0
            ))
            break
        except:
            time.sleep(10)
    time.sleep(2)
    break
