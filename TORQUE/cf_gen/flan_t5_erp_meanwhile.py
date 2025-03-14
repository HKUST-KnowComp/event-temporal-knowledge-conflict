import os
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ht_bias_dict = np.load("data/dataset_bias_new/erp_bias_dict.npy", allow_pickle=True)[()]

# model

model_name = 

tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="balanced")

from tqdm import tqdm
keys = []
results = []

for (h, t), (p_dict, freq) in tqdm(ht_bias_dict.items()):
    if p_dict["meanwhile"] < 0.25 and freq >= 3 and p_dict["meanwhile"] > 0.0:
        # e1 and e2 happens in the meantime is less frequent
        e1 = h
        e2 = t
        r  = "during"
        prompt = f"Write a story where \"{e1}\" and \"{e2}\" happen in the meantime:"
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = model.generate(do_sample=True, num_return_sequences=30, top_p=0.8, max_length=110, min_length=40, early_stopping=True, **inputs)
        output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        keys.append((h, t, p_dict["meanwhile"]))
        results.append(output)

np.save('data/flan_t5_new/meanwhile_30', list(zip(keys, results)))
