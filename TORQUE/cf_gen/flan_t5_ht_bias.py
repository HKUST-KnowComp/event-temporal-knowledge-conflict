import os
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ht_bias_dict = np.load("data/dataset_bias/ht_bias_dict.npy", allow_pickle=True)[()]

# model

model_name = 

tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="balanced")

# ht_bias_dict[(h, r)] = #(h, "what happend before h?", t) / (#(h, "what happend before h?", t) + #(h, "what happend after h?", t))

from tqdm import tqdm
keys = []
results = []
for (h, t), p in tqdm(ht_bias_dict.items()):
    if p > 0.6:
    # if p < 0.4:
        e1 = h
        e2 = t
        # frequent pattern is what happend before h? t (h, is after, t)
        r  = "before"
        # r  = "after"
        prompt = f"Write a story where \"{e1}\" happens {r} \"{e2}\":"
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = model.generate(do_sample=True, num_return_sequences=10, top_p=0.8, max_length=128, min_length=40, early_stopping=True, **inputs)
        output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        keys.append((h, t, p))
        results.append(output)

np.save('data/flan-t5/before_min_40_num_10_p_0.8', list(zip(keys, results)))
# np.save('data/flan-t5/after_min_40_num_10_p_0.8', list(zip(keys, results)))
