import numpy as np
import os
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ht_bias_pairs = np.load("dataset_bias/ht_prob_dict.npy", allow_pickle=True)[()]

model_name = 

tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="balanced")

from tqdm import tqdm
keys = []
results = []

r = "before"

for (h, t), (p_dict, freq) in tqdm(ht_bias_pairs.items()):
    p = p_dict[r]
    if p < 0.3 and freq >= 2:
        e1 = h
        e2 = t
        prompt = f"Write a story where \"{e1}\" happens {r} \"{e2}\":"
        
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        outputs = model.generate(do_sample=True, num_return_sequences=20, top_p=0.8, max_length=80, 
                                 min_length=40, early_stopping=True, **inputs)
        output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        keys.append((h, t, p))
        results.append(output)

os.makedirs("data/flan_t5", exist_ok=True)
np.save('data/flan_t5/before', list(zip(keys, results)))
