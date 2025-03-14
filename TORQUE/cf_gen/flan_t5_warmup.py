import os
import numpy as np
import torch
import random
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

seed = 100
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# sampled_events_by_r = np.load("data/dataset_bias/warmup_cf_events.npy", allow_pickle=True)[()]
sampled_events_by_r = np.load("data/dataset_bias_new/warmup_cf_events_happened.npy", allow_pickle=True)[()]

# model

model_name = 

tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="balanced")

# sampled_events_by_r[r] r in ["future", "happening", "happened"]
# 

r = "happened"

relation_description = {
    "future": "will happen",
    "happened": "has happened",
    "happening": "are happening",
}

from tqdm import tqdm
keys = []
results = []
for event_list in tqdm(sampled_events_by_r[r]):

    events = ", ".join([f'"{e}"' for e in event_list])
    prompt = f"Write a story where {events} {relation_description[r]}:"

    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(do_sample=True, num_return_sequences=10, top_p=0.8, max_length=128, min_length=40, early_stopping=True, **inputs)
    output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    keys.append(("-".join(event_list), r))
    results.append(output)

np.save(f"data/flan_t5_new/warmup_happened_new", list(zip(keys, results)))
