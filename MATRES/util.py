import torch
import datetime
import random
import numpy as np
from scipy.stats import entropy

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def collate_fn(batch, mask_in_input_ids, mask_in_input_mask):
    max_len = max([len(f['input_ids']) for f in batch])
    input_ids = [f['input_ids'] + [0] * (max_len - len(f['input_ids'])) for f in batch]
    if mask_in_input_ids:
        input_ids_new = []
        for f_id, f in enumerate(input_ids):
            for event_id, start in enumerate(batch[f_id]['event_pos']):
                end = batch[f_id]['event_pos_end'][event_id]
                for token_id in range(start, end): # needs verification
                    f[token_id] = 67
            input_ids_new.append(f)
        input_ids = input_ids_new
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = [[1.0] * len(f['input_ids']) + [0.0] * (max_len - len(f['input_ids'])) for f in batch]
    if mask_in_input_mask:
        input_mask_new = []
        for f_id, f in enumerate(input_mask):
            for event_id, start in enumerate(batch[f_id]['event_pos']):
                end = batch[f_id]['event_pos_end'][event_id]
                for token_id in range(start, end): # needs verification
                    f[token_id] = 0.0
            input_mask_new.append(f)
        input_mask = input_mask_new
    # Updated on May 17, 2022    
    input_mask_eo = [[0.0] * max_len for f in batch]
    for f_id, f in enumerate(input_mask_eo):
        for event_id, start in enumerate(batch[f_id]['event_pos']):
            end = batch[f_id]['event_pos_end'][event_id]
            for token_id in range(start, end): # needs verification
                f[token_id] = 1.0
    # Updated on Jun 14, 2022
    input_mask_xbar = [[0.0] * max_len for f in batch]
    input_mask_xbar = torch.tensor(input_mask_xbar, dtype=torch.float)
    input_mask_eo = torch.tensor(input_mask_eo, dtype=torch.float)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    event_pos = [f['event_pos'] for f in batch]
    event_pos_end = [f['event_pos_end'] for f in batch]
    event_pair = [f['event_pair'] for f in batch]
    labels = [f['labels'] for f in batch]
    output = (input_ids, input_mask, event_pos, event_pos_end, event_pair, labels, input_mask_eo, input_mask_xbar)
    return output
