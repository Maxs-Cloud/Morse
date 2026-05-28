import torch
import numpy as np
from torchaudio.functional import edit_distance

def ctc_greedy_decode(log_probs, blank=0):
    pred = torch.argmax(log_probs, dim=-1)
    decoded = []
    prev = blank
    for idx in pred:
        if idx != blank and idx != prev:
            decoded.append(idx.item())
        prev = idx
    return decoded

def decode_to_text(indices, idx_to_char):
    return ''.join([idx_to_char[i] for i in indices])

def levenshtein_metric(pred_texts, true_texts):
    return np.mean([edit_distance(p, t) for p, t in zip(pred_texts, true_texts)])