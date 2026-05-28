import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

def build_alphabet(train_csv):
    import pandas as pd
    df = pd.read_csv(train_csv)
    texts = df['message'].values
    chars = set()
    for t in texts:
        chars.update(t)
    alphabet = sorted(list(chars))
    if ' ' not in alphabet:
        alphabet = [' '] + alphabet
    char_to_idx = {ch: i+1 for i, ch in enumerate(alphabet)}
    idx_to_char = {i+1: ch for i, ch in enumerate(alphabet)}
    idx_to_char[0] = '<blank>'
    return alphabet, char_to_idx, idx_to_char

def spec_augment(mel_spec, freq_mask_param=15, time_mask_param=40, noise_level=0.02):
    freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
    mel_spec = freq_mask(mel_spec.unsqueeze(0)).squeeze(0)
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
    mel_spec = time_mask(mel_spec.unsqueeze(0)).squeeze(0)
    mel_spec += torch.randn_like(mel_spec) * noise_level
    return mel_spec

class FileMorseDataset(Dataset):
    def __init__(self, cache_dirs, subset, ids, labels, char_to_idx, augment=False):
        self.cache_dirs = cache_dirs if isinstance(cache_dirs, list) else [cache_dirs]
        self.subset = subset
        self.ids = ids
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.augment = augment

        self.id_to_path = {}
        for file_id in self.ids:
            found = False
            for base_dir in self.cache_dirs:
                candidate = os.path.join(base_dir, self.subset, f"{file_id}.pt")
                if os.path.exists(candidate):
                    self.id_to_path[file_id] = candidate
                    found = True
                    break
            if not found:
                print(f" Файл {file_id}.pt не найден в {self.cache_dirs}")

        self.ids = [fid for fid in self.ids if fid in self.id_to_path]
        if self.labels is not None:
            id2label = {id_: lbl for id_, lbl in zip(ids, labels)}
            self.labels = [id2label[fid] for fid in self.ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        file_id = self.ids[idx]
        file_path = self.id_to_path[file_id]
        log_mel = torch.load(file_path, weights_only=True)
        if self.augment:
            log_mel = spec_augment(log_mel)
        if self.labels is not None:
            target = torch.tensor([self.char_to_idx[c] for c in self.labels[idx]], dtype=torch.long)
        else:
            target = torch.tensor([], dtype=torch.long)
        return log_mel, target

def collate_fn(batch):
    mels, targets = zip(*batch)
    mel_lengths = [m.shape[1] for m in mels]
    max_len = max(mel_lengths)
    padded_mels = [F.pad(m, (0, max_len - m.shape[1])) for m in mels]
    mel_batch = torch.stack(padded_mels)
    mel_batch = mel_batch.permute(0, 2, 1)
    targets_concat = torch.cat(targets, dim=0)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    input_lengths = torch.tensor([(l // 8) for l in mel_lengths], dtype=torch.long)
    input_lengths = torch.clamp(input_lengths, min=1)
    return mel_batch, input_lengths, targets_concat, target_lengths