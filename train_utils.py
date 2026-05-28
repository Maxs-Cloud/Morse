import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from zoneinfo import ZoneInfo

from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, DEVICE, CHECKPOINT_BASE_DIR
)
from data_utils import FileMorseDataset, collate_fn
from model import MorseCRNN_v2
from decode_utils import ctc_greedy_decode, decode_to_text, levenshtein_metric
from checkpoint_utils import save_checkpoint, load_checkpoint

def train_model(train_cache_dirs, test_cache_dir, train_csv, test_csv, submission_path,
                char_to_idx, idx_to_char, num_classes, resume_from=None):
    labels_df = pd.read_csv(train_csv)
    train_ids = labels_df['id'].astype(str).tolist()
    train_labels = labels_df['message'].tolist()
    print(f"Всего примеров в CSV: {len(train_ids)}")

    num_train = len(train_ids)
    indices = np.random.permutation(num_train)
    val_size = int(num_train * 0.1)
    valid_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_ids_split = [train_ids[i] for i in train_indices]
    train_lbl_split = [train_labels[i] for i in train_indices]
    valid_ids_split = [train_ids[i] for i in valid_indices]
    valid_lbl_split = [train_labels[i] for i in valid_indices]

    train_dataset = FileMorseDataset(train_cache_dirs, "train", train_ids_split, train_lbl_split,
                                     char_to_idx, augment=True)
    valid_dataset = FileMorseDataset(train_cache_dirs, "train", valid_ids_split, valid_lbl_split,
                                     char_to_idx, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=2)

    test_pt_dir = os.path.join(test_cache_dir, "test")
    test_ids = []
    if os.path.exists(test_pt_dir):
        test_ids = [os.path.splitext(f)[0] for f in os.listdir(test_pt_dir) if f.endswith('.pt')]
        test_ids.sort()
        print(f"Тестовых примеров: {len(test_ids)}")

    model = MorseCRNN_v2(num_classes).to(DEVICE)
    ctc_loss = nn.CTCLoss(blank=0, reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    start_epoch = 0
    train_losses, valid_dists = [], []
    best_valid_dist = float('inf')
    run_timestamp = datetime.now(ZoneInfo('Europe/Moscow')).strftime("%Y%m%d_%H%M")
    current_rundir = os.path.join(CHECKPOINT_BASE_DIR, f'run_{run_timestamp}')

    if resume_from:
        start_epoch, train_losses, valid_dists, best_valid_dist = load_checkpoint(
            resume_from, model, optimizer, scheduler)
        current_rundir = os.path.dirname(resume_from)
    else:
        os.makedirs(current_rundir, exist_ok=True)
        print(f'Новый запуск: {current_rundir}')

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} train")
        for batch in pbar:
            mel_batch, input_lengths, target_concat, target_lengths = [b.to(DEVICE) for b in batch]
            optimizer.zero_grad()
            logits = model(mel_batch)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = ctc_loss(log_probs, target_concat, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        all_pred_texts, all_true_texts = [], []
        with torch.no_grad():
            for batch in valid_loader:
                mel_batch, input_lengths, target_concat, target_lengths = [b.to(DEVICE) for b in batch]
                logits = model(mel_batch)
                log_probs = F.log_softmax(logits, dim=-1)
                for i in range(mel_batch.size(0)):
                    T_cnn = input_lengths[i].item()
                    single_logits = logits[:T_cnn, i, :]
                    decoded = ctc_greedy_decode(single_logits)
                    pred_text = decode_to_text(decoded, idx_to_char)
                    start = sum(target_lengths[:i])
                    end = start + target_lengths[i]
                    true_indices = target_concat[start:end].cpu().tolist()
                    true_text = decode_to_text(true_indices, idx_to_char)
                    all_pred_texts.append(pred_text)
                    all_true_texts.append(true_text)

        valid_dist = levenshtein_metric(all_pred_texts, all_true_texts)
        valid_dists.append(valid_dist)
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, valid_dist={valid_dist:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()

        if valid_dist < best_valid_dist:
            best_valid_dist = valid_dist
            torch.save(model.state_dict(), os.path.join(current_rundir, 'best_model.pt'))

        save_checkpoint(epoch, model, optimizer, scheduler, train_losses, valid_dists, best_valid_dist, current_rundir)

    if test_ids:
        model.load_state_dict(torch.load(os.path.join(current_rundir, 'best_model.pt'), map_location=DEVICE))
        model.eval()
        test_dataset = FileMorseDataset([test_cache_dir], "test", test_ids, None, char_to_idx, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2)
        predictions = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test inference"):
                mel_batch, input_lengths, _, _ = [b.to(DEVICE) for b in batch]
                logits = model(mel_batch)
                log_probs = F.log_softmax(logits, dim=-1).squeeze(1)
                T_cnn = input_lengths[0].item()
                decoded = ctc_greedy_decode(log_probs[:T_cnn])
                pred_text = decode_to_text(decoded, idx_to_char)
                predictions.append(pred_text)
        sub_df = pd.read_csv(submission_path)
        id2pred = dict(zip(test_ids, predictions))
        sub_df['label'] = sub_df['id'].apply(lambda x: id2pred.get(str(x).split('.')[0], ""))
        sub_df.to_csv(os.path.join(current_rundir, "submission_v2.csv"), index=False)
        print(f" Результат сохранён в {os.path.join(current_rundir, 'submission_v2.csv')}")

    return model, train_losses, valid_dists, current_rundir

def plot_morse_history(train_losses, valid_dists, save_path=None):
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(range(1,len(train_losses)+1), train_losses, 'b-o', label='Train Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('CTC Loss')
    ax2 = ax1.twinx()
    ax2.plot(range(1,len(valid_dists)+1), valid_dists, 'r-s', label='Valid Dist')
    ax2.set_ylabel('Levenshtein Distance')
    ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
    plt.title('Morse CRNN v2 Training')
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()