
import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.functional import edit_distance
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from zoneinfo import ZoneInfo
import warnings
warnings.filterwarnings('ignore')

SR = 8000
N_MELS = 80
HOP_LENGTH = 80
WIN_LENGTH = 200
F_MIN = 300
F_MAX = 1500
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Пути
TRAIN_CSV = r"C:\Users\marhr\PycharmProjects\PythonProject9\morse_cache\train_labels.csv"
TEST_CSV = r"C:\Users\marhr\PycharmProjects\PythonProject9\test.csv"
SAMPLE_SUBMISSION = r"C:\Users\marhr\PycharmProjects\PythonProject9\sample_submission.csv"
TRAIN_CACHE_DIRS = r"C:\Users\marhr\PycharmProjects\PythonProject9\morse_cache"
TEST_CACHE_DIR = r"C:\Users\marhr\PycharmProjects\PythonProject9\morse_cache\test"
CHECKPOINT_BASE_DIR = r"C:\Users\marhr\PycharmProjects\PythonProject9\morse_checkpoints_v2"
os.makedirs(CHECKPOINT_BASE_DIR, exist_ok=True)

def build_alphabet(train_csv):
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

# формируем батч
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


class MorseCRNN_v2(nn.Module):
    def __init__(self, num_classes, n_mels=80, hidden_size=256):
        super().__init__()
        self.conv1a = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1b = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.1)

        self.conv2a = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2b = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.1)

        self.conv3a = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3b = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.1)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.LSTM(128, hidden_size, num_layers=3, bidirectional=True,
                           dropout=0.3, batch_first=False)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1).permute(0, 1, 3, 2)  # (B,1,n_mels,T)
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x); x = self.drop1(x)
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x); x = self.drop2(x)
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x); x = self.drop3(x)
        x = self.adaptive_pool(x)                # (B,128,1,T_cnn)
        x = x.squeeze(2).permute(2,0,1)         # (T_cnn,B,128)
        x, _ = self.rnn(x)                      # (T_cnn,B,512)
        logits = self.classifier(x)
        return logits

# декодим
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

# загрузка сохранение
def save_checkpoint(epoch, model, optimizer, scheduler, train_losses, valid_dists, best_dist, checkpoint_dir):
    filename = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1:03d}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'valid_dists': valid_dists,
        'best_dist': best_dist
    }, filename)
    print(f'Чекпоинт сохранён: {filename}')

def load_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    valid_dists = checkpoint['valid_dists']
    best_dist = checkpoint['best_dist']
    print(f'Загружен чекпоинт из epoch {epoch+1}, best valid dist = {best_dist:.4f}')
    return epoch + 1, train_losses, valid_dists, best_dist

def select_checkpoint(base_dir):
    if not os.path.exists(base_dir):
        return None
    runs = sorted(glob.glob(os.path.join(base_dir, 'run_*')))
    if not runs:
        return None
    print("\nДоступные запуски:")
    for i, run in enumerate(runs):
        run_name = os.path.basename(run)
        checkpoints = sorted(glob.glob(os.path.join(run, 'checkpoint_epoch_*.pth')))
        if checkpoints:
            epochs = [int(os.path.basename(cp).split('_epoch_')[1].split('.')[0]) for cp in checkpoints]
            print(f"{i+1}. {run_name} (эпохи: {min(epochs)}-{max(epochs)})")
        else:
            print(f"{i+1}. {run_name} (нет чекпоинтов)")
    choice = input("Введите номер запуска (0 - новое обучение): ").strip()
    if choice == '0':
        return None
    try:
        idx = int(choice) - 1
        selected_run = runs[idx]
    except:
        return None
    checkpoints = sorted(glob.glob(os.path.join(selected_run, 'checkpoint_epoch_*.pth')))
    if not checkpoints:
        return None
    epochs = [int(os.path.basename(cp).split('_epoch_')[1].split('.')[0]) for cp in checkpoints]
    epoch_choice = input(f"Введите номер эпохи (Enter для последней {max(epochs)}): ").strip()
    if not epoch_choice:
        return checkpoints[-1]
    try:
        epoch_num = int(epoch_choice)
        for cp in checkpoints:
            if int(os.path.basename(cp).split('_epoch_')[1].split('.')[0]) == epoch_num:
                return cp
    except:
        pass
    return checkpoints[-1]

def train_model(train_cache_dirs, test_cache_dir, train_csv, test_csv, submission_path,
                char_to_idx, idx_to_char, num_classes, resume_from=None):
    # Загружаем метки из combined_train_clean.csv
    labels_df = pd.read_csv(train_csv)
    train_ids = labels_df['id'].astype(str).tolist()
    train_labels = labels_df['message'].tolist()
    print(f"Всего примеров в CSV: {len(train_ids)}")

    # train/valid split
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

    # Тест
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

        # Validation
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

    # Инференс
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

# ------------------------------ 9. График ------------------------------
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

if __name__ == "__main__":
    print("Построение алфавита")
    alphabet, char_to_idx, idx_to_char = build_alphabet(TRAIN_CSV)
    num_classes = len(alphabet) + 1
    print(f"Алфавит ({len(alphabet)} символов): {alphabet}")

    resume_path = select_checkpoint(CHECKPOINT_BASE_DIR)

    model, train_losses, valid_dists, run_folder = train_model(
        TRAIN_CACHE_DIRS, TEST_CACHE_DIR, TRAIN_CSV, TEST_CSV, SAMPLE_SUBMISSION,
        char_to_idx, idx_to_char, num_classes, resume_from=resume_path)

    plot_morse_history(train_losses, valid_dists,
                       save_path=os.path.join(run_folder, 'training_plot.png'))