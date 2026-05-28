import os
import time
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader, ConcatDataset, SubsetRandomSampler
from torchaudio.functional import edit_distance
from tqdm import tqdm

warnings.filterwarnings("ignore")

# cuda
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.enabled = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")
print(f"TORCH: {torch.__version__}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {total_mem:.1f} GB")


N_MELS = 80
BATCH_SIZE = 24
EPOCHS = 80
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 15
GRAD_CLIP = 1.0
NUM_WORKERS = 2
TIME_DOWNSAMPLE = 2

# Пути
REAL_CACHE_DIR = r"C:\Users\marhr\PycharmProjects\PythonProject9\morse_cache"
SYNTH_CACHE_DIR = r"C:\Users\marhr\PycharmProjects\PythonProject9\synthetic_cache"
COMBINED_CSV = r"C:\Users\marhr\PycharmProjects\PythonProject9\combined_train_clean.csv"
CHECKPOINTS_DIR = r"C:\Users\marhr\PycharmProjects\PythonProject9\morse_checkpoints_v3"

os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

def build_alphabet(csv_path):
    df = pd.read_csv(csv_path)
    chars = set()
    for text in df["message"].astype(str).values:
        chars.update(text)
    alphabet = sorted(list(chars))
    if " " not in alphabet:
        alphabet = [" "] + alphabet
    char_to_idx = {ch: i+1 for i, ch in enumerate(alphabet)}
    idx_to_char = {i+1: ch for i, ch in enumerate(alphabet)}
    idx_to_char[0] = "<blank>"
    return alphabet, char_to_idx, idx_to_char

class DiskCacheDataset(Dataset):
    def __init__(self, cache_dir, ids, labels, char_to_idx):
        self.cache_dir = cache_dir
        valid_ids = []
        valid_labels = []
        skipped = 0
        print(f"Фильтрация коротких примеров в {cache_dir}...")
        for sample_id, lbl in tqdm(zip(ids, labels), total=len(ids), desc="Filtering"):
            file_path = os.path.join(cache_dir, "train", f"{sample_id}.pt")
            try:
                mel = torch.load(file_path, map_location="cpu", weights_only=True)
                T_orig = mel.shape[1]
                T_down = T_orig // TIME_DOWNSAMPLE
                T_out = T_down // 8
                if T_out < 1:
                    T_out = 1
                if T_out >= len(lbl):
                    valid_ids.append(sample_id)
                    valid_labels.append(lbl)
                else:
                    skipped += 1
            except Exception as e:
                print(f"Ошибка загрузки {sample_id}: {e}")
                skipped += 1
        print(f"Оставлено: {len(valid_ids)}, пропущено: {skipped}")
        self.ids = valid_ids
        self.targets = [
            torch.tensor([char_to_idx[c] for c in lbl], dtype=torch.long)
            for lbl in valid_labels
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        file_path = os.path.join(self.cache_dir, "train", f"{sample_id}.pt")
        mel = torch.load(file_path, map_location="cpu", weights_only=True).float()
        mel = mel[:, ::TIME_DOWNSAMPLE]
        return mel, self.targets[idx]

# ---------- Collate ----------
def collate_fn(batch):
    mels, targets = zip(*batch)
    lengths = [m.shape[1] for m in mels]
    max_len = max(lengths)
    padded = torch.zeros(len(mels), N_MELS, max_len)
    for i, mel in enumerate(mels):
        padded[i, :, :mel.shape[1]] = mel
    mel_batch = padded.permute(0, 2, 1)
    targets_cat = torch.cat(targets)
    mel_lengths = torch.tensor(lengths, dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    out_lengths = mel_lengths // 8
    out_lengths = torch.clamp(out_lengths, min=1)
    assert (out_lengths >= target_lengths).all(), \
        f"Нарушение CTC: out_lengths={out_lengths}, target_lengths={target_lengths}"
    return mel_batch, mel_lengths, targets_cat, target_lengths

class FastSpecAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_mask = torchaudio.transforms.TimeMasking(10)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(5)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        if torch.rand(1).item() > 0.5:
            x = self.time_mask(x)
        if torch.rand(1).item() > 0.5:
            x = self.freq_mask(x)
        if torch.rand(1).item() > 0.7:
            x += torch.randn_like(x) * 0.003
        return x.permute(0, 2, 1)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y + x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += identity
        out = F.relu(out)
        return out

# ---------- Позиционное кодирование ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)
    def forward(self, x):
        return x + self.pe[:x.size(0)]


class MorseTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            nn.MaxPool2d(2, 2)
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.proj = nn.Sequential(
            nn.Conv1d(128, 192, 1),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True)
        )
        self.positional = PositionalEncoding(192)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=192, nhead=4, dim_feedforward=1024,
            dropout=0.1, activation="gelu", batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(192, 192),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(192, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_output_lengths(self, lengths):
        for _ in range(3):
            lengths = torch.div(lengths, 2, rounding_mode="floor")
        return torch.clamp(lengths, min=1)

    def forward(self, x, lengths):
        x = x.unsqueeze(1).permute(0, 1, 3, 2)  # [B,1,F,T]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.freq_pool(x)
        x = x.squeeze(2)
        x = self.proj(x)
        x = x.permute(2, 0, 1)
        x = self.positional(x)
        x = self.transformer(x)
        x = self.dropout(x)
        x = self.classifier(x)
        output_lengths = self.get_output_lengths(lengths)
        return x, output_lengths


class GreedyDecoder:
    def __init__(self, blank=0):
        self.blank = blank
    def decode(self, log_probs, idx_to_char):
        pred = torch.argmax(log_probs, dim=-1).cpu().numpy()
        result = []
        prev = self.blank
        for p in pred:
            if p != self.blank and p != prev:
                result.append(idx_to_char.get(p, "?"))
            prev = p
        return "".join(result)

def train():
    df = pd.read_csv(COMBINED_CSV)
    all_ids = df["id"].astype(str).tolist()
    all_messages = df["message"].tolist()
    real_mask = [not x.startswith("synth_") for x in all_ids]
    synth_mask = [x.startswith("synth_") for x in all_ids]
    real_ids = [all_ids[i] for i, m in enumerate(real_mask) if m]
    real_labels = [all_messages[i] for i, m in enumerate(real_mask) if m]
    synth_ids = [all_ids[i] for i, m in enumerate(synth_mask) if m]
    synth_labels = [all_messages[i] for i, m in enumerate(synth_mask) if m]

    print(f"\nREAL: {len(real_ids)}")
    print(f"SYNTH: {len(synth_ids)}")

    alphabet, char_to_idx, idx_to_char = build_alphabet(COMBINED_CSV)
    num_classes = len(alphabet) + 1
    print(f"ALPHABET: {len(alphabet)}")

    real_dataset = DiskCacheDataset(REAL_CACHE_DIR, real_ids, real_labels, char_to_idx)
    synth_dataset = DiskCacheDataset(SYNTH_CACHE_DIR, synth_ids, synth_labels, char_to_idx)
    combined = ConcatDataset([real_dataset, synth_dataset])

    total = len(combined)
    indices = np.random.permutation(total)
    val_size = int(total * 0.1)
    train_indices = indices[val_size:]
    valid_indices = indices[:val_size]

    train_loader = DataLoader(
        combined, batch_size=BATCH_SIZE,
        sampler=SubsetRandomSampler(train_indices),
        collate_fn=collate_fn, num_workers=NUM_WORKERS,
        pin_memory=True, drop_last=True
    )
    valid_loader = DataLoader(
        combined, batch_size=BATCH_SIZE,
        sampler=SubsetRandomSampler(valid_indices),
        collate_fn=collate_fn, num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = MorseTransformer(num_classes).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"\nPARAMS: {params:,}")

    ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    scaler = torch.amp.GradScaler("cuda")
    decoder = GreedyDecoder(blank=0)
    spec_augment = FastSpecAugment()

    run_dir = os.path.join(
        CHECKPOINTS_DIR,
        f"run_{datetime.now(ZoneInfo('Europe/Moscow')).strftime('%Y%m%d_%H%M')}"
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nRUN: {run_dir}")

    start_epoch = 0
    best_dist = float("inf")
    patience = 0

    checkpoint_files = sorted([f for f in os.listdir(CHECKPOINTS_DIR) if f.endswith(".pt")])
    if len(checkpoint_files) > 0:
        print("\nAVAILABLE CHECKPOINTS:\n")
        for i, ckpt in enumerate(checkpoint_files):
            print(f"[{i}] {ckpt}")
        print("\n[-1] Start new training")
        selected = int(input("\nSelect checkpoint: "))
        if selected >= 0:
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, checkpoint_files[selected])
            print(f"\nLOADING: {checkpoint_files[selected]}")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
            start_epoch = checkpoint["epoch"] + 1
            best_dist = checkpoint["best_dist"]
            patience = checkpoint["patience"]
            scheduler.last_epoch = start_epoch - 1  # ← важно!
            print(f"\nRESUMED FROM EPOCH {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for mel_batch, mel_lengths, targets, target_lengths in pbar:
            mel_batch = mel_batch.to(DEVICE, non_blocking=True)
            mel_lengths = mel_lengths.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            target_lengths = target_lengths.to(DEVICE, non_blocking=True)

            mel_batch = spec_augment(mel_batch)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                logits, input_lengths = model(mel_batch, mel_lengths)
                log_probs = F.log_softmax(logits, dim=-1).contiguous()
                loss = ctc_loss(
                    log_probs, targets,
                    input_lengths.cpu(), target_lengths.cpu()
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if pbar.n % 20 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "T": int(input_lengths.max())})

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Валидация
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for mel_batch, mel_lengths, targets, target_lengths in valid_loader:
                mel_batch = mel_batch.to(DEVICE, non_blocking=True)
                mel_lengths = mel_lengths.to(DEVICE, non_blocking=True)
                logits, input_lengths = model(mel_batch, mel_lengths)
                log_probs = F.log_softmax(logits, dim=-1)

                for i in range(mel_batch.size(0)):
                    Ti = input_lengths[i].item()
                    pred = decoder.decode(log_probs[:Ti, i, :], idx_to_char)
                    preds.append(pred)
                    s = torch.sum(target_lengths[:i]).item()
                    true = "".join([
                        idx_to_char.get(idx.item(), "?")
                        for idx in targets[s:s + target_lengths[i].item()]
                    ])
                    trues.append(true)

        valid_dist = np.mean([edit_distance(p, t) for p, t in zip(preds, trues)])
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch+1} | {elapsed:.0f}s | Loss: {avg_loss:.4f} | Dist: {valid_dist:.4f}")
        for i in range(min(3, len(preds))):
            print(f"{trues[i]:<25} -> {preds[i]}")

        checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"epoch_{epoch+1:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_dist": best_dist,
            "patience": patience
        }, checkpoint_path)
        print(f"\nSAVED: {checkpoint_path}")

        if valid_dist < best_dist:
            best_dist = valid_dist
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
            print(f"\nBEST: {best_dist:.4f}")
            patience = 0
        else:
            patience += 1
            print(f"Patience: {patience}/{EARLY_STOP_PATIENCE}")
            if patience >= EARLY_STOP_PATIENCE:
                print("\nEARLY STOPPING")
                break

    print("TRAINING COMPLETE")
    print(f"BEST DISTANCE: {best_dist:.4f}")


if __name__ == "__main__":
    train()