import os
import torch
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

TRAIN_CSV = r"C:\Users\marhr\PycharmProjects\PythonProject9\morse_cache\train_labels.csv"
TEST_CSV = r"C:\Users\marhr\PycharmProjects\PythonProject9\test.csv"
SAMPLE_SUBMISSION = r"C:\Users\marhr\PycharmProjects\PythonProject9\sample_submission.csv"
TRAIN_CACHE_DIRS = r"C:\Users\marhr\PycharmProjects\PythonProject9\morse_cache"
TEST_CACHE_DIR = r"C:\Users\marhr\PycharmProjects\PythonProject9\morse_cache\test"
CHECKPOINT_BASE_DIR = r"C:\Users\marhr\PycharmProjects\PythonProject9\morse_checkpoints_v2"
os.makedirs(CHECKPOINT_BASE_DIR, exist_ok=True)