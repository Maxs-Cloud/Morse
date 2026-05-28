import os
from config import (
    TRAIN_CSV, TEST_CSV, SAMPLE_SUBMISSION,
    TRAIN_CACHE_DIRS, TEST_CACHE_DIR, CHECKPOINT_BASE_DIR
)
from data_utils import build_alphabet
from train_utils import train_model, plot_morse_history
from checkpoint_utils import select_checkpoint

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