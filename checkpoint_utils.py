import os
import glob
import torch
from config import DEVICE

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