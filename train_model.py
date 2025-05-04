import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
from DCNN import DCNN
from dataset_loader import CustomDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

# Parameters
NUM_EPOCHS = 100
BATCH_SIZE = 32
NUM_WORKERS = 4
K_FOLDS = 5
LEARNING_RATE = 0.001
H_in, W_in = 12, 1280
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_fold(model, train_loader, val_loader, fold):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1).to(DEVICE).float()
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.unsqueeze(1).to(DEVICE).float()
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"[Fold {fold}] Epoch [{epoch+1}/{NUM_EPOCHS}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"best_model_fold_{fold}.pth")

    return train_losses, val_losses

def cross_validate():
    dataset = CustomDataset(data_dir='dataset_5s/train/train_data.npz')
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    all_train_losses, all_val_losses = [], []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n--- Fold {fold + 1}/{K_FOLDS} ---")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        model = DCNN(H_in, W_in).to(DEVICE)
        train_losses, val_losses = train_one_fold(model, train_loader, val_loader, fold + 1)

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

    # Plot averaged loss curves
    avg_train = np.mean([np.array(losses) for losses in all_train_losses], axis=0)
    avg_val = np.mean([np.array(losses) for losses in all_val_losses], axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(avg_train, label='Average Train Loss')
    plt.plot(avg_val, label='Average Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{K_FOLDS}-Fold Cross-Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig("cross_val_loss_plot.png")
    plt.show()

if __name__ == "__main__":
    cross_validate()
