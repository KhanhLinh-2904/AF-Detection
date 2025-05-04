import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from DCNN import DCNN
from dataset_loader import CustomDataset

# Model input dimensions
H_in, W_in = 12, 1280
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(checkpoint_path, H_in, W_in, device):
    """Load model from checkpoint."""
    model = DCNN(H_in, W_in).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def compute_metrics(model, dataloader, device):
    """Return accuracy, sensitivity, and specificity of the model on given dataloader."""
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.unsqueeze(1).to(device).float()
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    TN, FP, FN, TP = cm.ravel()
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    return accuracy, sensitivity, specificity

def print_stats(name, values):
    mean = np.mean(values)
    std = np.std(values)
    print(f"{name}: {mean:.4f} ± {std:.4f}")
    
def main():
    # Load full training and test dataset
    full_train_dataset = CustomDataset(data_dir='dataset_5s/train/train_data.npz')
    test_dataset = CustomDataset(data_dir='dataset_5s/test/test_data.npz')

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # KFold to reconstruct same splits
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    train_accuracies, test_accuracies = [], []
    train_sensitivities, test_sensitivities = [], []
    train_specificities, test_specificities = [], []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_train_dataset)):
        checkpoint_path = f"best_model_fold_{fold+1}.pth"
        if not os.path.exists(checkpoint_path):
            print(f"Missing checkpoint: {checkpoint_path}")
            continue

        print(f"\nEvaluating Fold {fold + 1}")

        model = load_model(checkpoint_path, H_in, W_in, device)

        # Use train subset to measure training accuracy
        train_subset = Subset(full_train_dataset, train_idx)
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=False, num_workers=4)

        train_acc, train_sens, train_spec = compute_metrics(model, train_loader, device)
        test_acc, test_sens, test_spec = compute_metrics(model, test_loader, device)

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        train_sensitivities.append(train_sens)
        test_sensitivities.append(test_sens)
        train_specificities.append(train_spec)
        test_specificities.append(test_spec)


        print(f"Fold {fold+1} - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    # Plot training vs. testing accuracy
    folds = np.arange(1, 6)
    plt.figure(figsize=(8, 5))
    plt.plot(folds, train_accuracies, marker='o', label='Training Accuracy')
    plt.plot(folds, test_accuracies, marker='s', label='Testing Accuracy')
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title("Training vs Testing Accuracy per Fold")
    plt.xticks(folds)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_per_fold.png")
    plt.show()
    
    print("\n=== Average Metrics Across Folds ===")
    print_stats("Train Accuracy", train_accuracies)
    print_stats("Test Accuracy", test_accuracies)
    print_stats("Train Sensitivity", train_sensitivities)
    print_stats("Test Sensitivity", test_sensitivities)
    print_stats("Train Specificity", train_specificities)
    print_stats("Test Specificity", test_specificities)
    


if __name__ == "__main__":
    main()
