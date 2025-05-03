import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import os

from DCNN import DCNN
from dataset_loader import CustomDataset

# Path to the best checkpoint
CHECKPOINT_PATH = "checkpoints/best_model.pth"

def load_model(checkpoint_path, H_in, W_in, device):
    """Load the best model from checkpoint."""
    model = DCNN(H_in, W_in).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    return model

def evaluate_model(model, test_loader, device):
    """Evaluate model on test data and compute Sensitivity & Specificity."""
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.unsqueeze(1).to(device).float()
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Get predicted class

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    TN, FP, FN, TP = cm.ravel()  # True Negative, False Positive, False Negative, True Positive

    # Compute Sensitivity (Recall) and Specificity
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    # Compute Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")

    return sensitivity, specificity, accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test dataset
    test_dataset = CustomDataset(data_dir='dataset_5s/test/test_data.npz')  # Update with actual test dataset
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Model input size (same as training)
    # H_in, W_in = 12, 1280
    H_in, W_in = 12, 256

    # Load best model checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        model = load_model(CHECKPOINT_PATH, H_in, W_in, device)
        sensitivity, specificity, accuracy = evaluate_model(model, test_loader, device)
        print("sensitivity, specificity, accuracy: ", sensitivity, specificity, accuracy)
    else:
        print(f"Checkpoint not found at {CHECKPOINT_PATH}")
