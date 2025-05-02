import os
import torch
import matplotlib.pyplot as plt
import re
from torch.utils.data import DataLoader
from DCNN import DCNN
from dataset_loader import CustomDataset
from performance import evaluate_model
import numpy as np
def load_model(checkpoint_path, H_in, W_in, device):
    """Load the best model from checkpoint."""
    model = DCNN(H_in, W_in).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])  # 🟢 Only load model weights
    model.eval()  # Set model to evaluation mode
    return model

def load_accuracy(folder_path,dataset_128):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load test dataset
    get_dataset = CustomDataset(data_dir=dataset_128)  # Update with actual test dataset
    get_loader = DataLoader(get_dataset, batch_size=32, shuffle=False, num_workers=4)
    H_in, W_in = 12, 192
    list_accs = []
    epochs = []

    for filename in os.listdir(folder_path):
        if filename.startswith("epoch_") and filename.endswith(".pth"):
            match = re.match(r"epoch_(\d+)\.pth", filename)
            if match:
                epoch_num = int(match.group(1))
                checkpoint_path = os.path.join(folder_path, filename)
                print("checkpoint path: ", checkpoint_path)
                model = load_model(checkpoint_path, H_in, W_in, device)

                # Calculate training accuracy
                _,_, accuracy = evaluate_model(model, get_loader, device)
                # print(f"Epoch {epoch_num}: Train Accuracy = {accuracy:.4f}")

                epochs.append(epoch_num)
                list_accs.append(accuracy)

    # Sort by epoch number
    sorted_data = sorted(zip(epochs, list_accs))
    epochs, list_accs = zip(*sorted_data)
    np.savez("accuracy_data.npz", epochs=np.array(epochs), accuracies=np.array(list_accs))
    print(f"Saved to accuracy_data.npz")
    return epochs, list_accs

def load_accuracy_npz(file_path="accuracy_data.npz"):
    data = np.load(file_path)
    epochs = data["epochs"]
    accuracies = data["accuracies"]
    return epochs, accuracies

def load_test_performance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = CustomDataset(data_dir='dataset_128/test/test.npz')  # Update with actual test dataset
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    H_in, W_in = 12, 192
    CHECKPOINT_PATH = "checkpoints_128/best_model.pth"
    # Load best model checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        model = DCNN(H_in, W_in).to(device)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        model.eval()  # Set model to evaluation mode
        sensitivity, specificity,accuracy = evaluate_model(model, test_loader, device)
        return sensitivity, specificity,accuracy
    else:
        print(f"Checkpoint not found at {CHECKPOINT_PATH}")
        
def plot_training_accuracy(epochs, train_accs, test_accuracy):
    # plt.figure(figsize=(8, 5))
    # plt.plot(epochs, train_accs, marker='o', label='Training Accuracy', color='blue')
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.title("Training Accuracy Over Epochs")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accs, label='Training Accuracy')
    plt.axhline(y=test_accuracy, color='r', linestyle='--', label='Test Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ =="__main__":
    # Path to checkpoints
    folder_path = "checkpoints_128/"
    dataset_128 = "dataset_128/train/train.npz"
    epochs, train_accs = load_accuracy_npz()
    sensitivity, specificity,accuracy = load_test_performance()
    print("sensitivity, specificity,accuracy: ", sensitivity, specificity,accuracy)
    plot_training_accuracy(epochs, train_accs, accuracy)
