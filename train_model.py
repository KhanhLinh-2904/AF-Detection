import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from DCNN import DCNN
from dataset_loader import CustomDataset
from torch.utils.data import DataLoader

# Directory to store checkpoints
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Early stopping parameters
PATIENCE = 20  # Stop training if no improvement after X epochs

def get_latest_checkpoint():
    """Find the latest checkpoint file."""
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort by epoch number
    return os.path.join(CHECKPOINT_DIR, checkpoints[-1])

# Training function with checkpoint saving
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start_epoch = 0
    best_loss = float("inf")
    best_epoch = 0
    loss_history = []
    no_improve_epochs = 0

    # Load latest checkpoint if available
    latest_checkpoint = get_latest_checkpoint()
    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', best_loss)
        best_epoch = checkpoint.get('best_epoch', best_epoch)
        print(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            # print("input: ", inputs.shape)
            inputs = inputs.unsqueeze(1)  # Ensure correct input shape
            inputs, labels = inputs.to(device).float(), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss,
            'best_epoch': best_epoch
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            no_improve_epochs = 0  # Reset counter
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"New best model saved at epoch {best_epoch} with loss {best_loss:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        # Early stopping condition
        if no_improve_epochs >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}. Best model was at epoch {best_epoch} with loss {best_loss:.4f}.")
            break

    # Visualization of training loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linestyle='-', label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "training_loss.png"))
    plt.show()
    print(f"Training visualization saved as training_loss.png")

# Load data
if __name__ == "__main__":
    # Create an instance of the dataset
    dataset = CustomDataset(data_dir='dataset/train/train_data.npz')

    # Create a DataLoader instance to load the dataset in batches
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # H_in, W_in = 12, 1280
    H_in, W_in = 12, 256
    model = DCNN(H_in, W_in)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, criterion, optimizer, num_epochs=200)