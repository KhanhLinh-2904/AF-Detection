import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from DCNN import DCNN
from dataset_loader import CustomDataset, normalize
from torch.utils.data import DataLoader

# Directory to store checkpoints
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

PATIENCE = 20  # Stop if training loss doesn't improve for 20 epochs

def get_latest_checkpoint():
    """Find the latest checkpoint file."""
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort by epoch number
    return os.path.join(CHECKPOINT_DIR, checkpoints[-1])

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start_epoch = 0
    best_train_loss = float("inf")
    best_epoch = 0
    train_losses = []
    val_losses = []
    no_improve_epochs = 0

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    latest_checkpoint = get_latest_checkpoint()
    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        # Optionally resume from previous
        # start_epoch = checkpoint['epoch'] + 1
        # best_train_loss = checkpoint.get('best_train_loss', best_train_loss)
        # best_epoch = checkpoint.get('best_epoch', best_epoch)

    for epoch in range(start_epoch, num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1).to(device).float()
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.unsqueeze(1).to(device).float()
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_train_loss': best_train_loss,
            'best_epoch': best_epoch
        }, checkpoint_path)

        # Track best training model
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_epoch = epoch + 1
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"New best model saved at epoch {best_epoch} with train loss {best_train_loss:.4f}")
        else:
            no_improve_epochs += 1

        # # Early stopping
        # if no_improve_epochs >= PATIENCE:
        #     print(f"Early stopping at epoch {epoch+1}. Best model at epoch {best_epoch} with train loss {best_train_loss:.4f}.")
        #     break

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plot_path = os.path.join(CHECKPOINT_DIR, "loss_plot.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Loss plot saved at {plot_path}")
# Load data
if __name__ == "__main__":
    # Create an instance of the dataset
    train_dataset = CustomDataset(data_dir='dataset/train/train.npz',transform=normalize)
    val_dataset =  CustomDataset(data_dir='dataset/val/val.npz', transform=normalize)
    # Create a DataLoader instance to load the dataset in batches
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=4)
    H_in, W_in = 12, 1280
    # H_in, W_in = 12, 192
    model = DCNN(H_in, W_in)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=200)