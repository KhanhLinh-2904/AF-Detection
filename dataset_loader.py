from SWT import compute_swt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_dir="test/test_data.npz", transform=None):
        """
        Args:
            data_dir (str): Path to the dataset directory containing the npz file with 'all_segments' and 'all_labels'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load the npz file containing 'all_segments' and 'all_labels'
        data = np.load(data_dir)
        print(data.files)  # Print the keys in the .npz file to understand its structure
        
        # Assuming 'all_segments' and 'all_labels' are the correct keys
        self.ecg_segments = data['all_segments']  # Update this if needed based on the printed keys
        self.labels = data['all_labels']  # Update this if needed based on the printed keys
        
    def __len__(self):
        # Return the number of samples in the dataset (number of ECG segments)
        return len(self.ecg_segments)   
    
    def __getitem__(self, idx):
        # Get the ECG segment at the given index
        ecg_segment = self.ecg_segments[idx]
        
        # Apply the SWT transformation to the ECG segment
        transformed_ecg_segment = compute_swt(ecg_segment)
        
        # Get the corresponding label for the segment
        label = self.labels[idx]
        
        # If a transform is provided, apply it (e.g., normalization)
        if self.transform:
            transformed_ecg_segment = self.transform(transformed_ecg_segment)
        
        # Return the transformed ECG segment and the corresponding label
        return (transformed_ecg_segment, label)

if __name__ == "__main__":
    # # Create an instance of the dataset
    # dataset = CustomDataset(data_dir='dataset_128/test/test.npz')

    # # Create a DataLoader instance to load the dataset in batches
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

    # # Iterate through the data
    # for batch_idx, (data, labels) in enumerate(data_loader):
    #     # Print the shapes of data and labels in each batch
    #     print(f"Batch {batch_idx + 1}: Data shape: {data.shape}, Labels shape: {labels.shape}")
    
    # Load the dataset
    data = np.load('dataset/train/train_data.npz')

    # Check the keys inside the file
    print("Keys in npz file:", data.files)

    # Assuming the data is stored under 'all_segments' and 'all_labels'
    segments = data['all_segments']
    labels = data['all_labels']

    # Print the number of instances
    print("Number of instances (samples):", len(labels))  # or len(segments), should be the same


