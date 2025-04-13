import wfdb
import numpy as np
import os
from sklearn.model_selection import train_test_split
from scipy.signal import filtfilt, ellip
from hr_analysis import classify_af

def bandpass_filter(ecg_signal, fs, lowcut=0.5, highcut=50, order=10):
    """Apply an elliptical band-pass filter to the ECG signal."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = ellip(order, 0.1, 40, [low, high], btype='band')
    return filtfilt(b, a, ecg_signal, axis=0)


def load_or_process_ecg_data(data_path="mit-bih-atrial-fibrillation-database-1.0.0/", output_dir="dataset"):
    """Load ECG data if already processed, otherwise process and store it.

    Args:
        data_path (str): Path to the directory containing raw ECG data.
        output_dir (str): Path to the directory where processed data is stored.

    Returns:
        tuple: (X_train, y_train, X_test, y_test) as NumPy arrays.
    """
    # Define train/test folders
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    train_file = os.path.join(train_dir, "train_data.npz")
    test_file = os.path.join(test_dir, "test_data.npz")

    # Check if data is already processed
    if os.path.exists(train_file) and os.path.exists(test_file):
        print("Processed data found. Loading data...")

        train_data = np.load(train_file)
        test_data = np.load(test_file)

        X_train, y_train = train_data["all_segments"], train_data["all_labels"]
        X_test, y_test = test_data["all_segments"], test_data["all_labels"]

        print("Data loaded successfully.")
        return X_train, y_train, X_test, y_test

    print("Processed data not found. Processing raw ECG data...")

    # Create output directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get all record names that have .dat files
    record_files = [f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.dat')]

    # Initialize lists to store segments and labels
    all_segments = []
    all_labels = []

    # Process each record
    for record_name in record_files:
        hea_path = os.path.join(data_path, record_name + ".hea")
        atr_path = os.path.join(data_path, record_name + ".atr")

        # Skip if the .hea or .atr file does not exist
        if not os.path.exists(hea_path) or not os.path.exists(atr_path):
            print(f"Skipping {record_name}: Missing .hea or .atr file.")
            continue  # Move to the next record

        print(f"Processing record: {record_name}")

        # Read the ECG signal data (both channels)
        record = wfdb.rdsamp(os.path.join(data_path, record_name))  
        ann = wfdb.rdann(os.path.join(data_path, record_name), 'atr')  

        # Extract both ECG signal channels
        ecg_signal = record[0]  # Shape: (samples, 2) for two channels
        fs = record[1]['fs']  # Sampling frequency

        # Define the segment length in samples (5 seconds per segment)
        segment_length = int(1 * fs)  
        # Calculate the number of segments
        num_segments = len(ecg_signal) // segment_length
        
        print("ecg_signal len: ", len(ecg_signal))
        print("segment_length: ", segment_length)
        print("num_segments: ", num_segments)

        print(f"Total segments in {record_name}: {num_segments}")

        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length

            # Extract the ECG segment (both channels)
            segment = ecg_signal[start_idx:end_idx, :]  # Shape: (5*fs, 2)
            
            for j in range(len(ann.sample)):
                if start_idx <= ann.sample[j] < end_idx:
                    label_note = ann.aux_note[j][1]
                    if label_note == "N":
                        label = 0
                    else:
                        label = 1
                    #    print("start_idx--- atr---- end_idx---- label: ",start_idx, ann.sample[j] , end_idx,label)
                    all_segments.append(segment[:, 0])
                    all_segments.append(segment[:, 1])
                    all_labels.append(label)
                    all_labels.append(label)
            
    # Convert lists to NumPy arrays
    all_segments = np.array(all_segments)  # Shape: (num_segments, 5*fs, 2)
    all_labels = np.array(all_labels)

    # Split data into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(all_segments, all_labels, test_size=0.2, random_state=42)

    # Save training data
    np.savez(train_file, all_segments=X_train, all_labels=y_train)
    print(f"Training data saved in {train_file}")

    # Save testing data
    np.savez(test_file, all_segments=X_test, all_labels=y_test)
    print(f"Testing data saved in {test_file}")

    return X_train, y_train, X_test, y_test

# Call the function
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_or_process_ecg_data()
    print("X_train: ", X_train.shape)
    num_N = [i for i in y_train if i ==0]
    num_A = [i for i in y_train if i ==1]

    print("N train:",len(num_N))
    print("A train:",len(num_A))

    print("X_test: ", X_test.shape)
    num_N = [i for i in y_test if i ==0]
    num_A = [i for i in y_test if i ==1]

    print("N test:",len(num_N))
    print("A test:",len(num_A))

