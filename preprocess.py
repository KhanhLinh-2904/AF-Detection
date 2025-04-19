import wfdb
import random
import numpy as np
import os
from sklearn.model_selection import train_test_split
from scipy.signal import filtfilt, ellip
from hr_analysis import remove_ectopic_beats, define_AF_or_none

def elliptical_bandpass_filter(segment, fs, order=10, lowcut=0.5, highcut=50):
    """
    Apply an elliptical bandpass filter to an ECG segment.
    
    Parameters:
        segment (np.ndarray): Shape (N,) or (N, C) where C is number of channels
        fs (int or float): Sampling frequency
        order (int): Filter order
        lowcut (float): Lower bound of passband (Hz)
        highcut (float): Upper bound of passband (Hz)
        
    Returns:
        np.ndarray: Filtered segment
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Elliptical filter: 1 dB ripple, 40 dB attenuation
    b, a = ellip(order, rp=1, rs=40, Wn=[low, high], btype='band')
    return filtfilt(b, a, segment, axis=0)

def load_or_process_ecg_data(data_path="mit-bih-atrial-fibrillation-database-1.0.0/", output_dir="dataset_128"):
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
    all_AF_labels = []
    all_non_labels = []
    all_AF_segments = []
    all_non_segments = []
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
        ecg1 = ecg_signal[:, 0]
        ecg2 = ecg_signal[:, 1]
        
        fs = record[1]['fs']  # Sampling frequency


        segment_length = 128  
        
        #Remove ectopic beats
        ecg1_clean = remove_ectopic_beats(ecg1)
        ecg2_clean = remove_ectopic_beats(ecg2)
        
        
        # Calculate the number of segments
        num_segments = len(ecg1_clean) // segment_length
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length

            # Extract the ECG segment (both channels)
            segment = ecg1_clean[start_idx:end_idx]  # Shape: (5*fs, 2)
            label = define_AF_or_none(segment)
            if label == 1:
                elliptical_bandpass_filter(segment, fs)
                all_AF_labels.append(label)
                all_AF_segments.append(segment)
            else:
                elliptical_bandpass_filter(segment, fs)
                all_non_labels.append(label)
                all_non_segments.append(segment)
            
        # Calculate the number of segments
        num_segments = len(ecg2_clean) // segment_length
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length

            # Extract the ECG segment (both channels)
            segment = ecg2_clean[start_idx:end_idx]  # Shape: (5*fs, 2)
            all_segments.append(segment)
            label = define_AF_or_none(segment)
            if label == 1:
                all_AF_labels.append(label)
                all_AF_segments.append(segment)
            else:
                all_non_labels.append(label)
                all_non_segments.append(segment)
    # Lưu dưới dạng .npz
    np.savez('af_data.npz',
         all_AF_labels=np.array(all_AF_labels),
         all_AF_segments=np.array(all_AF_segments),
         all_non_labels=np.array(all_non_labels),
         all_non_segments=np.array(all_non_segments))
    # Xử lý file af_data.npz
def read_af_data():
    all_segments = []
    all_labels = []
    data_dir = "af_data.npz"
    data = np.load(data_dir)
    all_AF_labels = data["all_AF_labels"]
    all_AF_segments = data["all_AF_segments"]
    all_non_labels = data["all_non_labels"]
    all_non_segments = data["all_non_segments"]
    print("all_AF_labels: ", len(all_AF_labels))   
    print("all_AF_segments: ", len(all_AF_segments))        
    print("all_non_labels: ", len(all_non_labels))   
    print("all_non_segments: ", len(all_non_segments))   
    total_samples = len(all_AF_labels) 
    print("total_samples: ", total_samples)       
    # Lấy random index
    random_indices = random.sample(range(len(all_non_labels)), total_samples)
    # Lấy mẫu dựa trên chỉ số
    new_non_labels = [all_non_labels[i] for i in random_indices]
    new_non_segments = [all_non_segments [i] for i in random_indices]
    print(" new_non_labels:  ", len(new_non_labels))
    
    all_segments.extend(new_non_segments)
    all_segments.extend(all_AF_segments)
    all_labels.extend(new_non_labels)
    all_labels.extend(all_AF_labels)
    
    # Convert lists to NumPy arrays
    all_segments = np.array(all_segments)  # Shape: (num_segments, 5*fs, 2)
    all_labels = np.array(all_labels)
    print("all_segments: ", all_segments.shape)
    print("all_labels: ", all_labels.shape)
    
    train_file = "dataset_128/train/train.npz"
    test_file = "dataset_128/test/test.npz"
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
    # X_train, y_train, X_test, y_test = load_or_process_ecg_data()
    # print("X_train: ", X_train.shape)
    # num_N = [i for i in y_train if i ==0]
    # num_A = [i for i in y_train if i ==1]

    # print("N train:",len(num_N))
    # print("A train:",len(num_A))

    # print("X_test: ", X_test.shape)
    # num_N = [i for i in y_test if i ==0]
    # num_A = [i for i in y_test if i ==1]

    # print("N test:",len(num_N))
    # print("A test:",len(num_A))

    # load_or_process_ecg_data()
    read_af_data()
