import wfdb
import random
import numpy as np
import os
from sklearn.model_selection import train_test_split
from scipy.signal import filtfilt, ellip
from remove_etopic import remove_ectopic_beats
from visualization import plot_ecg_segment

def extractData(ann_atr, ann_qrs, ecg):

    fs = 250
    # Define the segment length in samples (5 seconds per segment)
    segment_length = int(1 * fs)  
    # Calculate the number of segments
    num_segments = len(ecg) // segment_length
    all_segments = []
    all_labels = []
    # print("ecg: ", ecg.shape)
    # print("ground_truth_full: ", len(ground_truth_full))
    print("num_segments: ", num_segments)
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length

        # Extract the ECG segment (both channels)
        segment = ecg[start_idx:end_idx]
        for j in range(len(ann_atr.sample)):
            if start_idx <= ann_atr.sample[j] < end_idx:
                label_note = ann_atr.aux_note[j][1]
                if label_note == "N":
                    label = 0
                else:
                    label = 1
                #    print("start_idx--- atr---- end_idx---- label: ",start_idx, ann.sample[j] , end_idx,label)
                all_segments.append(segment)
                all_labels.append(label)

    return all_segments, all_labels

def elliptical_bandpass_filter(segment, fs=250, order=10, lowcut=0.5, highcut=50):
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
    val_dir = os.path.join(output_dir, "val")
    train_file = os.path.join(train_dir, "train_data.npz")
    test_file = os.path.join(test_dir, "test_data.npz")
    val_file = os.path.join(val_dir, "val_data.npz")

    # Check if data is already processed
    if os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(val_file):
        print("Processed data found. Loading data...")

        train_data = np.load(train_file)
        test_data = np.load(test_file)
        val_data = np.load(val_file)
        X_train, y_train = train_data["all_segments"], train_data["all_labels"]
        X_test, y_test = test_data["all_segments"], test_data["all_labels"]
        X_val, y_val = test_data["all_segments"], test_data["all_labels"]
        print("Data loaded successfully.")
        return X_train, y_train, X_test, y_test, X_val, y_val

    print("Processed data not found. Processing raw ECG data...")

    # Create output directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    

    all_labels = []
    all_segments = []
    record_files = [f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.dat')]
    for record_name in record_files:
        hea_path = os.path.join(data_path, record_name + ".hea")
        atr_path = os.path.join(data_path, record_name + ".atr")
        qrs_path = os.path.join(data_path, record_name + ".qrs")
        
    # ==============================================================================
    
        # Skip if the .hea or .atr file does not exist
        if not os.path.exists(hea_path) or not os.path.exists(atr_path) or not os.path.exists(qrs_path):
            print(f"Skipping {record_name}: Missing .hea or .atr or .qrs file.")
            continue  # Move to the next record
        print("====================================================")
        print(f"Processing record: {record_name}")

        # Read the ECG signal data (both channels)
        record1 = wfdb.rdrecord(os.path.join(data_path, record_name), channels=[0]) 
        # atr contains label AFIB, N
        ann_atr = wfdb.rdann(os.path.join(data_path,record_name), 'atr')  
        # contains index of QRS annotations
        ann_qrs = wfdb.rdann(os.path.join(data_path, record_name), 'qrs')
        ecg1 = record1.p_signal.flatten()
        fs = record1.fs # frequency
        segments, labels = extractData(ann_atr, ann_qrs, ecg1)
        
        all_segments.extend(segments)
        all_labels.extend(labels)
        # print("all_segments: ", all_segments.shape)
        # print("all_labels: ", all_labels.shape)

        
        
        record2 = wfdb.rdrecord(os.path.join(data_path, record_name), channels=[1]) 
        ecg2 = record2.p_signal.flatten()
        segments, labels  = extractData(ann_atr, ann_qrs, ecg2)
        all_segments.extend(segments)
        all_labels.extend(labels)
    # print("all_segments: ", all_segments)
    # print("all_labels: ", all_labels)
        
    # all_segments = np.vstack(all_segments)
    print("all_segments: ", len(all_segments)) 
    print("all_labels: ", len(all_labels) )
    all_AF_labels = []
    all_AF_segments = []
    all_non_labels = []
    all_non_segments = []
    for i in range(len(all_labels)):
        if all_labels[i]:
            all_AF_labels.append(1)
            all_AF_segments.append(all_segments[i])
        else:
            all_non_labels.append(0)
            all_non_segments.append(all_segments[i])
  
    np.savez('af_data.npz',
         all_AF_labels=np.array(all_AF_labels),
         all_AF_segments=np.array(all_AF_segments),
         all_non_labels=np.array(all_non_labels),
         all_non_segments=np.array(all_non_segments))
    
    print("Success!")
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
    # # Lấy random index
    # random_indices = random.sample(range(len(all_non_labels)), total_samples)
    # # Lấy mẫu dựa trên chỉ số
    # new_non_labels = [all_non_labels[i] for i in random_indices]
    # new_non_segments = [all_non_segments [i] for i in random_indices]
    # print(" new_non_labels:  ", len(new_non_labels))
    
    # all_segments.extend(new_non_segments)
    all_segments.extend(all_non_segments)
    all_segments.extend(all_AF_segments)
    # all_labels.extend(new_non_labels)
    all_labels.extend(all_non_labels)
    all_labels.extend(all_AF_labels)
    
    # Convert lists to NumPy arrays
    all_segments = np.array(all_segments)  # Shape: (num_segments, 5*fs, 2)
    all_labels = np.array(all_labels)
    print("all_segments: ", all_segments.shape)
    print("all_labels: ", all_labels.shape)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
    all_segments, all_labels, test_size=0.15, random_state=42, stratify=all_labels)

    # Step 2: Split the remaining into train (70%) and val (15%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )
    # 0.1765 ≈ 15 / 85 to maintain the 70/15/15 split

    # Save to .npz files
    train_file = "dataset/train/train.npz"
    val_file = "dataset/val/val.npz"
    test_file = "dataset/test/test.npz"

    np.savez(train_file, all_segments=X_train, all_labels=y_train)
    print(f"Training data saved in {train_file}")

    np.savez(val_file, all_segments=X_val, all_labels=y_val)
    print(f"Validation data saved in {val_file}")

    np.savez(test_file, all_segments=X_test, all_labels=y_test)
    print(f"Testing data saved in {test_file}")

    return X_train, y_train, X_test, y_test, X_val, y_val

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

    load_or_process_ecg_data()
    read_af_data()
