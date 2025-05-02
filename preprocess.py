import wfdb
import random
import numpy as np
import os
from sklearn.model_selection import train_test_split
from scipy.signal import filtfilt, ellip
from remove_etopic import remove_ectopic_beats
from visualization import plot_ecg_segment

def segment_rr_intervals_with_labels(rr_intervals, ground_truth, target_sum=5.0, tolerance=1e-2):
    rr_segments = []
    label_segments = []

    current_rr_segment = []
    current_label_segment = []
    current_sum = 0.0

    for rr, label in zip(rr_intervals, ground_truth):
        print("rr: ", rr)
        if current_sum + rr <= target_sum + tolerance:
            current_rr_segment.append(rr)
            current_label_segment.append(label)
            current_sum += rr
        else:
            if current_rr_segment:
                plot_ecg_segment(current_rr_segment, "N")
                rr_segments.append(current_rr_segment)
                label_segments.append(current_label_segment)

            current_rr_segment = [rr]
            current_label_segment = [label]
            current_sum = rr

    if current_rr_segment:
        rr_segments.append(current_rr_segment)
        label_segments.append(current_label_segment)

    return rr_segments, label_segments

def extractData(ann_atr, ann_qrs, ecg):
    # ann.sample contains the indices of the QRS annotations
    qrs_index = ann_qrs.sample
    max_indices = []
    for i in range(len(qrs_index) - 1):
        if qrs_index[i] > len(ecg) or qrs_index[i + 1] > len(ecg):
            break
        start_idx = qrs_index[i]
        end_idx = qrs_index[i + 1]
        if start_idx < 0:
            start_idx = 0
        segment = ecg[start_idx:end_idx]
        maximum_idx = np.argmax(segment)
        max_indices.append(start_idx + maximum_idx)
    fs = 250
    max_indices = np.array(max_indices)
    r_peaks = ecg[max_indices]
    RR_intervals = np.diff(max_indices) / fs
    RR_intervals_remove_etopic, kept_indices = remove_ectopic_beats(RR_intervals, return_indices=True)
    
    ground_truth_full = np.zeros(len(RR_intervals), dtype=int)
    ground_truth_index = ann_atr.sample
    for i in range(len(ground_truth_index) - 1):
        ground_idx1 = np.searchsorted(max_indices, ground_truth_index[i], side='right')
        ground_idx2 = np.searchsorted(max_indices, ground_truth_index[i + 1], side='right')
        label = ann_atr.aux_note[i].replace('(', '').replace(')', '')
        if label == 'AFIB':
            ground_truth_full[ground_idx1:ground_idx2] = 1
        else:
            ground_truth_full[ground_idx1:ground_idx2] = 0
    ground_truth_filtered = ground_truth_full[kept_indices]

    segment, label_segment = segment_rr_intervals_with_labels(RR_intervals_remove_etopic, ground_truth_filtered)
    # print("segment: ", len(segment))
    # print("****: ", label_segment)
    # print("label_segment: ", len(label_segment))
    for i in range(len(label_segment)):
        small_label = label_segment[i]
        if  np.sum(small_label) >= (len(small_label)/2):
            label_segment[i] = 1
        else:
            label_segment[i] = 0
    return segment, label_segment, RR_intervals_remove_etopic

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
    val_dir = os.path.join(output_dir, "val")
    train_file = os.path.join(train_dir, "train.npz")
    test_file = os.path.join(test_dir, "test.npz")
    val_file = os.path.join(val_dir, "val.npz")

    # # Check if data is already processed
    # if os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(val_file):
    #     print("Processed data found. Loading data...")

    #     train_data = np.load(train_file, allow_pickle=True)
    #     test_data = np.load(test_file, allow_pickle=True)
    #     val_data = np.load(val_file, allow_pickle=True)
    #     X_train, y_train = train_data["all_segments"], train_data["all_labels"]
    #     X_test, y_test = test_data["all_segments"], test_data["all_labels"]
    #     X_val, y_val = test_data["all_segments"], test_data["all_labels"]
    #     print("Data loaded successfully.")
    #     return X_train, y_train, X_test, y_test, X_val, y_val

    print("Processed data not found. Processing raw ECG data...")

    # Create output directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    

    manual_label = []
    segment5second = []
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
        segment_5s, label_5s, RR_intervals = extractData(ann_atr, ann_qrs, ecg1)
        
        segment5second.extend(segment_5s)
        manual_label.extend(label_5s)
    #     # print("segment_128: ", segment_128.shape)
    #     print("reshaped_truth: ", reshaped_truth)

        
        
        record2 = wfdb.rdrecord(os.path.join(data_path, record_name), channels=[1]) 
        ecg2 = record2.p_signal.flatten()
        segment_5s, label_5s, RR_intervals = extractData(ann_atr, ann_qrs, ecg2)
        # print("!!!!! segment: ", segment_128.shape)
        segment5second.extend(segment_5s)
        manual_label.extend(label_5s)
        
    # # segment5second = np.vstack(segment5second)
    # print("segment5second: ", len(segment5second)) 
    # print("manual_label: ", len(manual_label) )
    # np.savez('segments_labels.npz',
    #      segment5second=np.array(segment5second, dtype=object),
    #      manual_label=np.array(manual_label))
def classfify_data():
    data = np.load("segments_labels.npz", allow_pickle=True)
    # Access arrays
    segment5second = data['segment5second']
    manual_label = data['manual_label']

    # Example: print the first segment and label
    print("First segment:", segment5second[0])
    print("First label:", manual_label[0])
    all_AF_labels = []
    all_AF_segments = []
    all_non_labels = []
    all_non_segments = []
    for i in range(len(manual_label)):
        if manual_label[i]:
            all_AF_labels.append(1)
            all_AF_segments.append(segment5second[i])
        else:
            all_non_labels.append(0)
            all_non_segments.append(segment5second[i])
  
    np.savez('af_data_2.npz',
         all_AF_labels=np.array(all_AF_labels),
         all_AF_segments=np.array(all_AF_segments, dtype=object),
         all_non_labels=np.array(all_non_labels),
         all_non_segments=np.array(all_non_segments, dtype=object))
    # Xử lý file af_data.npz
def read_af_data():
    all_segments = []
    all_labels = []
    data_dir = "af_data_2.npz"
    data = np.load(data_dir, allow_pickle=True)
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
    all_segments = np.array(all_segments, dtype=object)  # Shape: (num_segments, 5*fs, 2)
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
    train_file = "dataset_128/train/train.npz"
    val_file = "dataset_128/val/val.npz"
    test_file = "dataset_128/test/test.npz"

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
    # classfify_data()
    # read_af_data()
