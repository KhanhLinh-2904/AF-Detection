import wfdb
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.signal import filtfilt, ellip
from SWT import compute_swt
from heart_rate_analysis import classify_af

def bandpass_filter(ecg_signal, fs, lowcut=0.5, highcut=50, order=10):
    """Apply an elliptical band-pass filter to the ECG signal."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = ellip(order, 0.1, 40, [low, high], btype='band')
    return filtfilt(b, a, ecg_signal, axis=0)


def load_or_process_ecg_data(data_path="mit-bih-atrial-fibrillation-database-1.0.0/", record_name="04015", folder="test"):
    # Initialize lists to store segments and labels
    all_segments = []
    all_labels = []
    test_file = os.path.join(folder, "test_data.npz")
    print(f"Processing record: {record_name}")
    
    # Read the ECG signal data (both channels)
    record = wfdb.rdsamp(os.path.join(data_path, record_name))  
    # ann_old = os.path.join(data_path, "old")
    ann = wfdb.rdann(os.path.join(data_path, record_name), 'atr')  
    print("ann: ", ann.aux_note)
    print("ann: ", ann.sample)
    
    # Extract both ECG signal channels
    ecg_signal = record[0]  # Shape: (samples, 2) for two channels
    print("ecg_signal: ", ecg_signal.shape)
    fs = record[1]['fs']  # Sampling frequency
    # Define the segment length in samples (5 seconds per segment)
    segment_length = int(fs)  
    # Calculate the number of segments
    num_segments = len(ecg_signal) // segment_length

    print(f"Total segments in {record_name}: {num_segments}")

    for i in range(num_segments):
        # print("***********************************")
        # print(" index: ", i)
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length
        # Extract the ECG segment (both channels)
        segment = ecg_signal[start_idx:end_idx, :]  # Shape: (5*fs, 2)
        # print("segment: ", segment)
        flag = False
        for j in range(len(ann.sample)):
            if start_idx <= ann.sample[j] < end_idx:
               label = ann.aux_note[j][1]
            #    print("start_idx--- atr---- end_idx---- label: ",start_idx, ann.sample[j] , end_idx,label)
               flag = True
               all_segments.append(segment[:, 0])
               all_segments.append(segment[:, 1])
               all_labels.append(label)
               all_labels.append(label)
        if flag == False:
            # print("start_idx--- end_idx---- label: ",start_idx, end_idx,"N")
            label ="N"
       
    
    # # Convert lists to NumPy arrays
    all_segments = np.array(all_segments)  
    all_labels = np.array(all_labels)

    return all_segments, all_labels


def plot_random_ecg_segment(segments, labels, fs=250):
    """Randomly selects an ECG segment and plots it."""
    idx = np.random.randint(len(segments))
    # segment = segments[idx]
    # label = labels[idx]
    time_axis = np.linspace(0, len(segments) / fs, len(segments))
    
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, segments, label=f"Label: {labels}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Random ECG Segment")
    plt.legend()
    plt.grid()
    plt.show()

    
# Call the function
if __name__ == "__main__":
    all_segments, all_labels = load_or_process_ecg_data()
    print("all_segments shape: ", all_segments.shape)
    print("all_labels shape: ", all_labels.shape)
    print(all_labels)
    N_labels = [i for i in all_labels if  i == 'N']
    A_labels = [i for i in all_labels if  i == 'A']
    
    print(len(N_labels))
    print(len(A_labels))
    
    # # plot_random_ecg_segment(all_segments, all_labels)
    
    #    # Read the ECG signal data (both channels)
    # data_path="mit-bih-atrial-fibrillation-database-1.0.0/"
    # record_name="04015"
    # record = wfdb.rdsamp(os.path.join(data_path, record_name))  
    # ann = wfdb.rdann(os.path.join(data_path, record_name), 'atr')  
    # ecg1 = record[0][:, 0]
    # norm = ecg1[0:250]
    # af_ecg = ecg1[102400:102650]
    # plot_random_ecg_segment(norm, "N")
    # plot_random_ecg_segment(af_ecg, "AFIB")
    # load_or_process_ecg_data()
    
