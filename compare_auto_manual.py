import wfdb
import random
import numpy as np
import os
from hr_analysis import print_attribute, remove_ectopic_beats, define_AF_or_none
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def get_label(start_idx, end_idx, ann):
    for j in range(len(ann.sample)):
        if start_idx <= ann.sample[j] <= end_idx:
            label_note = ann.aux_note[j][1]
            if label_note == "N":
                return True, 0, ann.sample[j]
            else:
                return True, 1, ann.sample[j]
        else:
            return False, 0, 0
        
def get_load_data(data_path="mit-bih-atrial-fibrillation-database-1.0.0/"):
    
    # CHỉ xử lý những label nào trong manual thôi, bỏ qua auto đi cho đỡ nặng máy 
    # Get all record names that have .dat files
    record_files = [f.split('.')[0] for f in os.listdir(data_path) if f.endswith('.dat')]

    # Initialize lists to store auto and manual label
    af_rmssd = []
    af_tpr = []
    af_se = []
    non_af_rmssd = []
    non_af_tpr = []
    non_af_se = []
    manual_label = []
    auto_label = []
    # Process each record
    for record_name in record_files:
        hea_path = os.path.join(data_path, record_name + ".hea")
        atr_path = os.path.join(data_path, record_name + ".atr")
    # ==============================================================================
    # record_name = "04015"
    # hea_path= "mit-bih-atrial-fibrillation-database-1.0.0/04015.hea"
    # atr_path = "mit-bih-atrial-fibrillation-database-1.0.0/04015.atr"
    
        # Skip if the .hea or .atr file does not exist
        if not os.path.exists(hea_path) or not os.path.exists(atr_path):
            print(f"Skipping {record_name}: Missing .hea or .atr file.")
            continue  # Move to the next record
        print("====================================================")
        print(f"Processing record: {record_name}")

        # Read the ECG signal data (both channels)
        record = wfdb.rdsamp(os.path.join(data_path, record_name))  
        ann = wfdb.rdann(os.path.join(data_path, record_name), 'atr')  

        # Extract both ECG signal channels
        ecg_signal = record[0]  # Shape: (samples, 2) for two channels
        ecg1 = ecg_signal[:, 0]
        ecg2 = ecg_signal[:, 1]
        
        fs = record[1]['fs']  # Sampling frequency

        print("ann.sample: ", ann.sample)
        

        segment_length = 128
        # segment_length = 250*5  
        
        #Remove ectopic beats
        ecg1_clean = remove_ectopic_beats(ecg1)
        ecg2_clean = remove_ectopic_beats(ecg2)
                
        
        print("***********ECG 1*************")
        # Calculate the number of segments
        num_segments = len(ecg1_clean) // segment_length
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length
            
            for j in range(len(ann.sample)):
                if start_idx <= ann.sample[j] <= end_idx:
                    print("start_idx vs end_idx vs ann.sample[j]: ", start_idx, end_idx, ann.sample[j])
                    
                    segment = ecg1_clean[start_idx:end_idx]  
               
                    rmssd, tpr, se = print_attribute(segment)
                    # au_label = define_AF_or_none(segment)
                    # auto_label.append(au_label)
                    label_note = ann.aux_note[j][1]
                    if label_note == "N":
                        # man_label = 0
                        # manual_label.append(0)
                        non_af_rmssd.append(rmssd)
                        non_af_tpr.append(tpr)
                        non_af_se.append(se)
                        # print("auto_label vs manual_label in ecg1: ", au_label, man_label)
                        
                    elif label_note == "A":
                        # man_label = 1
                        af_rmssd.append(rmssd)
                        af_tpr.append(tpr)
                        af_se.append(se)
                        # manual_label.append(1)
                        # print("auto_label vs manual_label in ecg1: ", au_label, man_label)
                        
        print("***********ECG 2*************")
        # Calculate the number of segments
        num_segments = len(ecg2_clean) // segment_length
        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length
            for j in range(len(ann.sample)):
                if start_idx <= ann.sample[j] <= end_idx:
                    print("start_idx vs end_idx vs ann.sample[j]: ", start_idx, end_idx, ann.sample[j])
                    
                    segment = ecg2_clean[start_idx:end_idx]  # Shape: (5*fs, 2)
                    
                    au_label = define_AF_or_none(segment)
                    auto_label.append(au_label)
                    label_note = ann.aux_note[j][1]
                    if label_note == "N":
                        # man_label = 0
                        # manual_label.append(0)
                        # print("auto_label vs manual_label in ecg1: ", au_label, man_label)
                        non_af_rmssd.append(rmssd)
                        non_af_tpr.append(tpr)
                        non_af_se.append(se)
                        
                    elif label_note == "A":
                        # man_label = 1
                        af_rmssd.append(rmssd)
                        af_tpr.append(tpr)
                        af_se.append(se)
                        # manual_label.append(1)
                        # print("auto_label vs manual_label in ecg1: ", au_label, man_label)
                        
        
    
    # Lưu dưới dạng comparision_auto_manual.npz
    np.savez('comparision_auto_manual.npz',
         non_af_rmssd=np.array(non_af_rmssd),
         non_af_tpr=np.array(non_af_tpr),
         non_af_se=np.array(non_af_se),
         af_rmssd=np.array(af_rmssd),
         af_tpr=np.array(af_tpr),
         af_se=np.array(af_se)
         )

def draw_barchart(af, non_af, title_name):
    # Create a figure with two subplots arranged side by side
    # Define bins
    # bins = np.arange(0, 1, 0.1)
    # Create stacked vertical plots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)

    # Top plot (AF)
    axes[0].hist(af, color='darkorange', label='AF')
    axes[0].legend()
    axes[0].set_ylabel('# of windows')

    # Bottom plot (Non-AF)
    axes[1].hist(non_af, color='steelblue', label='Non-AF')
    axes[1].legend()
    axes[1].set_ylabel('# of windows')
    axes[1].set_xlabel(title_name)

    # Adjust layout
    plt.tight_layout()
    plt.show()
def comparision_auto_manual_data():
  
    data_dir = "comparision_auto_manual.npz"
    data = np.load(data_dir)
    auto_label = data["auto_label"]
    manual_label = data["manual_label"]
    # Confusion matrix elements
    TP = np.sum((manual_label == 1) & (auto_label == 1))
    TN = np.sum((manual_label == 0) & (auto_label == 0))
    FP = np.sum((manual_label == 0) & (auto_label == 1))
    FN = np.sum((manual_label == 1) & (auto_label == 0))

    # Metrics
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    print("TP: ", TP)
    print("TN: ", TN)
    print("FP: ", FP) 
    print("FN: ", FN)
    
    print("Sensitivity (Recall):", sensitivity)
    print("Specificity:", specificity)
   # Calculate metrics
    precision = precision_score(manual_label, auto_label)
    recall = recall_score(manual_label, auto_label)
    f1 = f1_score(manual_label, auto_label)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


# Call the function
if __name__ == "__main__":
    get_load_data()
    # comparision_auto_manual_data()
    data_dir = "comparision_auto_manual.npz"
    data = np.load(data_dir)
    non_af_rmssd = data["non_af_rmssd"]
    af_rmssd = data["af_rmssd"]
    draw_barchart(af_rmssd, non_af_rmssd, "RMSSD")
    non_af_tpr = data["non_af_tpr"]
    af_tpr = data["af_tpr"]
    draw_barchart(non_af_tpr, af_tpr, "TPR")
    non_af_se = data["non_af_se"]
    af_se = data["af_se"]
    draw_barchart(non_af_se, af_se, "Shannon Entropy")
    
