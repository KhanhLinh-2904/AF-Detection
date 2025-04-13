import wfdb
import numpy as np
import os
import matplotlib.pyplot as plt

from SWT import compute_swt
from preprocess import load_or_process_ecg_data


def plot_ecg_segment(segment, label, fs=250):
    """Randomly selects an ECG segment and plots it."""
    time_axis = np.linspace(0, len(segment) / fs, len(segment))
    
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, segment, label=f"Label: {label}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("ECG Segment in Test dataset")
    plt.legend()
    plt.grid()
    plt.show()
    
def plot_swt_coeffs(coeff_matrix, time):
    num_coeffs = coeff_matrix.shape[0]
    fig, axes = plt.subplots(num_coeffs, 1, figsize=(10, 6), sharex=True)
    
    for i in range(num_coeffs):
        axes[i].plot(time, coeff_matrix[i], 'b', linewidth=0.75)
        axes[i].set_yticks([])  # Hide y-axis labels
        axes[i].set_ylabel(f'C_{i+1}' if i < num_coeffs // 2 else f'D_{i - num_coeffs // 2 + 1}', rotation=0, labelpad=20, fontsize=10)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
    
    axes[-1].set_xlabel("Time (second)")
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_or_process_ecg_data()
    
    for i in range(len(y_train)):
        if y_train[i] == 0:
            labelN_train = i
            break
    for i in range(len(y_train)):
        if y_train[i] == 1:
            labelA_train = i
           
    
    # plot_ecg_segment(X_train[labelN_train], "N")   
    # plot_ecg_segment(X_train[labelA_train], "A")  
     
    # for i in range(len(y_test)):
    #     if y_test[i] == 0:
    #         labelN_test = i
    # for i in range(len(y_test)):
    #     if y_test[i] == 1:
    #         labelA_test = i
    #         break
    
    # plot_ecg_segment(X_test[labelN_test], "N")   
    # plot_ecg_segment(X_test[labelA_test], "A")   
    
    coeff_matrix_normalized = compute_swt(X_train[labelN_train], J=6, wavelet='db5')
    num_samples = 256
    # Find the overall max and min values
    max_value = np.max(coeff_matrix_normalized)
    min_value = np.min(coeff_matrix_normalized)

    print(f"Max value: {max_value}")
    print(f"Min value: {min_value}")
    # print("coeff_matrix_normalized: ", coeff_matrix_normalized)
    time = time = np.linspace(0, 1, num_samples)
    plot_swt_coeffs(coeff_matrix_normalized, time)
    
    
    
