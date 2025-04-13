import numpy as np
import pywt
import matplotlib.pyplot as plt

# Function to perform SWT and return the coefficient matrix
def compute_swt(signal, J=6, wavelet='db5'):
    # Zero-padding the signal to make its length a multiple of 2^J
    signal_length = len(signal)
    pad_length = 2**J - (signal_length % (2**J))
    padded_signal = np.pad(signal, (0, pad_length), mode='constant')
    
    # List to store the coefficients
    detail_coeffs = []
    coarse_coeffs = []
    
    # Perform the SWT at each level
    test_coef = pywt.swt(padded_signal, wavelet, level=6)
    # Extract details and approximation coefficients
    detail_coeffs = [c[1] for c in test_coef]  # Detail coefficients
    coarse_coeffs = [c[0] for c in test_coef]  # Approximation coefficients
    
    # for j in range(1, J+1):
    #     # SWT decomposition
    #     coeffs = pywt.swt(padded_signal, wavelet, level=j)
    #     # print("******************")
    #     coeffs = np.array(coeffs)
    #     # print("coeffs : ", coeffs.shape)
    #     # print("coeffs: ", coeffs[-1].shape)
    #     # Detail coefficients (high-pass filter) and coarse coefficients (low-pass filter)
    #     detail_coeffs.append(coeffs[-1][1])  # Detail coefficients
    #     coarse_coeffs.append(coeffs[-1][0])  # Coarse coefficients
    
    # Stack the coefficients into a 2D matrix where each row corresponds to a coefficient time series
    coeff_matrix = np.vstack(detail_coeffs + coarse_coeffs)
    coeff_matrix_normalized = 2 * (coeff_matrix - np.min(coeff_matrix)) / np.ptp(coeff_matrix) - 1
    return coeff_matrix_normalized

if __name__ == "__main__":
    # Example signal (replace this with actual signal data)
    # Simulated sine wave with noise as an example signal
    fs = 250  # Sampling frequency (Hz)
    t = np.linspace(0, 5, fs * 5)  # 5 seconds of data
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(len(t))  # Sine wave + noise

    # # Plot the original 1D signal
    # plt.figure(figsize=(10, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(t, signal, label="Original Signal")
    # plt.title("Original 1D Signal")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.grid(True)

    # Compute the SWT coefficients for the signal
    coeff_matrix = compute_swt(signal, J=6, wavelet='db5')

    # Normalize the matrix to the range [-1, 1]
    coeff_matrix_normalized = 2 * (coeff_matrix - np.min(coeff_matrix)) / np.ptp(coeff_matrix) - 1
    print("coeff_matrix_normalized: ", coeff_matrix_normalized.shape)
    # Plot the 2D matrix of coefficients (you can consider this as an "image")
    plt.subplot(1, 2, 2)
    plt.imshow(coeff_matrix_normalized, aspect='auto', cmap='gray', origin='lower')
    plt.colorbar()
    plt.title("SWT Coefficients (Normalized) - 2D Representation")
    plt.xlabel("Time (samples)")
    plt.ylabel("Coefficient Series")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # coeff_matrix contains the 2J time series (6 detail + 6 coarse)
    # This is the 2D input that can be used in a DCNN model
