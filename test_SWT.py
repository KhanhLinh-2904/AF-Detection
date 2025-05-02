import numpy as np
import pywt
import matplotlib.pyplot as plt

from test_model import load_or_process_ecg_data

# Generate a sample signal (replace this with actual ECG data)
all_segments, all_labels = load_or_process_ecg_data()
ec_signal = all_segments[0]
# Set SWT parameters
wavelet = 'db5'  # Daubechies 5 wavelet
level = 6         # Number of decomposition levels
required_length = 2**6  # 6 levels need at least 2^6 = 64 samples
if len(ec_signal) < required_length:
    ec_signal = np.pad(ec_signal, (0, required_length - len(ec_signal)), 'constant')
# Recalculate max level after padding
max_level = pywt.swt_max_level(len(ec_signal))
level = min(6, max_level)

# Apply SWT
coeffs = pywt.swt(ec_signal, wavelet, level=level)

# Extract details and approximation coefficients
detail_coeffs = [c[1] for c in coeffs]  # Detail coefficients
approx_coeffs = [c[0] for c in coeffs]  # Approximation coefficients

# Plot the original signal and wavelet decomposition
plt.figure(figsize=(10, 6))
plt.subplot(level + 1, 1, 1)
plt.plot(ec_signal, color='black')
plt.title('Original ECG Signal')
plt.xlim([0, len(ec_signal)])

# Plot the decomposition results
for i in range(level):
    plt.subplot(level + 1, 1, i + 2)
    plt.plot(detail_coeffs[i], color='blue')
    plt.title(f'Detail Coefficients D_{i+1}')
    plt.xlim([0, len(detail_coeffs[i])])

plt.tight_layout()
plt.show()
