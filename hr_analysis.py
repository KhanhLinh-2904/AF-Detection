import numpy as np
from scipy.stats import entropy

def remove_ectopic_beats(rr_intervals):
    rr = np.array(rr_intervals, dtype=float)

    # Remove NaN, inf, and non-positive values
    rr = rr[np.isfinite(rr)]
    rr = rr[rr > 0]

    # Compute RR ratios
    rr_ratios = rr[1:] / rr[:-1]
    perc1 = np.percentile(rr_ratios, 1)
    perc99 = np.percentile(rr_ratios, 99)
    perc25 = np.percentile(rr_ratios, 25)

    clean_rr = [rr[0]]  # Always include the first RR
    i = 1
    while i < len(rr) - 2:
        r1 = rr[i] / rr[i - 1]
        r2 = rr[i + 1] / rr[i]
        r3 = rr[i + 1] / rr[i + 2]

        if r1 < perc1 and r2 > perc99 and r3 > perc25:
            # Suspected ectopic beat + compensatory beat — skip both
            clean_rr.append(np.nan)
            i += 2
        else:
            clean_rr.append(rr[i])
            i += 1

    # Optionally add the last valid value(s)
    if i < len(rr):
        clean_rr.append(rr[i])

    return np.array(clean_rr)

# def compute_rmssd(rr_segment):
#     rr = np.array(rr_segment, dtype=float)
#     rr = rr[~np.isnan(rr)]
#     # rr_minus_outliers = rr.copy()
#     # for _ in range(8):
#     #     if len(rr_minus_outliers) < 2:
#     #         break
#     #     rr_minus_outliers[np.argmax(rr_minus_outliers)] = np.nan
#     #     rr_minus_outliers[np.argmin(rr_minus_outliers)] = np.nan
#     # rr = rr_minus_outliers[~np.isnan(rr_minus_outliers)]
#     diff = np.diff(rr)
#     if len(diff) == 0:
#         return 0.0
#     rmssd = np.sqrt(np.sum(diff ** 2)/(len(rr)-1))
#     mean_rr = np.mean(rr)
#     if mean_rr <= 0:
#         return 0.0
#     return rmssd / mean_rr

def compute_rmssd(rr_segment):
    rr_segment = np.array(rr_segment, dtype=float)
    signal_minus_outliers = rr_segment.copy()

    # Remove 8 maximum values (outliers)
    for _ in range(8):
        max_index = np.argmax(signal_minus_outliers)
        signal_minus_outliers[max_index] = 0

    # Remove 8 minimum values (outliers)
    for _ in range(8):
        min_index = np.argmin(signal_minus_outliers)
        signal_minus_outliers[min_index] = 0

    # Remove all zeros (which are placeholder for removed outliers)
    signal_minus_outliers = signal_minus_outliers[signal_minus_outliers != 0]

    signal_length = len(signal_minus_outliers)

    if signal_length < 2:
        return np.nan  # Not enough data to compute RMSSD

    # Compute RMSSD
    diff = np.diff(signal_minus_outliers)
    rmssd = np.sqrt(np.sum(diff**2) / (signal_length - 1))

    return rmssd


# def compute_tpr(rr_segment):
#     rr_segment = np.array(rr_segment)
#     rr_segment = rr_segment[~np.isnan(rr_segment)]
#     # rr_minus_outliers = rr_segment.copy()
#     # for _ in range(8):
#     #     if len(rr_minus_outliers) < 2:
#     #         break
#     #     rr_minus_outliers[np.argmax(rr_minus_outliers)] = np.nan
#     #     rr_minus_outliers[np.argmin(rr_minus_outliers)] = np.nan
#     # rr_segment = rr_minus_outliers[~np.isnan(rr_minus_outliers)]
#     count = 0 
#     for i in range(1, len(rr_segment) - 1):
#         if (rr_segment[i] > rr_segment[i - 1] and rr_segment[i] > rr_segment[i + 1]) or \
#            (rr_segment[i] < rr_segment[i - 1] and rr_segment[i] < rr_segment[i + 1]):
#             count += 1  
#     if len(rr_segment) <= 2:
#         return 0.0
#     tpr_ratio = count / (len(rr_segment)-2)
#     return tpr_ratio

def compute_tpr(rr_segment):
    rr_segment = np.array(rr_segment, dtype=float)
    qrs_length = 128
    tp = np.zeros(qrs_length)

    # Find turning points
    for j in range(1, qrs_length - 1):
        if (rr_segment[j - 1] < rr_segment[j] > rr_segment[j + 1]) or \
           (rr_segment[j - 1] > rr_segment[j] < rr_segment[j + 1]):
            tp[j] = 1

    # Segment length
    l = len(tp)

    # Expected and actual values
    u_tp_expected = (2 * l - 4) / 3
    u_tp_actual = np.sum(tp)

    # Expected and real standard deviations
    sigma_tp_expected = np.sqrt((16 * l - 29) / 90)
    sigma_tp_real = np.std(tp)

    return u_tp_expected, u_tp_actual, sigma_tp_expected, sigma_tp_real

# def compute_shannon_entropy(rr_segment, bins=16):
#     rr = np.array(rr_segment)
#     rr = rr[~np.isnan(rr)]
#     rr_minus_outliers = rr.copy()
#     for _ in range(8):
#         if len(rr_minus_outliers) < 2:
#             break
#         rr_minus_outliers[np.argmax(rr_minus_outliers)] = np.nan
#         rr_minus_outliers[np.argmin(rr_minus_outliers)] = np.nan
#     rr_minus_outliers = rr_minus_outliers[~np.isnan(rr_minus_outliers)]
    
#     if len(rr_minus_outliers) == 0:
#         return 0.0
#     counts, _ = np.histogram(rr, bins=bins)
#     # total_counts = np.sum(counts)
#     # if total_counts == 0:
#     #     return 0.0
#     # probabilities = counts / total_counts
#     probabilities = counts / (len(rr)-16)
#     se = entropy(probabilities, base=2) 
#     # base=bins giúp chuẩn hóa entropy trong [0,1]
#     se = se/np.log(bins)
#     return se

def compute_shannon_entropy(rr_segment, bin_size=16, window_size=128):
    rr_segment = np.array(rr_segment, dtype=float)
    signal_minus_outliers = rr_segment.copy()

    # Remove 8 maximum values (outliers)
    for _ in range(8):
        max_index = np.argmax(signal_minus_outliers)
        signal_minus_outliers[max_index] = 0

    # Remove 8 minimum values (outliers)
    for _ in range(8):
        min_index = np.argmin(signal_minus_outliers)
        signal_minus_outliers[min_index] = 0

    # Remove all zeros (placeholders for removed outliers)
    signal_minus_outliers = signal_minus_outliers[signal_minus_outliers != 0]

    # Compute histogram (frequencies)
    counts, _ = np.histogram(signal_minus_outliers, bins=bin_size)

    # Avoid division by zero
    total_count = window_size - bin_size
    if total_count <= 0:
        return np.nan

    # Compute probabilities
    probabilities = counts / total_count

    # Compute Shannon entropy
    # Use base 1/16 = log2(prob)/log2(1/16) = -log2(prob)/4
    entropy = 0
    for p in probabilities:
        if p > 0:
            entropy += p * (np.log(p) / np.log(1 / bin_size))  # base 1/bin_size

    return entropy

def print_attribute(segment):
    rmssd = compute_rmssd(segment)
    tpr = compute_tpr(segment)
    se = compute_shannon_entropy(segment)
    return rmssd, tpr, se

def define_AF_or_none(segment):
    rmssd = compute_rmssd(segment)
    tpr = compute_tpr(segment)
    se = compute_shannon_entropy(segment)
    if (rmssd > 0.1) and 0.54 < tpr  and (se > 0.7):
        return 1    
    else:
        return 0

if __name__ == "__main__":
    # Giả sử bạn có RR intervals từ thiết bị đo ECG
    rr_intervals = np.random.normal(0.8, 0.05, 10000)  # ví dụ dữ liệu

    segments = remove_ectopic_beats(rr_intervals)

    for segment in segments:
        rmssd = compute_rmssd(segment)
        tpr = compute_tpr(segment)
        se = compute_shannon_entropy(segment)
        if (rmssd > 0.1) or (0.54 < tpr < 0.77) or (se > 0.7):
            print(f"AF")
            
        else:
            print(f"NON-AF")
        print(f"RMSSD/MeanRR: {rmssd:.4f}, TPR: {tpr:.4f}, Shannon Entropy: {se:.4f}")
