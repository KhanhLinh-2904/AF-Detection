import numpy as np

def remove_ectopic_beats(rr_intervals):
    rr = np.array(rr_intervals)
    # Bỏ các giá trị không hợp lệ
    rr = rr[~np.isnan(rr)]
    rr = rr[np.isfinite(rr)]
    rr = rr[rr > 0]

    rr_ratios = rr[1:] / rr[:-1]
    # print(rr_ratios)
    # Tính percentiles để phát hiện nhịp ngoại tâm thu
    perc1 = np.percentile(rr_ratios, 1)
    perc99 = np.percentile(rr_ratios, 99)
    perc25 = np.percentile(rr_ratios, 25)

    clean_rr = []
    i = 1
    while i < len(rr) - 2:
        r1 = rr[i] / rr[i - 1]
        r2 = rr[i + 1] / rr[i]
        r3 = rr[i + 1] / rr[i + 2]

        if r1 < perc1 and r2 > perc99 and r3 > perc25:
            i += 2  # Bỏ qua cả nhịp ngoại tâm thu và nhịp bù sau nó
        else:
            clean_rr.append(rr[i])
            i += 1

    return clean_rr

def compute_rmssd(rr_segment):
    rr = np.array(rr_segment)
    diff = np.diff(rr)
    rmssd = np.sqrt(np.mean(diff ** 2))
    mean_rr = np.mean(rr)
    return rmssd / mean_rr  # chuẩn hóa theo bài báo

def compute_tpr(rr_segment):
    count = 0
    for i in range(1, len(rr_segment) - 1):
        if (rr_segment[i] > rr_segment[i - 1] and rr_segment[i] > rr_segment[i + 1]) or \
           (rr_segment[i] < rr_segment[i - 1] and rr_segment[i] < rr_segment[i + 1]):
            count += 1
    tpr = count / (len(rr_segment) - 2)
    return tpr

def compute_shannon_entropy(rr_segment, bins=16):
    rr = np.array(rr_segment)
    # Loại bỏ 8 giá trị lớn nhất và nhỏ nhất (outliers)
    sorted_rr = np.sort(rr)
    trimmed_rr = sorted_rr[8:-8]
    
    hist, _ = np.histogram(trimmed_rr, bins=bins, density=False)
    probabilities = hist / np.sum(hist)
    se = -np.sum([p * np.log(p) for p in probabilities if p > 0]) / np.log(1/bins)
    return se

def define_AF_or_none(segment):
    rmssd = compute_rmssd(segment)
    tpr = compute_tpr(segment)
    se = compute_shannon_entropy(segment)
    if (rmssd > 0.1) or (0.54 < tpr < 0.77) and (se > 0.7):
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
