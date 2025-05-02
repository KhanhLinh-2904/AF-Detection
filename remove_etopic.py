import numpy as np

def remove_ectopic_beats(rr_intervals, return_indices=False):
    rr = np.array(rr_intervals, dtype=float)
    original_indices = np.arange(len(rr))

    # Remove NaN, inf, and non-positive values
    valid_mask = np.isfinite(rr) & (rr > 0)
    rr = rr[valid_mask]
    original_indices = original_indices[valid_mask]

    # Compute RR ratios
    rr_ratios = rr[1:] / rr[:-1]
    perc1 = np.percentile(rr_ratios, 1)
    perc99 = np.percentile(rr_ratios, 99)
    perc25 = np.percentile(rr_ratios, 25)

    clean_rr = [rr[0]]               # Always include the first RR
    kept_indices = [original_indices[0]]
    i = 1
    while i < len(rr) - 2:
        r1 = rr[i] / rr[i - 1]
        r2 = rr[i + 1] / rr[i]
        r3 = rr[i + 1] / rr[i + 2]

        if r1 < perc1 and r2 > perc99 and r3 > perc25:
            # Skip ectopic + compensatory
            i += 2
        else:
            clean_rr.append(rr[i])
            kept_indices.append(original_indices[i])
            i += 1

    # Optionally add the last value(s)
    if i < len(rr):
        clean_rr.append(rr[i])
        kept_indices.append(original_indices[i])

    clean_rr = np.array(clean_rr)
    kept_indices = np.array(kept_indices)
    print("kept_indices: ", kept_indices)
    print("len of kept_indices: ", len(kept_indices))
    
    if return_indices:
        return clean_rr, kept_indices
    else:
        return clean_rr
