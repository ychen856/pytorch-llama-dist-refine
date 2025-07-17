import numpy as np
def get_lm_head_idx(end_idx):

    lm_heads = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    lm_head = 1
    lm_head_idx = 0

    for i in range(0, len(lm_heads)):
        if lm_heads[i] > end_idx:
            #lm_head = lm_heads[i - 1]
            #lm_head_idx = lm_head_idx - 1
            break
        elif lm_heads[i] == end_idx:
            lm_head = lm_heads[i]
            lm_head_idx = i
            break

        lm_head = lm_heads[i]
        lm_head_idx = i

    lm_head_idx = lm_head_idx + 1


    return lm_head, lm_head_idx


def remove_latency_outliers_iqr(data):
    latencies = [latency for _, latency in data]
    q1 = np.percentile(latencies, 25)
    q3 = np.percentile(latencies, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    filtered = [(t, l) for (t, l) in data if lower <= l <= upper]
    return filtered

def fit_linear_model(ks, total_times):
    X = np.vstack([np.ones(len(ks)), ks]).T  # [1, k]
    y = np.array(total_times)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]  # returns [t0, delta_t]
    return coef[0], coef[1]