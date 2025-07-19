import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from utils.utils import segment_signal, sample_entropy, petrosian_fd

def extract_feats(arr):
    return np.stack([np.mean(arr, axis=1), np.std(arr, axis=1),
                     skew(arr, axis=1), kurtosis(arr, axis=1)], axis=1) if arr.ndim == 2 else np.zeros((arr.shape[0], 4))

def extract_all_features(df_list, labels, subject_ids, target_length):
    all_grf, all_cop, all_handcrafted, stride_labels, stride_subjects = [], [], [], [], []

    for i, (df, fname) in enumerate(df_list):
        fz_l, fz_r = df["Fz_L"].values, df["Fz_R"].values
        copx_l, copy_l = df["COPx_L"].values, df["COPy_L"].values
        copx_r, copy_r = df["COPx_R"].values, df["COPy_R"].values
        peaks_l = find_peaks(fz_l, height=300, distance=80)[0]
        peaks_r = find_peaks(fz_r, height=300, distance=80)[0]

        grf_l = segment_signal(fz_l, peaks_l, target_length)
        grf_r = segment_signal(fz_r, peaks_r, target_length)
        cop_l_x = segment_signal(copx_l, peaks_l, target_length)
        cop_l_y = segment_signal(copy_l, peaks_l, target_length)
        cop_r_x = segment_signal(copx_r, peaks_r, target_length)
        cop_r_y = segment_signal(copy_r, peaks_r, target_length)

        if len(grf_l) == 0 or len(grf_r) == 0:
            continue
        cop_l = np.stack([cop_l_x, cop_l_y], axis=1)
        cop_r = np.stack([cop_r_x, cop_r_y], axis=1)
        strides = min(len(grf_l), len(grf_r), len(cop_l), len(cop_r))

        grf = (grf_l[:strides] + grf_r[:strides]) / 2
        cop = (cop_l[:strides] + cop_r[:strides]) / 2
        handcrafted = np.hstack([
            extract_feats(grf_l[:strides]), extract_feats(grf_r[:strides]),
            extract_feats(cop_l_x[:strides]), extract_feats(cop_l_y[:strides]),
            extract_feats(cop_r_x[:strides]), extract_feats(cop_r_y[:strides])
        ])

        if len(peaks_l) > strides:
            stride_times = np.diff(peaks_l[:strides+1])
            mean_stride = np.mean(stride_times)
            std_stride = np.std(stride_times)
            cv_stride = std_stride / mean_stride if mean_stride != 0 else 0
        else:
            mean_stride = std_stride = cv_stride = 0

        irregular_feats = []
        for j in range(strides):
            segment = fz_l[peaks_l[j]:peaks_l[j+1]] if j+1 < len(peaks_l) else None
            if segment is not None and len(segment) > 10:
                sampen = sample_entropy(segment)
                fractal = petrosian_fd(segment)
            else:
                sampen = fractal = 0
            irregular_feats.append([mean_stride, std_stride, cv_stride, sampen, fractal])
        irregular_feats = np.array(irregular_feats)

        all_grf.append(grf)
        all_cop.append(cop)
        all_handcrafted.append(np.hstack([handcrafted, irregular_feats]))
        stride_labels.extend([labels[i]] * strides)
        stride_subjects.extend([subject_ids[i]] * strides)

    return np.vstack(all_grf), np.vstack(all_cop), np.vstack(all_handcrafted), np.array(stride_labels), np.array(stride_subjects)
