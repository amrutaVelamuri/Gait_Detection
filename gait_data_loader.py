import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from features.feature_extraction import extract_all_features
from utils.utils import get_label_from_filename, get_subject_id

def load_gait_data(data_folder, target_length=100):
    df_list, labels, subject_ids = [], [], []
    for root, _, files in os.walk(data_folder):
        for file in sorted(files):
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                try:
                    df_temp = pd.read_csv(path, delim_whitespace=True, header=None)
                    if df_temp.shape[1] == 19:
                        df_temp.columns = [
                            "Time", "Fx_L", "Fy_L", "Fz_L", "COPx_L", "COPy_L", "Mx_L", "My_L", "Mz_L",
                            "Fx_R", "Fy_R", "Fz_R", "COPx_R", "COPy_R", "Mx_R", "My_R", "Mz_R",
                            "Other1", "Other2"]
                        label = get_label_from_filename(file)
                        if label == -1:
                            continue
                        df_list.append((df_temp, file))
                        labels.append(label)
                        subject_ids.append(get_subject_id(file))
                except Exception:
                    continue
    return df_list, labels, subject_ids

def prepare_features(df_list, labels, subject_ids, target_length):
    return extract_all_features(df_list, labels, subject_ids, target_length)

def split_and_normalize(X_grf_np, X_cop_np, X_hand_np, y_np, groups):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_hand_np, y_np, groups))
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42, stratify=y_np[train_idx])

    X_grf_train, X_grf_val, X_grf_test = X_grf_np[train_idx], X_grf_np[val_idx], X_grf_np[test_idx]
    X_cop_train, X_cop_val, X_cop_test = X_cop_np[train_idx], X_cop_np[val_idx], X_cop_np[test_idx]
    X_hand_train, X_hand_val, X_hand_test = X_hand_np[train_idx], X_hand_np[val_idx], X_hand_np[test_idx]
    y_train, y_val, y_test = y_np[train_idx], y_np[val_idx], y_np[test_idx]

    scaler_hand = StandardScaler()
    X_hand_train = scaler_hand.fit_transform(X_hand_train)
    X_hand_val = scaler_hand.transform(X_hand_val)
    X_hand_test = scaler_hand.transform(X_hand_test)

    return (X_grf_train, X_grf_val, X_grf_test,
            X_cop_train, X_cop_val, X_cop_test,
            X_hand_train, X_hand_val, X_hand_test,
            y_train, y_val, y_test)
