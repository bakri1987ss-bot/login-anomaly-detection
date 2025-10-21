import os
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from datetime import datetime

# ==============================
# 🔹 Paths Setup
# ==============================
DATA_DIR = "/home/bakri/projects/login-anomaly/data"
MODEL_FILE = os.path.join(DATA_DIR, "random_forest_model_final.joblib")
FEATURE_COLS_FILE = os.path.join(DATA_DIR, "feature_columns.joblib")
FEATURE_FILE = os.path.join(DATA_DIR, "auth_features_large.csv")
PARSED_FILE = os.path.join(DATA_DIR, "auth_parsed_large.csv")

# ==============================
# 🔹 Load Data & Model
# ==============================
print("🔹 Loading model and data...")
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("❌ Model file not found! Train your model first.")

model_data = joblib.load(MODEL_FILE)
clf = model_data['model']
scaler = model_data['scaler']
features = joblib.load(FEATURE_COLS_FILE)

# Load base data
feat = pd.read_csv(FEATURE_FILE, parse_dates=['timestamp'], low_memory=False)
df = pd.read_csv(PARSED_FILE, parse_dates=['timestamp'], low_memory=False)

# Ensure both have same rows
feat = feat.iloc[:len(df)]

# ==============================
# 🔹 Feature Reconstruction
# ==============================
print("🔹 Reconstructing missing features...")

# Hour and is_night
feat['hour'] = df['timestamp'].dt.hour
feat['is_night'] = feat['hour'].isin([0,1,2,3,4,5,23]).astype(int)

# Avg interarrival
df_sorted = df.sort_values(['ip','timestamp'])
df_sorted['time_diff'] = df_sorted.groupby('ip')['timestamp'].diff().dt.total_seconds()
feat['avg_interarrival'] = df_sorted.groupby('ip')['time_diff'].transform('mean').fillna(0)

# Failed streak
df['failed_flag'] = (df['result'] == 'failed').astype(int)
def compute_failed_streak(x):
    return x.groupby((x == 0).cumsum()).cumsum()
df['failed_streak'] = df.groupby('user')['failed_flag'].transform(compute_failed_streak)
feat['failed_streak'] = df['failed_streak'].fillna(0).astype(int)

# Unique users last 5 attempts
from collections import deque
def unique_users_last_5(series):
    arr = series.tolist()
    counts = []
    dq = deque(maxlen=5)
    for user in arr:
        dq.append(user)
        counts.append(len(set(dq)))
    return pd.Series(counts, index=series.index)
df['unique_users_last_5'] = df.groupby('ip')['user'].transform(unique_users_last_5)
feat['unique_users_last_5'] = df['unique_users_last_5'].fillna(0).astype(int)

# Latitude / Longitude placeholders (if not present)
for col in ['lat','lon','geo_country']:
    if col not in feat.columns:
        feat[col] = 0

# ==============================
# 🔹 Prepare Data
# ==============================
df['result_bin'] = df['result'].apply(lambda x: 1 if x == 'success' else 0)
X = feat.reindex(columns=features, fill_value=0)
y = df['result_bin']

# ==============================
# 🔹 Evaluate Current Model
# ==============================
X_scaled = scaler.transform(X)
y_pred = clf.predict(X_scaled)
roc_score = roc_auc_score(y, y_pred)
print(f"✅ Current ROC AUC: {roc_score:.4f}")

# ==============================
# 🔹 Retrain if Needed
# ==============================
if roc_score < 0.95:
    print("⚠️ Model performance degraded — Retraining initiated...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_scaled, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
    
    clf_new = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1
    )
    clf_new.fit(X_train, y_train)
    
    y_pred_new = clf_new.predict(X_test)
    roc_new = roc_auc_score(y_test, y_pred_new)
    print(f"✅ New model ROC AUC: {roc_new:.4f}")
    
    if roc_new > roc_score:
        joblib.dump({'model': clf_new, 'scaler': scaler}, MODEL_FILE)
        print("🚀 Model updated successfully.")
    else:
        print("⚙️ New model not better. Keeping the current model.")
else:
    print("✅ Model is performing well. No retraining needed.")

print("🎯 Optimization check complete.")
