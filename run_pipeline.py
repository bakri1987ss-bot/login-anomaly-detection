# ==============================
# Login Anomaly Detection Full Pipeline (Enhanced with Feedback Handling)
# ==============================

import os
import pandas as pd
import numpy as np
import geoip2.database
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

# ==============================
# 0️⃣ Paths Setup
# ==============================
BASE_DIR = '/home/bakri/projects/login-anomaly/data'
AUTH_PARSED_FILE = os.path.join(BASE_DIR, 'auth_parsed_large.csv')
AUTH_FEATURES_FILE = os.path.join(BASE_DIR, 'auth_features_large.csv')
FEATURE_FILE_NEW = os.path.join(BASE_DIR, 'auth_features_new.csv')
GEOIP_FILE = os.path.join(BASE_DIR, 'GeoLite2-City.mmdb')
MODEL_FILE = os.path.join(BASE_DIR, 'random_forest_model_final.joblib')
FEATURE_COLS_FILE = os.path.join(BASE_DIR, 'feature_columns.joblib')
ALERT_FILE = os.path.join(BASE_DIR, 'alerts.csv')
FEEDBACK_FILE = os.path.join(BASE_DIR, 'alerts_feedback.csv')
METRICS_FILE = os.path.join(BASE_DIR, 'alerts_metrics.csv')

# ==============================
# 1️⃣ Load Raw Data
# ==============================
df = pd.read_csv(AUTH_PARSED_FILE, parse_dates=['timestamp'], low_memory=False)
feat = pd.read_csv(AUTH_FEATURES_FILE, parse_dates=['timestamp'], low_memory=False)

# ==============================
# 2️⃣ Check Data Quality
# ==============================
print("\n----- Missing Values: auth_parsed_large.csv -----")
print(df.isnull().mean() * 100)
print("\n----- Missing Values: auth_features_large.csv -----")
print(feat.isnull().mean() * 100)

print("\n----- Result Distribution -----")
print(df['result'].value_counts())

print("\n----- Numeric Summary -----")
print(feat.describe())

# Optional plots
numeric_cols = feat.select_dtypes(include='number').columns
plt.figure(figsize=(12,6))
sns.boxplot(data=feat[numeric_cols])
plt.title("Boxplot of Numeric Features")
plt.xticks(rotation=45)
plt.show()

feat[numeric_cols].hist(bins=30, figsize=(15,10))
plt.suptitle("Histograms of Numeric Features")
plt.show()

# ==============================
# 3️⃣ Prepare Target
# ==============================
df['result_bin'] = df['result'].apply(lambda x: 1 if x=='success' else 0)
y = df['result_bin'].astype(int)

# ==============================
# 4️⃣ Feature Engineering
# ==============================
# Hour & Night flag
feat['hour'] = df['timestamp'].dt.hour
feat['is_night'] = feat['hour'].isin([0,1,2,3,4,5,23]).astype(int)

# Avg interarrival
df_sorted = df.sort_values(['ip','timestamp'])
df_sorted['time_diff'] = df_sorted.groupby('ip')['timestamp'].diff().dt.total_seconds()
feat['avg_interarrival'] = df_sorted.groupby('ip')['time_diff'].transform('mean').fillna(0)

# Failed streak
df['failed_flag'] = (df['result']=='failed').astype(int)
def compute_failed_streak(x):
    return x.groupby((x==0).cumsum()).cumsum()
df['failed_streak'] = df.groupby('user')['failed_flag'].transform(compute_failed_streak)
feat['failed_streak'] = df['failed_streak'].fillna(0).astype(int)

# Unique users last 5 attempts
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

# GeoIP features
reader = geoip2.database.Reader(GEOIP_FILE)
geo_cache = {}
def geoip_lookup(ip):
    if ip in geo_cache:
        return geo_cache[ip]
    try:
        r = reader.city(ip)
        country = r.country.iso_code
        city = r.city.name
        lat = r.location.latitude
        lon = r.location.longitude
    except:
        country, city, lat, lon = 'NA','NA',0,0
    geo_cache[ip] = pd.Series([country, city, lat, lon])
    return geo_cache[ip]

unique_ips = df['ip'].unique()
geo_results = Parallel(n_jobs=-1)(delayed(geoip_lookup)(ip) for ip in unique_ips)
geo_df = pd.DataFrame(geo_results, index=unique_ips, columns=['geo_country','city','lat','lon'])
df = df.join(geo_df, on='ip')
feat['lat'] = df['lat'].fillna(0)
feat['lon'] = df['lon'].fillna(0)
feat['geo_country'] = df['geo_country']

# ==============================
# 5️⃣ Prepare Feature Matrix
# ==============================
numeric_cols = feat.select_dtypes(include=np.number).columns.tolist()
if 'timestamp' in numeric_cols: numeric_cols.remove('timestamp')
X = feat[numeric_cols].fillna(0)

# ==============================
# 6️⃣ Scale Features
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# 7️⃣ Handle Imbalance
# ==============================
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

# ==============================
# 8️⃣ Train/Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.3, random_state=42, stratify=y_res
)

# ==============================
# 9️⃣ Train Model
# ==============================
param_dist = {'n_estimators':[50,100,200],'max_depth':[5,10,20,None]}
rs = RandomizedSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
    param_distributions=param_dist, n_iter=10, cv=3, scoring='recall'
)
rs.fit(X_train, y_train)
clf = rs.best_estimator_
print("✅ Best hyperparameters:", rs.best_params_)

# ==============================
# 🔹 Evaluate Model
# ==============================
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
prec, rec, _ = precision_recall_curve(y_test, y_proba)
print("PR AUC:", auc(rec, prec))

# Feature Importance
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(x=importances[:10], y=importances.index[:10])
plt.title("Top 10 Important Features")
plt.show()

# ==============================
# 🔹 Save Model & Feature List
# ==============================
joblib.dump({'model': clf, 'scaler': scaler}, MODEL_FILE)
joblib.dump(X.columns.tolist(), FEATURE_COLS_FILE)
print("✅ Model and feature list saved.")

# ==============================
# 🔹 Predict on New Data & Generate Alerts
# ==============================
def predict_new_data(feature_file, threshold=0.5):
    if not os.path.exists(feature_file):
        print("⚠️ Feature file not found, skipping alerts.")
        return None
    
    df_new = pd.read_csv(feature_file, low_memory=False)
    
    # Ensure all trained features exist
    for c in X.columns:
        if c not in df_new.columns:
            df_new[c] = 0
    
    X_new_scaled = scaler.transform(df_new[X.columns])
    df_new['failed_prob'] = 1 - clf.predict_proba(X_new_scaled)[:,1]
    df_new['alert'] = (df_new['failed_prob'] >= threshold).astype(int)
    
    for col in ['timestamp','ip','user']:
        if col not in df_new.columns:
            df_new[col] = 'NA'
    
    df_alert = df_new[['timestamp','ip','user','failed_prob','alert']]
    
    if os.path.exists(ALERT_FILE):
        df_alert.to_csv(ALERT_FILE, mode='a', header=False, index=False)
    else:
        df_alert.to_csv(ALERT_FILE, index=False)
    
    return df_alert

alerts = predict_new_data(FEATURE_FILE_NEW)
if alerts is not None and not alerts.empty:
    print(f"✅ {len(alerts)} alerts generated. Check {ALERT_FILE}")
else:
    print("⚠️ No alerts generated.")
# ==============================
# 🔹 Handle Feedback Safely (Clean corrupted rows)
# ==============================
if os.path.exists(FEEDBACK_FILE):
    fixed_rows = []
    with open(FEEDBACK_FILE, 'r') as f:
        for line in f:
            if len(line.strip().split(',')) == 7:  # Keep only rows with 7 columns
                fixed_rows.append(line)
    # Save to a temporary cleaned file
    FEEDBACK_FIXED = os.path.join(BASE_DIR, 'alerts_feedback_fixed.csv')
    with open(FEEDBACK_FIXED, 'w') as f:
        f.writelines(fixed_rows)
    
    # Read the cleaned feedback
    df_feedback = pd.read_csv(FEEDBACK_FIXED, parse_dates=['timestamp'])
    print(f"✅ Feedback file loaded (cleaned): {FEEDBACK_FIXED}")
else:
    print("⚠️ Feedback file not found, skipping feedback matrix.")
    df_feedback = pd.DataFrame(columns=['timestamp','ip','user','failed_prob','alert','label'])


# ==============================
# 🔹 Record Production Metrics
# ==============================
metrics = {
    'total_alerts': len(alerts) if alerts is not None else 0,
    'true_positives': df_feedback['label'].sum() if 'label' in df_feedback else 0
}

pd.DataFrame([metrics]).to_csv(METRICS_FILE, index=False)
print(f"✅ Production metrics recorded in {METRICS_FILE}")
