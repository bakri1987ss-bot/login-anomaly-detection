#!/usr/bin/env python3
"""
train_and_save_model.py
يقرأ data/auth_features.csv، يبني نموذج LogisticRegression بسيطًا، ويحفظه في ../models/logreg_baseline.joblib
تشغيل: python3 scripts/train_and_save_model.py
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "auth_features.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "logreg_baseline.joblib")

# تأكد من وجود المجلد models
os.makedirs(MODELS_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    print(f"Error: file not found: {DATA_PATH}")
    sys.exit(1)

# قراءة البيانات
df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])

# عرض سريع للتأكد
print("Loaded data with shape:", df.shape)
print("Columns:", df.columns.tolist())

# أنشئ عمود target إن لم يكن موجودًا
if 'target' not in df.columns:
    # قاعدة مؤقتة: مشبوه إذا كان عدد الفشل أكبر من 3
    if 'fail_count' in df.columns:
        df['target'] = (df['fail_count'] > 3).astype(int)
        print("Created 'target' from 'fail_count' > 3")
    else:
        # لو لا يوجد 'fail_count' استخدم 'cnt_last_5m' كبديل
        if 'cnt_last_5m' in df.columns:
            df['target'] = (df['cnt_last_5m'] > 3).astype(int)
            print("Created 'target' from 'cnt_last_5m' > 3")
        else:
            print("Error: no suitable feature to create target (no fail_count or cnt_last_5m).")
            sys.exit(1)

# اختر ميزات متاحة — اختبر وجود كل عمود وإلا استخدم بدائل
possible_features = ['fail_count','success_count','fail_rate','cnt_last_1m','cnt_last_5m','cnt_last_15m']
features = [f for f in possible_features if f in df.columns]
if len(features) < 2:
    print("Error: not enough feature columns found. Found:", features)
    sys.exit(1)

print("Using features:", features)

X = df[features].fillna(0)
y = df['target'].astype(int)

# تقسيم وتدريب
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# تقييم سريع
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print("\nClassification report:\n", classification_report(y_test, y_pred))

# حفظ النموذج
joblib.dump(model, MODEL_PATH)
print(f"\n✅ Model saved to: {MODEL_PATH}")
