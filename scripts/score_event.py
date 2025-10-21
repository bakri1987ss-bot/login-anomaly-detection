#!/usr/bin/env python3
# score_event.py — نسخة ذكية تستخدم الميزات من النموذج مباشرة
import sys, pandas as pd, joblib
from datetime import datetime

if len(sys.argv) != 3:
    print("Usage: python3 score_event.py features.csv model.joblib")
    sys.exit(1)

feat_file = sys.argv[1]
model_file = sys.argv[2]

# قراءة البيانات
feat = pd.read_csv(feat_file, parse_dates=['timestamp'])

# تحميل النموذج
model = joblib.load(model_file)

# التحقق إن كان النموذج يحتوي على feature_names_in_ (سكيت-ليرن >= 0.24)
if hasattr(model, 'feature_names_in_'):
    features = list(model.feature_names_in_)
else:
    # fallback: استخدم كل الأعمدة ما عدا timestamp، ip، event_user، event_result
    exclude = ['timestamp','ip','event_user','event_result']
    features = [c for c in feat.columns if c not in exclude]

# تأكد أن جميع الميزات موجودة في البيانات
missing = [f for f in features if f not in feat.columns]
if missing:
    print(f"Error: missing columns in CSV: {missing}")
    sys.exit(1)

X = feat[features].values

# حساب احتمالات الشذوذ
probs = model.predict_proba(X)[:,1]
feat['score'] = probs

# طباعة الأحداث الأكثر خطورة
alerts = feat[feat['score']>0.7].sort_values('score', ascending=False)
if alerts.empty:
    print("No high-confidence alerts.")
else:
    print("Alerts:")
    print(alerts[['timestamp','ip','event_user','event_result','score']].head(20).to_string(index=False))
