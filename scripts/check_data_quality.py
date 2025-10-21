#!/usr/bin/env python3
import os
import pandas as pd

# ===============================
# 1️⃣ تحديد المسارات
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
REPORT_DIR = os.path.join(BASE_DIR, "..", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

PARSED_FILE = os.path.join(DATA_DIR, "auth_parsed_large.csv")
FEATURE_FILE = os.path.join(DATA_DIR, "auth_features_large.csv")

# ===============================
# 2️⃣ تحقق من وجود الملفات
# ===============================
if not os.path.exists(PARSED_FILE) or not os.path.exists(FEATURE_FILE):
    raise FileNotFoundError(f"❌ الملفات غير موجودة:\n{PARSED_FILE}\n{FEATURE_FILE}")

print("📥 تحميل الملفات الكبيرة...")
df_parsed = pd.read_csv(PARSED_FILE, parse_dates=['timestamp'], low_memory=False)
df_feat = pd.read_csv(FEATURE_FILE, parse_dates=['timestamp'], low_memory=False)

# ===============================
# 3️⃣ فحص جودة البيانات
# ===============================
print("\n📊 جودة ملف auth_parsed_large.csv:")
missing_parsed = df_parsed.isnull().mean().sort_values(ascending=False)
if 'result' in df_parsed.columns:
    result_counts = df_parsed['result'].value_counts()
else:
    result_counts = pd.Series(dtype=int)

# ===============================
# 4️⃣ فحص الميزات
# ===============================
print("\n📊 جودة ملف auth_features_large.csv:")
missing_feat = df_feat.isnull().mean().sort_values(ascending=False)

# ===============================
# 5️⃣ إنشاء تقارير
# ===============================
parsed_report_path = os.path.join(REPORT_DIR, "parsed_quality_report.html")
feature_report_path = os.path.join(REPORT_DIR, "feature_quality_report.html")

summary_csv_path = os.path.join(REPORT_DIR, "data_quality_summary.csv")

# HTML Reports
with open(parsed_report_path, "w", encoding="utf-8") as f:
    f.write("<h1>تقرير جودة البيانات - auth_parsed_large.csv</h1>")
    f.write("<h3>نسبة القيم المفقودة</h3>")
    f.write(missing_parsed.to_frame("missing_ratio").to_html(border=1))
    if not result_counts.empty:
        f.write("<h3>توزيع النتائج</h3>")
        f.write(result_counts.to_frame("count").to_html(border=1))
    f.write("<h3>وصف إحصائي</h3>")
    f.write(df_parsed.describe().to_html(border=1))

with open(feature_report_path, "w", encoding="utf-8") as f:
    f.write("<h1>تقرير جودة البيانات - auth_features_large.csv</h1>")
    f.write("<h3>نسبة القيم المفقودة</h3>")
    f.write(missing_feat.to_frame("missing_ratio").to_html(border=1))
    f.write("<h3>وصف إحصائي</h3>")
    f.write(df_feat.describe().to_html(border=1))

# CSV Summary
summary = pd.DataFrame({
    "dataset": ["auth_parsed_large.csv", "auth_features_large.csv"],
    "rows": [len(df_parsed), len(df_feat)],
    "columns": [len(df_parsed.columns), len(df_feat.columns)],
    "missing_avg": [missing_parsed.mean(), missing_feat.mean()]
})
summary.to_csv(summary_csv_path, index=False)

# ===============================
# 6️⃣ طباعة ملخص نهائي
# ===============================
print("\n✅ تم إنشاء تقارير جودة البيانات بنجاح!")
print(f"📄 HTML Reports:")
print(f"   - {parsed_report_path}")
print(f"   - {feature_report_path}")
print(f"📊 CSV Summary:")
print(f"   - {summary_csv_path}")

