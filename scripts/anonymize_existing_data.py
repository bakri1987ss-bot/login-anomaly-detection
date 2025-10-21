import os
import hashlib
import pandas as pd

# ==============================
# 🔹 تحميل مفتاح SALT من المتغير البيئي
# ==============================
SALT = os.getenv("LOGIN_ANOMALY_SALT")
if not SALT:
    raise EnvironmentError("❌ LOGIN_ANOMALY_SALT is not set! Please export it first.")

# ==============================
# 🔹 دالة التهشيم الآمن
# ==============================
def anonymize_value(value):
    """تهشيم القيمة النصية باستخدام SHA-256 مع salt ثابت."""
    if pd.isna(value):
        return value
    value = str(value)
    salted_value = f"{SALT}_{value}"
    return hashlib.sha256(salted_value.encode()).hexdigest()

# ==============================
# 🔹 مسارات الملفات
# ==============================
DATA_DIR = "/home/bakri/projects/login-anomaly/data"
FILES = [
    "auth_parsed_large.csv",
    "auth_features_large.csv",
    "auth_features_new.csv"
]

# ==============================
# 🔹 تنفيذ التهشيم
# ==============================
for filename in FILES:
    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        print(f"⚠️ Skipping: {file_path} not found.")
        continue

    print(f"🔹 Processing {filename} ...")
    df = pd.read_csv(file_path, low_memory=False)

    for col in ["ip", "user"]:
        if col in df.columns:
            print(f"  → Anonymizing column: {col}")
            df[col] = df[col].apply(anonymize_value)

    # حفظ الملف المجهول
    new_path = os.path.join(DATA_DIR, f"anon_{filename}")
    df.to_csv(new_path, index=False)
    print(f"✅ Saved anonymized file: {new_path}")

print("🎯 All available files anonymized successfully.")
