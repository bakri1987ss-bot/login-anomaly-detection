# security_config.py
"""
Security & privacy utilities for login-anomaly project.

Provides:
- anonymize_ip/ip/user via SHA256 + salt (configurable)
- set_secure_permissions for data and logs
- backup_files utility
- rotate_log simple helper
"""

import os
import hashlib
import shutil
from datetime import datetime

# Read salt from env var for safety. Set SECRET_SALT before running scripts.
SECRET_SALT = os.environ.get("LOGIN_ANOMALY_SALT", "change_this_in_production")

def _hash_value(val: str, length: int = 12) -> str:
    """Deterministic truncated SHA256 with salt. Returns hex prefix length chars."""
    if val is None:
        return ""
    if not isinstance(val, str):
        val = str(val)
    h = hashlib.sha256()
    h.update((SECRET_SALT + val).encode('utf-8'))
    return h.hexdigest()[:length]

def anonymize_ip(ip: str, length: int = 12) -> str:
    """Return anonymized representation of IP."""
    return _hash_value(ip, length)

def anonymize_user(user: str, length: int = 12) -> str:
    """Return anonymized representation of username."""
    return _hash_value("user:" + user, length)

def set_secure_permissions(path: str, user_only_files: bool = True):
    """
    Set secure permissions recursively:
    - files -> 600 (rw------)
    - dirs  -> 700 (rwx------)
    Use with caution.
    """
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o700)
        for f in files:
            full = os.path.join(root, f)
            if user_only_files:
                os.chmod(full, 0o600)
            else:
                os.chmod(full, 0o640)

def backup_files(src_paths, dst_dir):
    """
    Copy listed files/dirs to dst_dir with timestamp suffix.
    src_paths: list of file/dir paths
    dst_dir: directory to store backups
    """
    os.makedirs(dst_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    for src in src_paths:
        if not os.path.exists(src):
            continue
        base = os.path.basename(src.rstrip('/'))
        dst = os.path.join(dst_dir, f"{base}_{ts}")
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    return True

def rotate_log(log_path, max_bytes=10_000_000):
    """
    If log size exceeds max_bytes, move it to archived name and create empty file.
    """
    if not os.path.exists(log_path):
        return False
    size = os.path.getsize(log_path)
    if size < max_bytes:
        return False
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    archive = f"{log_path}.{ts}.bak"
    shutil.move(log_path, archive)
    # create empty file with strict permissions
    open(log_path, 'a').close()
    os.chmod(log_path, 0o600)
    return archive
