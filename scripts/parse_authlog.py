#!/usr/bin/env python3
# parse_authlog.py
import sys, re
from datetime import datetime
import pandas as pd

if len(sys.argv) != 3:
    print("Usage: python3 parse_authlog.py /path/to/auth.log output.csv")
    sys.exit(1)

infile = sys.argv[1]
outfile = sys.argv[2]

line_re = re.compile(
    r'^(?P<month>\w{3})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+'
    r'(?P<host>\S+)\s+(?P<service>[\w\/\-\[\]]+):\s+(?P<msg>.+)$'
)

failed_re = re.compile(r'Failed password for (invalid user )?(?P<user>\S+) from (?P<ip>\d{1,3}(?:\.\d{1,3}){3})')
accepted_re = re.compile(r'Accepted (password|publickey) for (?P<user>\S+) from (?P<ip>\d{1,3}(?:\.\d{1,3}){3})')
invalid_user_re = re.compile(r'Invalid user (?P<user>\S+) from (?P<ip>\d{1,3}(?:\.\d{1,3}){3})')
other_ip_re = re.compile(r'from (?P<ip>\d{1,3}(?:\.\d{1,3}){3})')

rows = []
current_year = datetime.now().year

with open(infile, 'r', errors='ignore') as f:
    for line in f:
        m = line_re.match(line.strip())
        if not m:
            continue
        month = m.group('month'); day = m.group('day'); ttime = m.group('time')
        host = m.group('host'); service = m.group('service'); msg = m.group('msg')
        dt_str = f"{month} {day} {ttime} {current_year}"
        try:
            timestamp = datetime.strptime(dt_str, "%b %d %H:%M:%S %Y")
        except Exception:
            timestamp = None

        result = "other"; user = None; ip = None

        m_failed = failed_re.search(msg)
        m_acc = accepted_re.search(msg)
        m_inv = invalid_user_re.search(msg)

        if m_failed:
            result = "failed"; user = m_failed.group('user'); ip = m_failed.group('ip')
        elif m_acc:
            result = "accepted"; user = m_acc.group('user'); ip = m_acc.group('ip')
        elif m_inv:
            result = "invalid_user"; user = m_inv.group('user'); ip = m_inv.group('ip')
        else:
            m_ip = other_ip_re.search(msg)
            if m_ip:
                ip = m_ip.group('ip')

        rows.append({
            "timestamp": timestamp,
            "hostname": host,
            "service": service,
            "raw_message": msg,
            "result": result,
            "user": user,
            "ip": ip
        })

df = pd.DataFrame(rows)
df.to_csv(outfile, index=False)
print(f"Wrote {len(df)} rows to {outfile}")
