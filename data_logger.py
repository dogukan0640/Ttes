import csv
from datetime import datetime

def log_signal(symbol, fr, oi, cvd, atr, label=None):
    row = [symbol, fr, oi, cvd, atr, label if label is not None else ""]
    with open("signals.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)