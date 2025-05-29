
import pandas as pd
import requests
from datetime import datetime, timedelta

def fetch_price(symbol):
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url)
        return float(response.json()["price"])
    except:
        return None

def auto_label():
    try:
        df = pd.read_csv("logs/signals_log.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        latest = df.sort_values("timestamp").tail(10)

        new_signals = []
        for _, row in latest.iterrows():
            symbol = row["symbol"]
            ts = row["timestamp"]
            pred = row["prediction"]
            price_at_signal = row["price"]

            now = datetime.utcnow()
            future_time = pd.to_datetime(ts) + timedelta(minutes=30)
            if now < future_time:
                continue  # henüz test edilecek zaman geçmemiş

            price_now = fetch_price(symbol)
            if price_now is None:
                continue

            price_change = (price_now - price_at_signal) / price_at_signal
            label = 1 if price_change > 0.01 else 0

            new_signals.append({
                "timestamp": ts,
                "symbol": symbol,
                "fr": row["fr"],
                "oi": row["oi"],
                "cvd": row["cvd"],
                "price": price_at_signal,
                "label": label
            })

        if new_signals:
            pd.DataFrame(new_signals).to_csv("signals.csv", mode='a', index=False, header=not pd.read_csv("signals.csv").shape[0])

    except Exception as e:
        print("Auto-label error:", e)

if __name__ == "__main__":
    auto_label()
