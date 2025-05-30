
import time
import requests
from data_collector import get_coin_data
from ai_model import load_model
from short_squeeze_detector import detect_squeeze
from telegram_notifier import send_telegram_message
from keep_alive import keep_alive
from datetime import datetime
import csv

model = load_model()

def log_signal(symbol, fr, oi, cvd, price, prediction):
    log_file = "logs/signals_log.csv"
    header = ["timestamp", "symbol", "fr", "oi", "cvd", "price", "prediction"]
    row = [datetime.utcnow().isoformat(), symbol, fr, oi, cvd, price, prediction]

    try:
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(header)
            writer.writerow(row)
    except Exception as e:
        print("Log error:", e)

def get_all_usdt_pairs():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        data = requests.get(url).json()
        symbols = [s["symbol"] for s in data["symbols"] if s["symbol"].endswith("USDT") and s["status"] == "TRADING"]
        return symbols
    except Exception as e:
        print("Symbol fetch error:", e)
        return []

def run_bot():
    send_telegram_message("🤖 Bot başlatıldı. Tüm USDT pariteleri taranıyor...")
    while True:
        coins = get_all_usdt_pairs()
        for symbol in coins:
            try:
                fr, oi, cvd, price = get_coin_data(symbol)
                if None in [fr, oi, cvd, price]:
                    continue
                features = [[fr, oi, cvd, price]]
                prediction = model.predict(features)[0]
                log_signal(symbol, fr, oi, cvd, price, prediction)
                if detect_squeeze(fr, oi, cvd):
                    send_telegram_message(f"⚠️ Squeeze sinyali: {symbol}\nFR: {fr}, OI: {oi}, CVD: {cvd}")
            except Exception as e:
                print(f"Hata ({symbol}):", e)
        time.sleep(900)  # 15 dakika

if __name__ == "__main__":
    keep_alive()
    run_bot()
