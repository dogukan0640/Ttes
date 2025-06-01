from squeeze_data_collector import get_squeeze_signals
from squeeze_config import TELEGRAM_TOKEN, CHAT_ID, INTERVAL
import requests
import time

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    requests.post(url, data=data)

def main():
    try:
        signals = get_squeeze_signals()
        for sig in signals:
            msg = f"💣 AI Squeeze Alarmı: {sig['symbol']}\n"
            msg += f"Fiyat: {sig['lastPrice']}\nFR: {sig['fundingRate']} | OI: %{sig['openInterestChange']}\n"
            msg += f"CVD: {sig['cvd'] / 1_000_000:.1f}M | ATR: {sig['atr']:.4f}\n"
            msg += f"AI Skoru: {sig['ai_score']}/100"
            send_telegram(msg)
    except Exception as e:
        send_telegram(f"Hata oluştu: {e}")

if __name__ == "__main__":
    while True:
        main()
        time.sleep(INTERVAL)