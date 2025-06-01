from squeeze_data_collector import get_squeeze_signals
from squeeze_config import TELEGRAM_TOKEN, CHAT_ID, INTERVAL
from predictor import predict_squeeze_score
from data_logger import log_signal
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
            score = predict_squeeze_score(sig['fundingRate'], sig['openInterestChange'], sig['cvd'], sig['atr'])
            log_signal(sig['symbol'], sig['fundingRate'], sig['openInterestChange'], sig['cvd'], sig['atr'], 1 if score >= 60 else 0)

            if score >= 80:
                msg = f"💣 AI Squeeze Alarmı: {sig['symbol']}\n"
                msg += f"Fiyat: {sig['lastPrice']}\nFR: {sig['fundingRate']} | OI: %{sig['openInterestChange']}\n"
                msg += f"CVD: {sig['cvd'] / 1_000_000:.1f}M | ATR: {sig['atr']:.4f}\n"
                msg += f"AI Skoru: {score}/100"
                send_telegram(msg)
    except Exception as e:
        send_telegram(f"Hata oluştu: {e}")

if __name__ == "__main__":
    while True:
        main()
        time.sleep(INTERVAL)