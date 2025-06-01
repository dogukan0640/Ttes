from utils import get_coin_data, evaluate_coin
from config import TELEGRAM_TOKEN, CHAT_ID, INTERVAL
import requests
import time

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print("Telegram hatası:", e)

def main():
    send_telegram_message("🔍 Bot taramaya başladı.")
    try:
        coins = get_coin_data()
        for symbol, data in coins.items():
            signal = evaluate_coin(symbol, data)
            if signal:
                send_telegram_message(signal)
    except Exception as e:
        send_telegram_message(f"Hata oluştu: {e}")

if __name__ == "__main__":
    while True:
        main()
        time.sleep(INTERVAL)