import time
from telegram import Bot
from squeeze import get_binance_metrics, detect_squeeze

TELEGRAM_TOKEN = "7744478523:AAEtRJar6uF7m0cxKfQh7r7TltXYxWwtmm0"
CHAT_ID = "1009868232"
bot = Bot(token=TELEGRAM_TOKEN)

def notify_telegram(msg):
    bot.send_message(chat_id=CHAT_ID, text=msg)

def main():
    notify_telegram("🚀 Real Squeeze Bot started.")
    while True:
        try:
            all_data = get_binance_metrics()
            for symbol, data in all_data.items():
                result = detect_squeeze(symbol, data)
                if result["score"] >= 80:
                    notify_telegram(result["message"])
        except Exception as e:
            notify_telegram(f"❌ Bot Error: {str(e)}")
        time.sleep(900)  # 15 dakikada bir
if __name__ == "__main__":
    main()
