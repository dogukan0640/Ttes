import requests

BOT_TOKEN = "7744478523:AAEtRJar6uF7m0cxKfQh7r7TltXYxWwtmm0"
CHAT_ID = "1009868232"

def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": message
        }
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print("✅ Telegram mesajı gönderildi.")
        else:
            print(f"⚠️ Telegram mesajı gönderilemedi: {response.text}")
    except Exception as e:
        print(f"❌ Telegram bağlantı hatası: {e}")
