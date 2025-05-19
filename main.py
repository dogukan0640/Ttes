from model_trainer import train_model
from predictor import load_model_and_predict
from telegram_notifier import send_telegram_message

import time

def main():
    print("🤖 Bot başlatıldı. Analiz ve tahmin döngüsü çalışıyor...")

    while True:
        try:
            print("🔁 Yeni döngü: Model eğitiliyor...")
            train_model()

            print("📊 Tahminler yapılıyor ve sinyaller hazırlanıyor...")
            message = load_model_and_predict()

            if message:
                print("📡 Sinyal gönderiliyor...")
                send_telegram_message(message)
            else:
                print("⚠️ Sinyal üretilemedi veya düşük güven skoru.")

        except Exception as e:
            print(f"❌ Hata oluştu: {e}")

        print("⏳ 4 saat bekleniyor...")
        time.sleep(4 * 60 * 60)

if __name__ == "__main__":
    main()
