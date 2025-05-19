import time
from telegram_notifier import send_telegram_message
from signal_generator import build_prediction_dataset
from model_trainer import train_model
from predictor import load_model_and_predict

def main():
    print("🤖 Bot başlatıldı.")
    send_telegram_message("🤖 Bot başlatıldı. Canlı veri analizi başlıyor...")

    while True:
        try:
            print("⏳ Canlı veri çekiliyor ve analiz hazırlanıyor...")
            build_prediction_dataset("BTCUSDT")

            print("📚 Model eğitiliyor...")
            train_model()

            print("🧠 Tahmin yapılıyor...")
            message = load_model_and_predict()

            if message:
                print(f"📡 Sinyal: {message}")
                send_telegram_message(message)
            else:
                print("⚠️ Güven eşiği altında, sinyal yok.")

        except Exception as e:
            print(f"❌ HATA: {e}")
            send_telegram_message(f"❌ Hata oluştu: {e}")

        print("🕒 4 saat uykuya geçiyor...\n")
        time.sleep(4 * 60 * 60)  # 4 saat bekle

if __name__ == "__main__":
    main()
