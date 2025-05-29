
import time
import threading
from train_model import train_model
from data_collector import get_all_data
from ai_model import load_model, predict
from short_squeeze_detector import detect_squeeze
from telegram_notifier import send_telegram_message
from keep_alive import keep_alive
import pandas as pd
from datetime import datetime

def run_training_schedule():
    while True:
        try:
            train_model()
        except Exception as e:
            send_telegram_message(f"⚠️ AI eğitimi hatası: {str(e)}")
        time.sleep(7200)  # 2 saat

def run_bot():
    print("Bot başlatıldı.")
    model = load_model()

    while True:
        try:
            print("Veriler alınıyor...")
            data = get_all_data()
            if not data.empty:
                predictions = predict(model, data)
                logs = []
                for idx, row in data.iterrows():
                    signal_strength = predictions[idx]
                    is_squeeze = detect_squeeze(row)
                    logs.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "symbol": row["symbol"],
                        "fr": row["fr"],
                        "oi": row["oi"],
                        "cvd": row["cvd"],
                        "price": row["price"],
                        "prediction": signal_strength,
                        "actual_squeeze": is_squeeze
                    })
                    if signal_strength >= 0.5 and is_squeeze:
                        send_telegram_message(f"⚠️ Güçlü Short Squeeze: {row['symbol']} | Güven: %{int(signal_strength * 100)}")
                pd.DataFrame(logs).to_csv("ai_log.csv", mode='a', index=False, header=False)
        except Exception as e:
            send_telegram_message(f"❌ Hata oluştu: {str(e)}")
        time.sleep(900)

if __name__ == "__main__":
    keep_alive()
    send_telegram_message("🤖 Bot başarıyla başlatıldı ve squeeze taramasına başladı.")
    threading.Thread(target=run_training_schedule).start()
    run_bot()
