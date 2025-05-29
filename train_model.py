import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from telegram_notifier import send_telegram_message
import os
from datetime import datetime
import shutil

def train_model():
    try:
        # Veriyi oku
        data = pd.read_csv("signals.csv")

        # Sadece sayısal sütunları al ve NaN'leri temizle
        X = data[["fr", "oi", "cvd", "price"]].apply(pd.to_numeric, errors="coerce").astype("float64")
        y = pd.to_numeric(data["label"], errors="coerce").astype("int")
        df = pd.concat([X, y], axis=1).dropna()

        X = df[["fr", "oi", "cvd", "price"]]
        y = df["label"]

        if len(X) < 10:
            send_telegram_message("⚠️ AI eğitimi başarısız: Yetersiz temiz veri.")
            return

        # Eğitim verisi ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Modeli eğit
        model = lgb.LGBMClassifier(max_depth=3, num_leaves=7)
        model.fit(X_train, y_train)

        # Doğruluk hesapla
        acc = accuracy_score(y_test, model.predict(X_test))

        # Eski model varsa versiyonla
        if os.path.exists("model.pkl"):
            version = datetime.now().strftime("model_v%Y%m%d_%H%M%S.pkl")
            shutil.move("model.pkl", version)

        # Yeni modeli kaydet
        joblib.dump(model, "model.pkl")

        # Eğitim kaydı yaz
        os.makedirs("logs", exist_ok=True)
        with open("logs/accuracy_log.txt", "a") as f:
            f.write(f"{datetime.now().isoformat()} - Accuracy: {acc:.4f} - Samples: {len(X)}\n")

        send_telegram_message(f"📊 AI eğitimi (LightGBM) tamamlandı. Doğruluk: %{int(acc * 100)}")

    except Exception as e:
        send_telegram_message(f"⚠️ AI eğitimi hatası: {str(e)}")

if __name__ == "__main__":
    train_model()
