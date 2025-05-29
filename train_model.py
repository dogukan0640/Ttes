import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from telegram_notifier import send_telegram_message

def train_model():
    try:
        # Veriyi oku
        data = pd.read_csv("signals.csv")

        # NaN satırları sil
        data.dropna(inplace=True)

        # Özellikleri float'a çevir
        X = data[["fr", "oi", "cvd", "price"]].astype("float64")
        y = data["label"].astype(int)

        if len(X) < 10:
            send_telegram_message("⚠️ AI eğitimi başarısız: Yetersiz veri.")
            return

        # Eğitim verisi ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Modeli eğit
        model = lgb.LGBMClassifier(max_depth=3, num_leaves=7)
        model.fit(X_train, y_train)

        # Doğruluk
        acc = accuracy_score(y_test, model.predict(X_test))

        # Kaydet
        joblib.dump(model, "model.pkl")

        # Bildirim
        send_telegram_message(f"📊 AI eğitimi (LightGBM) tamamlandı. Doğruluk: %{int(acc * 100)}")

    except Exception as e:
        send_telegram_message(f"⚠️ AI eğitimi hatası: {str(e)}")

if __name__ == "__main__":
    train_model()
