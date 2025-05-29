
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from telegram_notifier import send_telegram_message

def train_model():
    data = pd.read_csv("signals.csv")
    X = data[["fr", "oi", "cvd", "price"]]
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    joblib.dump(model, "model.pkl")
    send_telegram_message(f"📊 AI eğitimi (LightGBM) tamamlandı. Doğruluk: %{int(acc * 100)}")

if __name__ == "__main__":
    train_model()
