import pandas as pd
import joblib
import os

MODEL_FILE = "model.pkl"
CSV_FILE = "predictions.csv"

FEATURES = [
    "ob", "fvg", "eqh", "ema_trend",
    "volume_spike", "atr_value",
    "funding_rate", "open_interest"
]

def load_model_and_predict():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(CSV_FILE):
        print("Model veya veri dosyası eksik.")
        return None

    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=FEATURES)

    if df.empty:
        print("Tahmin için uygun veri yok.")
        return None

    X = df[FEATURES]
    preds = model.predict(X)
    confidence = model.predict_proba(X).max()

    if confidence < 0.6:
        return None

    direction = "LONG" if preds[-1] == 1 else "SHORT"
    return f"Yeni Sinyal: {direction} | Güven: {round(confidence * 100, 2)}%"
