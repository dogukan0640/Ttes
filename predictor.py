import pandas as pd
import joblib
import os

CSV_FILE = "predictions.csv"
MODEL_FILE = "model.pkl"

FEATURES = [
    "ob", "fvg", "eqh", "ema_trend",
    "volume_spike", "atr_value",
    "funding_rate", "open_interest"
]

def load_model_and_predict():
    if not os.path.exists(MODEL_FILE):
        print("Model dosyası bulunamadı.")
        return None

    if not os.path.exists(CSV_FILE):
        print("Tahmin için veri bulunamadı.")
        return None

    df = pd.read_csv(CSV_FILE)
    df = df.dropna()

    if df.empty:
        print("Tahmin için yeterli veri yok.")
        return None

    X = df[FEATURES]
    model = joblib.load(MODEL_FILE)
    df["prediction"] = model.predict(X)

    result = df.iloc[-1]
    signal = "LONG" if result["prediction"] == 1 else "SHORT"
    confidence = model.predict_proba([result[FEATURES]])[0].max()

    if confidence >= 0.6:
        return f"Yeni sinyal: {signal} | Güven: {round(confidence * 100, 2)}%"
    else:
        return None
