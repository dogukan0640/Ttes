import pandas as pd
import joblib
import os
from signal_generator import generate_signal

MODEL_FILE = "model.pkl"
CSV_FILE = "predictions.csv"

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
        print("Veri dosyası bulunamadı.")
        return None

    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=FEATURES)

    if df.empty:
        print("Tahmin için yeterli veri yok.")
        return None

    X = df[FEATURES]
    predictions = model.predict(X)
    df['prediction'] = predictions

    signal = generate_signal(df)
    return signal
