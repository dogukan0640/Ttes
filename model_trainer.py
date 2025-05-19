import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

CSV_FILE = "predictions.csv"
MODEL_FILE = "model.pkl"

FEATURES = [
    "ob", "fvg", "eqh", "ema_trend",
    "volume_spike", "atr_value",
    "funding_rate", "open_interest"
]

TARGET = "actual"

def train_model():
    if not os.path.exists(CSV_FILE):
        print("Veri dosyası bulunamadı.")
        return None

    df = pd.read_csv(CSV_FILE)
    df = df.dropna(subset=FEATURES)

    df["actual"] = np.random.randint(0, 2, size=len(df))  # Geçici çözüm, sonra gerçek sinyal etiketlemesi eklenecek

    X = df[FEATURES]
    y = df["actual"].astype(int)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)
    print(f"✅ Model kaydedildi: {MODEL_FILE}")
    return model
