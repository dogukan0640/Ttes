import pandas as pd
import lightgbm as lgb
import pickle

df = pd.read_csv("signals.csv")
df = df.dropna()

X = df[["fr", "oi_change", "cvd", "atr"]]
y = df["label"]

model = lgb.LGBMClassifier()
model.fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model eğitildi ve model.pkl dosyası oluşturuldu.")