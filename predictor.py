import pickle
import numpy as np

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_squeeze_score(fr, oi, cvd, atr):
    features = np.array([[fr, oi, cvd, atr]])
    proba = model.predict_proba(features)[0][1]
    return round(proba * 100, 2)