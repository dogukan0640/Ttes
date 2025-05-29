
import joblib
import pandas as pd

def load_model():
    return joblib.load("model.pkl")

def predict(model, df):
    return model.predict_proba(df[["fr", "oi", "cvd", "price"]])[:, 1]
