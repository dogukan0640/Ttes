import pandas as pd
import numpy as np
import requests

def fetch_binance_klines(symbol="BTCUSDT", interval="1h", limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
    ])

    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

def calculate_features(df):
    df["ob"] = (df["high"] - df["low"]) / df["close"]  # Order Block proxy
    df["fvg"] = (df["close"] - df["open"]).abs() / df["open"]
    df["eqh"] = (df["high"].rolling(3).max() == df["high"]).astype(int)
    df["ema_trend"] = df["close"].ewm(span=21).mean()
    df["volume_spike"] = (df["volume"] > df["volume"].rolling(20).mean()).astype(int)
    df["atr_value"] = (df["high"] - df["low"]).rolling(14).mean()
    df["funding_rate"] = np.random.uniform(-0.005, 0.005)  # Yerine gerçek FR eklenecek
    df["open_interest"] = np.random.randint(100, 1000)  # Yerine gerçek OI eklenecek
    return df

def build_prediction_dataset(symbol="BTCUSDT"):
    df = fetch_binance_klines(symbol)
    df = calculate_features(df)
    df["actual"] = np.nan  # Eğitim için etiketlenmemiş veri
    df.to_csv("predictions.csv", index=False)
    print("✅ Canlı veriyle predictions.csv güncellendi.")
