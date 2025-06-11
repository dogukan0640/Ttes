# train_model.py (Refactored for auto-training)
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import time
from datetime import datetime, timedelta

# --- Veri Çekme Fonksiyonu (Geçmiş Mum Verileri İçin) ---
def get_historical_klines(symbol, interval, limit):
    """Binance Futures API'sinden geçmiş mum verilerini çeker."""
    base_url = "https://fapi.binance.com/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        klines = response.json()
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        return df
    except requests.exceptions.RequestException as e:
        print(f"Hata: Geçmiş mum verileri alınamadı: {e}")
        print(f"Hata Detayı: {e.response.text if e.response else 'Yanıt Yok'}") 
        return None

# --- Özellik Mühendisliği ve Etiketleme Fonksiyonu ---
def prepare_data_for_training(df):
    """
    Model eğitimi için daha gelişmiş özellikleri çıkarır ve hedef değişkeni oluşturur.
    Hedef: Bir sonraki mumun kapanışı mevcut mumun kapanışından yüksek mi olacak? (1: Yükseliş, 0: Düşüş/Kararsızlık)
    """
    if df is None or df.empty:
        return None, None

    features = pd.DataFrame(index=df.index)

    # 1. Mum gövdesi ve gölge özellikleri
    features['body_size'] = abs(df['close'] - df['open'])
    features['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    features['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    features['candle_range'] = df['high'] - df['low']

    # 2. Oran bazlı özellikler
    features['body_to_range_ratio'] = features['body_size'] / features['candle_range'].replace(0, 1e-9) 
    features['upper_shadow_to_body_ratio'] = features['upper_shadow'] / features['body_size'].replace(0, 1e-9)
    features['lower_shadow_to_body_ratio'] = features['lower_shadow'] / features['body_size'].replace(0, 1e-9)
    features['high_low_ratio'] = df['high'] / df['low'].replace(0, 1e-9)
    features['close_to_open_ratio'] = df['close'] / df['open'].replace(0, 1e-9)

    # 3. Hacim ve Fiyat Değişim Yüzdeleri (Birden Fazla Periyot Üzerinden)
    features['volume_change_pct_1'] = df['volume'].pct_change().fillna(0) 
    features['volume_change_pct_3'] = df['volume'].pct_change(periods=3).fillna(0) 
    features['volume_change_pct_5'] = df['volume'].pct_change(periods=5).fillna(0) 

    features['price_change_pct_1'] = df['close'].pct_change().fillna(0) 
    features['price_change_pct_3'] = df['close'].pct_change(periods=3).fillna(0) 
    features['price_change_pct_5'] = df['close'].pct_change(periods=5).fillna(0) 

    # 4. Hareketli Ortalamalar (SMA)
    features['sma_5'] = df['close'].rolling(window=5).mean().fillna(0)
    features['sma_10'] = df['close'].rolling(window=10).mean().fillna(0)

    # 5. Volatilite (Standard Sapma)
    features['volatility_5'] = df['close'].rolling(window=5).std().fillna(0)
    features['volatility_10'] = df['close'].rolling(window=10).std().fillna(0)

    # Hedef değişken (etiket): Bir sonraki mumun kapanışı mevcut mumun kapanışından büyük mü?
    features['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # NaN veya sonsuz değerleri temizle
    features = features.dropna()
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

    X = features.drop('target', axis=1)
    y = features['target']
    
    return X, y

# --- Model Eğitimi ve Kaydetme Fonksiyonu ---
def train_and_save_model(X, y, model_path='price_direction_model.pkl'):
    """Modeli eğitir ve .pkl dosyasına kaydeder."""
    if X.empty or y.empty:
        print("Eğitim verisi boş, model eğitilemiyor.")
        return False

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Model Eğitim Başarısı:")
    print(f"Doğruluk (Accuracy): {accuracy_score(y_test, y_pred):.2f}")
    print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model başarıyla '{model_path}' olarak kaydedildi.")
    return True

# --- Ana Eğitim Sürecini Başlatma Fonksiyonu ---
def run_training_process(symbol="BTCUSDT", interval="1h", limit=1000):
    """
    Yapay zeka modelini eğitir ve kaydeder.
    Bu fonksiyon main.py tarafından çağrılacaktır.
    """
    print(f"\n--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Yapay Zeka Modeli Eğitimi Başladı ---")
    print(f"{symbol} için {interval} aralığında {limit} adet geçmiş mum verisi çekiliyor...")
    
    historical_df = get_historical_klines(symbol, interval, limit)

    if historical_df is not None and not historical_df.empty:
        print("Veriler çekildi, özellikler çıkarılıyor ve etiketleniyor...")
        X, y = prepare_data_for_training(historical_df)
        
        if X is not None and y is not None and not X.empty and not y.empty:
            print("Model eğitiliyor ve kaydediliyor...")
            if train_and_save_model(X, y):
                print("Yapay Zeka Modeli Eğitimi Başarıyla Tamamlandı.")
                return True
            else:
                print("Model Eğitimi Başarısız Oldu.")
                return False
        else:
            print("Veri hazırlığı sırasında problem oluştu (boş özellik/hedef), eğitim yapılamadı.")
            return False
    else:
        print("Geçmiş veri çekilemedi veya boş geldi, eğitim iptal edildi.")
        return False

# Eğer bu dosya doğrudan çalıştırılırsa, manuel eğitim yap
if __name__ == "__main__":
    # Manuel çalıştırma için varsayılan değerler
    DEFAULT_SYMBOL = "BTCUSDT"
    DEFAULT_INTERVAL = "1h"
    DEFAULT_LIMIT = 1000 
    run_training_process(DEFAULT_SYMBOL, DEFAULT_INTERVAL, DEFAULT_LIMIT)
