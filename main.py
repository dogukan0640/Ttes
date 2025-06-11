import os
import requests
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pickle
from train_model import run_training_process # train_model.py'deki eğitim fonksiyonunu import ettik

# --- Konfigürasyon ---
TELEGRAM_TOKEN = "7744478523:AAEtRJar6uF7m0cxKfQh7r7TltXYxWwtmm0" 
CHAT_ID = "1009868232" 

# Analiz için eşik değerleri
MIN_HACIM_USDT = 50_000_000  # Minimum 24s Hacim (50 Milyon USDT)
MIN_FIYAT_DEGISIM_YUZDE = 3.0 # Dikkate alınacak minimum 24s fiyat değişim yüzdesi (mutlak değer)

# Tarama sıklığı (saniye cinsinden)
TARAMA_SIKLIGI_DAKIKA = 30 
TARAMA_SIKLIGI_SANİYE = TARAMA_SIKLIGI_DAKIKA * 60

# Mum grafiği zaman dilimi ve çekilecek mum sayısı
CANDLESTICK_INTERVAL = "1h" 
# ML tahmini ve gösterge hesaplamaları için yeterli geçmiş mum gerekli
# En uzun periyot EMA 200 olduğu için en az 200 mum + buffer = 250 ideal
NUM_CANDLES_TO_FETCH = 250 

# Kalıcı disk yolu (Render'da belirlediğiniz Mount Path ile aynı olmalı)
PERSISTENT_STORAGE_PATH = "/opt/render/persist" 

# Yapay Zeka Modeli Dosyası (artık kalıcı disk yoluyla)
MODEL_PATH = os.path.join(PERSISTENT_STORAGE_PATH, 'price_direction_model.pkl')
# Son eğitim zamanı dosyası (artık kalıcı disk yoluyla)
LAST_TRAIN_TIME_FILE = os.path.join(PERSISTENT_STORAGE_PATH, 'last_train_time.txt')

ai_model = None 

# Otomatik eğitim konfigürasyonu
# Her dağıtımda eğitimi zorlamak için bu değeri 0'a çektik.
# Eğitim yapıldıktan sonra tekrar 2 olarak ayarlayabilirsiniz.
AUTO_TRAIN_INTERVAL_DAYS = 2 # HER ZAMAN EĞİTİMİ TETİKLE!
TRAIN_SYMBOL = "BTCUSDT" 
TRAIN_INTERVAL = "1h"    
TRAIN_LIMIT = 1000       

# Tüm özellik sütunlarının doğru ve tutarlı sırası
FEATURE_COLUMNS = [
    'body_size', 'upper_shadow', 'lower_shadow', 'candle_range',
    'body_to_range_ratio', 'upper_shadow_to_body_ratio', 'lower_shadow_to_body_ratio',
    'high_low_ratio', 'close_to_open_ratio',
    'volume_change_pct_1', 'volume_change_pct_3', 'volume_change_pct_5',
    'price_change_pct_1', 'price_change_pct_3', 'price_change_pct_5',
    'sma_5', 'sma_10', 'volatility_5', 'volatility_10',
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_middle', 'bb_upper', 'bb_lower', 'bb_position', 'bb_width',
    'ema_20', 'ema_50', 'ema_200', 
    'ema_20_slope', 
    'atr', 
    'roc_14', 
    'dist_from_5_high', 'dist_from_5_low', 
    'dist_from_20_high', 'dist_from_20_low', 
]

# --- Yardımcı Fonksiyonlar ---
def telegram_sinyal_gonder(mesaj):
    """Telegram kanalına formatlı bir sinyal mesajı gönderir."""
    api_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        response = requests.post(api_url, json={'chat_id': CHAT_ID, 'text': mesaj, 'parse_mode': 'Markdown'})
        response.raise_for_status() 
        print(f"Telegram'a sinyal gönderildi. Yanıt: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Telegram'a gönderirken hata oluştu: {e}")

def load_ai_model(model_path):
    """Kaydedilmiş yapay zeka modelini yükler."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Yapay zeka modeli '{model_path}' başarıyla yüklendi.")
        return model
    except FileNotFoundError:
        print(f"HATA: Yapay zeka modeli '{model_path}' bulunamadı. Lütfen önce modeli eğitin ve kaydedin.")
        return None
    except Exception as e:
        print(f"Yapay zeka modeli yüklenirken hata oluştu: {e}")
        return None

def get_last_train_time():
    """Son eğitim zamanını dosyadan okur."""
    try:
        # Klasör yoksa oluştur (Render kalıcı disk için gerekli)
        os.makedirs(os.path.dirname(LAST_TRAIN_TIME_FILE), exist_ok=True)
        with open(LAST_TRAIN_TIME_FILE, 'r') as f:
            timestamp_str = f.read().strip()
            return datetime.fromisoformat(timestamp_str)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Son eğitim zamanı okunurken hata oluştu: {e}")
        return None

def save_last_train_time(timestamp):
    """Son eğitim zamanını dosyaya kaydeder."""
    try:
        # Klasör yoksa oluştur (Render kalıcı disk için gerekli)
        os.makedirs(os.path.dirname(LAST_TRAIN_TIME_FILE), exist_ok=True)
        with open(LAST_TRAIN_TIME_FILE, 'w') as f:
            f.write(timestamp.isoformat())
        print(f"Son eğitim zamanı kaydedildi: {timestamp}")
    except Exception as e:
        print(f"Son eğitim zamanı kaydedilirken hata oluştu: {e}")

# --- Binance API'den Veri Çekme Yardımcı Fonksiyonu ---
def _fetch_klines_and_ticker(base_url, market_type, symbol_suffix="USDT"):
    """Belirtilen Binance API base_url'inden klines ve 24hr ticker verilerini çeker."""
    all_symbols = {}
    try:
        tickers_url = f"{base_url}ticker/24hr"
        tickers_response = requests.get(tickers_url)
        tickers_response.raise_for_status()
        tickers_data = {item['symbol']: item for item in tickers_response.json()}
    except requests.exceptions.RequestException as e:
        print(f"Hata: {market_type} 24hr Ticker verileri alınamadı: {e}")
        return {}

    usdt_symbols = [s for s in tickers_data.keys() if s.endswith(symbol_suffix)]
    
    for symbol in usdt_symbols:
        try:
            klines_url = f"{base_url}klines"
            klines_params = {
                "symbol": symbol,
                "interval": CANDLESTICK_INTERVAL,
                "limit": NUM_CANDLES_TO_FETCH 
            }
            klines_response = requests.get(klines_url, params=klines_params)
            klines_response.raise_for_status()
            klines_data = klines_response.json()
            
            if symbol in tickers_data and klines_data and len(klines_data) == NUM_CANDLES_TO_FETCH:
                ticker_info = tickers_data[symbol]
                all_symbols[symbol] = {
                    "market_type": market_type, 
                    "lastPrice": float(ticker_info.get("lastPrice", 0)),
                    "quoteVolume": float(ticker_info.get("quoteVolume", 0)),
                    "priceChangePercent": float(ticker_info.get("priceChangePercent", 0)),
                    "klines": []
                }
                for kline in klines_data:
                    all_symbols[symbol]["klines"].append({
                        "open": float(kline[1]),
                        "high": float(kline[2]),
                        "low": float(kline[3]),
                        "close": float(kline[4]),
                        "volume": float(kline[5]),
                        "close_time": kline[6]
                    })
            
        except requests.exceptions.RequestException as e:
            print(f"Hata: {market_type} Sembol {symbol} için mum verileri alınamadı: {e}")
            continue 
        except (ValueError, TypeError, IndexError) as e:
            print(f"Hata: {market_type} Sembol {symbol} mum verileri işlenirken hata oluştu: {e}")
            continue
            
    return all_symbols

def binance_tum_veri_cek_spot_futures():
    """Hem Binance Spot hem de Futures USDT pariteleri için veri çeker."""
    all_market_data = {}

    print("Spot piyasa verileri çekiliyor...")
    spot_data = _fetch_klines_and_ticker("https://api.binance.com/api/v3/", "SPOT")
    all_market_data.update(spot_data) 

    print("Futures piyasa verileri çekiliyor...")
    futures_data = _fetch_klines_and_ticker("https://fapi.binance.com/fapi/v1/", "FUTURES")
    all_market_data.update(futures_data) 

    return all_market_data

# --- Mum Formasyonu Tanıma Fonksiyonları ---
def is_doji(candles):
    """Doji mumu formasyonunu tanır."""
    if not candles: return False
    last_candle = candles[-1]
    body = abs(last_candle["close"] - last_candle["open"])
    candle_range = last_candle["high"] - last_candle["low"]
    if candle_range == 0: return False
    return body < (candle_range * 0.1) 

def is_hammer(candles):
    """Çekiç mumu formasyonunu tanır."""
    if len(candles) < 1: return False
    c = candles[-1] 
    body = abs(c["close"] - c["open"])
    upper_shadow = c["high"] - max(c["open"], c["close"])
    lower_shadow = min(c["open"], c["close"]) - c["low"]
    if body > 0 and lower_shadow >= (body * 2) and upper_shadow < (body * 0.5): return True
    return False

def is_inverted_hammer(candles):
    """Ters Çekiç mumu formasyonunu tanır."""
    if len(candles) < 1: return False
    c = candles[-1] 
    body = abs(c["close"] - c["open"])
    upper_shadow = c["high"] - max(c["open"], c["close"])
    lower_shadow = min(c["open"], c["close"]) - c["low"]
    if body > 0 and upper_shadow >= (body * 2) and lower_shadow < (body * 0.5): return True
    return False

def is_bullish_engulfing(candles):
    """Boğa Yutan Boğa formasyonunu tanır."""
    if len(candles) < 2: return False
    prev_c = candles[-2] 
    curr_c = candles[-1] 
    if prev_c["close"] < prev_c["open"] and curr_c["close"] > curr_c["open"]:
        if curr_c["close"] >= prev_c["open"] and curr_c["open"] <= prev_c["close"]: return True
    return False

def is_morning_star(candles):
    """Sabah Yıldızı formasyonunu tanır."""
    if len(candles) < 3: return False
    c1 = candles[-3] 
    c2 = candles[-2] 
    c3 = candles[-1] 
    is_c1_bearish_long = c1["close"] < c1["open"] and (c1["open"] - c1["close"]) > (c1["high"] - c1["low"]) * 0.6
    is_c2_small_body = abs(c2["close"] - c2["open"]) < (c2["high"] - c2["low"]) * 0.3
    is_c3_bullish_long = c3["close"] > c3["open"] and (c3["close"] - c3["open"]) > (c3["high"] - c3["low"]) * 0.6
    c1_midpoint = (c1["open"] + c1["close"]) / 2
    if is_c1_bearish_long and is_c2_small_body and is_c3_bullish_long and c3["close"] > c1_midpoint: return True
    return False

def is_three_white_soldiers(candles):
    """Üç Beyaz Asker formasyonunu tanır."""
    if len(candles) < 3: return False
    c1 = candles[-3]
    c2 = candles[-2]
    c3 = candles[-1]
    if not (c1["close"] > c1["open"] and c2["close"] > c2["open"] and c3["close"] > c3["open"]): return False
    if not (c2["close"] > c1["close"] and c3["close"] > c2["close"]): return False
    if not (c2["open"] >= c1["open"] and c2["open"] <= c1["close"] and \
            c3["open"] >= c2["open"] and c3["open"] <= c2["close"]): return False
    avg_range = sum([c["high"] - c["low"] for c in candles[-3:]]) / 3
    if not (abs(c1["close"] - c1["open"]) > avg_range * 0.3 and \
            abs(c2["close"] - c2["open"]) > avg_range * 0.3 and \
            abs(c3["close"] - c3["open"]) > avg_range * 0.3): return False
    return True

def is_piercing_pattern(candles):
    """Delici Mum Formasyonu'nu tanır."""
    if len(candles) < 2: return False
    prev_c = candles[-2] 
    curr_c = candles[-1] 
    if not (prev_c["close"] < prev_c["open"]): return False
    if not (curr_c["close"] > curr_c["open"]): return False
    if not (curr_c["open"] < prev_c["close"]): return False
    midpoint_prev_c = (prev_c["open"] + prev_c["close"]) / 2
    if not (curr_c["close"] > midpoint_prev_c and curr_c["close"] < prev_c["open"]): return True
    return False

def is_bullish_harami(candles):
    """Boğa Harami formasyonunu tanır."""
    if len(candles) < 2: return False
    prev_c = candles[-2] 
    curr_c = candles[-1] 
    if not (prev_c["close"] < prev_c["open"] and abs(prev_c["close"] - prev_c["open"]) > (prev_c["high"] - prev_c["low"]) * 0.4): return False
    if not (curr_c["close"] > curr_c["open"] and abs(curr_c["close"] - curr_c["open"]) < (curr_c["high"] - curr_c["low"]) * 0.5): return False
    if not (curr_c["open"] > prev_c["close"] and curr_c["close"] < prev_c["open"]): return True
    return False

def is_three_inside_up(candles):
    """Üç İçeriye Yükseliş formasyonunu tanır."""
    if len(candles) < 3: return False
    c1 = candles[-3] 
    c2 = candles[-2] 
    c3 = candles[-1] 
    if not is_bullish_harami(candles[:-1] + [c2]): return False
    if not (c3["close"] > c3["open"]): return False
    if not (c3["close"] > c1["open"]): return True
    return False

def is_rising_three_methods(candles):
    """Yükselen Üç Metot formasyonunu tanır."""
    if len(candles) < 5: return False
    c1 = candles[-5] 
    c2 = candles[-4] 
    c3 = candles[-3] 
    c4 = candles[-2] 
    c5 = candles[-1] 
    if not (c1["close"] > c1["open"] and abs(c1["close"] - c1["open"]) > (c1["high"] - c1["low"]) * 0.6): return False
    for c in [c2, c3, c4]:
        if not (c["close"] < c["open"] and \
                abs(c["close"] - c["open"]) < (c["high"] - c["low"]) * 0.5 and \
                c["high"] <= c1["high"] and c["low"] >= c1["low"]): return False
    if not (c5["close"] > c5["open"] and abs(c5["close"] - c5["open"]) > (c5["high"] - c5["low"]) * 0.6): return False
    if not (c5["open"] > c4["close"] and c5["close"] > c1["close"]): return True
    return False

# --- Özellik Mühendisliği (Tahmin için canlı veriden) ---
def extract_features_for_prediction(klines_data):
    """
    Canlı mum verilerinden modelin beklediği özellikleri çıkarır.
    Bu fonksiyon, train_model.py'deki prepare_data_for_training ile AYNI MANTIKTA olmalı.
    """
    # En uzun EMA 200 periyot gerektiriyor, ROC/ATR 14, BB 20 -> en az 200 mum + buffer
    if len(klines_data) < 200: 
        print(f"Uyarı: ML tahmini için yeterli mum verisi yok ({len(klines_data)} yerine en az 200 gerekli).")
        return None

    df = pd.DataFrame(klines_data)
    
    features = pd.DataFrame(index=df.index)

    features['body_size'] = abs(df['close'] - df['open'])
    features['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    features['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    features['candle_range'] = df['high'] - df['low'] 

    features['body_to_range_ratio'] = features['body_size'] / features['candle_range'].replace(0, 1e-9)
    features['upper_shadow_to_body_ratio'] = features['upper_shadow'] / features['body_size'].replace(0, 1e-9)
    features['lower_shadow_to_body_ratio'] = features['lower_shadow'] / features['body_size'].replace(0, 1e-9)
    features['high_low_ratio'] = df['high'] / df['low'].replace(0, 1e-9)
    features['close_to_open_ratio'] = df['close'] / df['open'].replace(0, 1e-9)

    features['volume_change_pct_1'] = df['volume'].pct_change().fillna(0)
    features['volume_change_pct_3'] = df['volume'].pct_change(periods=3).fillna(0)
    features['volume_change_pct_5'] = df['volume'].pct_change(periods=5).fillna(0)

    features['price_change_pct_1'] = df['close'].pct_change().fillna(0)
    features['price_change_pct_3'] = df['close'].pct_change(periods=3).fillna(0)
    features['price_change_pct_5'] = df['close'].pct_change(periods=5).fillna(0)

    features['sma_5'] = df['close'].rolling(window=5).mean().fillna(0)
    features['sma_10'] = df['close'].rolling(window=10).mean().fillna(0)

    features['volatility_5'] = df['close'].rolling(window=5).std().fillna(0)
    features['volatility_10'] = df['close'].rolling(window=10).std().fillna(0)

    # --- Mevcut Gelişmiş Özellikler (Teknik Göstergeler) ---
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-9)
    features['rsi'] = 100 - (100 / (1 + rs))
    features['rsi'] = features['rsi'].fillna(0)

    # MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    features['macd'] = exp12 - exp26
    features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
    features['macd_hist'] = features['macd'] - features['macd_signal']
    features[['macd', 'macd_signal', 'macd_hist']] = features[['macd', 'macd_signal', 'macd_hist']].fillna(0)

    # Bollinger Bantları
    window_bb = 20
    num_std_dev_bb = 2
    features['bb_middle'] = df['close'].rolling(window=window_bb).mean()
    std_dev_bb = df['close'].rolling(window=window_bb).std()
    features['bb_upper'] = features['bb_middle'] + (std_dev_bb * num_std_dev_bb)
    features['bb_lower'] = features['bb_middle'] - (std_dev_bb * num_std_dev_bb)
    features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower']).replace(0, 1e-9)
    features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle'].replace(0, 1e-9)
    features[['bb_middle', 'bb_upper', 'bb_lower', 'bb_position', 'bb_width']] = features[['bb_middle', 'bb_upper', 'bb_lower', 'bb_position', 'bb_width']].fillna(0)

    # --- YENİ EKLENEN TEKNİK GÖSTERGELER (Trend, Momentum, Volatilite, Fibonacci-vari) ---
    # EMA (Exponential Moving Averages)
    features['ema_20'] = df['close'].ewm(span=20, adjust=False).mean().fillna(0)
    features['ema_50'] = df['close'].ewm(span=50, adjust=False).mean().fillna(0)
    features['ema_200'] = df['close'].ewm(span=200, adjust=False).mean().fillna(0)
    
    # EMA Eğimi (Slope)
    features['ema_20_slope'] = features['ema_20'].diff().fillna(0)

    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift())
    low_prev_close = abs(df['low'] - df['close'].shift())
    tr = pd.DataFrame({'hl': high_low, 'hpc': high_prev_close, 'lpc': low_prev_close}).max(axis=1)
    features['atr'] = tr.rolling(window=14).mean().fillna(0)

    # ROC (Rate of Change) - Momentum
    features['roc_14'] = df['close'].pct_change(periods=14).fillna(0)

    # Fiyatın N-periyot Yüksek/Düşük seviyelerine göre uzaklığı (Fibonacci-vari)
    features['dist_from_5_high'] = (df['high'].rolling(window=5).max() - df['close']) / df['close'].replace(0, 1e-9)
    features['dist_from_5_low'] = (df['close'] - df['low'].rolling(window=5).min()) / df['close'].replace(0, 1e-9)
    features['dist_from_20_high'] = (df['high'].rolling(window=20).max() - df['close']) / df['close'].replace(0, 1e-9)
    features['dist_from_20_low'] = (df['close'] - df['low'].rolling(window=20).min()) / df['close'].replace(0, 1e-9)
    features[['dist_from_5_high', 'dist_from_5_low', 'dist_from_20_high', 'dist_from_20_low']] = features[['dist_from_5_high', 'dist_from_5_low', 'dist_from_20_high', 'dist_from_20_low']].fillna(0)

    final_features_df = features.iloc[-1:] 
    
    # Önemli: Özellik sütunlarını FEATURE_COLUMNS listesine göre sırala
    final_features_df = final_features_df[FEATURE_COLUMNS]

    final_features_df = final_features_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return final_features_df


# --- Ana Analiz Fonksiyonu ---
def piyasayi_tara_ve_analiz_et():
    """Binance piyasalarını tarar, yükseliş potansiyeli olan formasyonları ve ML tahminlerini bulur ve sinyal gönderir."""
    global ai_model 

    current_time = datetime.now()

    # Model yüklü değilse VEYA eğitilmesi gerekiyorsa (2 gün geçtiyse) eğitimi tetikle
    last_train_time = get_last_train_time()
    
    should_train = False
    if ai_model is None: # Model bellekte yüklü değilse (ilk başlangıç veya önceki hata)
        print("Yapay zeka modeli bellekte yüklü değil. Yüklü model dosyası kontrol ediliyor...")
        if not os.path.exists(MODEL_PATH): # Kalıcı diskte model dosyası yoksa
            print(f"Model dosyası '{MODEL_PATH}' bulunamadı. Eğitim tetikleniyor.")
            should_train = True
        else: # Model dosyası varsa yükle
            ai_model = load_ai_model(MODEL_PATH)
            if ai_model is None: # Yükleme başarısız olursa eğitim tetikle
                print("Yüklü model dosyası bozuk veya yüklenemedi. Eğitim tetikleniyor.")
                should_train = True
    
    # Model yüklü ise ve eğitim süresi dolmuşsa
    if ai_model is not None and last_train_time:
        if (current_time - last_train_time) > timedelta(days=AUTO_TRAIN_INTERVAL_DAYS):
            print(f"Son eğitim {AUTO_TRAIN_INTERVAL_DAYS} günden daha eski. Yeniden eğitim tetikleniyor.")
            should_train = True
        else:
            print(f"Son eğitim {last_train_time.strftime('%Y-%m-%d %H:%M:%S')} tarihinde yapılmış. Yeniden eğitim gerekli değil.")
    elif ai_model is not None and last_train_time is None: # Model yüklü ama zaman dosyası yoksa (ilk kez kalıcı disk ile başlatma)
        print("Model yüklü ancak son eğitim zamanı bulunamadı. İlk eğitim tetikleniyor.")
        should_train = True # İlk kez kalıcı disk ile çalışırken zamanı kaydetmek için eğit.

    if should_train:
        print("Yapay zeka modeli eğitimi başlatılıyor...")
        telegram_sinyal_gonder(f"⏳ **Yapay Zeka Modeli Eğitiliyor/Yeniden Eğitiliyor!** ⏳\nBu işlem biraz sürebilir.")
        
        success = run_training_process(TRAIN_SYMBOL, TRAIN_INTERVAL, TRAIN_LIMIT) 
        
        if success:
            ai_model = load_ai_model(MODEL_PATH) # Eğitilen yeni modeli yükle
            save_last_train_time(current_time) # Başarılı eğitim sonrası zamanı kaydet
            telegram_sinyal_gonder(f"✅ **Yapay Zeka Modeli Başarıyla Eğitildi ve Yüklendi!** ✅")
        else:
            telegram_sinyal_gonder(f"❌ **Yapay Zeka Modeli Eğitimi Başarısız Oldu!** ❌\nAI tahmini devre dışı kalacak.")
            print("Eğitim başarısız, AI tahmini devre dışı kalacak.")
            ai_model = None 

    print(f"\n--- {current_time.strftime('%Y-%m-%d %H:%M:%S')} - Piyasa Taraması Başlandı ---")
    
    potansiyel_adaylar = []

    piyasa_verileri = binance_tum_veri_cek_spot_futures()
    if piyasa_verileri is None or not piyasa_verileri:
        print("Piyasa verileri çekilemedi veya boş geldi, tarama atlanıyor.")
        return

    for symbol, data in piyasa_verileri.items(): 
        try:
            market_type = data["market_type"] 
            hacim_24s = data["quoteVolume"]
            son_fiyat = data["lastPrice"]
            fiyat_degisim_yuzde = data["priceChangePercent"]
            klines = data["klines"]

            if len(klines) < NUM_CANDLES_TO_FETCH or \
               hacim_24s < MIN_HACIM_USDT or \
               abs(fiyat_degisim_yuzde) < MIN_FIYAT_DEGISIM_YUZDE:
                continue
            
            tespit_edilen_formasyonlar = []
            ml_tahmin = "Yok" 
            ml_tahmin_puan = 0 
            
            if ai_model is not None:
                features_for_pred = extract_features_for_prediction(klines)
                if features_for_pred is not None and not features_for_pred.empty:
                    try:
                        prediction = ai_model.predict(features_for_pred)[0]
                        prediction_proba = ai_model.predict_proba(features_for_pred)[0] 
                        
                        if prediction == 1 and prediction_proba[1] > 0.60: 
                            ml_tahmin = f"AI Tahmini: Yükseliş ({prediction_proba[1]*100:.1f}%)"
                            ml_tahmin_puan = 3 
                        elif prediction == 0 and prediction_proba[0] > 0.60: 
                             ml_tahmin = f"AI Tahmini: Düşüş ({prediction_proba[0]*100:.1f}%)"
                        else:
                            ml_tahmin = f"AI Tahmini: Kararsız ({prediction_proba[0]*100:.1f}% Düşüş, {prediction_proba[1]*100:.1f}% Yükseliş)"

                    except Exception as ml_e:
                        print(f"Sembol {symbol} için AI tahmini yapılırken hata oluştu: {ml_e}")
                        ml_tahmin = "AI Tahmini: Hata"
            
            if is_hammer(klines):
                tespit_edilen_formasyonlar.append("Çekiç (Hammer) - Dönüş")
            if is_inverted_hammer(klines):
                tespit_edilen_formasyonlar.append("Ters Çekiç (Inverted Hammer) - Dönüş")
            if is_bullish_engulfing(klines):
                tespit_edilen_formasyonlar.append("Boğa Yutan Boğa (Bullish Engulfing) - Dönüş")
            if is_morning_star(klines):
                tespit_edilen_formasyonlar.append("Sabah Yıldızı (Morning Star) - Dönüş")
            if is_three_white_soldiers(klines):
                tespit_edilen_formasyonlar.append("Üç Beyaz Asker (Three White Soldiers) - Devam/Dönüş")
            
            if is_piercing_pattern(klines):
                tespit_edilen_formasyonlar.append("Delici Mum Formasyonu (Piercing Pattern) - Dönüş")
            if is_bullish_harami(klines):
                tespit_edilen_formasyonlar.append("Boğa Harami (Bullish Harami) - Dönüş")
            if is_three_inside_up(klines):
                tespit_edilen_formasyonlar.append("Üç İçeriye Yükseliş (Three Inside Up) - Dönüş")
            if is_rising_three_methods(klines):
                tespit_edilen_formasyonlar.append("Yükselen Üç Metot (Rising Three Methods) - Devam")

            if is_doji(klines): 
                if "Doji" not in tespit_edilen_formasyonlar: 
                    tespit_edilen_formasyonlar.append("Doji (Kararsızlık)")
            
            if tespit_edilen_formasyonlar or (ml_tahmin_puan > 0):
                puan = len(tespit_edilen_formasyonlar) * 2 
                if hacim_24s > 100_000_000:
                    puan += 1
                if abs(fiyat_degisim_yuzde) > 7:
                    puan += 1
                puan += ml_tahmin_puan 

                aday_bilgisi = {
                    "symbol": symbol,
                    "market_type": market_type, 
                    "puan": puan,
                    "son_fiyat": son_fiyat,
                    "hacim_24s": f"{hacim_24s:,.0f} USDT",
                    "fiyat_degisim_yuzde": f"{fiyat_degisim_yuzde:.2f}%",
                    "formasyonlar": ", ".join(tespit_edilen_formasyonlar) if tespit_edilen_formasyonlar else "Yok",
                    "ml_tahmin": ml_tahmin
                }
                potansiyel_adaylar.append(aday_bilgisi)

        except KeyError as e:
            print(f"Eksik veri anahtarı için {symbol} ({market_type}): {e}")
            continue
        except Exception as e:
            print(f"Sembol {symbol} ({market_type}) işlenirken beklenmeyen hata oluştu: {e}")
            continue

    potansiyel_adaylar.sort(key=lambda x: x['puan'], reverse=True)

    if potansiyel_adaylar:
        mesaj = f"📈 **Yükseliş Potansiyeli Formasyon ve Yapay Zeka Sinyalleri ({CANDLESTICK_INTERVAL})** 📈\n_{current_time.strftime('%d-%m-%Y %H:%M')} itibarıyla_\n"
        mesaj += "--------------------------------------\n"
        for aday in potansiyel_adaylar:
            mesaj += f"*{aday['symbol']}* ({aday['market_type']}) | Puan: *{aday['puan']}*\n" 
            mesaj += f"▫️ Fiyat: `{aday['son_fiyat']}`\n"
            mesaj += f"▫️ 24s Değişim: `{aday['fiyat_degisim_yuzde']}`\n"
            mesaj += f"▫️ Hacim (24s): `{aday['hacim_24s']}`\n"
            mesaj += f"▫️ Tespit Edilen Formasyonlar: `{aday['formasyonlar']}`\n"
            mesaj += f"▫️ AI Tahmini: `{aday['ml_tahmin']}`\n" 
            mesaj += "--------------------------------------\n"
        
        telegram_sinyal_gonder(mesaj)
    else:
        print("Koşulları sağlayan potansiyel aday bulunamadı.")
        
    print("Tarama tamamlandı.")


# --- Botu Çalıştırma ---
if __name__ == "__main__":
    if not TELEGRAM_TOKEN or not CHAT_ID or TELEGRAM_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
        print("HATA: Lütfen TELEGRAM_TOKEN ve CHAT_ID'yi main.py dosyası içinde kendi bilgilerinizle güncelleyin.")
        exit() 

    # Render kalıcı diskinin klasörünün mevcut olduğundan emin olun
    os.makedirs(PERSISTENT_STORAGE_PATH, exist_ok=True)

    # Bot ilk çalıştığında model yüklenemezse veya eğitim süresi dolduysa eğitim tetiklenecek.
    # AUTO_TRAIN_INTERVAL_DAYS = 0 ayarı sayesinde her deployda eğitim tetiklenecek.
    ai_model = load_ai_model(MODEL_PATH)
    last_train_time = get_last_train_time()
    current_time = datetime.now()

    should_train_on_start = False
    if ai_model is None: # Model hiç yüklenememişse (ilk defa veya dosya bozuksa)
        should_train_on_start = True
    elif last_train_time is None: # Model yüklü ama zaman damgası yoksa (ilk defa kalıcı diskle)
        should_train_on_start = True
    elif (current_time - last_train_time) > timedelta(days=AUTO_TRAIN_INTERVAL_DAYS):
        should_train_on_start = True

    if should_train_on_start:
        print("Bot başlatılırken yapay zeka modeli eğitimi/yeniden eğitimi tetikleniyor...")
        telegram_sinyal_gonder(f"⏳ **Yapay Zeka Modeli Eğitiliyor/Yeniden Eğitiliyor!** ⏳\nBu işlem biraz sürebilir.")
        success = run_training_process(TRAIN_SYMBOL, TRAIN_INTERVAL, TRAIN_LIMIT)
        if success:
            ai_model = load_ai_model(MODEL_PATH)
            save_last_train_time(current_time)
            telegram_sinyal_gonder(f"✅ **Yapay Zeka Modeli Başarıyla Eğitildi ve Yüklendi!** ✅")
        else:
            telegram_sinyal_gonder(f"❌ **Yapay Zeka Modeli Eğitimi Başarısız Oldu!** ❌\nAI tahmini devre dışı kalacak.")
            ai_model = None

    baslangic_mesaji = f"🚀 **Kripto Formasyon & AI Botu Başlatıldı (Spot & Futures)!** 🚀\nTarama her {TARAMA_SIKLIGI_DAKIKA} dakikada bir yapılacak. Mum aralığı: {CANDLESTICK_INTERVAL}"
    if ai_model: 
        baslangic_mesaji += "\nYapay Zeka tahmini aktif!"
    telegram_sinyal_gonder(baslangic_mesaji)
    print("Bot başlatıldı ve Telegram'a bildirim gönderildi.")

    try: 
        while True:
            print(f"{TARAMA_SIKLIGI_DAKIKA} dakika sonraki tarama için bekleniyor...")
            time.sleep(TARAMA_SIKLIGI_SANİYE)
            piyasayi_tara_ve_analiz_et()
    except Exception as e:
        critical_error_msg = f"🔥🔥🔥 **KRİTİK HATA! Bot Durdu!** 🔥🔥🔥\nLütfen Render loglarını kontrol edin.\nHata Detayı: `{type(e).__name__}: {e}`"
        telegram_sinyal_gonder(critical_error_msg)
        print(f"KRİTİK SİSTEM HATASI: {e}")
