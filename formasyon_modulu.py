
import numpy as np

def detect_desc_triangle(df):
    look = df.tail(120)
    idx = np.arange(len(look))
    slope_high = np.polyfit(idx, look['high'], 1)[0] / np.mean(look['high'])
    slope_low  = np.polyfit(idx, look['low'], 1)[0]  / np.mean(look['low'])
    if slope_high < -0.001 and abs(slope_low) < 0.0003:
        return {'name': 'Alçalan Üçgen', 'desc': 'Düşen zirveler + yatay destek; düşüş riski'}
    return None

def detect_asc_triangle(df):
    look = df.tail(120)
    idx = np.arange(len(look))
    slope_low  = np.polyfit(idx, look['low'], 1)[0] / np.mean(look['low'])
    slope_high = np.polyfit(idx, look['high'], 1)[0] / np.mean(look['high'])
    if slope_low > 0.001 and abs(slope_high) < 0.0003:
        return {'name': 'Yükselen Üçgen', 'desc': 'Yükselen dipler + yatay direnç; breakout ihtimali'}
    return None

def detect_sym_triangle(df):
    look = df.tail(120)
    idx = np.arange(len(look))
    slope_high = np.polyfit(idx, look['high'], 1)[0] / np.mean(look['high'])
    slope_low  = np.polyfit(idx, look['low'], 1)[0]  / np.mean(look['low'])
    if slope_high < -0.001 and slope_low > 0.001:
        return {'name': 'Simetrik Üçgen', 'desc': 'Sıkışan fiyat aralığı; yön kırılımı beklenir'}
    return None

def detect_double_top(df):
    tops = df['high'].rolling(5).max()
    if (abs(tops.iloc[-1] - tops.iloc[-3]) / tops.iloc[-3]) < 0.005:
        return {'name': 'Double Top', 'desc': 'Çift tepe formasyonu; geri çekilme ihtimali'}
    return None

def detect_double_bottom(df):
    lows = df['low'].rolling(5).min()
    if (abs(lows.iloc[-1] - lows.iloc[-3]) / lows.iloc[-3]) < 0.005:
        return {'name': 'Double Bottom', 'desc': 'Çift dip formasyonu; yükseliş ihtimali'}
    return None

ALL_PATTERNS = [
    detect_desc_triangle,
    detect_asc_triangle,
    detect_sym_triangle,
    detect_double_top,
    detect_double_bottom
]
