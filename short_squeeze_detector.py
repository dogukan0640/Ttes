
import requests

def detect_squeeze(row):
    symbol = row["symbol"].replace("/", "")
    try:
        # 5m ve 15m funding rate
        fr_url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=5"
        fr_data = requests.get(fr_url).json()
        fr_5m = float(fr_data[-1]["fundingRate"]) if len(fr_data) >= 1 else 0.0
        fr_15m = float(fr_data[-3]["fundingRate"]) if len(fr_data) >= 3 else 0.0

        # Open interest 5m ve 15m karşılaştırması
        oi_url = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={symbol}&period=5m&limit=4"
        oi_data = requests.get(oi_url).json()
        if len(oi_data) >= 3:
            oi_5m = float(oi_data[-1]["sumOpenInterest"]) / float(oi_data[-2]["sumOpenInterest"])
            oi_15m = float(oi_data[-1]["sumOpenInterest"]) / float(oi_data[-3]["sumOpenInterest"])
        else:
            oi_5m = oi_15m = 1.0

        # Ana şartlar
        main_condition = row["fr"] < -0.01 and row["oi"] > 1.05 and row["cvd"] < 0

        # Uyumlu zaman dilimi analizi (FR düşüyor, OI artıyor)
        multi_tf_match = (fr_5m < 0 and fr_15m < 0) and (oi_5m > 1.01 and oi_15m > 1.01)

        return main_condition and multi_tf_match

    except Exception as e:
        print(f"Zaman dilimi uyumu alınamadı: {symbol}, Hata: {e}")
        return False
