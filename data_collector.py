
import requests
import pandas as pd
import time

def get_all_data():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]  # örnek olarak birkaç coin, dinamik yapılabilir
    rows = []

    for symbol in symbols:
        try:
            # Funding Rate
            fr_url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"
            fr_data = requests.get(fr_url).json()
            funding_rate = float(fr_data[0]["fundingRate"]) if fr_data else 0.0

            # Open Interest
            oi_url = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={symbol}&period=5m&limit=2"
            oi_data = requests.get(oi_url).json()
            if len(oi_data) >= 2:
                oi_change = float(oi_data[-1]["sumOpenInterest"]) / float(oi_data[-2]["sumOpenInterest"])
            else:
                oi_change = 1.0

            # Price and Volume
            kline_url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1m&limit=2"
            kline_data = requests.get(kline_url).json()
            last_price = float(kline_data[-1][4])
            volume_change = float(kline_data[-1][5]) - float(kline_data[-2][5])

            # CVD approximation = volume difference
            cvd = volume_change

            rows.append({
                "symbol": symbol,
                "fr": funding_rate,
                "oi": oi_change,
                "cvd": cvd,
                "price": last_price
            })

            time.sleep(0.1)  # rate limit için kısa bekleme
        except Exception as e:
            print(f"Veri alınamadı: {symbol}, Hata: {e}")

    return pd.DataFrame(rows)
