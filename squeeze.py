import requests

def get_binance_metrics():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    results = {}
    for symbol in symbols:
        price_data = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}").json()
        funding_data = requests.get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1").json()
        oi_data = requests.get(f"https://fapi.binance.com/futures/data/openInterestHist?symbol={symbol}&period=5m&limit=2").json()
        try:
            price_change = 100 * (float(oi_data[-1]["sumOpenInterest"]) - float(oi_data[-2]["sumOpenInterest"])) / float(oi_data[-2]["sumOpenInterest"])
        except:
            price_change = 0
        results[symbol] = {
            "price": float(price_data["price"]),
            "funding_rate": float(funding_data[0]["fundingRate"]),
            "open_interest_change": round(price_change, 2),
            "cvd": -1.5,  # Placeholder
            "liquidation": "short-heavy"  # Placeholder
        }
    return results

def detect_squeeze(symbol, data):
    score = 0
    if data["funding_rate"] < -0.005:
        score += 30
    if data["open_interest_change"] > 2:
        score += 20
    if data["cvd"] < -1:
        score += 20
    if data["liquidation"] == "short-heavy":
        score += 10

    message = (
        f"[SQUEEZE ALERT] {symbol}\n"
        f"Price: {data['price']}\n"
        f"Funding Rate: {data['funding_rate']}\n"
        f"Open Interest Δ: {data['open_interest_change']}%\n"
        f"CVD: {data['cvd']}\n"
        f"Likidasyonlar: {data['liquidation']}\n"
        f"Durum: Çok net SHORT SQUEEZE ihtimali!"
    )
    return {"score": score, "message": message}
