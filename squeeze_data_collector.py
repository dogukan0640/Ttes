import requests

def get_binance_data():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    response = requests.get(url)
    return response.json()

def get_mock_funding_oi_data(symbol):
    # Gerçek API verileri yerine örnek/mock değerler (ileride Coinglass/Coinalyze API eklenecek)
    return {
        "fundingRate": -0.015,
        "openInterestChange": 6.2,
        "cvd": -82_000_000
    }

def analyze_squeeze(symbol, coin_data):
    last_price = float(coin_data["lastPrice"])
    high = float(coin_data["highPrice"])
    low = float(coin_data["lowPrice"])
    if last_price == 0:
    return None  # Hatalı veri varsa atla
atr_like = (high - low) / last_price

    volume = float(coin_data["volume"])
    price_change = float(coin_data["priceChangePercent"])

    if volume < 10000000 or atr_like > 0.015:
        return None

    extra = get_mock_funding_oi_data(symbol)
    if extra["fundingRate"] < -0.01 and extra["openInterestChange"] > 4 and extra["cvd"] < 0:
        return {
            "symbol": symbol,
            "lastPrice": last_price,
            "fundingRate": extra["fundingRate"],
            "openInterestChange": extra["openInterestChange"],
            "cvd": extra["cvd"],
            "atr": atr_like,
            "price_change": price_change,
            "ai_score": 91  # şimdilik sabit; sonraki sürümde modelden alınacak
        }
    return None

def get_squeeze_signals():
    results = []
    all_data = get_binance_data()
    for coin in all_data:
        symbol = coin["symbol"]
        if not symbol.endswith("USDT") or any(x in symbol for x in ["UP", "DOWN", "BULL", "BEAR"]):
            continue
        result = analyze_squeeze(symbol, coin)
        if result:
            results.append(result)
    return results
