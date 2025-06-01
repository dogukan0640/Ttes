import requests

def get_coin_data():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    response = requests.get(url)
    data = response.json()
    coin_data = {}
    for coin in data:
        if coin['symbol'].endswith('USDT') and not any(ex in coin['symbol'] for ex in ['DOWN', 'UP', 'BULL', 'BEAR']):
            coin_data[coin['symbol']] = {
                'priceChangePercent': float(coin['priceChangePercent']),
                'volume': float(coin['volume']),
                'lastPrice': float(coin['lastPrice'])
            }
    return coin_data

def evaluate_coin(symbol, data):
    if data['priceChangePercent'] > 3 and data['volume'] > 10000000:
        return f"🚀 {symbol} LONG sinyali \nFiyat: {data['lastPrice']} \n24h Değişim: %{data['priceChangePercent']:.2f} \nHacim: {data['volume']:.2f}"
    elif data['priceChangePercent'] < -3 and data['volume'] > 10000000:
        return f"🔻 {symbol} SHORT sinyali \nFiyat: {data['lastPrice']} \n24h Değişim: %{data['priceChangePercent']:.2f} \nHacim: {data['volume']:.2f}"
    return None