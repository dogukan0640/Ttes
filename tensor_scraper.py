import requests
from bs4 import BeautifulSoup

def get_tensor_cvd(symbol):
    symbol = symbol.replace("USDT", "")
    url = f"https://www.tensorcharts.com/?symbol=BINANCE:{symbol}USDT"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        script_tags = soup.find_all("script")
        for script in script_tags:
            if 'cvd' in script.text.lower():
                text = script.text
                lines = text.splitlines()
                for line in lines:
                    if "cvd" in line and ":" in line:
                        try:
                            value = float(line.strip().split(":")[-1].strip().replace(",", "").replace("}", ""))
                            return round(value, 2)
                        except:
                            continue
    except:
        pass
    return 0.0

def get_tensor_liquidation_trend(symbol):
    """
    TensorCharts üzerinde coin için hangi tarafın daha fazla likide olduğunu tahmin etmeye çalışır.
    Gerçek scraping yapılır, sadeleştirilmiş etiket döner: 'short-heavy', 'long-heavy', 'balanced'
    """
    symbol = symbol.replace("USDT", "")
    url = f"https://www.tensorcharts.com/?symbol=BINANCE:{symbol}USDT"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text().lower()
        short_count = text.count("short liquidation")
        long_count = text.count("long liquidation")
        if short_count > long_count * 1.5:
            return "short-heavy"
        elif long_count > short_count * 1.5:
            return "long-heavy"
        else:
            return "balanced"
    except:
        return "unknown"
