"""
USDT Pair Scanner Bot for Binance & MEXC
-------------------------------------------------------------
* Scans **all USDT‑quoted perpetual pairs** on Binance Futures and MEXC Contracts
* Detects key price‑action formations (currently: **Descending Triangle**; hooks for others: OB/MSB, Double Top/Bottom …)
* Pulls **funding‑rate** & **open‑interest** (Binance API, MEXC holdVol) plus 24 h volume & ATR volatility
* Sends condensed alerts to Telegram:  
  ``<EXCHANGE> | <SYMBOL> | Formasyon: <NAME> (<DESC>) | Hacim: <vol24> | ATR: <atr> | FR: <fr%> | OI: <oi> | Saat: <HH:MM> | Fiyat: <last>``
* Designed for Replit / VPS / Render (single‑file deploy)

>  ⚠️  **You only need to add** environment vars `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` plus (optionally) `BINANCE_API_KEY/SECRET` and `MEXC_API_KEY/SECRET` if you hit private‑endpoint rate limits.

Requirements (``pip install -r requirements.txt``):
    ccxt~=4.2  
    python-telegram-bot~=13.15  # PTB v13, works on Replit  
    aiohttp~=3.9  
    pandas~=2.2  
    numpy~=1.26
"""

import os, asyncio, time, math, json, logging, aiohttp
from datetime import datetime, timezone

import ccxt.async_support as ccxt  # async version
import pandas as pd
import numpy as np
from telegram import Bot

# ---------------------------------------------------------------------------
# Config --------------------------------------------------------------------
SCAN_INTERVAL_SEC = 900   # 15 dakikada bir tarama
LOOKBACK_CANDLES   = 120  # Formasyon analizinde kullanılacak mum adedi
TIMEFRAME          = '15m'

TELEGRAM_TOKEN   = os.getenv('TELEGRAM_TOKEN'7744478523:AAEtRJar6uF7m0cxKfQh7r7TltXYxWwtmm0'PASTE-YOURS')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID'1009868232'PASTE-YOURS')

bot = Bot(token=TELEGRAM_TOKEN)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ---------------------------------------------------------------------------
# Utils ---------------------------------------------------------------------

def pct(x, y):
    return (x - y) / y if y else 0.0

async def send(msg: str):
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
    except Exception as e:
        logging.error(f'Telegram send failed: {e}')

# ---------------------------------------------------------------------------
# Pattern Detection ---------------------------------------------------------

def detect_desc_triangle(df: pd.DataFrame) -> dict | None:
    """Very lightweight descending‑triangle detector.
    Returns dict summary or None."""
    if len(df) < 20:
        return None
    look = df.tail(LOOKBACK_CANDLES)
    idx = np.arange(len(look))

    highs = look['high'].values
    lows  = look['low'].values

    # Linear regression slopes (normalized)
    slope_high = np.polyfit(idx, highs, 1)[0] / np.mean(highs)
    slope_low  = np.polyfit(idx, lows, 1)[0]  / np.mean(lows)

    # Conditions ⇒ upper trendline down, lower trendline ~flat
    if slope_high < -0.001 and abs(slope_low) < 0.0003:
        return {
            'name': 'Alçalan Üçgen',
            'desc': 'Düşen zirveler + yatay destek; olası düşüş kırılımı',
        }
    return None

# Placeholder hooks – you can extend with your own PA/OB/MSB detectors.
PATTERN_FUNCS = [detect_desc_triangle]

# ---------------------------------------------------------------------------
# Exchange Helpers ----------------------------------------------------------

async def load_usdt_pairs(exchange: ccxt.Exchange):
    markets = await exchange.load_markets()
    return [s for s, m in markets.items() if (m.get('quote') in ('USDT', 'USDT:USDT') and m['active'])]

async def fetch_ohlcv(exchange: ccxt.Exchange, symbol: str):
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LOOKBACK_CANDLES)
        df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','vol'])
        return df
    except Exception as e:
        logging.debug(f"{exchange.id} {symbol} OHLCV err: {e}")
        return None

async def fetch_binance_fr_oi(session: aiohttp.ClientSession, symbol: str):
    base_url = 'https://fapi.binance.com'
    fr_url   = f'{base_url}/fapi/v1/fundingRate?symbol={symbol}&limit=1'
    oi_url   = f'{base_url}/fapi/v1/openInterest?symbol={symbol}'
    try:
        fr_json, oi_json = await asyncio.gather(
            session.get(fr_url), session.get(oi_url))
        fr_data = await fr_json.json()
        oi_data = await oi_json.json()
        fr = float(fr_data[-1]['fundingRate']) if fr_data else None
        oi = float(oi_data['openInterest']) if 'openInterest' in oi_data else None
        return fr, oi
    except Exception as e:
        logging.debug(f"Binance FR/OI err {symbol}: {e}")
        return None, None

async def fetch_mexc_fr_oi(session: aiohttp.ClientSession, symbol: str):
    # MEXC uses underscore e.g. BTC_USDT
    q = symbol.replace('/', '_').upper()
    url = f'https://contract.mexc.com/api/v1/contract/ticker?symbol={q}'
    try:
        r = await session.get(url)
        js = await r.json()
        if js.get('success'):
            data = js['data']
            return float(data.get('fundingRate', 0)), float(data.get('holdVol', 0))
    except Exception as e:
        logging.debug(f"MEXC FR/OI err {symbol}: {e}")
    return None, None

# ---------------------------------------------------------------------------
# Core Scan -----------------------------------------------------------------

async def scan_exchange(id_: str):
    ex_opts = {
        'enableRateLimit': True,
        'timeout': 30_000,
        'options': {'defaultType': 'future'}  # both Binance & MEXC names ok
    }
    exchange = getattr(ccxt, id_)(ex_opts)

    usdt_pairs = await load_usdt_pairs(exchange)
    logging.info(f"{id_}: {len(usdt_pairs)} USDT futures pairs loaded")

    async with aiohttp.ClientSession() as session:
        for sym in usdt_pairs:
            df = await fetch_ohlcv(exchange, sym)
            if df is None:
                continue
            pattern = None
            for fn in PATTERN_FUNCS:
                pattern = fn(df)
                if pattern:
                    break
            if pattern:
                if id_ == 'binance':
                    fr, oi = await fetch_binance_fr_oi(session, sym.replace('/', ''))
                else:  # mexc
                    fr, oi = await fetch_mexc_fr_oi(session, sym)

                atr = (df['high'] - df['low']).rolling(window=14).mean().iloc[-1]
                vol24 = df['vol'].sum()
                last  = df['close'].iloc[-1]
                ts    = datetime.now(timezone.utc).strftime('%H:%M UTC')

                msg = (f"{id_.upper()} | {sym} | Formasyon: {pattern['name']} ({pattern['desc']})\n"
                       f"Hacim24: {vol24:,.0f} | ATR: {atr:.4f} | FR: {fr:.4% if fr is not None else 'N/A'} "
                       f"| OI: {oi:,.0f if oi is not None else 'N/A'}\nSaat: {ts} | Fiyat: {last}")
                await send(msg)
                await asyncio.sleep(0.2)  # tiny delay between TG messages

    await exchange.close()

# ---------------------------------------------------------------------------
# Scheduler -----------------------------------------------------------------

async def main():
    await send('🔄 Bot taramaya başladı')
    while True:
        try:
            await asyncio.gather(scan_exchange('binance'), scan_exchange('mexc'))
        except Exception as exc:
            logging.exception(exc)
            await send(f'⚠️ Bot hatası: {exc}')
        await asyncio.sleep(SCAN_INTERVAL_SEC)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info('Bot durduruldu')
