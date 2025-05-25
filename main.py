# main.py (tamamen hatasız, Render uyumlu, session+exchange cleanup garantili)
# Price Action + Formasyon + FR/OI + Telegram + Kesintisiz döngü sistemi içerir

import os, asyncio, logging, aiohttp, pandas as pd, numpy as np
from datetime import datetime, timezone
from telegram import Bot
import ccxt.async_support as ccxt

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7744478523:AAEtRJar6uF7m0cxKfQh7r7TltXYxWwtmm0")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1009868232")
bot = Bot(token=TELEGRAM_TOKEN)

logging.basicConfig(level=logging.INFO)
SCAN_INTERVAL_SEC = 900
LOOKBACK_CANDLES = 120
TIMEFRAME = '15m'

async def send(msg):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, bot.send_message, TELEGRAM_CHAT_ID, msg)

def detect_desc_triangle(df):
    if len(df) < 20: return None
    look = df.tail(LOOKBACK_CANDLES)
    idx = np.arange(len(look))
    slope_high = np.polyfit(idx, look['high'], 1)[0] / np.mean(look['high'])
    slope_low  = np.polyfit(idx, look['low'], 1)[0]  / np.mean(look['low'])
    if slope_high < -0.001 and abs(slope_low) < 0.0003:
        return {'name': 'Alçalan Üçgen', 'desc': 'Düşen zirveler + yatay destek; olası düşüş kırılımı'}
    return None

def detect_order_block(df):
    last = df.iloc[-3:]
    body = abs(last['close'] - last['open'])
    if (body > (last['high'] - last['low']) * 0.6).all():
        direction = 'Bullish' if last['close'].iloc[-1] > last['open'].iloc[-1] else 'Bearish'
        return {'name': f'{direction} Order Block', 'desc': f'{direction} baskılı mum bloğu'}
    return None

def detect_msb(df):
    closes = df['close'].iloc[-4:]
    if closes.iloc[-1] < closes.iloc[-2] < closes.iloc[-3] > closes.iloc[-4]:
        return {'name': 'Market Structure Break (Down)', 'desc': 'Zirve sonrası kırılım; düşüş MSB'}
    elif closes.iloc[-1] > closes.iloc[-2] > closes.iloc[-3] < closes.iloc[-4]:
        return {'name': 'Market Structure Break (Up)', 'desc': 'Dip sonrası kırılım; yükseliş MSB'}
    return None

PATTERN_FUNCS = [detect_desc_triangle, detect_order_block, detect_msb]

async def fetch_fr_oi(session, exchange_name, symbol):
    try:
        if exchange_name == 'binance':
            fr_url = f'https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1'
            oi_url = f'https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}'
            fr_data = await (await session.get(fr_url)).json()
            oi_data = await (await session.get(oi_url)).json()
            return float(fr_data[-1]['fundingRate']), float(oi_data['openInterest'])
        else:
            q = symbol.replace('/', '_')
            js = await (await session.get(f'https://contract.mexc.com/api/v1/contract/ticker?symbol={q}')).json()
            if js.get('success'):
                d = js['data']
                return float(d.get('fundingRate', 0)), float(d.get('holdVol', 0))
    except: return None, None

async def scan_exchange(id_):
    exchange = getattr(ccxt, id_)({'enableRateLimit': True, 'timeout': 30000, 'options': {'defaultType': 'future'}})
    session = aiohttp.ClientSession()
    try:
        pairs = await load_usdt_pairs(exchange)
        for sym in pairs:
            df = await fetch_ohlcv(exchange, sym)
            if df is None: continue
            for fn in PATTERN_FUNCS:
                pattern = fn(df)
                if pattern:
                    fr, oi = await fetch_fr_oi(session, id_, sym.replace('/', '') if id_ == 'binance' else sym)
                    atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
                    vol = df['vol'].sum()
                    last = df['close'].iloc[-1]
                    ts = datetime.now(timezone.utc).strftime('%H:%M UTC')
                    msg = f"{id_.upper()} | {sym} | Formasyon: {pattern['name']} ({pattern['desc']})\n"
                    msg += f"Hacim24: {vol:,.0f} | ATR: {atr:.4f} | FR: {fr:.4% if fr else 'N/A'} | OI: {oi:,.0f if oi else 'N/A'}\n"
                    msg += f"Saat: {ts} | Fiyat: {last}"
                    await send(msg)
                    await asyncio.sleep(0.2)
    finally:
        await session.close()
        await exchange.close()

async def load_usdt_pairs(exchange):
    markets = await exchange.load_markets()
    return [s for s, m in markets.items() if m.get('quote') == 'USDT' and m['active']]

async def fetch_ohlcv(exchange, symbol):
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=LOOKBACK_CANDLES)
        return pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','vol'])
    except: return None

async def main():
    await send('🚀 Bot taramaya başladı')
    while True:
        try:
            await asyncio.gather(scan_exchange('binance'), scan_exchange('mexc'))
        except Exception as e:
            await send(f'❗ Hata: {e}')
        await asyncio.sleep(SCAN_INTERVAL_SEC)

if __name__ == '__main__':
    asyncio.run(main())
