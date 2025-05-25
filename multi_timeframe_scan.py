
import pandas as pd
import numpy as np
import aiohttp
import ccxt.async_support as ccxt
from datetime import datetime, timezone
from formasyon_modulu import ALL_PATTERNS

TIMEFRAMES = ['5m', '15m', '1h', '4h', '1d']
LOOKBACK_CANDLES = 120

async def fetch_ohlcv(exchange, symbol, tf):
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=tf, limit=LOOKBACK_CANDLES)
        return pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','vol'])
    except: return None

async def scan_multi_timeframes(exchange_id, send_fn):
    exchange = getattr(ccxt, exchange_id)({'enableRateLimit': True, 'timeout': 30000, 'options': {'defaultType': 'future'}})
    session = aiohttp.ClientSession()
    try:
        pairs = await exchange.load_markets()
        usdt_pairs = [s for s, m in pairs.items() if m.get('quote') == 'USDT' and m['active']]

        for sym in usdt_pairs:
            for tf in TIMEFRAMES:
                df = await fetch_ohlcv(exchange, sym, tf)
                if df is None or df.empty: continue

                for fn in ALL_PATTERNS:
                    result = fn(df)
                    if result:
                        ts = datetime.now(timezone.utc).strftime('%H:%M UTC')
                        msg = (
                            f"[{exchange_id.upper()}] {sym} | TF: {tf}\n"
                            f"Formasyon: {result['name']} → {result['desc']}\n"
                            f"Saat: {ts}"
                        )
                        await send_fn(msg)
    finally:
        await session.close()
        await exchange.close()
