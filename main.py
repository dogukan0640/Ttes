
import os, asyncio, logging
from telegram import Bot
from multi_timeframe_scan import scan_multi_timeframes

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7744478523:AAEtRJar6uF7m0cxKfQh7r7TltXYxWwtmm0")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1009868232")
SCAN_INTERVAL_SEC = 900

bot = Bot(token=TELEGRAM_TOKEN)
logging.basicConfig(level=logging.INFO)

async def send(msg):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, bot.send_message, TELEGRAM_CHAT_ID, msg)

async def main():
    await send('🚀 Çoklu zamanlı formasyon botu başladı')
    while True:
        try:
            await asyncio.gather(
                scan_multi_timeframes('binance', send),
                scan_multi_timeframes('mexc', send)
            )
        except Exception as e:
            await send(f'❗ Hata: {e}')
        await asyncio.sleep(SCAN_INTERVAL_SEC)

if __name__ == '__main__':
    asyncio.run(main())
