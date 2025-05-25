# ✅ Nihai sade, stabil ve havuz taşmayan main.py
import os, asyncio, logging
from telegram import Bot
from multi_timeframe_scan import scan_multi_timeframes

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7744478523:AAEtRJar6uF7m0cxKfQh7r7TltXYxWwtmm0")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1009868232")
SCAN_INTERVAL_SEC = 900

bot = Bot(token=TELEGRAM_TOKEN)
message_queue = asyncio.Queue()

# Sadece sıraya ekle
def queue_message(msg):
    message_queue.put_nowait(msg)

# Tek bir bağlantıdan sırayla gönder
async def telegram_worker():
    while True:
        msg = await message_queue.get()
        try:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
        except Exception as e:
            print(f"Telegram send error: {e}")
        await asyncio.sleep(1.2)
        message_queue.task_done()

async def main():
    asyncio.create_task(telegram_worker())
    queue_message('🚀 Çoklu zamanlı formasyon botu (sade versiyon) başladı')
    while True:
        try:
            await asyncio.gather(
                scan_multi_timeframes('binance', queue_message),
                scan_multi_timeframes('mexc', queue_message)
            )
        except Exception as e:
            queue_message(f'❗ Hata: {e}')
        await asyncio.sleep(SCAN_INTERVAL_SEC)

if __name__ == '__main__':
    asyncio.run(main())
