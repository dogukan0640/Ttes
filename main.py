# 📛 Telegram bağlantı havuzu taşma sorunu → kesin çözüm: asyncio.Queue + tek worker

import asyncio
from telegram import Bot
import os

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7744478523:AAEtRJar6uF7m0cxKfQh7r7TltXYxWwtmm0")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "1009868232")
bot = Bot(token=TELEGRAM_TOKEN)
message_queue = asyncio.Queue()

# 🔄 Her mesajı sıraya al
def queue_message(msg):
    message_queue.put_nowait(msg)

# 🧵 Kuyruktaki mesajları gönderir
async def telegram_worker():
    while True:
        msg = await message_queue.get()
        try:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
        except Exception as e:
            print(f"Telegram send error: {e}")
        await asyncio.sleep(1.2)
        message_queue.task_done()

# 🧠 Kullanım:
# 1. main.py'de: asyncio.create_task(telegram_worker())
# 2. send(msg) yerine: queue_message(msg)
