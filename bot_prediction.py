from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from telegram import Update, InputMediaPhoto, InlineKeyboardButton, InlineKeyboardMarkup
from influxdb_client import InfluxDBClient
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    ConversationHandler,
)

INFLUX_URL = "http://localhost:8087"
INFLUX_ORG = "myorg"
INFLUX_BUCKET = "mybucket"
INFLUX_TOKEN = "" 

_influx_client = InfluxDBClient(
    url=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG
)
_query_api = _influx_client.query_api()

TOKEN = "" #Insert token of Telegram API 

ASK_DEVICE, ASK_TIMERANGE = range(2)



DEVICES_MAP = {
    "Dishwasher":"appliance_state_helped",
    "Microwave":"appliance_state",
    "TumbleDryer":"appliance_state",
    "WashingMachine":"appliance_state"
}



# ---------------- FUNZIONI ----------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Benvenuto nel bot della SmartHouse!")

async def predizioni(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Dishwasher", callback_data="Dishwasher")],
        [InlineKeyboardButton("Washing Machine", callback_data="WashingMachine")],
        [InlineKeyboardButton("Tumble Dryer", callback_data="TumbleDryer")],
        [InlineKeyboardButton("Microwave", callback_data="Microwave")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Seleziona il dispositivo di cui vuoi vedere le predizioni:",
        reply_markup=reply_markup
    )
    return ASK_DEVICE

async def handle_device_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    context.user_data["device"] = query.data

    keyboard = [
        [InlineKeyboardButton("Oggi", callback_data="oggi")],
        [InlineKeyboardButton("Ieri", callback_data="ieri")],
        [InlineKeyboardButton("Ultime 2 ore", callback_data="ultime_2_ore")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        f"Hai selezionato: {query.data}\nSeleziona la fascia oraria:",
        reply_markup=reply_markup
    )
    return ASK_TIMERANGE

async def handle_timerange_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    device = context.user_data["device"]
    timerange = query.data




    if timerange == "oggi":
        start, stop = "today()", "now()"
    elif timerange == "ieri":
        ieri = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        start_ieri = ieri.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + "Z"
        stop_ieri = ieri.replace(hour=23, minute=59, second=59, microsecond=0).isoformat() + "Z"
        
        start, stop = start_ieri, stop_ieri
    elif timerange == "ultime_2_ore":
        start, stop = "-2h", "now()"

    flux_query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: {start}, stop: {stop})
      |> filter(fn: (r) => r._measurement == "{DEVICES_MAP[device]}")
      |> filter(fn: (r) => r.device == "{device}")
      |> filter(fn: (r) => r._field == "state")
      |> sort(columns: ["_time"])
    '''
    tables = _query_api.query(flux_query)
            
    times, states = [], []
    for table in tables:
        for record in table.records:
            times.append(record.get_time())
            states.append(record.get_value())

    if not times:
        await query.edit_message_text("Nessun dato in questa fascia oraria.")
        return ConversationHandler.END

    plt.figure(figsize=(10, 4))
    plt.plot(times, states, marker="o", linewidth=1)
    plt.yticks([0, 1], ["OFF", "ON"])
    plt.xlabel("Tempo")
    plt.ylabel("Stato")
    plt.title(f"Predizioni: {device} ({timerange})")
    plt.grid(True)

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()

    await query.edit_message_media(media=InputMediaPhoto(buffer), reply_markup=None)
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Operazione annullata.")
    return ConversationHandler.END

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("predizioni", predizioni)],
        states={
            ASK_DEVICE: [CallbackQueryHandler(handle_device_callback)],
            ASK_TIMERANGE: [CallbackQueryHandler(handle_timerange_callback)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv_handler)

    app.run_polling()
