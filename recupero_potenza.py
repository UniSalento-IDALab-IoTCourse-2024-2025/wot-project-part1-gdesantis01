#!/usr/bin/env python3
import math
import csv
from datetime import datetime
import time
import numpy as np
import pandas as pd
import requests
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from zoneinfo import ZoneInfo

URL = "" #URL of Smart House's API for getting power values
JWT = "" #Bearer Token

# CONFIGURAZIONE INFLUXDB
INFLUX_URL = "http://localhost:8087"
INFLUX_ORG = "myorg"
INFLUX_BUCKET = "mybucket"
INFLUX_TOKEN = ""

_influx_client = InfluxDBClient(
    url=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG
)
_write_api = _influx_client.write_api(write_options=SYNCHRONOUS)

def fix_nan(x):
    return x if (x is None or isinstance(x, float) and math.isnan(x) or x is np.nan) else float(x)

def save_potenze():
    try:
        headers = {
            "Authorization": f"Bearer {JWT}",
            "Accept": "application/json"
        }
        response = requests.get(URL, headers=headers, timeout=5)

        response.raise_for_status()  # Lancia un'eccezione per errori HTTP
        data = response.json()
        if response.status_code == 200:
            data = response.json()  # Converte il JSON in un dizionario Python
            #print(data)
        else:
            print(f"Errore nel recupero dei dati: {response.status_code}")
            return

        interested_entities_names_dict = {
            "sensor.quadro_primo_terra_channel_1_power" : "Aggregate",
            "sensor.asciugatrice_power"                 : "TumbleDryer",
            "sensor.lavatrice_switch_0_power"           : "WashingMachine",
            "sensor.lavastoviglie_switch_0_power"          : "Dishwasher"
        }

        filtered_entities = [item for item in data if item["entity_id"] in interested_entities_names_dict]
        #print(filtered_entities)

        #print("-----------DATI RILEVATI--------------")
        valori = {}  # qui mettiamo i dati da scrivere nel CSV

        for entity in filtered_entities:
            nome_colonna = interested_entities_names_dict[entity["entity_id"]]
            valore = entity["state"] if entity["state"] != "unavailable" else np.nan
            print(f"{nome_colonna}: {valore} W")
            valori[nome_colonna] = valore

        # --- SALVATAGGIO IN CSV ---
        csv_filename = "dati_potenze.csv"

        # Prepara l'intestazione (colonne)
        colonne = ["Time", "Unix", "Aggregate", "TumbleDryer", "WashingMachine", "Dishwasher"]

        # Aggiungi timestamp corrente
        now = datetime.now()
        valori["Time"] = now.strftime("%Y-%m-%d %H:%M:%S")  # leggibile
        valori["Unix"] = int(time.time())  # numero epoch
        
        valori2 = {}
        col2 = ["Time", "Unix", "Aggregate", "Issues"]
        valori2["Time"] = valori["Time"]
        valori2["Unix"] = valori["Unix"]
        valori2["Aggregate"] = valori["Aggregate"]
        valori2["Issues"] = 0

        # Scrivi su CSV (append, creando il file se non esiste)
        try:
            with open(csv_filename, "x", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=colonne)
                writer.writeheader()
                writer.writerow(valori)
        except FileExistsError:
            with open(csv_filename, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=colonne)
                writer.writerow(valori)
        print(f"Scrittura eseguita su {csv_filename}")      
        csv2_filename = "MyCamAL_Predictions/data/dati_filtrato.csv"
        
        try:
            with open(csv2_filename, "x", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=col2)
                writer.writeheader()
                writer.writerow(valori2)
        except FileExistsError:
            with open(csv2_filename, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=col2)
                writer.writerow(valori2)
        print(f"Scrittura eseguita su {csv2_filename}")

        try:
            ts = pd.to_datetime(valori["Time"]).tz_localize("Europe/Rome")
            p = (
                Point("Power_monitoring")
                .field("Aggregate", fix_nan(valori["Aggregate"]))
                .field("Dishwasher", fix_nan(valori["Dishwasher"]))
                .field("TumbleDryer", fix_nan(valori["TumbleDryer"]))
                .field("WashingMachine", fix_nan(valori["WashingMachine"]))
                .time(ts, WritePrecision.NS)
            )
            _write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=p)
            print(f"Riga inviata a Influx: [{valori}] con Timestamp {ts}")
        except Exception as e:
                print(f"Errore invio scrittura a Influx: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Errore nella richiesta: {e}")
while True:
    start = time.time()
    save_potenze()
    end = time.time()
    if (end-start) < 10:
    	time.sleep(10 - (end-start))
