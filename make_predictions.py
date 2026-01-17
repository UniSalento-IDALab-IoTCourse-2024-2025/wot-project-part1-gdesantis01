import csv
import os
import time
from time import sleep
import torch
import numpy as np
from src.helpers.data_processing import REFIT_DataBuilder, nilmdataset_to_clfdataset
from src.helpers.torch_dataset import NILMscaler
from src.models.camal.core import CamAL


from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
from zoneinfo import ZoneInfo

SECONDS = [10,20,50]
t = 0

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

window_idx = 0
APPLIANCES = ['WashingMachine','Dishwasher', 'Microwave','TumbleDryer']

def function(expes_config,
             my_house_path):

    # costruzione del Data Builder
    global window_idx
    global APPLIANCES
    data_builder = REFIT_DataBuilder(
        data_path='data/refit/RAW_DATA_CLEAN/',
        mask_app=[],
        sampling_rate=expes_config['sampling_rate'],
        window_size=expes_config['window_size_test'],
        use_status_from_kelly_paper=expes_config['use_kelly_status']
    )
    # inizializzazione dello scaler
    scaler = NILMscaler(
        power_scaling_type=expes_config['power_scaling'],
        appliance_scaling_type=expes_config['appliance_scaling']
    )
    
    # caricamento delle ultime 30 righe del csv sulle potenze
    check_error, data_test, st_date_test = data_builder.get_nilm_dataset(my_house_path)
    if (check_error == -3):
        return check_error
    data_test_scaled = scaler.fit_transform(data_test)

    torch.serialization.default_restore_location = lambda storage, loc: storage.cpu()
    
    for appliance in APPLIANCES:
        #Info specifico appliance
        expes_config['appliance'] = appliance
        expes_config['appliance_mean_on_power'] = dataset_config[appliance]['appliance_mean_on_power']
        
        # Caricamento del modello di CamAL
        model_path = f"models/model_{appliance}.pkl"
        camal = CamAL(device=expes_config['device'])
        camal.load(model_path)
        camal.eval()

        # predizione su quella finestra
        soft_label, prob_detect = camal.predict(data_test_scaled[:, 0, 0, :], return_prob=True)

        # Per la classificazione usiamo prob_detect (probabilità che l'appliance sia presente)
        y_prob = prob_detect.ravel()  # Probabilità continua [0, 1]
        threshold = 0.5
        y_pred = (y_prob >= threshold).astype(int)  # Predizione binaria con threshold 0.5

        window_data = {
            'window_id': window_idx,
            'start_date': st_date_test.iloc[0]['start_date'],
            'y_pred': 'ON' if int(y_pred[0]) == 1 else 'OFF',
            'y_prob': float(y_prob[0])
        }

        # Controlla se il file esiste per decidere se scrivere l'intestazione
        file_name = f"predizioni_{appliance}.csv"
        file_exists = os.path.isfile(file_name)

        # Apre in modalità append
        with open(file_name, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=window_data.keys())
            if not file_exists:
                writer.writeheader()  # Scrive l'intestazione solo la prima volta
            writer.writerow(window_data)  # Scrive la nuova riga
            
            try:
                print("Sono entrato nel try")
                y_pred = 1 if window_data["y_pred"] == "ON" else 0
                ts = pd.to_datetime(window_data["start_date"]).tz_localize("Europe/Rome")
                #ts_2 = ts.tz_convert("UTC")
                print("Sono dopo il ts_2")
                #epoch_ns = int(ts_2.timestamp() * 1e9)

                p = (
                    Point("appliance_state")
                    .tag("device", appliance)
                    .field("state", y_pred)
                    .field("prob", float(window_data["y_prob"]))
                    .time(ts, WritePrecision.NS)
                )
                _write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=p)
                print(f"[{appliance}] scritto  su Influx window_id={window_data['window_id']}, con data {ts}")
            except Exception as e:
                print(f"[{appliance}] errore invio  a Influx: {e}")

    window_idx+=1


if __name__ == '__main__':
    
    seed = 123

    from src.helpers.other import load_config
    dataset_config = load_config('configs/config_refit_data.yaml')
    expes_config = load_config('configs/config_expes.yaml')

    while True:
        start = time.time()
        if (function(expes_config, "data/dati_filtrato.csv") == -3):
            t += 1
            if (0 < t and t <= 3):
                print(f"Errore -3: Attendo {SECONDS[0]} secondi")
                sleep(SECONDS[0])
            elif(3 < t and t <= 5):
                print(f"Errore -3: Attendo {SECONDS[1]} secondi")
                sleep(SECONDS[1])
            else:
                print(f"Errore -3: Attendo {SECONDS[2]} secondi")
                sleep(SECONDS[2])
            continue
        print("Sono dopo il while")
        t = 0
        end = time.time()

        elapsed = end - start
        print(f"Tempo di esecuzione: {elapsed:.6f} secondi")
        sleep(max(0, 10 - elapsed))

