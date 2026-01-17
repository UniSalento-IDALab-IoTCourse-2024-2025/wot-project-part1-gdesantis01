import csv
import pandas as pd
import requests
import time
from collections import deque

APPLIANCES = ['Microwave', 'Dishwasher', 'WashingMachine','TumbleDryer']

STATES = {
    'Microwave': False,
    'Dishwasher': False,
    'WashingMachine': False,
    'TumbleDryer': False
}
#state = False


TOKEN = "" #Insert token of Telegram API
CHAT_ID = 0  #Insert ID of chat in which bot joined
url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"


def get_last_rows(csv_filename, num_rows):
    with open(csv_filename, "r") as f:
        last_n_lines = deque(f, num_rows)

    csv_snippet = "".join(last_n_lines)
    return csv_snippet


def check_state(csv_snippet: str, appliance: str):
    global state 
    lines = csv_snippet.strip().split("\n")

    # Extract only the state values (3rd column: ON/OFF)
    states = [line.split(",")[2].strip() for line in lines if line.strip()]

    n_rows = len(states)
    print(f"Sono in check_state e analizzo {n_rows} righe")

    # Count ON and OFF
    on_count = states.count("ON")
    off_count = states.count("OFF")

    perc_on = on_count / n_rows
    
    # Check if last 2 states are ON
    #last_two_on = len(states) >= 2 and states[-1] == "ON" and states[-2] == "ON"

    global state

    final_state = ""

    if perc_on > 0.7:#on_count > off_count:# and last_two_on:
        if(STATES[appliance] == False):
            STATES[appliance] = True
            print(f"Invio notifica stato {STATES[appliance]}")
            timestamp = [line.split(",")[1].strip() for line in lines if line.strip()][20] # prendo il ventunesimo elemento della finestra
            notificate(STATES[appliance], appliance, timestamp, n_rows)
        final_state = "ON"
    elif perc_on < 0.3:
        if (STATES[appliance] == True):
            STATES[appliance] = False
            print(f"Invio notifica stato {STATES[appliance]}")
            timestamp = [line.split(",")[1].strip() for line in lines if line.strip()][20]
            notificate(STATES[appliance], appliance, timestamp, n_rows)
        final_state = "OFF"
    else:
        final_state = "Not_Sure!"
        print(f"Not Sure! Percentage ON:\t{perc_on}")


    print(f"{appliance}:\tON: {on_count}, OFF: {off_count} → Stato finale: {final_state}")
    return final_state

def notificate(status, appliance: str, timestamp, n_rows):
    MESSAGE = ""
    if status == True:
        MESSAGE = f"{appliance} risulta: ACCESO\n{timestamp}\nMetodo con {n_rows} righe"
    else:
        MESSAGE = f"{appliance} risulta: SPENTO\n{timestamp}\nMetodo con {n_rows} righe"

    payload = {"chat_id": CHAT_ID, "text": MESSAGE}
    r = requests.post(url, json=payload)
    print(r.status_code, r.text)


if __name__ == "__main__":

    payload = {"chat_id": CHAT_ID, "text": "Lo script di invio delle predizioni è stato riavviato ora"}
    r = requests.post(url, json=payload)
    print(r.status_code, r.text)
    while True:
        for appliance in APPLIANCES:
            #last_10rows = get_last_rows(f"predizioni_{appliance}.csv", 10)
            last_70rows = get_last_rows(f"predizioni_{appliance}.csv", 70)
            
            #check_state(last_10rows, appliance)
            check_state(last_70rows, appliance)
            
        time.sleep(10)
    
