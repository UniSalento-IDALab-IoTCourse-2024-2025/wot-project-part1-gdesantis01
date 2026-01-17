# wot-NILMSmartHouse-RaspberryPi-RolloDeSantis
## Descrizione del progetto
This project *NILM Smart House" implements a solution for house appliances monitoring in a non-intrusive way, in order to increase the awareness of inhabitants about their households consumption without installing any power sensors as Shelly Plugs on every devices.

Il sistema si basa su una rete di machine learning, in particolare su un ensemble di reti ResNet, in grado di percepeire se il dispositivo è stato acceso o meno in una determinata fascia oraria. L'addestramento dei modelli è stato effettuato sui dati del dataset REFIT, che hanno dispositivi e impianti con caratteristiche elettriche europee, quindi simili alla Smart House.

Le previsioni sono salvate in un database opportuno, InfluxDB, e presentati all'utente mediante una dashboard che rappresenta i grafici delle fasce orarie di consumo per ogni elettrodomestico. Inoltre, grazie ad uno script Python, viene gestito un bot telegram che permette la consultazione dei dati da remoto.

## Architettura del sistema
```bash
├── MyCamAL_Predictions/
│   ├── bot_prediction.py
│   ├── configs/
│   │   ├── config_expes.yaml
│   │   └── config_refit_data.yaml
│   ├── data/
│   │   ├── HOUSES_Labels
│   │   └── dati_filtrato.csv
│   ├── make_predictions.py
│   ├── partial_fit_make_predictions.py
│   ├── requirements-predictions.txt
│   ├── send_prediction.py
│   └── src
│       └── ...
├── grafana
│   ├── dashboard_grafana.json
│   └── docker-compose.yml
├── recupero_potenza.py
└── requirements-recupero_potenza.txt
```

L’architettura del sistema è composta dai seguenti elementi:

1. **Raccolta dei dati dalla Smart House**: lo script *recupero-potenza.py* si occupa del recupero dei valori di potenza aggregata e delle potenze dei singoli dispositivi mediante le API della Smart House;
2. **Predizione dei carichi attivi**: dopo un pre-processing dei dati, i modelli generano predizioni sullo stato di ogni dispositivo risultato attivo in una finestra temporale. In particolare, i modelli consistono in ensemble di reti *ResNets*. Gli scripts *make_predictions.py* e *partial_fit_make_predictions.py* servono ad effettuare le predizioni utilizzando rispettivamente i modelli che sono stati addestrati solo sul dataset REFIT e i modelli a cui è stato effettuato un partial fit con i dati della Smart House raccolti;
3. **Visualizzazione dei dati su Grafana**: Grafana presenta una dashboard user-friendly che consente di monitorare lo storico delle predizioni per ogni dispositivo;
4. **Monitoring delle predizioni da remoto con Telegram**: mediante un bot Telegram è possibile consultare le predizioni sui dispositivi nelle fasce temporali "Giorno precedente", "Giorno corrente", "Ultime 2 ore".

![Architettura](Soluzione_proposta.png)
