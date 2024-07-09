import hopsworks
import pandas as pd
import requests
import json

# Connettersi a Hopsworks
project = hopsworks.login()
serving = project.get_model_serving()

# Definire la funzione di previsione
def predict(data):
    # Ottieni l'endpoint di inferenza
    endpoint = serving.get_deployment("flight_delay_serving").url

    # Effettua la richiesta di previsione
    response = requests.post(endpoint + "/predict", json={"instances": data})

    # Gestisci la risposta
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Errore durante la previsione: {response.status_code}, {response.text}")

# Carica un esempio di dati da prevedere (sostituisci con i tuoi dati reali)
example_data = {
    "MONTH": 1,
    "DAY_OF_WEEK": 5,
    "DISTANCE_GROUP": 1,
    "DEP_BLOCK": 1,
    "SEGMENT_NUMBER": 1,
    "CONCURRENT_FLIGHTS": 1,
    "NUMBER_OF_SEATS": 100,
    "CARRIER_NAME": "CarrierX",
    "AIRPORT_FLIGHTS_MONTH": 500,
    "AIRLINE_FLIGHTS_MONTH": 200,
    "AIRLINE_AIRPORT_FLIGHTS_MONTH": 50,
    "AVG_MONTHLY_PASS_AIRPORT": 10000,
    "AVG_MONTHLY_PASS_AIRLINE": 5000,
    "FLT_ATTENDANTS_PER_PASS": 0.01,
    "GROUND_SERV_PER_PASS": 0.02,
    "PLANE_AGE": 10,
    "DEPARTING_AIRPORT": "JFK",
    "LATITUDE": 40.6413,
    "LONGITUDE": -73.7781,
    "PREVIOUS_AIRPORT": "LAX",
    "PRCP": 0.0,
    "SNOW": 0.0,
    "SNWD": 0.0,
    "TMAX": 30,
    "AWND": 5.0
}

# Effettua la previsione
prediction = predict([example_data])
print(f"Previsione: {prediction}")