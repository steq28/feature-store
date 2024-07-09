import os
import numpy as np
import hsfs
import joblib

class Predict(object):

    def __init__(self):     
        # Get feature store handle
        fs_conn = hsfs.connection()
        self.fs = fs_conn.get_feature_store()
        
        # Get feature view
        self.fv = self.fs.get_feature_view(
            name="flight_delay_online_fv", 
            version=1,
        )
        
        # Initialize serving
        self.fv.init_serving(1)

        # Load the trained model
        self.model = joblib.load(os.environ["ARTIFACT_FILES_PATH"] + "/nn_model.pkl")
        print("Initialization Complete")

    def predict(self, inputs):
        print("DDDAGAKGAMGJKOIA")
        # Extract input values
        (month, day_of_week, distance_group, dep_block, segment_number, concurrent_flights,
         number_of_seats, carrier_name, airport_flights_month, airline_flights_month,
         airline_airport_flights_month, avg_monthly_pass_airport, avg_monthly_pass_airline,
         flt_attendants_per_pass, ground_serv_per_pass, plane_age, departing_airport,
         latitude, longitude, previous_airport, prcp, snow, snwd, tmax, awnd) = inputs[0]

        # Create feature vector dictionary
        feature_dict = {
            "MONTH": month,
            "DAY_OF_WEEK": day_of_week,
            "DISTANCE_GROUP": distance_group,
            "DEP_BLOCK": dep_block,
            "SEGMENT_NUMBER": segment_number,
            "CONCURRENT_FLIGHTS": concurrent_flights,
            "NUMBER_OF_SEATS": number_of_seats,
            "CARRIER_NAME": carrier_name,
            "AIRPORT_FLIGHTS_MONTH": airport_flights_month,
            "AIRLINE_FLIGHTS_MONTH": airline_flights_month,
            "AIRLINE_AIRPORT_FLIGHTS_MONTH": airline_airport_flights_month,
            "AVG_MONTHLY_PASS_AIRPORT": avg_monthly_pass_airport,
            "AVG_MONTHLY_PASS_AIRLINE": avg_monthly_pass_airline,
            "FLT_ATTENDANTS_PER_PASS": flt_attendants_per_pass,
            "GROUND_SERV_PER_PASS": ground_serv_per_pass,
            "PLANE_AGE": plane_age,
            "DEPARTING_AIRPORT": departing_airport,
            "LATITUDE": latitude,
            "LONGITUDE": longitude,
            "PREVIOUS_AIRPORT": previous_airport,
            "PRCP": prcp,
            "SNOW": snow,
            "SNWD": snwd,
            "TMAX": tmax,
            "AWND": awnd
        }

        # Get feature vector from feature store
        feature_vector = self.fv.get_feature_vector(feature_dict)

        # Predict using the model
        prediction = self.model.predict(np.asarray(feature_vector).reshape(1, -1)).tolist()
        return prediction
