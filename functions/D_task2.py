import os
import pickle
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from functions import A_preprocessing as preprocessing
import pandas as pd

def test2(wf, wc, ff, fc):
  print(wc, fc)
  # df = pd.DataFrame(wf, wc).join(pd.DataFrame(ff, fc))
  # weather_feature_descriptions.join(flight_airport_feature_descriptions)
  # df = weather_feature_descriptions
  supportVec = LinearSVC(dual="auto", tol=1e-5, C=1)
  X_train, X_test, y_train, y_test = preprocessing.prepare_data_for_ML_model(df, predCol="PRCP")
  supportVec.fit(X_train, y_train)
  y_pred = supportVec.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuratezza: {accuracy}\n\n")
  print(classification_report(y_test, y_pred))
