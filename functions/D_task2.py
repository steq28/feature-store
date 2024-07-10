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
from functions import B_create_feature_store as feature_store

def svm(X_train, X_test, y_train, y_test):
  supportVec = LinearSVC(dual="auto", tol=1e-5, C=1)
  
  # make task binary
  y_train = y_train > 0.0
  y_test = y_test > 0.0

  supportVec.fit(X_train, y_train)
  y_pred = supportVec.predict(X_test)

  accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuratezza: {accuracy}\n\n")
  print(classification_report(y_test, y_pred))