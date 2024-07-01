import os
import pickle
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

########################################################################################################################################################################################
# Neural Network
########################################################################################################################################################################################

def define_NN_model(X_train):
    nn_model=Sequential()
    nn_model.add(Dense(32, input_shape=(1, X_train.shape[2]), activation='relu'))
    nn_model.add(Dense(32, activation='relu'))
    nn_model.add(Dropout(0.2))
    nn_model.add(Dense(1, activation='sigmoid'))
    nn_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

    return nn_model

def train_NN_model(neural_model, X_train, X_test, Y_train, Y_test):

    history = neural_model.fit(X_train, Y_train, epochs=3, batch_size=200, verbose=1, validation_data=(X_test, Y_test))

    _, train_acc = neural_model.evaluate(X_train, Y_train, verbose=0)
    _, test_acc = neural_model.evaluate(X_test, Y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    return history


def show_NN_performance(history):
    plt.figure(figsize=(18,8))
    plt.subplot(121)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()

    plt.subplot(122)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()

########################################################################################################################################################################################
# MLP Classifier
########################################################################################################################################################################################

def define_MLP_classifier():
    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=300, random_state=42)
    return mlp

def train_MLP_classifier(mlp, X_train, X_test, Y_train, Y_test):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for i in range(10):
        mlp.partial_fit(X_train, Y_train, classes=np.unique(Y_train))
        
        # Calculate metrics on training set
        train_loss = mlp.loss_
        train_losses.append(train_loss)
        train_accuracy = accuracy_score(Y_train, mlp.predict(X_train))
        train_accuracies.append(train_accuracy)
        
        # Calculate metrics on test set
        test_loss = np.mean(mlp.loss_curve_)
        test_losses.append(test_loss)
        test_accuracy = accuracy_score(Y_test, mlp.predict(X_test))
        test_accuracies.append(test_accuracy)

        print(f"Epoch {i+1}/{10} - Train Loss: {train_loss:.4f} - Train Acc: {train_accuracy:.4f} - Test Loss: {test_loss:.4f} - Test Acc: {test_accuracy:.4f}")

    return train_losses, test_losses, train_accuracies, test_accuracies


def show_MLP_performance(train_losses, test_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def save_model(model, filename):
    if not os.path.isdir("./data/model/"):
        os.mkdir("./data/model/")

    with open(f"./data/model/{filename}.pkl", 'wb') as f:
        pickle.dump(model, f)