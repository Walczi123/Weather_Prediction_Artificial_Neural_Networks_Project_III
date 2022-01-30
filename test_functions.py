import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def classification_test(x_train, y_train, x_test, y_test, hidden_layer_sizes = (100,20)):
    mlpclf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=hidden_layer_sizes, random_state=1)
    mlpclf.fit(x_train, y_train)
    print("Classification accuracy: ",accuracy_score(mlpclf.predict(x_test), y_test))

def classification_test_keras(x_train, y_train, x_test, y_test, hidden_layer_sizes=(4,8), epochs=15):
    model = Sequential()
    for layer_size in hidden_layer_sizes:
        model.add(Dense(layer_size, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))
    model.compile(loss=['mse','sparse_categorical_crossentropy'], optimizer='adam')
    model.fit(x_train, y_train, epochs=15, batch_size=32, verbose=2)
    yhat = model.predict(x_test)
    yhat = np.argmax(yhat, axis=-1).astype('int')
    acc = accuracy_score(y_test, yhat)
    print('Keras classification accuracy: %.3f' % acc)
    
def create_sklearn_classifier(x_train, y_train, hidden_layer_sizes=(100,20), random_state=1):
    model = MLPClassifier( alpha=1e-5, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)
    model.fit(x_train, y_train)
    return model

def predict_sklearn_classifier(model, x_test, y_test):
    acc = accuracy_score(model.predict(x_test), y_test)
    print("Classification accuracy: ",acc)
    return acc

def create_keras_classifier(x_train, y_train, hidden_layer_sizes=(32,16), epochs=15):
    model = Sequential()
    for layer_size in hidden_layer_sizes:
        model.add(Dense(layer_size, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))
    model.compile(loss=['mse','sparse_categorical_crossentropy'], optimizer='adam')
    model.fit(x_train, y_train, epochs=15, batch_size=32, verbose=2)
    return model

def predict_keras_classifier(model, x_test, y_test):
    yhat = model.predict(x_test)
    yhat = np.argmax(yhat, axis=-1).astype('int')
    acc = accuracy_score(y_test, yhat)
    print('Keras classification accuracy: %.3f' % acc)
    return acc

def run_classification(x_train, y_train, x_test, y_test ,hidden_layer_sizes = (100,20)):
    model_sk = create_sklearn_classifier(x_train, y_train)
    model_ker = create_keras_classifier(x_train, y_train)
    sk = 0
    ker = 0
    for i in range(10):
        sk += predict_sklearn_classifier(model_sk, x_test, y_test)
        ker += predict_keras_classifier(model_ker, x_test, y_test)
    print("SKLEARN: " + str((sk / 10)))
    print("KERAS: " + str((ker / 10)))

    def regression_test(x_train, y_train, x_test, y_test, hidden_layer_sizes = (100,20), solver='lbfgs'):
        mlpregr = MLPRegressor(solver=solver, alpha=1e-5, hidden_layer_sizes=hidden_layer_sizes, random_state=1)
        mlpregr.fit(x_train, y_train)
        err = abs(mlpregr.predict(x_test) - y_test)
        MAE = round(np.mean(err),2)
        RMSE = round(np.sqrt(((err)**2).mean()),2)
        print("Regression errors - MAE:", MAE, "RMSE:", RMSE)

def regression_test_keras(x_train, y_train, x_test, y_test, hidden_layer_sizes=(32,16), epochs=15):
    model = Sequential()
    for layer_size in hidden_layer_sizes:
        model.add(Dense(layer_size, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=2)
    yhat = model.predict(x_test)
    error = mean_absolute_error(y_test, yhat)
    print('MAE: %.3f' % error)

def create_sklearn_regressor(x_train, y_train, hidden_layer_sizes=(100,20), random_state=1):
    model = MLPRegressor( alpha=1e-5, hidden_layer_sizes=hidden_layer_sizes, random_state=1)
    model.fit(x_train, y_train)
    return model

def predict_sklearn_regressor(model, x_test, y_test):
    err = abs(model.predict(x_test) - y_test)
    MAE = round(np.mean(err),2)
    RMSE = round(np.sqrt(((err)**2).mean()),2)
    print("Regression errors - MAE:", MAE, "RMSE:", RMSE)
    return MAE

def create_keras_regressor(x_train, y_train, hidden_layer_sizes=(32,16), epochs=15):
    model = Sequential()
    for layer_size in hidden_layer_sizes:
        model.add(Dense(layer_size, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=2)
    return model

def predict_keras_regressor(model, x_test, y_test):
    yhat = model.predict(x_test)
    error = mean_absolute_error(y_test, yhat)
    print('MAE: %.3f' % error)
    return error

def run_regression(x_train, y_train, x_test, y_test ,hidden_layer_sizes = (100,20)):
    model_sk = create_sklearn_regressor(x_train, y_train)
    model_ker = create_keras_regressor(x_train, y_train)
    sk = 0
    ker = 0
    for i in range(10):
        sk += predict_sklearn_regressor(model_sk, x_test, y_test)
        ker += predict_keras_regressor(model_ker, x_test, y_test)
    print("SKLEARN: " + str((sk / 10)))
    print("KERAS: " + str((ker / 10)))