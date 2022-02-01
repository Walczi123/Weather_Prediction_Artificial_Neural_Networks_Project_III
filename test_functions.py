import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

EPOCHS = 100
RANGE = 5
HIDDEN_LAYERS = (128, 8)

def classification_test(x_train, y_train, x_test, y_test, hidden_layer_sizes = HIDDEN_LAYERS):
    mlpclf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=hidden_layer_sizes, random_state=1)
    mlpclf.fit(x_train, y_train)
    print("Classification accuracy: ",accuracy_score(mlpclf.predict(x_test), y_test))

def classification_test_keras(x_train, y_train, x_test, y_test, hidden_layer_sizes=(4,8), epochs=EPOCHS):
    model = Sequential()
    for layer_size in hidden_layer_sizes:
        model.add(Dense(layer_size, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))
    model.compile(loss=['sparse_categorical_crossentropy'], optimizer='adam')
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=2)
    yhat = model.predict(x_test)
    yhat = np.argmax(yhat, axis=-1).astype('int')
    acc = accuracy_score(y_test, yhat)
    print('Keras classification accuracy: %.3f' % acc)
    
def create_sklearn_classifier(x_train, y_train, hidden_layer_sizes=HIDDEN_LAYERS, random_state=1):
    model = MLPClassifier( alpha=1e-5, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)
    model.fit(x_train, y_train)
    return model

def predict_sklearn_classifier(model, x_test, y_test):
    pred = model.predict(x_test)
    acc = accuracy_score(pred, y_test)
    print("Classification accuracy: ",acc)
    conv_matrix = confusion_matrix(y_test, pred)
    return (acc, conv_matrix)

def create_keras_classifier(x_train, y_train, hidden_layer_sizes=HIDDEN_LAYERS, epochs=EPOCHS):
    model = Sequential()
    for layer_size in hidden_layer_sizes:
        model.add(Dense(layer_size, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))
    model.compile(loss=['sparse_categorical_crossentropy'], optimizer='adam')
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=2)
    return model

def predict_keras_classifier(model, x_test, y_test):
    yhat = model.predict(x_test)
    yhat = np.argmax(yhat, axis=-1).astype('int')
    acc = accuracy_score(y_test, yhat)
    print('Keras classification accuracy: %.3f' % acc)
    conv_matrix = confusion_matrix(y_test, yhat)
    return (acc, conv_matrix)

def run_classification(x_train, y_train, x_test, y_test ,hidden_layer_sizes = HIDDEN_LAYERS):
    sk = 0
    ker = 0
    sk_m = np.zeros([2, 2])
    ker_m = np.zeros([2, 2])
    for i in range(RANGE):
        model_sk = create_sklearn_classifier(x_train, y_train, hidden_layer_sizes=hidden_layer_sizes, random_state=i)
        model_ker = create_keras_classifier(x_train, y_train, hidden_layer_sizes=hidden_layer_sizes)
        s1, s2 = predict_sklearn_classifier(model_sk, x_test, y_test)
        k1, k2 = predict_keras_classifier(model_ker, x_test, y_test)
        sk += s1
        ker += k1
        sk_m += s2
        ker_m += k2
    print("SKLEARN: " + str((sk / RANGE)))
    print(sk_m/RANGE)
    print("KERAS: " + str((ker / RANGE)))
    print(ker_m/RANGE)

def regression_test(x_train, y_train, x_test, y_test, hidden_layer_sizes = HIDDEN_LAYERS, solver='lbfgs', random_state =1):
    mlpregr = MLPRegressor(solver=solver, alpha=1e-5, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)
    mlpregr.fit(x_train, y_train)
    err = abs(mlpregr.predict(x_test) - y_test)
    MAE = round(np.mean(err),2)
    RMSE = round(np.sqrt(((err)**2).mean()),2)
    print("Regression errors - MAE:", MAE, "RMSE:", RMSE)

def regression_test_keras(x_train, y_train, x_test, y_test, hidden_layer_sizes=HIDDEN_LAYERS, epochs=EPOCHS):
    model = Sequential()
    for layer_size in hidden_layer_sizes:
        model.add(Dense(layer_size, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=2)
    yhat = model.predict(x_test)
    error = mean_absolute_error(y_test, yhat)
    print('MAE: %.3f' % error)

def create_sklearn_regressor(x_train, y_train, hidden_layer_sizes=HIDDEN_LAYERS, random_state=1):
    model = MLPRegressor( alpha=1e-5, hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)
    model.fit(x_train, y_train)
    return model

def predict_sklearn_regressor(model, x_test, y_test, denom = None):
    err = abs(model.predict(x_test) - y_test)
    if denom is not None:
        err = abs(denom(model.predict(x_test)) - denom(y_test))
    print("Error was smaller than 2 degree in :", np.count_nonzero(err < 2) / len(err))
    MAE = round(np.mean(err),2)
    RMSE = round(np.sqrt(((err)**2).mean()),2)
    print("Regression errors - MAE:", MAE, "RMSE:", RMSE)
    return (MAE, np.count_nonzero(err < 2) / len(err))

def create_keras_regressor(x_train, y_train, hidden_layer_sizes=HIDDEN_LAYERS, epochs=EPOCHS):
    model = Sequential()
    for layer_size in hidden_layer_sizes:
        model.add(Dense(layer_size, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=2)
    return model

def predict_keras_regressor(model, x_test, y_test, denom=None):
    yhat = model.predict(x_test)
    err = abs(model.predict(x_test).reshape(1,-1)[0] - y_test)
    if denom is not None:
        err = abs(denom(model.predict(x_test).reshape(1,-1)[0]) - denom(y_test))

    print("Error was smaller than 2 degree in :", np.count_nonzero(err < 2) / len(err))
    error = round(np.mean(err),2)
    print('MAE: %.3f' % error)
    return (error, np.count_nonzero(err < 2) / len(err))

def run_regression(x_train, y_train, x_test, y_test ,hidden_layer_sizes = HIDDEN_LAYERS, denom = None):
    sk = 0
    ker = 0
    sk_p = 0
    ker_p = 0
    for i in range(RANGE):
        model_sk = create_sklearn_regressor(x_train, y_train, hidden_layer_sizes=hidden_layer_sizes, random_state=i)
        model_ker = create_keras_regressor(x_train, y_train, hidden_layer_sizes=hidden_layer_sizes)
        s1, s2 = predict_sklearn_regressor(model_sk, x_test, y_test, denom = denom)
        sk += s1
        sk_p += s2
        k1, k2 = predict_keras_regressor(model_ker, x_test, y_test ,denom = denom)
        ker += k1
        ker_p += k2
    print(f"SKLEARN: {str((sk / RANGE))} and error was smaller than 2 degree in {sk_p/RANGE}")
    print(f"KERAS: {str((ker / RANGE))} and error was smaller than 2 degree in {ker_p/RANGE}")

def mean_normalization(df_data):
    return ((df_data-df_data.mean())/df_data.std(), lambda x: (x * df_data.std() + df_data.mean()))

def minmax_normalization(df_data):
    return ((df_data-df_data.min())/(df_data.max()-df_data.min()), lambda x: (x * (df_data.max()-df_data.min()) + df_data.min()))


def classification_study_issue_5(x_train, y_train, y_train2, x_test, y_test2 ,hidden_layer_sizes = HIDDEN_LAYERS):
    sk = 0
    ker = 0
    sk_m = np.zeros([2, 2])
    ker_m = np.zeros([2, 2])
    for i in range(RANGE):
        model_sk = create_sklearn_classifier(x_train, y_train, hidden_layer_sizes=hidden_layer_sizes)
        model_ker = create_keras_classifier(x_train, y_train, hidden_layer_sizes=hidden_layer_sizes)
        predicted1 = model_sk.predict(x_train)
        predicted2 = model_ker.predict(x_train)
        model_sk2 = create_sklearn_classifier(predicted1.reshape(-1, 1), y_train2, hidden_layer_sizes=hidden_layer_sizes)
        model_ker2 = create_keras_classifier(np.argmax(predicted2, axis=-1).astype('int'), y_train2, hidden_layer_sizes=hidden_layer_sizes)
        predicted_sk1 = model_sk.predict(x_test)
        predicted_ker1 = model_ker.predict(x_test)
        s1, s2 = predict_sklearn_classifier(model_sk2, predicted_sk1.reshape(-1, 1), y_test2)
        k1, k2 = predict_keras_classifier(model_ker2, np.argmax(predicted_ker1, axis=-1).astype('int'), y_test2)
        sk += s1
        ker += k1
        sk_m += s2
        ker_m += k2
    print("SKLEARN: " + str((sk / RANGE)))
    print(sk_m/RANGE)
    print("KERAS: " + str((ker / RANGE)))
    print(ker_m/RANGE)

def classification_study_issue_5_real(x_train, y_train, y_train2, x_test, y_test2 ,hidden_layer_sizes = HIDDEN_LAYERS):
    sk = 0
    ker = 0
    sk_m = np.zeros([2, 2])
    ker_m = np.zeros([2, 2])
    for i in range(RANGE):
        model_sk = create_sklearn_classifier(x_train, y_train, hidden_layer_sizes=hidden_layer_sizes)
        model_ker = create_keras_classifier(x_train, y_train, hidden_layer_sizes=hidden_layer_sizes)
        model_sk2 = create_sklearn_classifier(y_train.to_numpy().reshape(-1,1), y_train2, hidden_layer_sizes=hidden_layer_sizes)
        model_ker2 = create_keras_classifier(y_train, y_train2, hidden_layer_sizes=hidden_layer_sizes)
        predicted_sk1 = model_sk.predict(x_test)
        predicted_ker1 = model_ker.predict(x_test)
        s1, s2 = predict_sklearn_classifier(model_sk2, predicted_sk1.reshape(-1,1), y_test2)
        k1, k2 = predict_keras_classifier(model_ker2, np.argmax(predicted_ker1, axis=-1).astype('int'), y_test2)
        sk += s1
        ker += k1
        sk_m += s2
        ker_m += k2
    print("SKLEARN: " + str((sk / RANGE)))
    print(sk_m/RANGE)
    print("KERAS: " + str((ker / RANGE)))
    print(ker_m/RANGE)


def regression_study_issue_5(x_train, y_train, y_train2, x_test, y_test2 ,hidden_layer_sizes = HIDDEN_LAYERS):
    sk = 0
    ker = 0
    sk_p = 0
    ker_p = 0
    for i in range(RANGE):
        model_sk = create_sklearn_regressor(x_train, y_train, hidden_layer_sizes=hidden_layer_sizes, random_state=i)
        model_ker = create_keras_regressor(x_train, y_train, hidden_layer_sizes=hidden_layer_sizes)
        predicted1 = model_sk.predict(x_train)
        predicted2 = model_ker.predict(x_train)
        model_sk2 = create_sklearn_regressor(predicted1.reshape(-1, 1), y_train2, hidden_layer_sizes=hidden_layer_sizes, random_state=i)
        model_ker2 = create_keras_regressor(predicted2.reshape(-1, 1), y_train2, hidden_layer_sizes=hidden_layer_sizes)
        predicted_sk1 = model_sk.predict(x_test)
        predicted_ker1 = model_ker.predict(x_test)
        s1, s2 = predict_sklearn_regressor(model_sk2, predicted_sk1.reshape(-1, 1), y_test2)
        sk += s1
        sk_p += s2
        k1, k2 = predict_keras_regressor(model_ker2, predicted_ker1, y_test2)
        ker += k1
        ker_p += k2
    print(f"SKLEARN: {str((sk / RANGE))} and error was smaller than 2 degree in {sk_p/RANGE}")
    print(f"KERAS: {str((ker / RANGE))} and error was smaller than 2 degree in {ker_p/RANGE}")

def regression_study_issue_5_real(x_train, y_train, y_train2, x_test, y_test2 ,hidden_layer_sizes = HIDDEN_LAYERS):
    sk = 0
    ker = 0
    sk_p = 0
    ker_p = 0
    for i in range(RANGE):
        model_sk = create_sklearn_regressor(x_train, y_train, hidden_layer_sizes=hidden_layer_sizes)
        model_ker = create_keras_regressor(x_train, y_train, hidden_layer_sizes=hidden_layer_sizes)
        model_sk2 = create_sklearn_regressor(y_train.to_numpy().reshape(-1,1), y_train2, hidden_layer_sizes=hidden_layer_sizes)
        model_ker2 = create_keras_regressor(y_train, y_train2, hidden_layer_sizes=hidden_layer_sizes)
        predicted_sk1 = model_sk.predict(x_test)
        predicted_ker1 = model_ker.predict(x_test)
        s1, s2 = predict_sklearn_regressor(model_sk2, predicted_sk1.reshape(-1, 1), y_test2)
        sk += s1
        sk_p += s2
        k1, k2 = predict_keras_regressor(model_ker2, predicted_ker1, y_test2)
        ker += k1
        ker_p += k2
    print(f"SKLEARN: {str((sk / RANGE))} and error was smaller than 2 degree in {sk_p/RANGE}")
    print(f"KERAS: {str((ker / RANGE))} and error was smaller than 2 degree in {ker_p/RANGE}")