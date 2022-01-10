import pandas as pd

def read_data(data_path):
    file_names = ["humidity", "pressure", "temperature", "weather_description", "wind_direction", "wind_speed"]

    data_train = dict()
    data_test = dict()

    for file_name in file_names:
        print(file_name)
        data_train[file_name] = pd.read_csv(f'{data_path}/train/{file_name}_train.csv', header=0, sep=';') 
        data_test[file_name] = pd.read_csv(f'{data_path}/test/{file_name}_test.csv', header=0, sep=';') 

    return data_train, data_test
