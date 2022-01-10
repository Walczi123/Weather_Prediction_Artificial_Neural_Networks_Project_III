import pandas as pd

def read_data(data_path, remove_nulls = True):
    file_names = ["humidity", "pressure", "temperature", "weather_description", "wind_direction", "wind_speed"]

    data_train = dict()
    data_test = dict()

    for file_name in file_names:
        print(file_name)
        data_train[file_name] = pd.read_csv(f'{data_path}/train/{file_name}_train.csv', header=0, sep=';') 
        data_test[file_name] = pd.read_csv(f'{data_path}/test/{file_name}_test.csv', header=0, sep=';') 

    if remove_nulls:
        data_train = remove_null_values(data_train)
        data_test = remove_null_values(data_test)

    return data_train, data_test

def remove_null_values(data_dict):
  for i in data_dict:
    data_dict[i] = data_dict[i].dropna()
  return data_dict
