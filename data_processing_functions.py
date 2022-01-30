import numpy as np
import pandas as pd
from copy import deepcopy

FILE_NAMES = ["humidity", "pressure", "temperature", "weather_description", "wind_direction", "wind_speed"]
DAY_COLUMNS = ['weather_description_day', 'wind_speed_max_day', 'wind_speed_mean_day', 'wind_direction_day', 'temperature_day', 'humidity_day', 'pressure_day']

def read_datas(data_path, remove_nulls:int = 1, file_names:list = FILE_NAMES):
  data_train = dict()
  data_test = dict()

  for file_name in file_names:
      data_train[file_name] = pd.read_csv(f'{data_path}/train/{file_name}_train.csv', header=0, sep=';') 
      data_test[file_name] = pd.read_csv(f'{data_path}/test/{file_name}_test.csv', header=0, sep=';') 

  if remove_nulls == 1:
      data_train = remove_null_values(data_train)
      data_test = remove_null_values(data_test)
  elif remove_nulls == 2:
      data_train = fill_null_values_with_means(data_train)
      data_test = fill_null_values_with_means(data_test)
  elif remove_nulls == 3:
      data_train = fill_null_values(data_train)
      data_test = fill_null_values(data_test)

  return data_train, data_test

def read_train_data(data_path, remove_nulls = True, file_names:list = FILE_NAMES):
  data_train = dict()

  for file_name in file_names:
      data_train[file_name] = pd.read_csv(f'{data_path}/train/{file_name}_train.csv', header=0, sep=';') 

  if remove_nulls:
      data_train = remove_null_values(data_train)

  return data_train

def read_test_data(data_path, remove_nulls = True, file_names:list = FILE_NAMES):
  data_test = dict()
  for file_name in file_names:
    data_test[file_name] = pd.read_csv(f'{data_path}/test/{file_name}_test.csv', header=0, sep=';') 

  if remove_nulls:
    data_test = remove_null_values(data_test)

  return data_test

def remove_null_values(data_dict):
  for i in data_dict:
    data_dict[i] = data_dict[i].dropna()
  return data_dict

def fill_null_values(data_dict):
  for i in data_dict:
    data_dict[i] = data_dict[i].ffill()
    data_dict[i] = data_dict[i].bfill()
  return data_dict

def fill_null_values_with_mean(df):
    for date in np.unique(df["datetime"].str[:10]):
        df_tmp = df[df["datetime"].str.startswith(date)]
        df[df["datetime"].str.startswith(date)] = df_tmp.fillna(df_tmp.mean())
    df = df.dropna()
    return df

def fill_null_values_with_means(data_dict):
    for i in data_dict:
        data_dict[i] = fill_null_values_with_mean(data_dict[i])
    return data_dict

def pivot_data(data_dict, file_names:list = FILE_NAMES):
  i = 0
  for df in data_dict.values():
    df =  df.T.reset_index()
    new_header = df.iloc[0] 
    df = df[1:] 
    df.columns = new_header 
    df = df.melt(id_vars="datetime", 
        var_name="date", 
        value_name=file_names[i])
    if i == 0:
      data_combined = pd.DataFrame(df.iloc[:,:-1])
    data_combined[file_names[i]] = df.iloc[:,-1]
    i+=1

  data_combined.rename(columns={"datetime": "city"}, inplace=True)
  return data_combined

def aggregate_by_day(data):
  data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y/%m/%d")
  data["wind_speed_max"] = pd.to_numeric(data["wind_speed"],downcast='unsigned')
  data["wind_speed_mean"] = pd.to_numeric(data["wind_speed"],downcast='unsigned')
  # data_combined["humidity"] = pd.to_numeric(data_combined["humidity"],downcast='unsigned')
  # data_combined["temperature"] = pd.to_numeric(data_combined["temperature"],downcast='float')
  # data_combined["pressure"] = pd.to_numeric(data_combined["pressure"],downcast='unsigned')
  # data_combined["wind_direction"] = pd.to_numeric(data_combined["wind_direction"],downcast='unsigned')
  # data_combined["weather_description"] = data_combined["weather_description"].astype("category")
  data_aggregated = data.groupby(by=["city", "date"], as_index=False).agg({'weather_description': lambda x: pd.Series.mode(x)[0],
                                                      'wind_speed_max': 'max',
                                                      'wind_speed_mean': 'mean',
                                                      "wind_direction": 'mean',
                                                      "temperature": 'mean',
                                                      "humidity": 'mean',
                                                      "pressure": 'mean'})
  return data_aggregated.sort_values(by=["city", "date"])

def check_date(diffs, dates):
    for d in diffs[1:]:
      if d != pd.Timedelta("1 days"):
        return True
    if dates[-1]-dates[2] != pd.Timedelta("2 days") or dates[-2]-dates[2] != pd.Timedelta("2 days"):
      return True
    return False

def generate_days_columns_names(amount_of_days:int, day_columns:list = DAY_COLUMNS):
  result = []
  for i in range(amount_of_days):
    result.extend([f'{column}{str(i+1)}' for column in day_columns])
  return result

def generate_dfs(data:pd.DataFrame, amount_of_days:int):
  dfs = [data[:-(amount_of_days-1)].reset_index(drop=True)]
  for i in range(1, amount_of_days-1):
    dfs.append(data[i:(-(amount_of_days-1)+i)].reset_index(drop=True))

  dfs.append(data[(amount_of_days-1):].reset_index(drop=True))
  return dfs

def combine_days_series(data:pd.DataFrame, amount_of_days:int = 3):
  data.date = pd.to_datetime(data.date, yearfirst=True)
  data["date_diff"] = data.date.diff()

  dfs = generate_dfs(data, amount_of_days)

  y_temp = data[["temperature","date", "city"]]
  y_temp = y_temp[4:len(dfs[0])].reset_index(drop=True)

  y_wind = data[["wind_speed_max","date","city"]]
  y_wind = y_wind[4:len(dfs[0])].reset_index(drop=True)

  dfs.extend([y_temp, y_wind])
  data_flattened = pd.concat(dfs, axis=1)

  data_flattened["to_del"] = data_flattened.apply(lambda row: len(set(row["city"])) != 1 or check_date(row["date_diff"], row["date"]), axis=1)
  data_flattened = data_flattened[~data_flattened["to_del"]].reset_index(drop=True)

  cities = data_flattened['city'].iloc[:, 0]
  data_flattened = data_flattened.drop(['city', 'date', 'date_diff', 'to_del'], axis=1)

  data_flattened = pd.concat([cities, data_flattened], axis=1)
  data_flattened.columns = ['city'] + generate_days_columns_names(amount_of_days) + ['y_temperature', 'y_wind_speed']                        
  return data_flattened  

def categorize_wind_data(data:pd.DataFrame, border:int = 8):
  bins = [-np.inf, border, np.inf]
  names = [f'below{border}', f'above{border}']
  data['y_wind_speed'] = pd.cut(data['y_wind_speed'], bins, labels=names, right=False)
  data = pd.get_dummies(data=data, columns=["y_wind_speed"],drop_first=True)
  return data

def my_convert_str_variable(data:pd.DataFrame, str_vaiable:str, rename_dict:dict = None, drop:bool=True):
  data = deepcopy(data)
  if rename_dict is None:
      str_vaiable_names = np.unique(data[str_vaiable])
      str_vaiable_names.sort()
      rename_dict = dict()
      i = 0
      for str_vaiable_name in str_vaiable_names:
          rename_dict[str_vaiable_name] = i
          i += 1
  else:
    all_variable_names = np.unique(data[str_vaiable])
    i = len(rename_dict)
    missing_variable = [x for x in all_variable_names if x not in rename_dict]
    for str_variable_name in missing_variable:
      rename_dict[str_variable_name] = i
      i+=1

  converted = []
  for d in data[str_vaiable]:
      converted.append(rename_dict[d])
  data.insert(data.columns.get_loc(str_vaiable), f'converted {str_vaiable}', converted, True)

  if drop:
      data = data.drop(columns=[str_vaiable])

  return data, rename_dict

def convert_str_variable(data:pd.DataFrame, amount_of_days:int = 3, columns:list = None, dicts:list = None, drop:bool=True):
  if columns is None:
    columns = ['city']
    columns.extend(generate_days_columns_names(amount_of_days, day_columns=["weather_description_day"]))

  if dicts is None:
    dicts = []
    for column in columns:
      data, d = my_convert_str_variable(data, column)
      dicts.append(d)
  else:
    for i in range(len(columns)):
      data, d = my_convert_str_variable(data, columns[i], rename_dict=dicts[i])

  return data, columns, dicts

def separate_x_and_y(data:pd.DataFrame, border:int = 8):
  x = data.drop(['y_temperature', f'y_wind_speed_above{border}'],axis=1)
  y_wind = data[f'y_wind_speed_above{border}']
  y_temperature = data['y_temperature']
  return x, y_wind, y_temperature

def get_train_data(data_path:str, amount_of_days:int = 3, wind_border:int = 8, convert_str_variable_flag:bool = True):
  data_train= read_train_data(data_path)
  data_train = pivot_data(data_train)
  data_train_agg = aggregate_by_day(data_train)
  data_train_flattened = combine_days_series(data_train_agg, amount_of_days)
  data_train_cat = categorize_wind_data(data_train_flattened, wind_border)
  x_data_train, y_data_wind_train, y_data_temperature_train = separate_x_and_y(data_train_cat, wind_border)
  
  if convert_str_variable_flag:
    x_data_train = convert_str_variable(x_data_train, amount_of_days)[0]
  
  return x_data_train, y_data_wind_train, y_data_temperature_train

def get_test_data(data_path:str, amount_of_days:int = 3, wind_border:int = 8, convert_str_variable_flag:bool = True):
  data_test= read_test_data(data_path)
  data_test = pivot_data(data_test)
  data_test_agg = aggregate_by_day(data_test)
  data_test_flattened = combine_days_series(data_test_agg, amount_of_days)
  data_test_cat = categorize_wind_data(data_test_flattened, wind_border)
  x_data_test, y_data_wind_test, y_data_temperature_test = separate_x_and_y(data_test_cat, wind_border)
  
  if convert_str_variable_flag:
    x_data_test = convert_str_variable(x_data_test, amount_of_days)[0]

  return x_data_test, y_data_wind_test, y_data_temperature_test

def get_train_and_test_data(data_path:str, amount_of_days:int = 3, wind_border:int = 8, convert_str_variable_flag:bool = True, remove_nulls:int = 1):
  data_train, data_test= read_datas(data_path, remove_nulls)

  data_train = pivot_data(data_train)
  data_train_agg = aggregate_by_day(data_train)
  if remove_nulls == 2:
    data_train_agg = data_train_agg.dropna()
  data_train_flattened = combine_days_series(data_train_agg, amount_of_days)
  data_train_cat = categorize_wind_data(data_train_flattened, wind_border)
  x_data_train, y_data_wind_train, y_data_temperature_train = separate_x_and_y(data_train_cat, wind_border)

  data_test = pivot_data(data_test)
  data_test_agg = aggregate_by_day(data_test)
  if remove_nulls == 2:
    data_test_agg = data_test_agg.dropna()
  data_test_flattened = combine_days_series(data_test_agg, amount_of_days)
  data_test_cat = categorize_wind_data(data_test_flattened, wind_border)
  x_data_test, y_data_wind_test, y_data_temperature_test = separate_x_and_y(data_test_cat, wind_border)
  
  if convert_str_variable_flag:
    x_data_train, columns, dicts = convert_str_variable(x_data_train, amount_of_days)
    x_data_test = convert_str_variable(x_data_test, amount_of_days, columns=columns, dicts=dicts)[0]
  
  return ((x_data_train, y_data_wind_train, y_data_temperature_train), (x_data_test, y_data_wind_test, y_data_temperature_test))

def aggregate_by_3days(data):
  for column in DAY_COLUMNS[1:]:
    data[column + "1"] = (data[column + "1"] + data[column + "2"] + data[column + "3"])/3
    data = data.drop(columns=[column + "2", column+"3"])
  data.drop(columns=[DAY_COLUMNS[0]+"2", DAY_COLUMNS[0]+"3"])
  return data

def get_train_and_test_data_by_3_days(data_path:str, amount_of_days:int = 3, wind_border:int = 8, convert_str_variable_flag:bool = True):
  data_train, data_test= read_datas(data_path)
  data_train = pivot_data(data_train)
  data_train_agg = aggregate_by_day(data_train)
  data_train_flattened = combine_days_series(data_train_agg, amount_of_days)
  data_train_cat = categorize_wind_data(data_train_flattened, wind_border)
  data_train_cat2 = aggregate_by_3days(data_train_cat)
  x_data_train, y_data_wind_train, y_data_temperature_train = separate_x_and_y(data_train_cat2, wind_border)

  data_test = pivot_data(data_test)
  data_test_agg = aggregate_by_day(data_test)
  data_test_flattened = combine_days_series(data_test_agg, amount_of_days)
  data_test_cat = categorize_wind_data(data_test_flattened, wind_border)
  data_test_cat2 = aggregate_by_3days(data_test_cat)
  x_data_test, y_data_wind_test, y_data_temperature_test = separate_x_and_y(data_test_cat2, wind_border)
  
  if convert_str_variable_flag:
    print('convert_str_variable_flag')
    print(amount_of_days)
    x_data_train, columns, dicts = convert_str_variable(x_data_train, amount_of_days)
    print('columns', columns)
    print('dicts', dicts)
    x_data_test = convert_str_variable(x_data_test, amount_of_days, columns=columns, dicts=dicts)[0]
  
  return ((x_data_train, y_data_wind_train, y_data_temperature_train), (x_data_test, y_data_wind_test, y_data_temperature_test))

def season(date):
    md = date.month * 100 + date.day
    if ((md > 320) and (md < 622)):
        s = 0 #spring
    elif ((md > 621) and (md < 923)):
        s = 1 #summer
    elif ((md > 922) and (md < 1223)):
        s = 2 #fall
    else:
        s = 3 #winter
    return s

def aggregate_by_season(data):
  agregated_by_season = dict()
  data["season"] = data.apply(lambda row: season(pd.to_datetime(row['date'])), axis=1)
  for i in range(4):
    agregated_by_season[i] = data[data["season"] == i]
  return agregated_by_season

def get_train_and_test_data_by_season(data_path:str, amount_of_days:int = 3, wind_border:int = 8, convert_str_variable_flag:bool = True, remove_nulls:int = 1):
  data_train, data_test= read_datas(data_path, remove_nulls)
  data_train = pivot_data(data_train)
  data_test = pivot_data(data_test)
  agregated_by_season_train = aggregate_by_season(data_train)
  agregated_by_season_test = aggregate_by_season(data_test)
  data_by_seasons = dict()
  for i in range(4):
    data_train = agregated_by_season_train[i]
    data_train_agg = aggregate_by_day(data_train)
    data_train_flattened = combine_days_series(data_train_agg, amount_of_days)
    data_train_cat = categorize_wind_data(data_train_flattened, wind_border)
    x_data_train, y_data_wind_train, y_data_temperature_train = separate_x_and_y(data_train_cat, wind_border)

    data_test = agregated_by_season_test[i]
    data_test_agg = aggregate_by_day(data_test)
    data_test_flattened = combine_days_series(data_test_agg, amount_of_days)
    data_test_cat = categorize_wind_data(data_test_flattened, wind_border)
    x_data_test, y_data_wind_test, y_data_temperature_test = separate_x_and_y(data_test_cat, wind_border)
    
    if convert_str_variable_flag:
      x_data_train, columns, dicts = convert_str_variable(x_data_train, amount_of_days)
      x_data_test = convert_str_variable(x_data_test, amount_of_days, columns=columns, dicts=dicts)[0]
    
    data_by_seasons[i] = ((x_data_train, y_data_wind_train, y_data_temperature_train), (x_data_test, y_data_wind_test, y_data_temperature_test))
  
  return data_by_seasons

def combine_days_series_separate_day(data:pd.DataFrame, amount_of_days:int = 3):
  data.date = pd.to_datetime(data.date, yearfirst=True)
  data["date_diff"] = data.date.diff()

  dfs = generate_dfs(data, amount_of_days)

  y_temp1 = data[["temperature","date", "city"]]
  y_temp1 = y_temp1[3:len(dfs[0])].reset_index(drop=True)

  y_wind1 = data[["wind_speed_max","date","city"]]
  y_wind1 = y_wind1[3:len(dfs[0])].reset_index(drop=True)

  y_temp2 = data[["temperature","date", "city"]]
  y_temp2 = y_temp2[4:len(dfs[0])].reset_index(drop=True)

  y_wind2 = data[["wind_speed_max","date","city"]]
  y_wind2 = y_wind2[4:len(dfs[0])].reset_index(drop=True)

  dfs.extend([y_temp1, y_wind1, y_temp2, y_wind2])
  data_flattened = pd.concat(dfs, axis=1)

  data_flattened["to_del"] = data_flattened.apply(lambda row: len(set(row["city"])) != 1 or check_date(row["date_diff"], row["date"]), axis=1)
  data_flattened = data_flattened[~data_flattened["to_del"]].reset_index(drop=True)

  cities = data_flattened['city'].iloc[:, 0]
  data_flattened = data_flattened.drop(['city', 'date', 'date_diff', 'to_del'], axis=1)

  data_flattened = pd.concat([cities, data_flattened], axis=1)
  data_flattened.columns = ['city'] + generate_days_columns_names(amount_of_days) + ['y_temperature1', 'y_wind_speed1', 'y_temperature2', 'y_wind_speed2']                        
  return data_flattened  

def categorize_wind_data_with_separate_day(data:pd.DataFrame, border:int = 8):
  bins = [-np.inf, border, np.inf]
  names = [f'below{border}', f'above{border}']
  data['y_wind_speed1'] = pd.cut(data['y_wind_speed1'], bins, labels=names, right=False)
  data = pd.get_dummies(data=data, columns=["y_wind_speed1"],drop_first=True)
  data['y_wind_speed2'] = pd.cut(data['y_wind_speed2'], bins, labels=names, right=False)
  data = pd.get_dummies(data=data, columns=["y_wind_speed2"],drop_first=True)
  return data

def separate_x_and_y_with_separate_day(data:pd.DataFrame, border:int = 8):
  x = data.drop(['y_temperature2', f'y_wind_speed2_above{border}'],axis=1)
  y_wind2 = data[f'y_wind_speed2_above{border}']
  y_temperature2 = data['y_temperature2']

  x = x.drop(['y_temperature1', f'y_wind_speed1_above{border}'],axis=1)
  y_wind1 = data[f'y_wind_speed1_above{border}']
  y_temperature1 = data['y_temperature1']
  return x, y_wind1, y_temperature1, y_wind2, y_temperature2

def get_train_and_test_data_with_separate_day(data_path:str, amount_of_days:int = 3, wind_border:int = 8, convert_str_variable_flag:bool = True, remove_nulls:int = 1):
  data_train, data_test= read_datas(data_path, remove_nulls)

  data_train = pivot_data(data_train)
  data_train_agg = aggregate_by_day(data_train)

  data_train_flattened = combine_days_series_separate_day(data_train_agg, amount_of_days)
  data_train_cat = categorize_wind_data_with_separate_day(data_train_flattened, wind_border)
  x_data_train, y_data_wind_train1, y_data_temperature_train1, y_data_wind_train2, y_data_temperature_train2 = separate_x_and_y_with_separate_day(data_train_cat, wind_border)

  data_test = pivot_data(data_test)
  data_test_agg = aggregate_by_day(data_test)
  data_test_flattened = combine_days_series_separate_day(data_test_agg, amount_of_days)
  data_test_cat = categorize_wind_data_with_separate_day(data_test_flattened, wind_border)
  x_data_test, y_data_wind_test1, y_data_temperature_test1, y_data_wind_test2, y_data_temperature_test2 = separate_x_and_y_with_separate_day(data_test_cat, wind_border)
  
  if convert_str_variable_flag:
    x_data_train, columns, dicts = convert_str_variable(x_data_train, amount_of_days)
    x_data_test = convert_str_variable(x_data_test, amount_of_days, columns=columns, dicts=dicts)[0]
  
  return ((x_data_train, y_data_wind_train1, y_data_temperature_train1, y_data_wind_train2, y_data_temperature_train2), (x_data_test, y_data_wind_test1, y_data_temperature_test1, y_data_wind_test2, y_data_temperature_test2))
