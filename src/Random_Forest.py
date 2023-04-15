#%% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from lightgbm import LGBMRegressor
from hyperopt import fmin, tpe, hp, Trials

from scipy.signal import savgol_filter as sg
from scipy.stats import randint as sp_randint
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import make_regression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score

import joblib
from joblib import parallel_backend

from pathlib import Path
import gc

# %% Import the data
train_df = pd.read_pickle('./data/train_df.pkl')

train_df = train_df.drop(['meter_reading'], axis=1) # drop meter_reading
print("Sum of Null Values Before filling NaN with 0 Values",train_df.isnull().sum())

train_df.fillna(0, inplace=True)
print("Sum of Null Values After filling NaN with 0 Values",train_df.isnull().sum())

# %% Select features
category_cols = ['building_id', 'site_id', 'primary_use',
                 'IsHoliday', 'groupNum_train']  # , 'meter'
feature_cols = ['square_feet_np_log1p', 'year_built'] + [
    'hour', 'weekend',
    'day',  'month',
    'dayofweek',
    'square_feet'
] + [
    'air_temperature', 'cloud_coverage',
    'dew_temperature', 'precip_depth_1_hr',
    'sea_level_pressure',
    'wind_direction', 'wind_speed',
    'air_temperature_mean_lag72',
    'air_temperature_max_lag72', 'air_temperature_min_lag72',
    'air_temperature_std_lag72', 'cloud_coverage_mean_lag72',
    'dew_temperature_mean_lag72', 'precip_depth_1_hr_mean_lag72',
    'sea_level_pressure_mean_lag72',
    'wind_direction_mean_lag72',
    'wind_speed_mean_lag72',
    'air_temperature_mean_lag3',
    'air_temperature_max_lag3',
    'air_temperature_min_lag3', 'cloud_coverage_mean_lag3',
    'dew_temperature_mean_lag3',
    'precip_depth_1_hr_mean_lag3',
    'sea_level_pressure_mean_lag3',
    'wind_direction_mean_lag3', 'wind_speed_mean_lag3',
    'floor_area',
    'year_cnt', 'bid_cnt',
    'dew_smooth', 'air_smooth',
    'dew_diff', 'air_diff',
    'dew_diff2', 'air_diff2'
]


# %% Define a function to create X and y
def create_X_y(train_df, groupNum_train):

    target_train_df = train_df[train_df['groupNum_train']
                               == groupNum_train].copy()

    X_train = target_train_df[feature_cols + category_cols]
    y_train = target_train_df['meter_reading_log1p'].values

    del target_train_df
    return X_train, y_train

# %% Encode categorical features and use MaxMinScaler to scale the features
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Encode categorical features
for col in category_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])

# Scale features
scaler = MinMaxScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])




# %% Train the model
# Define a random forest regression model
rf = RandomForestRegressor()

# Define a hyperparameter space
param_grid = {
    'n_estimators': [1000],
    'max_depth': [7,10],
    'min_samples_split': [2,4],
    'min_samples_leaf': [2,4]
   
}
folds = 3
kf = StratifiedKFold(n_splits=folds)

print(X_train['groupNum_train'].unique())
print(X_test['groupNum_train'].unique())

for groupNum_train in X_train['groupNum_train'].unique():
        target_train_df = train_df[train_df['groupNum_train']
                                == groupNum_train].copy()

        X_train = target_train_df[feature_cols + category_cols]
        y_train = target_train_df['meter_reading_log1p'].values
        del target_train_df
        
        y_valid_pred_total = np.zeros(X_train.shape[0])

        gc.collect()
        print('groupNum_train', groupNum_train, X_train.shape)

        cat_features = [X_train.columns.get_loc(
            cat_col) for cat_col in category_cols]
        print('cat_features', cat_features)

        exec('models' + str(groupNum_train) + '=[]')

        
        best_rf = RandomForestRegressor(max_depth=10, random_state=0,n_estimators=1000,min_samples_split=2,min_samples_leaf=2,n_jobs=8)
        best_rf.fit(X_train, y_train)
        print('fit end')
        filename_reg='./model/rf_grid' + str(groupNum_train) +'.sav'
        joblib.dump(best_rf,filename_reg)
      
           
        del X_train, y_train
        gc.collect()

        print('-------------------------------------------------------------')
# %% Test the model
for groupNum_train in X_test['groupNum_train'].unique():
    target_test_df = train_df[train_df['groupNum_train']
                             == groupNum_train].copy()

    X_test = target_test_df[feature_cols + category_cols]
    y_test = target_test_df['meter_reading_log1p']
    best_rf=joblib.load('./model/rf_grid' + str(groupNum_train) +'.sav')
    y_pred = best_rf.predict(X_test)
   
    #Call the functions
    mae = mean_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(groupNum_train)
    print("Grid Search Test MSE:", mse)
    print("Grid Search Test MAE:", mae)
    print("Grid Search Test r2:", r2)

