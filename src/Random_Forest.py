#%% Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from lightgbm import LGBMRegressor
from hyperopt import fmin, tpe, hp, Trials

from scipy.stats import uniform,randint as sp_randint
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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

# %% Encode categorical features and use MaxMinScaler to scale the features
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Encode categorical features
for col in category_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])

# Scale features
scaler = MinMaxScaler()
train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

# %% Print the head of the data with the encoded categorical features
print(train_df[category_cols].head())

# %% Print the head of the data with the scaled features
print(train_df[feature_cols].head())


# %% Define a function to create X and y
def create_X_y(train_df, groupNum_train):

    target_train_df = train_df[train_df['groupNum_train']
                               == groupNum_train].copy()

    X_train = target_train_df[feature_cols + category_cols]
    y_train = target_train_df['meter_reading_log1p'].values

    del target_train_df
    return X_train, y_train

# %% Define a function to train the model
def train_model(X_train, y_train, groupNum_train):
     
    cat_features = [X_train.columns.get_loc(
        cat_col) for cat_col in category_cols]
    print('cat_features', cat_features)

    exec('models' + str(groupNum_train) + '=[]')

    # Define a random forest regression model
    rf = RandomForestRegressor()

    # Define a hyperparameter space
    param_dist = {
    'n_estimators': sp_randint(10, 100),
    'max_depth': [10, 20, 30, 40, None],
    'min_samples_split': uniform(0, 1),
    'min_samples_leaf': uniform(0, 0.5),
    'max_features': [1.0,'sqrt', 'log2'],
    'bootstrap': [True, False],
    #'criterion': ['mse', 'mae']
    }
    
    #kf = StratifiedKFold(n_splits=3)

    # Define a RandomizedSearchCV object
    model = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=100,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    cv=5,
    random_state=42,
    verbose=1)

    # Fit the grid search
    model.fit(X_train, y_train)#, cat_features=cat_features

    # Print the best parameters and lowest RMSE
    print('Best parameters found by grid search are:', model.best_params_)
    print('Best RMSE found by grid search is:', np.sqrt(
        -model.best_score_))

    # Save the best model
    exec('models' + str(groupNum_train) + '.append(model.best_estimator_)')
    filename_reg='./model/rf_grid' + str(groupNum_train) +'.sav'
    joblib.dump(model.best_estimator_, filename_reg)

    return model.best_estimator_
# %% Train the model
for groupNum_train in train_df['groupNum_train'].unique():
    X_train, y_train = create_X_y(train_df, groupNum_train)
    best_rf = train_model(X_train, y_train, groupNum_train)
    print(groupNum_train)


# %% Delete the dataframes to free up memory
del train_df
gc.collect()

# %% Load the test data
test_df = pd.read_pickle('./data/test_df.pkl')

# %% Fill NaN values with 0
test_df.fillna(0, inplace=True)

# %% Encode categorical features and use MaxMinScaler to scale the features
for col in category_cols:
    le = LabelEncoder()
    test_df[col] = le.fit_transform(test_df[col])

test_df[feature_cols] = scaler.transform(test_df[feature_cols])

# %% Print the head of the data with the encoded categorical features
print(test_df[category_cols].head())

# %% Print the head of the data with the scaled features
print(test_df[feature_cols].head())



# %% Test the model
def test_model(test_df, groupNum_test):
    target_test_df = test_df[test_df['groupNum_test']
                             == groupNum_test].copy()

    X_test = target_test_df[feature_cols + category_cols]
    

    del target_test_df
    return X_test

# %% Load the submission file
submission_df = pd.read_csv('./data/sample_submission.csv')

# %% Make predictions
for groupNum_test in test_df['groupNum_test'].unique():
    X_test = test_model(test_df, groupNum_test)
    exec('best_rf = models' + str(groupNum_test) + '[0]')
    y_pred = best_rf.predict(X_test)
    submission_df.loc[test_df['groupNum_test']
                      == groupNum_test, 'meter_reading'] = np.expm1(y_pred)

# %% Save the predictions to a csv file
submission_df.to_csv('./data/submission_RF.csv', index=False)

# %% Delete the dataframes to free up memory
del test_df
del submission_df
gc.collect()





