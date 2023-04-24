#%% Import libraries
import pandas as pd
import numpy as np
from scipy.stats import uniform,randint as sp_randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pandas.api.types import is_categorical_dtype
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import joblib
import gc

# %% Import the data
train_df = pd.read_pickle('/workspace/Ashrae-Energy-Prediction-III/src/data/train_df.pkl')

train_df = train_df.drop(['meter_reading'], axis=1) # drop meter_reading
print("Sum of Null Values Before filling NaN with 0 Values",train_df.isnull().sum())

train_df.fillna(0, inplace=True)
print("Sum of Null Values After filling NaN with 0 Values",train_df.isnull().sum())

# %% Select features
# Feature Selection
print("Feature Selection...")

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


# %% Memory management

def reduce_mem_usage(df, use_float16=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            # skip datetime type or categorical type
            continue
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))

    return df

# %% Reduce memory usage
train_df = reduce_mem_usage(train_df, use_float16=True)



# %% Define a function to create X and y
def create_X_y(train_df, groupNum_train):

    target_train_df = train_df[train_df['groupNum_train']
                               == groupNum_train].copy()

    X_train = target_train_df[feature_cols + category_cols]
    y_train = target_train_df['meter_reading_log1p']

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
    
    # Use a smaller subset of the data for training
    X_train_sample = X_train.sample(frac=0.1, random_state=42)
    y_train_sample = y_train.sample(frac=0.1, random_state=42)

    # Define a RandomizedSearchCV object
    model = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,
    scoring='neg_mean_squared_error',
    n_jobs= -1, # -1 means use all processors 
    cv=3,
    random_state=42,
    verbose=1)

    # Fit the grid search
    model.fit(X_train_sample, y_train_sample)#, cat_features=cat_features

    # Print the best parameters and lowest RMSE
    print('Best parameters found by grid search are:', model.best_params_)
    print('Best RMSE found by grid search is:', np.sqrt(
        -model.best_score_))

    # Save the best model
    exec('models' + str(groupNum_train) + '.append(model.best_estimator_)')
    filename_reg='/workspace/Ashrae-Energy-Prediction-III/model/rf_grid' + str(groupNum_train) +'.sav'
    joblib.dump(model.best_estimator_, filename_reg)

    return model.best_estimator_
# %% Train the model
for groupNum_train in train_df['groupNum_train'].unique():
    print(groupNum_train)
    X_train, y_train = create_X_y(train_df, groupNum_train)
    # Reduce the memory usage of the dataframes
    X_train = reduce_mem_usage(X_train, use_float16=True)
    best_rf = train_model(X_train, y_train, groupNum_train)
    del X_train, y_train
    gc.collect()
    


# %% Load the test data
test_df = pd.read_pickle('/workspace/Ashrae-Energy-Prediction-III/src/data/test_df.pkl')

# %% Load the building metadata and weather test data
building_metadata_df = pd.read_pickle(
    '/workspace/Ashrae-Energy-Prediction-III/src/data/building_meta_df.pkl')
weather_test_df = pd.read_pickle(
    '/workspace/Ashrae-Energy-Prediction-III/src/data/weather_test_df.pkl')

# %% Merge the test data with the building metadata and weather test data
target_test_df = test_df.copy()
target_test_df = target_test_df.merge(
        building_metadata_df, on=['building_id', 'meter', 'groupNum_train', 'square_feet'], how='left')
target_test_df = target_test_df.merge(
    weather_test_df, on=['site_id', 'timestamp'], how='left')
X_test = target_test_df[feature_cols + category_cols]

del target_test_df
gc.collect()

# %% Reduce the memory usage of the dataframes
X_test = reduce_mem_usage(X_test, use_float16=True)


# %% Fill NaN values with 0
X_test.fillna(0, inplace=True)

# %% Print the head of the data
print(X_test.head())

# %% Encode categorical features and use MaxMinScaler to scale the features
for col in category_cols:
    le = LabelEncoder()
    X_test[col] = le.fit_transform(X_test[col])

#scaler = MinMaxScaler()
X_test[feature_cols] = scaler.transform(X_test[feature_cols])

# %% Print the head of the data with the encoded categorical features
print(X_test[category_cols].head())

# %% Print the unique values in the groupNum_train column of the X_test dataframe
print(X_test['groupNum_train'].unique())

# %% Print the head of the data with the scaled features
print(X_test[feature_cols].head())



# %% # Train the entire dataset with the saved best features.
for groupNum_train in train_df['groupNum_train'].unique():
    print(groupNum_train)
    X_train, y_train = create_X_y(train_df, groupNum_train)
    # Reduce the memory usage of the dataframes
    #X_train = reduce_mem_usage(X_train, use_float16=True)
    
    # load the best estimator from disk
    best_estimator = joblib.load('/workspace/Ashrae-Energy-Prediction-III/model/rf_grid' + str(groupNum_train)+ '.sav')
    
    # Use the best estimator from the saved model to train the model
    model = RandomForestRegressor(n_estimators=best_estimator.n_estimators,
                              max_depth=best_estimator.max_depth,
                              min_samples_split=best_estimator.min_samples_split,
                              min_samples_leaf=best_estimator.min_samples_leaf,
                              max_features=best_estimator.max_features,
                              bootstrap=best_estimator.bootstrap,
                              random_state=42)
    model.fit(X_train, y_train)

    # save the model to the disk using sklearn
    filename_reg='/workspace/Ashrae-Energy-Prediction-III/model/Model_best/rf_grid' + str(groupNum_train) +'.joblib'
    joblib.dump(model, filename_reg)
    
    del X_train, y_train
    gc.collect()

# %% Load the submission file
submission_df = pd.read_csv('/workspace/Ashrae-Energy-Prediction-III/src/data/sample_submission.csv')

# %% Make predictions
for groupNum in X_test['groupNum_train'].unique():
    print('Group Number: ', groupNum)
    # Select the Features in the test dataset
    X_test_group = X_test[X_test['groupNum_train']
                            == groupNum][feature_cols + category_cols]
    
    
    # Load the model
    model = joblib.load('/workspace/Ashrae-Energy-Prediction-III/model/Model_best/rf_grid' + str(groupNum)+ '.joblib')

    # Predict the meter_reading_log1p
    y_pred = model.predict(X_test_group)

    # convert the meter_reading_log1p to meter_reading
    y_pred = np.expm1(y_pred)

    # Save the meter_reading to the sample_submission_df by matching the index of the X_test dataset with the row_id of the sample_submission_df dataset
    submission_df.loc[X_test[X_test['groupNum_train'] == groupNum].index,
                             'meter_reading'] = y_pred

    # Delete the model
    del model, X_test_group, y_pred
    gc.collect()

# %% Save the predictions to a csv file
submission_df.to_csv('/workspace/Ashrae-Energy-Prediction-III/src/data/submission_RF.csv', index=False)

# %% Show the head of the submission_df
print(submission_df.head())
# %% Print the unique meter_reading values in the submission_df
print(submission_df['meter_reading'].unique())






# %%
