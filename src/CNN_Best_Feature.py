# %% Import the libraries to train a CNN model
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import keras as k
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.utils import plot_model # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import gc
import warnings
warnings.filterwarnings('ignore')

# %% Import the dataset
train_df = pd.read_pickle(
    '/workspace/Ashrae-Energy-Prediction-III/src/data/train_df.pkl')

# %% Print the shape of the dataset
print('Shape of the training dataset: ', train_df.shape)


# %% Print the first 5 rows of the dataset
train_df.head()

# %% Drop Meter_reading column from the train dataset
train_df = train_df.drop(['meter_reading'], axis=1)


# %% Split the dataset into the Training set and validation set and make sure that the validation set is from the same groupNum_train
X_train, X_val = train_test_split(
    train_df, test_size=0.2, random_state=42, shuffle=True, stratify=train_df['groupNum_train'])
y_train = X_train['meter_reading_log1p'] # type: ignore
y_val = X_val['meter_reading_log1p'] # type: ignore 
X_train = X_train.drop(['meter_reading_log1p'], axis=1) # type: ignore
X_val = X_val.drop(['meter_reading_log1p'], axis=1) # type: ignore

# %% Print the unique values of the groupNum_train in X_train and X_val and sort them
print('Unique values of the groupNum_train in X_train: ', np.sort(
    X_train['groupNum_train'].unique()))
print('Unique values of the groupNum_train in X_val: ', np.sort(
    X_val['groupNum_train'].unique()))


# %% Print the shape of the dataset
print('Shape of the training dataset: ', X_train.shape)
print('Shape of the validation dataset: ', X_val.shape)
print('Shape of the training labels: ', y_train.shape)
print('Shape of the validation labels: ', y_val.shape)

del train_df
gc.collect()

# %% Print a correlation matrix for a random subset of the features

X_train_subset = X_train.sample(frac=0.1, random_state=42)
corr = X_train_subset.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr, annot=False, fmt='.2f', cmap='coolwarm')
plt.show()

del X_train_subset
gc.collect()


# %% Selecting the features best found by the LightGBM model
print("Feature Selection...")

category_cols = ['building_id', 'site_id', 'primary_use',
                 'IsHoliday']  # , 'groupNum_train']  # , 'meter'
feature_cols = ['square_feet_np_log1p', 'year_built'] + [
    'hour', 'weekend',
    #    'day', # 'month' ,
    #    'dayofweek',
    #    'building_median'
    #    'square_feet'
] + [
    'air_temperature', 'cloud_coverage',
    'dew_temperature', 'precip_depth_1_hr',
    'sea_level_pressure',
    # 'wind_direction', 'wind_speed',
    'air_temperature_mean_lag72',
    'air_temperature_max_lag72', 'air_temperature_min_lag72',
    'air_temperature_std_lag72', 'cloud_coverage_mean_lag72',
    'dew_temperature_mean_lag72', 'precip_depth_1_hr_mean_lag72',
    'sea_level_pressure_mean_lag72',
    # 'wind_direction_mean_lag72',
    'wind_speed_mean_lag72',
    'air_temperature_mean_lag3',
    'air_temperature_max_lag3',
    'air_temperature_min_lag3', 'cloud_coverage_mean_lag3',
    'dew_temperature_mean_lag3',
    'precip_depth_1_hr_mean_lag3',
    'sea_level_pressure_mean_lag3',
    #    'wind_direction_mean_lag3', 'wind_speed_mean_lag3',
    #    'floor_area',
    'year_cnt', 'bid_cnt',
    'dew_smooth', 'air_smooth',
    'dew_diff', 'air_diff',
    'dew_diff2', 'air_diff2'
]

# %% Check the correlation between the features using fraction of the dataset which has the features selected by the LightGBM model
X_train_subset = X_train[feature_cols + category_cols].sample(
    frac=0.1, random_state=42)
corr = X_train_subset.corr()
plt.figure(figsize=(20, 20))
sns.heatmap(corr, annot=False, fmt='.2f', cmap='coolwarm')
plt.show()

del X_train_subset
gc.collect()

#%% Normalize the features
print("Normalizing the features...")
scaler = StandardScaler()
X_train[feature_cols] = scaler.fit_transform(X_train[feature_cols])
X_val[feature_cols] = scaler.transform(X_val[feature_cols])

#%% Encode the categorical features using label encodin
print("label encoding the categorical features...")
for col in category_cols:
   le = LabelEncoder()
   X_train[col] = le.fit_transform(X_train[col])
   X_val[col] = le.transform(X_val[col])

#%% Print the dtype of the categorical features after label encoding
print('Dtype of the categorical features after label encoding: ', X_train[category_cols].dtypes)

# # %% One-hot encode the categorical features
# print("One-hot encoding the categorical features...")
# X_train = pd.get_dummies(X_train, columns=category_cols)
# X_val = pd.get_dummies(X_val, columns=category_cols)

#%% Scale the categorical features
print("Scaling the categorical features...")
scaler = StandardScaler()
X_train[category_cols] = scaler.fit_transform(X_train[category_cols])
X_val[category_cols] = scaler.transform(X_val[category_cols])

# # %% Encode categorical features and use MaxMinScaler to scale the features

# # Encode categorical features
# for col in category_cols:
#     le = LabelEncoder()
#     train_df[col] = le.fit_transform(train_df[col])

# # Scale features
# scaler = MinMaxScaler()
# train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])

# # %% Print the head of the data with the encoded categorical features
# print(train_df[category_cols].head())

# # %% Print the head of the data with the scaled features
# print(train_df[feature_cols].head())

# %% Print the shape of the dataset
print('Shape of the training dataset: ', X_train[feature_cols + category_cols].shape)
print('Shape of the validation dataset: ', X_val[feature_cols + category_cols].shape)
print('Shape of the training labels: ', y_train.shape)
print('Shape of the validation labels: ', y_val.shape)




# %% define the model with batch normalization and dropout layers


def create_model(input_shape):
    # Build 1D CNN autoencoder
    model = Sequential()
    model.add(Conv1D(16, kernel_size=1,
              activation='sigmoid', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size=1, activation='sigmoid'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=1, activation='sigmoid'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, kernel_size=1, activation='sigmoid'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    return model 
    # Add gradient clipping
    


# %% Run a loop to train the model based on the groupNum_train
for groupNum in X_train['groupNum_train'].unique():
    print('Group Number: ', groupNum)
    # Trim the dataset based on the selected features and groupNum_train
    X_train_group = X_train[X_train['groupNum_train']
                            == groupNum][feature_cols + category_cols]
    X_val_group = X_val[X_val['groupNum_train']
                        == groupNum][feature_cols + category_cols]

    # Get the labels
    y_train_group = y_train[X_train['groupNum_train'] == groupNum]
    y_val_group = y_val[X_val['groupNum_train'] == groupNum]

    
    # Convert the dataset to numpy array
    X_train_group = X_train_group.values
    X_val_group = X_val_group.values

    # Reshape the dataset to batch_size, timesteps, features
    X_train_group = X_train_group.reshape(
        X_train_group.shape[0], X_train_group.shape[1], 1)
    X_val_group = X_val_group.reshape(
        X_val_group.shape[0], X_val_group.shape[1], 1)

    # Reshape the labels to batch_size, timesteps, features
    y_train_group = y_train_group.values.reshape(
        y_train_group.shape[0], 1, 1)
    y_val_group = y_val_group.values.reshape(
        y_val_group.shape[0], 1, 1)

    # Get the input shape
    input_shape = (X_train_group.shape[1], X_train_group.shape[2])

    #  Create the model
    model = create_model(input_shape)
    # Apply Gradient clipping
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    #gvs = optimizer.get_gradients(model.total_loss, model.trainable_weights)
    #capped_gvs = [tf.clip_by_value(grad, -1., 1.) for grad in gvs]
    #optimizer.apply_gradients(zip(capped_gvs, model.trainable_weights))
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])
    model.summary()



    # Early stopping and model checkpoint
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=3, mode='min')
    model_checkpoint = ModelCheckpoint(
        'model.h5', monitor='val_loss', save_best_only=True, mode='min')

    #  Train the model
    history = model.fit(X_train_group, y_train_group, epochs=100, batch_size=512, validation_data=(
        X_val_group, y_val_group), callbacks=[early_stopping, model_checkpoint])
    
    #  Plot the training and validation loss
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [meter_reading_log1p]')
    plt.legend()
    plt.grid(True)
    plt.show()


    #  Plot the training and validation accuracy
    plt.plot(history.history['mse'], label='mse')
    plt.plot(history.history['val_mse'], label='val_mse')
    plt.xlabel('Epoch')
    plt.ylabel('Error [meter_reading_log1p]')
    plt.legend()
    plt.grid(True)
    plt.show()


    #  Predict the meter_reading_log1p
    y_pred = model.predict(X_val_group)

    #  Calculate the RMSE
    # print('RMSE: ', np.sqrt(mean_squared_error(y_val_group, y_pred)))

    #  Calculate the MAE
    # print('MAE: ', mean_absolute_error(y_val_group, y_pred))

    # Save the model based on the groupNum_train in the model folder
    model.save('model/model_' + str(groupNum) + '.h5')

    """ #  Delete the model
    del model
    gc.collect()

    #  Delete the X_train_group and X_val_group dataset
    del X_train_group, X_val_group
    gc.collect()

    #  Delete the y_train_group and y_val_group dataset
    del y_train_group, y_val_group
    gc.collect() """


 # %% Load the test dataset
test_df = pd.read_pickle(
    '/workspace/Ashrae-Energy-Prediction-III/src/data/test_df.pkl')
building_meta_df = pd.read_pickle(
    '/workspace/Ashrae-Energy-Prediction-III/src/data/building_meta_df.pkl')
weather_test_df = pd.read_pickle(
    '/workspace/Ashrae-Energy-Prediction-III/src/data/weather_test_df.pkl')
#%%
print(weather_test_df.columns)

# %% print the shape of the dataset
print('Shape of the test dataset: ', test_df.shape)
print('Shape of the building_meta_df dataset: ', building_meta_df.shape)
print('Shape of the weather_test_df dataset: ', weather_test_df.shape)

# %% Merge the test data with the building metadata and weather test data
target_test_df = test_df.copy()
target_test_df = target_test_df.merge(
        building_meta_df, on=['building_id', 'meter', 'groupNum_train', 'square_feet'], how='left')
target_test_df = target_test_df.merge(
    weather_test_df, on=['site_id', 'timestamp'], how='left')
test_df = target_test_df[feature_cols + category_cols]
del target_test_df, building_meta_df, weather_test_df
gc.collect()

# %% Print the shape of the Test dataset after merging 
print('Shape of the test dataset after merging: ', test_df.shape)


# %% Print the columns of the test dataset
print('Columns of the test dataset: ', test_df.columns)

# #%% Merge the test dataset with the building_meta_df
# target_test_df = test_df[test_df['groupNum_train']
#                             == groupNum_train].copy()
# target_test_df = target_test_df.merge(
#    building_meta_df, on=['building_id', 'meter', 'groupNum_train', 'square_feet'], how='left')
# target_test_df = target_test_df.merge(
#    weather_test_df, on=['site_id', 'timestamp'], how='left')
# X_test = target_test_df[feature_cols + category_cols]

# %% Normalize the features in the test data
print("Normalizing the features in the test data...")
scaler = StandardScaler()
test_df[feature_cols] = scaler.fit_transform(test_df[feature_cols])

# %% Encode the categorical features using label encoding
print("label encoding the categorical features in the test data...")
for col in category_cols:
   le = LabelEncoder()
   test_df[col] = le.fit_transform(test_df[col])
    
# %% Scale the categorical features
print("Scaling the categorical features in the test data...")
scaler = StandardScaler()
test_df[category_cols] = scaler.fit_transform(test_df[category_cols])

# # %% Encode categorical features and use MaxMinScaler to scale the features
# for col in category_cols:
#     le = LabelEncoder()
#     test_df[col] = le.fit_transform(test_df[col])

# scaler = MinMaxScaler()
# test_df[feature_cols] = scaler.transform(test_df[feature_cols])

# %% Print the head of the data with the encoded categorical features
print(test_df[category_cols].head())

# %% Print the head of the data with the scaled features
print(test_df[feature_cols].head())
# %% Read the sample_submission.csv file
sample_submission_df = pd.read_feather(
    '/workspace/Ashrae-Energy-Prediction-III/src/data/sample_submission.feather')

# %% print the shape of the dataset
print('Shape of the sample_submission_df dataset: ', sample_submission_df.shape)

# %% Print the first 5 rows of the dataset
sample_submission_df.head()

# %% Print the first 5 rows of the test dataset
test_df.head()

#%% Read the groupNum_train from the test dataset saved in the pickle file
test_df_GroupNumTrain = pd.read_pickle(
    '/workspace/Ashrae-Energy-Prediction-III/src/data/test_df.pkl')
test_df_GroupNumTrain = test_df_GroupNumTrain['groupNum_train']

# Adding the groupNum_train to the test_df
test_df['groupNum_train'] = test_df_GroupNumTrain

del test_df_GroupNumTrain
gc.collect()

#print the head of the test_df
print(test_df.head())

# %% Run a loop to predict the meter_reading_log1p based on the groupNum_train

for groupNum in test_df['groupNum_train'].unique():
    print('Group Number: ', groupNum)
    # Select the Features in the test dataset
    X_test_group = test_df[test_df['groupNum_train']
                            == groupNum][feature_cols + category_cols]
    
    # Reshape the X_test_group dataset
    X_test_group = X_test_group.values.reshape(
        X_test_group.shape[0], X_test_group.shape[1], 1)

    # Load the model
    model = load_model('model/model_' + str(groupNum) + '.h5')

    # Predict the meter_reading_log1p
    y_pred = model.predict(X_test_group)

    # convert the meter_reading_log1p to meter_reading
    y_pred = np.expm1(y_pred)

    # Save the meter_reading to the sample_submission_df by matching the index of the X_test dataset with the row_id of the sample_submission_df dataset
    sample_submission_df.loc[test_df[test_df['groupNum_train'] == groupNum].index,
                             'meter_reading'] = y_pred.reshape(-1)

    # Delete the model
    del model, X_test_group, y_pred
    gc.collect()






# %% Print the first 5 rows of the sample_submission_df dataset
sample_submission_df.head()

# %% Print the unique values of the meter_reading the sample_submission_df dataset
print('Unique values of the meter_reading in the sample_submission_df dataset: ', sample_submission_df['meter_reading'].unique())

# %% Save the sample_submission_df dataset to the submission folder
sample_submission_df.to_csv('/workspace/Ashrae-Energy-Prediction-III/src/data/submission_final_CNN.csv', index=False)

# %% Push the CNN_Best_Feature.py file to the github repository remotes/origin/17-feature-addition-cnn using git
# !git add CNN_Best_Feature.py
# !git commit -m "Added the CNN_Best_Feature.py file"
# !git push origin 17-feature-addition-cnn 
# !git status
#  
