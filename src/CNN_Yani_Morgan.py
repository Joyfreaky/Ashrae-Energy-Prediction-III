# %% Import the libraries to train a CNN model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers,layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error

import gc
import warnings
warnings.filterwarnings('ignore')

# %% Import the dataset
train_df = pd.read_pickle(
    '/workspace/Ashrae-Energy-Prediction-III/src/data/train_df.pkl')
# test_df = pd.read_pickle('/workspace/Ashrae-Energy-Prediction-III/src/data/test_df.pkl')

# %% Print the shape of the dataset
print('Shape of the training dataset: ', train_df.shape)
# print('Shape of the testing dataset: ', test_df.shape)

# %% Import the weather dataset and building metadata
# weather_test = pd.read_pickle('/workspace/Ashrae-Energy-Prediction-III/src/data/weather_test_df.pkl')
# building_metadata = pd.read_pickle('/workspace/Ashrae-Energy-Prediction-III/src/data/building_meta_df.pkl')

# %% Print the shape of the dataset
# print('Shape of the weather dataset: ', weather_test.shape)
# print('Shape of the building metadata: ', building_metadata.shape)

# %% Print the columns of the dataset
# print('Columns of the training dataset: ', train_df.columns)
# print('Columns of the testing dataset: ', test_df.columns)
# print('Columns of the weather dataset: ', weather_test.columns)
# print('Columns of the building metadata: ', building_metadata.columns)

# %% Drop Meter_reading column from the train dataset
train_df = train_df.drop(['meter_reading'], axis=1)

# %% Split the dataset into the Training set and validation set and make sure that the validation set is from the same groupNum_train
X_train, X_val = train_test_split(
    train_df, test_size=0.2, random_state=42, shuffle=True, stratify=train_df['groupNum_train']) # each every unit groupNum produces x_train, x_val
y_train = X_train['meter_reading_log1p']
y_val = X_val['meter_reading_log1p']
X_train = X_train.drop(['meter_reading_log1p'], axis=1)
X_val = X_val.drop(['meter_reading_log1p'], axis=1)
#%% print the unique value of the GroupNum train in X_train, X_val
print('unique value of the GroupNum_train in X_train',
     X_train['groupNum_train'].unique())               
print('unique value of the GroupNum_train in X_val',
     X_val['groupNum_train'].unique())         # all these unique value is also in X_train but is also in X_val which means the data is same as the training data 
# %% Print the shape of the dataset
print('Shape of the training dataset: ', X_train.shape)
print('Shape of the validation dataset: ', X_val.shape)
print('Shape of the training labels: ', y_train.shape)
print('Shape of the validation labels: ', y_val.shape)

del train_df
gc.collect()
#%% print a correlation matrix 
import seaborn as sns

X_train_subset=X_train.sample(frac=0.1,random_state=42)
corr=X_train_subset.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr,annot=False, fmt='2f',cmap='coolwarm') # annot = number 
# for exmaple, the column of month is highly corrlated with index btw i dont need both of the column
# function to check if the correlation between two features is greater than a threshold and drop one of them 
#def check_corr(df,threshold):
#    corr_matrix-df.corr().abs()
# threshold anyting which is below 0.4 or 0.8
# %% Selecting the features best found by the LightGBM model (there are around 50 percent of features in here )
print("Feature Selection...")
#1. we need to do feature selection based on our cnn model
#2.    
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

# %% define the model
def create_model(input_shape):
    # Build 1D CNN autoencoder
    model = Sequential()
    model.add(Conv1D(16, kernel_size=3,
              activation='sigmoid', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size=3, activation='sigmoid'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=3, activation='sigmoid'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(128, kernel_size=3, activation='sigmoid'))
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
    # Feature selection pipeline
    X_train = X_train[category_cols + feature_cols]
    X_val = X_val[category_cols + feature_cols]
    gc.collect()
    # Set the input shape for the model
    input_shape = (X_train[X_train['groupNum_train'] == groupNum].drop(
            'groupNum_train', axis=1).shape[1], 1) # do not change this 
    #Create the model
    model = create_model(input_shape)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), 
    loss='mse',
    metrics='mae')
    model.summary()
    #tf.keras.optimizers.SGD(learning_rate=0.001)
    #tf.keras.optimizers.Adam(clipnorm=1.0)
    #Reshape the X_train and X_val dataset to fit the CNN model input shape (n, 1)
    X_train_group = X_train[X_train['groupNum_train']
                                == groupNum].drop('groupNum_train', axis=1).copy()
    X_val_group = X_val[X_val['groupNum_train'] ==
                            groupNum].drop('groupNum_train', axis=1).copy()
    X_train_group = X_train_group.values.reshape(
            X_train_group.shape[0], X_train_group.shape[1], 1)
    X_val_group = X_val_group.values.reshape(
            X_val_group.shape[0], X_val_group.shape[1], 1)

    #Reshape the y_train and y_val dataset to fit the CNN model output shape (1)
    y_train_group = y_train[X_train['groupNum_train']
                                == groupNum].values.reshape(-1, 1)
    y_val_group = y_val[X_val['groupNum_train']
                            == groupNum].values.reshape(-1, 1)

    #Early stopping and model checkpoint
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min') # if the accuracy does not increase after iterations, it gonna early stop
    model_checkpoint = ModelCheckpoint(
            'model.h5', monitor='val_loss', save_best_only=True, mode='min')

    #  Train the model
    #history = model.fit(X_train_group, y_train_group, epochs=100, batch_size=128,
    #                        validation_data=(X_val_group, y_val_group),
    #                        callbacks=[early_stopping, model_checkpoint])
    history =  model.fit(X_train_group, y_train_group, epochs=10, batch_size=32, validation_data=(X_val_group, y_val_group),
                        callbacks=[early_stopping, model_checkpoint])

    # threshold = 1.0

    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    # grads_and_vars = optimizer.compute_gradients(loss, var_list=variable)
    # capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
    #             for grad, var in grads_and_vars]
    # training_op = optimizer.apply_gradients(capped_gvs)
    # #  Plot the training and validation loss
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [meter_reading_log1p]')
    plt.legend()
    plt.grid(True)
    plt.show()

    #  Plot the training and validation accuracy
    plt.plot(history.history['mae'], label='mae')
    plt.plot(history.history['val_mae'], label='val_mae')
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

    #  Delete the model
    del model
    gc.collect()

    #  Delete the X_train_group and X_val_group dataset
    del X_train_group, X_val_group
    gc.collect()

    #  Delete the y_train_group and y_val_group dataset
    del y_train_group, y_val_group
    gc.collect()


# %% Load the test dataset
test_df = pd.read_pickle(
    '/workspace/Ashrae-Energy-Prediction-III/src/data/train_df.pkl')
building_meta_df = pd.read_pickle(
    '/workspace/Ashrae-Energy-Prediction-III/src/data/building_meta_df.pkl')
weather_test_df = pd.read_pickle(
    '/workspace/Ashrae-Energy-Prediction-III/src/data/weather_test_df.pkl')

# %% print the shape of the dataset
print('Shape of the test dataset: ', test_df.shape)
print('Shape of the building_meta_df dataset: ', building_meta_df.shape)
print('Shape of the weather_test_df dataset: ', weather_test_df.shape)

# %% Merge the test dataset with the building_meta_df
# target_test_df = test_df[test_df['groupNum_train']
#                             == groupNum_train].copy()
# target_test_df = target_test_df.merge(
#    building_meta_df, on=['building_id', 'meter', 'groupNum_train', 'square_feet'], how='left')
# target_test_df = target_test_df.merge(
#    weather_test_df, on=['site_id', 'timestamp'], how='left')
X_test = test_df[feature_cols + category_cols]

# %% Read the sample_submission.csv file
sample_submission_df = pd.read_feather(
    '/workspace/Ashrae-Energy-Prediction-III/src/data/sample_submission.feather')

# %% print the shape of the dataset
print('Shape of the sample_submission_df dataset: ', sample_submission_df.shape)

# %% Print the first 5 rows of the dataset
sample_submission_df.head()

# %% Print the first 5 rows of the test dataset
X_test.head()

# %% Run a loop to predict the meter_reading_log1p based on the groupNum_train
i = 0
for groupNum in X_test['groupNum_train'].unique():
    # Run the loop only two times
    i += 1
    if i > 1:
        break
    # Drop the groupNum_train column
    X_test_group = X_test[X_test['groupNum_train']
                          == groupNum].drop('groupNum_train', axis=1).copy()
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
    sample_submission_df.loc[X_test[X_test['groupNum_train'] == groupNum].index,
                             'meter_reading'] = y_pred.reshape(-1)

    # Delete the model
    del model, X_test_group, y_pred
    gc.collect()


# %% Print the first 5 rows of the sample_submission_df dataset
sample_submission_df.head()

# %% save the sample_submission_df dataset to the submission.csv file
sample_submission_df.to_csv('submission.csv', index=False)
