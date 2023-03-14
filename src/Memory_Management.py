# %% [code]
"""This script is used to reduce the memory usage of the dataset and save it in a feather format. 
   Before running this script, please change the root variable to the location of the raw dataset.
   or you can just run the script in the Gitpod environment.
   Also if you are running this script in the Gitpod environment, please run the /workspace/Ashrae-Energy-Prediction-III-21-22/src/File_import_api_git.ipynb script first."""
import os
import gc

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.api.types import is_datetime64_any_dtype as is_datetime

# %% [code]
# Copy from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
# Modified to support timestamp type
# Modified to add option to use float16 or not. feather format does not support float16.


def reduce_mem_usage(df, use_float16=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        if is_datetime(df[col]):
            # skip datetime type
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


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df


# Read data...
# Change the Location of the raw of the dataset before run.

print('Reading data...')
root = '/workspace/Ashrae-Energy-Prediction-III/data'

train_df = pd.read_csv(os.path.join(root, 'train.csv'))
weather_train_df = pd.read_csv(os.path.join(root, 'weather_train.csv'))
test_df = pd.read_csv(os.path.join(root, 'test.csv'))
weather_test_df = pd.read_csv(os.path.join(root, 'weather_test.csv'))
building_meta_df = pd.read_csv(os.path.join(root, 'building_metadata.csv'))
sample_submission = pd.read_csv(os.path.join(root, 'sample_submission.csv'))

print('Converting to datetime...')

train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
weather_train_df['timestamp'] = pd.to_datetime(weather_train_df['timestamp'])
weather_test_df['timestamp'] = pd.to_datetime(weather_test_df['timestamp'])

print("Reducing memory usage...")

reduce_mem_usage(train_df)
reduce_mem_usage(test_df)
reduce_mem_usage(building_meta_df)
reduce_mem_usage(weather_train_df)
reduce_mem_usage(weather_test_df)

# Save the data in feather format for faster loading in the data folder.

print('Saving data...')
train_df.to_feather('data/train.feather')
test_df.to_feather('data/test.feather')
weather_train_df.to_feather('data/weather_train.feather')
weather_test_df.to_feather('data/weather_test.feather')
building_meta_df.to_feather('data/building_metadata.feather')
sample_submission.to_feather('data/sample_submission.feather')
# %% [code]

gc.collect()  # Garbage Collection

print("Script executed Sucessfully.")
