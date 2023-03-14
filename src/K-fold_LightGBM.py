# %% [code]
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from scipy.signal import savgol_filter as sg
import lightgbm as lgb
import holidays
from IPython.core.display import display, HTML
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
from tqdm import tqdm_notebook as tqdm
from pandas.api.types import is_categorical_dtype
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import os
import gc
import sys
import random
from pathlib import Path
import warnings
from sklearn.model_selection import StratifiedKFold
print("Importing libraries...")


warnings.filterwarnings('ignore')


# %% [code]
print("Declaring global variables...")

win = 3  # 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
black_day = 10
outlier = True
rescale = True

debug = True
Train = False
num_rounds = 200

clip0 = True

folds = 3  # 3, 6, 12
# 3:
# 6:
# 12:

use_ucf = True
ucf_clip = True

ucf_year = [2017, 2018]  # ucf data year used in train

predmode = 'all'  # 'valid', 'train', 'all'

zone_dict = {0: 4, 1: 0, 2: 7, 3: 4, 4: 7, 5: 0, 6: 4, 7: 4,
             8: 4, 9: 5, 10: 7, 11: 4, 12: 0, 13: 5, 14: 4, 15: 4}

# Site Specific Holidays
en_holidays = holidays.UK()
ir_holidays = holidays.Ireland()
ca_holidays = holidays.Canada()
us_holidays = holidays.UnitedStates()

# %% [code]
print("Declaring functions...")
# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
# Modified to support timestamp type, categorical type
# Modified to add option to use float16 or not. feather format does not support float16.


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


def set_local(df):
    for sid, zone in zone_dict.items():
        sids = df.site_id == sid
        df.loc[sids, 'timestamp'] = df[sids].timestamp - pd.offsets.Hour(zone)


def add_holiyday(df_weather):
    en_idx = df_weather.query('site_id == 1 or site_id == 5').index
    ir_idx = df_weather.query('site_id == 12').index
    ca_idx = df_weather.query('site_id == 7 or site_id == 11').index
    us_idx = df_weather.query(
        'site_id == 0 or site_id == 2 or site_id == 3 or site_id == 4 or site_id == 6 or site_id == 8 or site_id == 9 or site_id == 10 or site_id == 13 or site_id == 14 or site_id == 15').index

    df_weather['IsHoliday'] = 0
    df_weather.loc[en_idx, 'IsHoliday'] = df_weather.loc[en_idx,
                                                         'timestamp'].apply(lambda x: en_holidays.get(x, default=0))
    df_weather.loc[ir_idx, 'IsHoliday'] = df_weather.loc[ir_idx,
                                                         'timestamp'].apply(lambda x: ir_holidays.get(x, default=0))
    df_weather.loc[ca_idx, 'IsHoliday'] = df_weather.loc[ca_idx,
                                                         'timestamp'].apply(lambda x: ca_holidays.get(x, default=0))
    df_weather.loc[us_idx, 'IsHoliday'] = df_weather.loc[us_idx,
                                                         'timestamp'].apply(lambda x: us_holidays.get(x, default=0))

    holiday_idx = df_weather['IsHoliday'] != 0
    df_weather.loc[holiday_idx, 'IsHoliday'] = 1
    df_weather['IsHoliday'] = df_weather['IsHoliday'].astype(np.uint8)


def preprocess(df):
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["weekend"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.dayofweek

# Adding some lag feature


def add_lag_feature(weather_df, window=3):
    group_df = weather_df.groupby('site_id')
    cols = ['air_temperature', 'cloud_coverage', 'dew_temperature',
            'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']
    rolled = group_df[cols].rolling(window=window, min_periods=0)
    lag_mean = rolled.mean().reset_index().astype(np.float16)
    lag_max = rolled.max().reset_index().astype(np.float16)
    lag_min = rolled.min().reset_index().astype(np.float16)
    lag_std = rolled.std().reset_index().astype(np.float16)
    for col in cols:
        weather_df[f'{col}_mean_lag{window}'] = lag_mean[col]
        weather_df[f'{col}_max_lag{window}'] = lag_max[col]
        weather_df[f'{col}_min_lag{window}'] = lag_min[col]
        weather_df[f'{col}_std_lag{window}'] = lag_std[col]

# SG Filter for Weather


def add_sg(df):
    w = 11
    p = 2
    for si in df.site_id.unique():
        index = df.site_id == si
        df.loc[index, 'air_smooth'] = sg(df[index].air_temperature, w, p)
        df.loc[index, 'dew_smooth'] = sg(df[index].dew_temperature, w, p)

        df.loc[index, 'air_diff'] = sg(df[index].air_temperature, w, p, 1)
        df.loc[index, 'dew_diff'] = sg(df[index].dew_temperature, w, p, 1)

        df.loc[index, 'air_diff2'] = sg(df[index].air_temperature, w, p, 2)
        df.loc[index, 'dew_diff2'] = sg(df[index].dew_temperature, w, p, 2)


# %% [code]
print("Importing data...")

# Change the path to the source file path, use Memory_Management.py to generate files in feather format
root = Path('/workspace/Ashrae-Energy-Prediction-III/src/data')
train_df = pd.read_feather(root/'train.feather')
weather_train_df = pd.read_feather(root/'weather_train.feather')
building_meta_df = pd.read_feather(root/'building_metadata.feather')

# %% [code]
print("Merging data...")

building_meta_df = building_meta_df.merge(
    train_df[['building_id', 'meter']].drop_duplicates(), on='building_id')

# %% [code]
print("Adding features...")
# Set group  (site-meter) for training models
building_meta_df['groupNum_train'] = building_meta_df['site_id'].astype(
    'int')*10 + building_meta_df['meter'].astype('int')  # Add group number

building_meta_df['floor_area'] = building_meta_df.square_feet / \
    building_meta_df.floor_count  # Add floor area


set_local(weather_train_df)  # Set local time

add_holiyday(weather_train_df)  # Add holiday

# %% [code]
print('Shape of Train Data:', train_df.shape)
print('Shape of Building Data:', building_meta_df.shape)
print('Shape of Weather Train Data:', weather_train_df.shape)


# %% [Markdown]
# ## Data Cleaning
print("Cleaning data...")

# %% [code]
train_df[train_df.building_id == 954].meter_reading.plot()

# %% [code]
(train_df[train_df.building_id == 954].meter_reading == 0).astype(int).plot()

# %% [code]
train_df[train_df.building_id == 1221].meter_reading.plot()

# %% [code]
train_df = train_df.query('not (building_id == 954 & meter_reading == 0)')
train_df = train_df.query('not (building_id == 1221 & meter_reading == 0)')

# %% [code]
train_df[train_df.building_id == 954].meter_reading.plot()

# %% [code]
train_df[train_df.building_id == 1221].meter_reading.plot()

# %% [code]
# Removing buildings with meter == 0 which has zero reading, before first initial reading.
print("Removing buildings with meter == 0 which has zero reading, before first usage.")
train_df = train_df.query(
    'not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20 18")')
train_df = train_df.query(
    'not (building_id == 681 & meter == 0 & timestamp <= "2016-04-27")')
train_df = train_df.query(
    'not (building_id == 761 & meter == 0 & timestamp <= "2016-09-02")')
train_df = train_df.query(
    'not (building_id == 799 & meter == 0 & timestamp <= "2016-09-02")')
train_df = train_df.query(
    'not (building_id == 802 & meter == 0 & timestamp <= "2016-08-24")')
train_df = train_df.query(
    'not (building_id == 1073 & meter == 0 & timestamp <= "2016-10-26")')
train_df = train_df.query(
    'not (building_id == 1094 & meter == 0 & timestamp <= "2016-09-08")')
train_df = train_df.query(
    'not (building_id == 29 & meter == 0 & timestamp <= "2016-08-10")')
train_df = train_df.query(
    'not (building_id == 40 & meter == 0 & timestamp <= "2016-06-04")')
train_df = train_df.query(
    'not (building_id == 45 & meter == 0 & timestamp <= "2016-07")')
train_df = train_df.query(
    'not (building_id == 106 & meter == 0 & timestamp <= "2016-11")')
train_df = train_df.query(
    'not (building_id == 107 & meter == 0 & timestamp >= "2016-11-10")')
train_df = train_df.query(
    'not (building_id == 112 & meter == 0 & timestamp < "2016-10-31 15")')
train_df = train_df.query(
    'not (building_id == 144 & meter == 0 & timestamp > "2016-05-14" & timestamp < "2016-10-31")')
train_df = train_df.query(
    'not (building_id == 147 & meter == 0 & timestamp > "2016-06-05 19" & timestamp < "2016-07-18 15")')
train_df = train_df.query(
    'not (building_id == 171 & meter == 0 & timestamp <= "2016-07-05")')
train_df = train_df.query(
    'not (building_id == 177 & meter == 0 & timestamp > "2016-06-04" & timestamp < "2016-06-25")')
train_df = train_df.query(
    'not (building_id == 258 & meter == 0 & timestamp > "2016-09-26" & timestamp < "2016-12-12")')
train_df = train_df.query(
    'not (building_id == 258 & meter == 0 & timestamp > "2016-08-30" & timestamp < "2016-09-08")')
train_df = train_df.query(
    'not (building_id == 258 & meter == 0 & timestamp > "2016-09-18" & timestamp < "2016-09-25")')
train_df = train_df.query(
    'not (building_id == 260 & meter == 0 & timestamp <= "2016-05-11")')
train_df = train_df.query(
    'not (building_id == 269 & meter == 0 & timestamp > "2016-06-04" & timestamp < "2016-06-25")')
train_df = train_df.query(
    'not (building_id == 304 & meter == 0 & timestamp >= "2016-11-20")')
train_df = train_df.query(
    'not (building_id == 545 & meter == 0 & timestamp > "2016-01-17" & timestamp < "2016-02-10")')
train_df = train_df.query(
    'not (building_id == 604 & meter == 0 & timestamp < "2016-11-21")')
train_df = train_df.query(
    'not (building_id == 693 & meter == 0 & timestamp > "2016-09-07" & timestamp < "2016-11-23")')
train_df = train_df.query(
    'not (building_id == 693 & meter == 0 & timestamp > "2016-07-12" & timestamp < "2016-05-29")')
train_df = train_df.query(
    'not (building_id == 723 & meter == 0 & timestamp > "2016-10-06" & timestamp < "2016-11-22")')
train_df = train_df.query(
    'not (building_id == 733 & meter == 0 & timestamp > "2016-05-29" & timestamp < "2016-06-22")')
train_df = train_df.query(
    'not (building_id == 733 & meter == 0 & timestamp > "2016-05-19" & timestamp < "2016-05-20")')
train_df = train_df.query(
    'not (building_id == 803 & meter == 0 & timestamp > "2016-9-25")')
train_df = train_df.query(
    'not (building_id == 815 & meter == 0 & timestamp > "2016-05-17" & timestamp < "2016-11-17")')
train_df = train_df.query(
    'not (building_id == 848 & meter == 0 & timestamp > "2016-01-15" & timestamp < "2016-03-20")')
train_df = train_df.query(
    'not (building_id == 857 & meter == 0 & timestamp > "2016-04-13")')
train_df = train_df.query(
    'not (building_id == 909 & meter == 0 & timestamp < "2016-02-02")')
train_df = train_df.query(
    'not (building_id == 909 & meter == 0 & timestamp < "2016-06-23")')
train_df = train_df.query(
    'not (building_id == 1008 & meter == 0 & timestamp > "2016-10-30" & timestamp < "2016-11-21")')
train_df = train_df.query(
    'not (building_id == 1113 & meter == 0 & timestamp < "2016-07-27")')
train_df = train_df.query(
    'not (building_id == 1153 & meter == 0 & timestamp < "2016-01-20")')
train_df = train_df.query(
    'not (building_id == 1169 & meter == 0 & timestamp < "2016-08-03")')
train_df = train_df.query(
    'not (building_id == 1170 & meter == 0 & timestamp > "2016-06-30" & timestamp < "2016-07-05")')
train_df = train_df.query(
    'not (building_id == 1221 & meter == 0 & timestamp < "2016-11-04")')
train_df = train_df.query(
    'not (building_id == 1225 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id == 1234 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id >= 1233 & building_id <= 1234 & meter == 0 & timestamp > "2016-01-13 22" & timestamp < "2016-03-08 12")')
train_df = train_df.query(
    'not (building_id == 1241 & meter == 0 & timestamp > "2016-07-14" & timestamp < "2016-11-19")')
train_df = train_df.query(
    'not (building_id == 1250 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id == 1255 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id == 1264 & meter == 0 & timestamp > "2016-08-23")')
train_df = train_df.query(
    'not (building_id == 1265 & meter == 0 & timestamp > "2016-05-06" & timestamp < "2016-05-26")')
train_df = train_df.query(
    'not (building_id == 1272 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id >= 1275 & building_id <= 1280 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id == 1283 & meter == 0 & timestamp > "2016-07-08" & timestamp < "2016-08-03")')
train_df = train_df.query(
    'not (building_id >= 1291 & building_id <= 1302 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id == 1303 & meter == 0 & timestamp > "2016-07-25 22" & timestamp < "2016-07-27 16")')
train_df = train_df.query(
    'not (building_id == 1303 & meter == 0 & timestamp > "2016-01-26" & timestamp < "2016-06-02 12")')
train_df = train_df.query(
    'not (building_id == 1319 & meter == 0 & timestamp > "2016-05-17 16" & timestamp < "2016-06-07 12")')
train_df = train_df.query(
    'not (building_id == 1319 & meter == 0 & timestamp > "2016-08-18 14" & timestamp < "2016-09-02 14")')
train_df = train_df.query(
    'not (building_id == 1322 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')

# %% [Markdown]
# ## Remove outliers
print("Removing Outliers...")

# %% [code] {"scrolled":true}
train_df = train_df.query(
    'not (building_id >= 874 & building_id <= 997 & meter == 0 & timestamp > "2016-10-14 22" & timestamp < "2016-10-17 08")')
train_df = train_df.query(
    'not (building_id >= 874 & building_id <= 997 & meter == 0 & timestamp > "2016-07-01 14" & timestamp < "2016-07-05 06")')
train_df = train_df.query(
    'not (building_id >= 874 & building_id <= 997 & meter == 1 & timestamp > "2016-10-14 22" & timestamp < "2016-10-17 08")')
train_df = train_df.query(
    'not (building_id >= 874 & building_id <= 997 & meter == 1 & timestamp > "2016-07-01 14" & timestamp < "2016-07-05 06")')
train_df = train_df.query(
    'not (building_id >= 874 & building_id <= 997 & meter == 2 & timestamp > "2016-10-14 22" & timestamp < "2016-10-17 08")')
train_df = train_df.query(
    'not (building_id >= 874 & building_id <= 997 & meter == 2 & timestamp > "2016-07-01 14" & timestamp < "2016-07-05 06")')
train_df = train_df.query(
    'not (building_id == 1272 & meter == 1 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id >= 1291 & building_id <= 1297 & meter == 1 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id == 1300 & meter == 1 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id == 1302 & meter == 1 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id >= 1291 & building_id <= 1299 & meter == 2 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id == 1221 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id >= 1225 & building_id <= 1226 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id >= 1233 & building_id <= 1234 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id == 1241 & meter == 0 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id == 1223 & meter == 1 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id == 1226 & meter == 1 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id >= 1233 & building_id <= 1234 & meter == 1 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id >= 1225 & building_id <= 1226 & meter == 2 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id == 1305 & meter == 2 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id == 1307 & meter == 2 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id == 1223 & meter == 3 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id == 1231 & meter == 3 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id >= 1233 & building_id <= 1234 & meter == 3 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id == 1272 & meter == 3 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id >= 1275 & building_id <= 1297 & meter == 3 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id == 1300 & meter == 3 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id == 1302 & meter == 3 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id == 1293 & meter == 3 & timestamp > "2016-09-28 07" & timestamp < "2016-10-25 12")')
train_df = train_df.query(
    'not (building_id == 1302 & meter == 3 & timestamp > "2016-09-28 07" & timestamp < "2016-10-25 12")')
train_df = train_df.query(
    'not (building_id == 1223 & meter == 0 & timestamp > "2016-9-28 07" & timestamp < "2016-10-11 18")')
train_df = train_df.query(
    'not (building_id == 1225 & meter == 1 & timestamp > "2016-8-22 23" & timestamp < "2016-10-11 14")')
train_df = train_df.query(
    'not (building_id == 1230 & meter == 1 & timestamp > "2016-8-22 08" & timestamp < "2016-10-05 18")')
train_df = train_df.query(
    'not (building_id == 904 & meter == 0 & timestamp < "2016-02-17 08")')
train_df = train_df.query(
    'not (building_id == 986 & meter == 0 & timestamp < "2016-02-17 08")')
train_df = train_df.query(
    'not (building_id == 954 & meter == 0 & timestamp < "2016-08-08 11")')
train_df = train_df.query(
    'not (building_id == 954 & meter == 0 & timestamp < "2016-06-23 08")')
train_df = train_df.query(
    'not (building_id >= 745 & building_id <= 770 & meter == 1 & timestamp > "2016-10-05 01" & timestamp < "2016-10-10 09")')
train_df = train_df.query(
    'not (building_id >= 774 & building_id <= 787 & meter == 1 & timestamp > "2016-10-05 01" & timestamp < "2016-10-10 09")')


# %% [code] {"scrolled":true}
# 3rd cleaning hourly spikes
print('Cleaning hourly spikes...')
train_df = train_df.query(
    'not (building_id >= 874 & building_id <= 997 & meter == 0 & timestamp > "2016-05-11 09" & timestamp < "2016-05-12 01")')
train_df = train_df.query(
    'not (building_id >= 874 & building_id <= 997 & meter == 1 & timestamp > "2016-05-11 09" & timestamp < "2016-05-12 01")')
train_df = train_df.query(
    'not (building_id >= 874 & building_id <= 997 & meter == 2 & timestamp > "2016-05-11 09" & timestamp < "2016-05-12 01")')

train_df = train_df.query(
    'not (building_id >= 874 & building_id <= 997 & meter == 0 & timestamp == "2016-02-26 01")')
train_df = train_df.query(
    'not (building_id >= 874 & building_id <= 997 & meter == 1 & timestamp == "2016-02-26 01")')
train_df = train_df.query(
    'not (building_id >= 874 & building_id <= 997 & meter == 2 & timestamp == "2016-02-26 01")')

train_df = train_df.query(
    'not (building_id >= 874 & building_id <= 997 & meter == 0 & timestamp > "2016-03-29 10" & timestamp < "2016-03-30 12")')
train_df = train_df.query(
    'not (building_id >= 874 & building_id <= 997 & meter == 1 & timestamp > "2016-03-29 10" & timestamp < "2016-03-30 12")')
train_df = train_df.query(
    'not (building_id >= 874 & building_id <= 997 & meter == 2 & timestamp > "2016-03-29 10" & timestamp < "2016-03-30 12")')

train_df = train_df.query(
    'not (building_id >= 874 & building_id <= 997 & meter == 0 & timestamp > "2016-01-19 23" & timestamp < "2016-01-28 15")')
train_df = train_df.query(
    'not (building_id >= 874 & building_id <= 997 & meter == 1 & timestamp > "2016-01-19 23" & timestamp < "2016-01-28 15")')
train_df = train_df.query(
    'not (building_id >= 874 & building_id <= 997 & meter == 2 & timestamp > "2016-01-19 23" & timestamp < "2016-01-28 15")')

train_df = train_df.query(
    'not (building_id != 1227 & building_id != 1281 & building_id != 1314 & building_id >=1223 & building_id < 1335 & meter==0 & meter_reading==0)')

# %% [Markdown]
# ## 2.3. Remove missing values
print("Remove missing values...")
train_df = train_df.query(
    'not (building_id >= 1223 & building_id <= 1324 & meter==1 & timestamp > "2016-07-16 04" & timestamp < "2016-07-19 11")')
train_df = train_df.query(
    'not (building_id == 107 & meter == 0 & timestamp <= "2016-07-06")')
train_df = train_df.query(
    'not (building_id == 180 & timestamp >= "2016-02-17 12")')
train_df = train_df.query('not (building_id == 182 & meter == 0)')
train_df = train_df.query(
    'not (building_id == 191 & meter == 0 & timestamp >= "2016-12-22 09")')
train_df = train_df.query(
    'not (building_id == 192 & meter == 1 & timestamp >= "2016-05-09 18")')
train_df = train_df.query(
    'not (building_id == 192 & meter == 3 & timestamp >= "2016-03-29 05" & timestamp <= "2016-04-04 08")')
train_df = train_df.query(
    'not (building_id == 207 & meter == 1 & timestamp > "2016-07-02 20" & timestamp < "2016-08-25 12")')
train_df = train_df.query(
    'not (building_id == 258 & timestamp > "2016-09-18" & timestamp < "2016-12-12 13")')
train_df = train_df.query(
    'not (building_id == 258 & timestamp > "2016-08-29 08" & timestamp < "2016-09-08 14")')
train_df = train_df.query(
    'not (building_id == 257 & meter == 1 & timestamp < "2016-03-25 16")')
train_df = train_df.query(
    'not (building_id == 260 & meter == 1 & timestamp > "2016-05-10 17" & timestamp < "2016-08-17 11")')
train_df = train_df.query(
    'not (building_id == 260 & meter == 1 & timestamp > "2016-08-28 01" & timestamp < "2016-10-31 13")')
train_df = train_df.query(
    'not (building_id == 220 & meter == 1 & timestamp > "2016-09-23 01" & timestamp < "2016-09-23 12")')
train_df = train_df.query(
    'not (building_id == 281 & meter == 1 & timestamp > "2016-10-25 08" & timestamp < "2016-11-04 15")')
train_df = train_df.query(
    'not (building_id == 273 & meter == 1 & timestamp > "2016-04-03 04" & timestamp < "2016-04-29 15")')
train_df = train_df.query(
    'not (building_id == 28 & meter == 0 & timestamp < "2016-10-14 20")')
train_df = train_df.query(
    'not (building_id == 71 & meter == 0 & timestamp < "2016-08-18 20")')
train_df = train_df.query(
    'not (building_id == 76 & meter == 0 & timestamp > "2016-06-04 09" & timestamp < "2016-06-04 14")')
train_df = train_df.query(
    'not (building_id == 101 & meter == 0 & timestamp > "2016-10-12 13" & timestamp < "2016-10-12 18")')
train_df = train_df.query(
    'not (building_id == 7 & meter == 1 & timestamp > "2016-11-03 09" & timestamp < "2016-11-28 14")')
train_df = train_df.query(
    'not (building_id == 9 & meter == 1 & timestamp > "2016-12-06 08")')
train_df = train_df.query(
    'not (building_id == 43 & meter == 1 & timestamp > "2016-04-03 08" & timestamp < "2016-06-06 13")')
train_df = train_df.query(
    'not (building_id == 60 & meter == 1 & timestamp > "2016-05-01 17" & timestamp < "2016-05-01 21")')
train_df = train_df.query(
    'not (building_id == 75 & meter == 1 & timestamp > "2016-08-05 13" & timestamp < "2016-08-26 12")')
train_df = train_df.query(
    'not (building_id == 95 & meter == 1 & timestamp > "2016-08-08 10" & timestamp < "2016-08-26 13")')
train_df = train_df.query(
    'not (building_id == 97 & meter == 1 & timestamp > "2016-08-08 14" & timestamp < "2016-08-25 14")')
train_df = train_df.query(
    'not (building_id == 1232 & meter == 1 & timestamp > "2016-06-23 16" & timestamp < "2016-08-31 20")')
train_df = train_df.query(
    'not (building_id == 1236 & meter == 1 & meter_reading >= 3000)')
train_df = train_df.query(
    'not (building_id == 1239 & meter == 1 & timestamp > "2016-03-11 16" & timestamp < "2016-03-27 17")')
train_df = train_df.query(
    'not (building_id == 1264 & meter == 1 & timestamp > "2016-08-22 17" & timestamp < "2016-09-22 20")')
train_df = train_df.query(
    'not (building_id == 1264 & meter == 1 & timestamp > "2016-09-28 07" & timestamp < "2016-10-20 13")')
train_df = train_df.query(
    'not (building_id == 1269 & meter == 1 & meter_reading >= 2000)')
train_df = train_df.query(
    'not (building_id == 1272 & meter == 1 & timestamp > "2016-08-11 12" & timestamp < "2016-08-30 19")')
train_df = train_df.query(
    'not (building_id == 1273 & meter == 1 & timestamp > "2016-05-31 14" & timestamp < "2016-06-17")')
train_df = train_df.query(
    'not (building_id == 1276 & meter == 1 & timestamp < "2016-02-03 23")')
train_df = train_df.query(
    'not (building_id == 1280 & meter == 1 & timestamp > "2016-05-18" & timestamp < "2016-05-26 09")')
train_df = train_df.query(
    'not (building_id == 1280 & meter == 1 & timestamp > "2016-02-28 23" & timestamp < "2016-05-02 05")')
train_df = train_df.query(
    'not (building_id == 1280 & meter == 1 & timestamp > "2016-06-12 01" & timestamp < "2016-7-07 06")')
train_df = train_df.query(
    'not (building_id == 1288 & meter == 1 & timestamp > "2016-07-07 15" & timestamp < "2016-08-12 17")')
train_df = train_df.query(
    'not (building_id == 1311 & meter == 1 & timestamp > "2016-04-25 18" & timestamp < "2016-05-13 14")')
train_df = train_df.query('not (building_id == 1099 & meter == 2)')

train_df = train_df.query(
    'not (building_id == 1329 & meter == 0 & timestamp > "2016-04-28 00" & timestamp < "2016-04-28 07")')
train_df = train_df.query(
    'not (building_id == 1331 & meter == 0 & timestamp > "2016-04-28 00" & timestamp < "2016-04-28 07")')
train_df = train_df.query(
    'not (building_id == 1427 & meter == 0 & timestamp > "2016-04-11 10" & timestamp < "2016-04-11 14")')
train_df = train_df.query(
    'not (building_id == 1426 & meter == 2 & timestamp > "2016-05-03 09" & timestamp < "2016-05-03 14")')
train_df = train_df.query(
    'not (building_id == 1345 & meter == 0 & timestamp < "2016-03-01")')
train_df = train_df.query(
    'not (building_id == 1346 & timestamp < "2016-03-01")')
train_df = train_df.query(
    'not (building_id == 1359 & meter == 0 & timestamp > "2016-04-25 17" & timestamp < "2016-07-22 14")')
train_df = train_df.query(
    'not (building_id == 1365 & meter == 0 & timestamp > "2016-08-19 00" & timestamp < "2016-08-19 07")')
train_df = train_df.query(
    'not (building_id == 1365 & meter == 0 & timestamp > "2016-06-18 22" & timestamp < "2016-06-19 06")')

train_df = train_df.query(
    'not (building_id == 18 & meter == 0 & timestamp > "2016-06-04 09" & timestamp < "2016-06-04 16")')
train_df = train_df.query(
    'not (building_id == 18 & meter == 0 & timestamp > "2016-11-05 05" & timestamp < "2016-11-05 15")')
train_df = train_df.query(
    'not (building_id == 101 & meter == 0 & meter_reading > 800)')

train_df = train_df.query(
    'not (building_id == 1384 & meter == 0 & meter_reading == 0 )')
train_df = train_df.query(
    'not (building_id >= 1289 & building_id <= 1301 & meter == 2 & meter_reading == 0)')
train_df = train_df.query(
    'not (building_id == 1243 & meter == 2 & meter_reading == 0)')
train_df = train_df.query(
    'not (building_id == 1263 & meter == 2 & meter_reading == 0)')
train_df = train_df.query(
    'not (building_id == 1284 & meter == 2 & meter_reading == 0)')
train_df = train_df.query(
    'not (building_id == 1286 & meter == 2 & meter_reading == 0)')
train_df = train_df.query(
    'not (building_id == 1263 & meter == 0 & timestamp > "2016-11-10 11" & timestamp < "2016-11-10 15")')

train_df = train_df.query(
    'not (building_id == 1238 & meter == 2 & meter_reading == 0)')
train_df = train_df.query(
    'not (building_id == 1329 & meter == 2 & timestamp > "2016-11-21 12" & timestamp < "2016-11-29 12")')
train_df = train_df.query(
    'not (building_id == 1249 & meter == 2 & meter_reading == 0)')

train_df = train_df.query(
    'not (building_id == 1250 & meter == 2 & meter_reading == 0)')
train_df = train_df.query(
    'not (building_id == 1256 & meter == 2 & timestamp > "2016-03-05 18" & timestamp < "2016-03-05 22")')
train_df = train_df.query(
    'not (building_id == 1256 & meter == 2 & timestamp > "2016-03-27 00" & timestamp < "2016-03-27 23")')
train_df = train_df.query(
    'not (building_id == 1256 & meter == 2 & timestamp > "2016-04-11 09" & timestamp < "2016-04-13 03")')
train_df = train_df.query(
    'not (building_id == 1256 & meter == 2 & timestamp > "2016-04-29 00" & timestamp < "2016-04-30 15")')
train_df = train_df.query(
    'not (building_id == 1303 & meter == 2 & timestamp < "2016-06-06 19")')
train_df = train_df.query(
    'not (building_id >= 1223 & building_id <= 1324 & meter == 1 & timestamp > "2016-08-11 17" & timestamp < "2016-08-12 17")')
train_df = train_df.query(
    'not (building_id >= 1223 & building_id <= 1324 & building_id != 1296 & building_id != 129 & building_id != 1298 & building_id != 1299 & meter == 2 & timestamp > "2016-08-11 17" & timestamp < "2016-08-12 17")')
train_df = train_df.query(
    'not (building_id >= 1223 & building_id <= 1324 & meter == 3 & timestamp > "2016-08-11 17" & timestamp < "2016-08-12 17")')

train_df = train_df.reset_index()

print('after', len(train_df))

gc.collect()


# %% [code]
print('Site 0 buildings: Correcting meter readings for meter 0 .....')
# https://www.kaggle.com/c/ashrae-energy-prediction/discussion/119261#latest-684102
site_0_bids = building_meta_df[building_meta_df.site_id ==
                               0].building_id.unique()

print(len(site_0_bids), len(
    train_df[train_df.building_id.isin(site_0_bids)].building_id.unique()))
print("Before correction:")
print(train_df[(train_df.building_id.isin(site_0_bids))
               & (train_df.meter == 0)].head(10))

train_df.loc[(train_df.building_id.isin(site_0_bids)) & (train_df.meter == 0), 'meter_reading'] = train_df[(
    train_df.building_id.isin(site_0_bids)) & (train_df.meter == 0)]['meter_reading'] * 0.2931
# %% [code]
print("After correction:")

print(train_df[(train_df.building_id.isin(site_0_bids))
               & (train_df.meter == 0)].head(10))

# %% [code]
# Add timestamp features and log transform of meter_reading
print("Addition of timestamp features...")

train_df['date'] = train_df['timestamp'].dt.date

print("transforming meter_reading...")
train_df['meter_reading_log1p'] = np.log1p(train_df['meter_reading'])

# %% [Markdown]
# Add time feature

"""Some features introduced in https://www.kaggle.com/ryches/simple-lgbm-solution by @ryches
Features that are likely predictive:"""

# Buildings:-

# primary_use,
# square_feet,
# year_built,
# floor_count (may be too sparse to use),
# Weather,

# time of day :-
# holiday,
# weekend,
# cloud_coverage + lags,
# dew_temperature + lags,
# precip_depth + lags,
# sea_level_pressure + lags,
# wind_direction + lags,
# wind_speed + lags,

# Train:-

# max, mean, min, std of the specific building historically,
# number of meters,
# number of buildings at a siteid.

# %% [code]
# Preprocessing time features
print("Preprocessing time features...")
preprocess(train_df)

# %% [code]
# Sort Train Data
print("Sorting Training Data...")
if use_ucf:
    train_df = train_df.sort_values('month')
    train_df = train_df.reset_index(drop=True)
# %% [code]
# Fill Nan value in weather dataframe by interpolation
print("Fill Nan value in weather dataframe by interpolation...")
weather_train_df.head()

# %% [code]
weather_train_df.describe()

# %% [code]
print("Before interpolation:")
print(weather_train_df.isna().sum())

# %% [code]
print("Shape before interpolation:")
print(weather_train_df.shape)

# %% [code]
print("Group by site_id Missing values before interpolation:")
print(weather_train_df.groupby('site_id').apply(
    lambda group: group.isna().sum()))

# %% [code]
print("Interpolate missing values...")
weather_train_df = weather_train_df.groupby('site_id').apply(
    lambda group: group.interpolate(method='ffill', limit_direction='forward'))

# %% [code]
print("After interpolation:")
print(weather_train_df.groupby('site_id').apply(
    lambda group: group.isna().sum()))

# %% [code]
# Checking how many data points in each site_id
print("Checking how many data points in each site_id...")
print(weather_train_df.groupby('site_id').apply(lambda group: group.shape))

# %% [code]
print("Adding lag features for weather data...")

add_lag_feature(weather_train_df, window=3)
add_lag_feature(weather_train_df, window=72)

# %% [code]
weather_train_df.head()

# %% [code]
weather_train_df.columns

# %% [code]
# count encoding
print("Count encoding...")
year_map = building_meta_df.year_built.value_counts()
building_meta_df['year_cnt'] = building_meta_df.year_built.map(year_map)

bid_map = train_df.building_id.value_counts()
train_df['bid_cnt'] = train_df.building_id.map(bid_map)
# %% [code]
# categorize primary_use column to reduce memory on merge...
print("Categorizing primary_use column to reduce memory on merge...")

primary_use_list = building_meta_df['primary_use'].unique()
primary_use_dict = {key: value for value, key in enumerate(primary_use_list)}
print('primary_use_dict: ', primary_use_dict)
building_meta_df['primary_use'] = building_meta_df['primary_use'].map(
    primary_use_dict)

gc.collect()

# %% [code]
# Reduce memory usage
print("Reducing memory usage...")
train_df = reduce_mem_usage(train_df, use_float16=True)
building_meta_df = reduce_mem_usage(building_meta_df, use_float16=True)
weather_train_df = reduce_mem_usage(weather_train_df, use_float16=True)

# %% [code]
print("Smoothing weather data...")

add_sg(weather_train_df)

# %% [Markdown]
# # Plotting smoothed data

# %% [code]
print("Air temperature vs smoothed air temperature for site_id 1")
weather_train_df[weather_train_df.site_id == 1].air_temperature[:100].plot()
weather_train_df[weather_train_df.site_id == 1].air_smooth[:100].plot()
# Show the plots in the console
plt.show()


# %% [code]
print("Dew temperature vs smoothed dew temperature for site_id 2")
weather_train_df[weather_train_df.site_id == 2].dew_temperature[:100].plot()
weather_train_df[weather_train_df.site_id == 2].dew_smooth[:100].plot()
plt.show()

# %% [code]
print("Difference between air temperature and smoothed air temperature for site_id 0")
weather_train_df[weather_train_df.site_id == 0].dew_diff[:100].plot()
weather_train_df[weather_train_df.site_id == 0].air_diff[:100].plot()
plt.show()

# %% [code]
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

# %% [code]
# Colums in train_df
train_df.columns
# %% [code]
# Colums in building_meta_df
building_meta_df.columns
# %% [code]

print("Merging dataframes...")

train_df = train_df.merge(building_meta_df, on=[
                          'building_id', 'meter'], how='left')
# %% [code]
train_df = reduce_mem_usage(train_df, use_float16=True)
weather_train_df = reduce_mem_usage(weather_train_df, use_float16=True)
# %% [code]
train_df = train_df.merge(weather_train_df, on=[
                          'site_id', 'timestamp'], how='left')

# %% [code]
print("Transforming Square Feet to log1p...")
train_df['square_feet_np_log1p'] = np.log1p(train_df['square_feet'])

# %% [code]
print("Reducing memory usage...")
train_df = reduce_mem_usage(train_df, use_float16=True)
gc.collect()  # Copy till here for data preprocessing.

# %% [code]
# Delete unnecessary dataframes to free up memory
# del building_meta_df
# del weather_train_df
# gc.collect()

# %% [code]
# Save preprocessed data
print("Saving preprocessed train data...")
train_df.to_pickle(
    '/workspace/Ashrae-Energy-Prediction-III/src/data/train_df.pkl')
weather_train_df.to_pickle(
    '/workspace/Ashrae-Energy-Prediction-III/src/data/weather_train_df.pkl')


# %% [code]
# Model Implementation and Training
print("Model Implementation and Training...")

# %% [code]
# Create X_train and y_train


def create_X_y(train_df, groupNum_train):

    target_train_df = train_df[train_df['groupNum_train']
                               == groupNum_train].copy()

    X_train = target_train_df[feature_cols + category_cols]
    y_train = target_train_df['meter_reading_log1p'].values

    del target_train_df
    return X_train, y_train

# %% [code]

# Define modelling parameters and training model


def fit_lgbm(train, val, devices=(-1,), seed=None, cat_features=None, num_rounds=1500, lr=0.1, bf=0.1):
    """Train Light GBM model"""
    X_train, y_train = train
    X_valid, y_valid = val
    metric = 'rmse'
    params = {'num_leaves': 31,
              'objective': 'regression',
              'max_depth': -1,
              'learning_rate': lr,
              "boosting": "gbdt",
              "bagging_freq": 5,
              "bagging_fraction": bf,
              "feature_fraction": 0.9,
              "metric": metric,
              "verbosity": -1,
              'reg_alpha': 0.1,
              'reg_lambda': 0.3
              }
    device = devices[0]
    if device == -1:
        # use cpu
        pass
    else:
        # use gpu
        print(f'using gpu device_id {device}...')
        params.update({'device': 'gpu', 'gpu_device_id': device})

    params['seed'] = seed

    early_stop = 20
    verbose_eval = 50

    d_train = lgb.Dataset(X_train, label=y_train,
                          categorical_feature=cat_features)
    d_valid = lgb.Dataset(X_valid, label=y_valid,
                          categorical_feature=cat_features)
    watchlist = [d_train, d_valid]

    print('training LGB:')
    model = lgb.train(params,
                      train_set=d_train,
                      num_boost_round=num_rounds,
                      valid_sets=watchlist,
                      verbose_eval=verbose_eval,
                      early_stopping_rounds=early_stop)

    # predictions
    y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)

    print('best_score', model.best_score)
    log = {'train/rmse': model.best_score['training']['rmse'],
           'valid/rmse': model.best_score['valid_1']['rmse']}
    return model, y_pred_valid, log


# %% [code]
# Stratified K-flod Cross Validation with shuffling
seed = 21099797
shuffle = True  # Try with shuffling as well and check accuracy
kf = StratifiedKFold(n_splits=folds)

# %% [markdown]
# Train model by each group # (site-meter)

# %% [code]

# plot feature importance


def plot_feature_importance(model, feature_cols):
    """Plot feature importance"""
    feature_importance_df = pd.DataFrame()
    feature_importance_df['feature'] = feature_cols
    feature_importance_df['importance'] = model.feature_importance()
    feature_importance_df = feature_importance_df.sort_values(
        'importance', ascending=False).reset_index(drop=True)
    plt.figure(figsize=(8, 16))
    sns.barplot(data=feature_importance_df, x='importance', y='feature')
    plt.title('Feature importance (averaged/folds)')
    plt.tight_layout()
    plt.show()


if Train:
    for groupNum_train in building_meta_df['groupNum_train'].unique():
        X_train, y_train = create_X_y(train_df, groupNum_train=groupNum_train)
        y_valid_pred_total = np.zeros(X_train.shape[0])

        gc.collect()
        print('groupNum_train', groupNum_train, X_train.shape)

        cat_features = [X_train.columns.get_loc(
            cat_col) for cat_col in category_cols]
        print('cat_features', cat_features)

        exec('models' + str(groupNum_train) + '=[]')

        train_df_site = train_df[train_df['groupNum_train']
                                 == groupNum_train].copy()

        for train_idx, valid_idx in kf.split(train_df_site, train_df_site['building_id']):
            train_data = X_train.iloc[train_idx, :], y_train[train_idx]
            valid_data = X_train.iloc[valid_idx, :], y_train[valid_idx]

            mindex = train_df_site.iloc[valid_idx, :].month.unique()
            print(mindex)

            print('train', len(train_idx), 'valid', len(valid_idx))

            model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols,
                                                num_rounds=num_rounds, lr=0.05, bf=0.7)
            y_valid_pred_total[valid_idx] = y_pred_valid

            exec('models' + str(groupNum_train) + '.append([mindex, model])')
            gc.collect()
            if debug:
                break

        try:
            sns.distplot(y_train)
            sns.distplot(y_valid_pred_total)
            plt.show()
        except:
            pass

        # Feature importance
        plot_feature_importance(model, X_train.columns)

        del X_train, y_train
        gc.collect()

        print('-------------------------------------------------------------')

else:
    print('Skip Training')

# %% [code]

# Preprocessing on test data

print("Preprocessing test data...")

print('loading...')
test_df = pd.read_feather(root/'test.feather')
weather_test_df = pd.read_feather(root/'weather_test.feather')

print('Before Preprocessing ....')
print('Shape of test data: ', test_df.shape)
print('Shape of Weather test data: ', weather_test_df.shape)

weather_test_df = weather_test_df.drop_duplicates(['timestamp', 'site_id'])
set_local(weather_test_df)
add_holiyday(weather_test_df)

print('preprocessing timestamp...')
test_df['date'] = test_df['timestamp'].dt.date
preprocess(test_df)


print('preprocessing weather...')
weather_test_df = weather_test_df.groupby('site_id').apply(
    lambda group: group.interpolate(method='ffill', limit_direction='forward'))
weather_test_df.groupby('site_id').apply(lambda group: group.isna().sum())

add_sg(weather_test_df)

add_lag_feature(weather_test_df, window=3)
add_lag_feature(weather_test_df, window=72)

test_df['bid_cnt'] = test_df.building_id.map(bid_map)

test_df = test_df.merge(building_meta_df[['building_id', 'meter', 'groupNum_train', 'square_feet']], on=[
                        'building_id', 'meter'], how='left')

test_df['square_feet_np_log1p'] = np.log1p(test_df['square_feet'])

print('reduce mem usage...')
test_df = reduce_mem_usage(test_df, use_float16=True)
weather_test_df = reduce_mem_usage(weather_test_df, use_float16=True)

gc.collect()

# %% [code]
# Save preprocessing test data in pickle format
print("Saving test data...")
test_df.to_pickle(
    '/workspace/Ashrae-Energy-Prediction-III/src/data/test_df.pkl')
weather_test_df.to_pickle('/workspace/Ashrae-Energy-Prediction-III/src/data/weather_test_df.pkl')
building_meta_df.to_pickle('/workspace/Ashrae-Energy-Prediction-III/src/data/building_meta_df.pkl')
gc.collect()


# %% [code]

if not debug:
    print('Reading Sample Submission...')
    sample_submission = pd.read_feather(
        os.path.join(root, 'sample_submission.feather'))
    reduce_mem_usage(sample_submission)

    print(sample_submission.shape)

# %% [code]
# Create X_test and y_test

print("Creating X_test and y_test...")


def create_X(test_df, groupNum_train):

    target_test_df = test_df[test_df['groupNum_train']
                             == groupNum_train].copy()
    target_test_df = target_test_df.merge(
        building_meta_df, on=['building_id', 'meter', 'groupNum_train', 'square_feet'], how='left')
    target_test_df = target_test_df.merge(
        weather_test_df, on=['site_id', 'timestamp'], how='left')
    X_test = target_test_df[feature_cols + category_cols]

    return X_test

# %% [code]
# Run a loop to merge data groupNum_train wise and define model prediction.


def pred_all(X_test, models, batch_size=1000000):
    iterations = (X_test.shape[0] + batch_size - 1) // batch_size
    print('iterations', iterations)

    y_test_pred_total = np.zeros(X_test.shape[0])
    for i, (mindex, model) in enumerate(models):
        print(f'predicting {i}-th model')
        for k in tqdm(range(iterations)):
            y_pred_test = model.predict(
                X_test[k*batch_size:(k+1)*batch_size], num_iteration=model.best_iteration)
            y_test_pred_total[k*batch_size:(k+1)*batch_size] += y_pred_test

    y_test_pred_total /= len(models)
    return y_test_pred_total

# define model prediction for train data


def pred(X_test, models, batch_size=1000000):
    if predmode == 'valid':
        print('valid pred')
        return pred_valid(X_test, models, batch_size=1000000)
    elif predmode == 'train':
        print('train pred')
        return pred_train(X_test, models, batch_size=1000000)
    else:
        print('all pred')
        return pred_all(X_test, models, batch_size=1000000)


# %% [code]
# Check columns of test data
print('test_df columns: ', test_df.columns)
print('weather_test_df columns: ', weather_test_df.columns)
print('Building_meta_df columns: ', building_meta_df.columns)


# %% [code]
# Call Model prediction
if Train:
    for groupNum_train in building_meta_df['groupNum_train'].unique():
        print('groupNum_train: ', groupNum_train)
        X_test = create_X(test_df, groupNum_train=groupNum_train)
        gc.collect()

        exec('y_test= pred(X_test, models' + str(groupNum_train) + ')')

        sns.distplot(y_test)
        plt.show()

        print(X_test.shape, y_test.shape)
        sample_submission.loc[test_df["groupNum_train"] ==
                              groupNum_train, "meter_reading"] = np.expm1(y_test)

        del X_test, y_test
        gc.collect()
else:
    print('Skip prediction')

if not debug:
    # %% [markdown]
    # site-0 correction

    # %% [code]
    # https://www.kaggle.com/c/ashrae-energy-prediction/discussion/119261#latest-684102
    sample_submission.loc[(test_df.building_id.isin(site_0_bids)) & (test_df.meter == 0), 'meter_reading'] = sample_submission[(
        test_df.building_id.isin(site_0_bids)) & (test_df.meter == 0)]['meter_reading'] * 3.4118

    # %% [code]
    sample_submission.head()

    # %% [code]
    sample_submission.tail()

    # %% [code]
    print('Shape of Sample Submission', sample_submission.shape)
else:
    print('Debug Mode')

# %% [code]
if not debug:
    sample_submission.to_csv(
        'k_fold_GBM_test_Submission.csv', index=False, float_format='%.4f')

# %% [code]
# Histogram of submission file.
if not debug:
    np.log1p(sample_submission['meter_reading']).hist(bins=100)

# Run the script to generate submission file.
# python3 K-fold_LightGBM.py

print("Script Executed Sucessfully...")