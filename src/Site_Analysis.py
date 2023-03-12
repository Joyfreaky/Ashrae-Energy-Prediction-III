#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px

import os
from pathlib import Path
# %%
# merge weather_train & weather_test
root= Path('/home/joydipb/Documents/CMT307-Coursework-2-Group-19')
weather_train = pd.read_csv(root/'weather_train.csv')
weather_test = pd.read_csv(root/'weather_test.csv')
weather_merged = weather_train.append(weather_test)
weather_merged['timestamp'] = pd.to_datetime(weather_merged['timestamp'])
weather_merged.set_index('timestamp', inplace=True)

weather_merged
# %%
#extract temperature from weather data
df_temperature_pivot = weather_merged.reset_index().pivot_table(index='timestamp', columns='site_id', values='air_temperature')
df_temperature_pivot.columns = 'site_'+df_temperature_pivot.columns.astype('str')

df_temperature_pivot

# %%
#load external temperature data
temperature_external = pd.read_csv(root/"temperature.csv")
temperature_external['datetime'] = pd.to_datetime(temperature_external['datetime'])
temperature_external.set_index('datetime', inplace=True)
temperature_external = temperature_external-273.15
temperature_external = temperature_external.merge(df_temperature_pivot, left_index=True, right_index=True, how='inner')
temperature_external = temperature_external.dropna()

temperature_external
# %%
#calculate correlations between sites
df_corr = temperature_external.corr(method='spearman')
list_site = df_temperature_pivot.columns
df_corr = df_corr[list_site]
df_corr = df_corr.drop(list_site)
df_corr

# %%
#sns heat map
fig, ax = plt.subplots(figsize=(30,15))   
sns.heatmap(df_corr, annot=True, cmap="YlGnBu", vmin=0.8, vmax=1.0)
# %%
#Get cities!
df_findCity = pd.concat([df_corr.idxmax(),df_corr.max()], axis=1).reset_index().rename(columns={'index':'site',0:'city',1:'corr'})
df_findCity
# %%
#compare sites & cities in plot
for city, site, corr in zip(df_findCity['city'],df_findCity['site'],df_findCity['corr']):
    if corr > 0.9:
        print('City: ' + city)
        print('Site: ' + site)   
        temperature_external[[city,site]].loc['2016'].plot(figsize=(15,8), alpha=0.5)
        plt.show()  
# %%
