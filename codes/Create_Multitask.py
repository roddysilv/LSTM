# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 15:58:55 2021

@author: rodrigo
"""

'''
Casas no msm range:
    2015 - 2018:
        3, 4, 5, 6, 7, 8, 9, 10, 
        11, 12, 13, 14, 15, 18, 
        19, 20
    
'''
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

# path = '../LSTM/data/daily'

c1 = 5

c2 = 14

path = 'csv_merged/Horário/Residential_'

h1 = path + str(c1) + '.csv'

h2 = path + str(c2) + '.csv'

# h3 = path + '/Residential_16_daily_sum.csv'

df1 = pd.read_csv(h1)
df1 = df1.rename(columns={'energy_kWh':'energy_kWh_' + str(c1)})

df2 = pd.read_csv(h2)
df2 = df2.rename(columns={'energy_kWh':'energy_kWh_' + str(c2)})

# df3 = pd.read_csv(h3)

cols = list(df1.columns)

df = pd.merge(df1,
              df2,
              on = cols[0:-1])

# df = pd.merge(df,
#               df3,
#               on = ['date',
#                     'hour',
#                     'temperature',
#                     'humidity',
#                     'pressure',
#                     'dc_output',
#                     'ac_output',
#                     'weather'])

df.index = df.date
df.pop('date')
# mx=np.mean(df['energy_kWh_x'])

# my=np.mean(df['energy_kWh_y'])

# df['energy_kWh_x'] = df['energy_kWh_x'] / mx

# df['energy_kWh_y'] = df['energy_kWh_y'] / my

cols = list(df.columns)

# df[[cols[-2],cols[-1]]][:].plot()

# plt.plot(df['energy_kWh_x']-df['energy_kWh_y'])

# print(df.corr())

df.to_csv('multitask.csv',index=True)