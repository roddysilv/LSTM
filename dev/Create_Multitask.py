# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 15:58:55 2021

@author: rodri
"""

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

# path = '../LSTM/data/daily'

path = '../LSTM/data/separated_by_house_type/all'

h1 = path + '/Residential_6_treated.csv'

h2 = path + '/Residential_12_treated.csv'

# h3 = path + '/Residential_16_daily_sum.csv'

df1 = pd.read_csv(h1)

df2 = pd.read_csv(h2)

# df3 = pd.read_csv(h3)

df = pd.merge(df1,
              df2,
              on = ['date',
                    'hour',
                    'temperature',
                    'humidity',
                    'pressure',
                    'dc_output',
                    'ac_output',
                    'weather'
                    ])

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

mx=np.mean(df['energy_kWh_x'])

my=np.mean(df['energy_kWh_y'])

df['energy_kWh_x'] = df['energy_kWh_x'] / mx

df['energy_kWh_y'] = df['energy_kWh_y'] / my

df[['energy_kWh_x','energy_kWh_y']][:].plot()

# plt.plot(df['energy_kWh_x']-df['energy_kWh_y'])

# df.to_csv('multitask.csv',index=False)