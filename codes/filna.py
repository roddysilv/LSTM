# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 10:55:00 2021

@author: rodri
"""

import pandas as pd

path = '../data/csv/'

for i in range(1,29):

    df = pd.read_csv(path + 'Residential_' + str(2) + '.csv')

    dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in df['date']])
    dates['hour']=df['hour']
    df['date'] = pd.to_datetime(dates)
    
    df.index = pd.to_datetime(df.date)
    
    # df.index = pd.to_datetime(df.hour)
    
    df1 = df.interpolate(method='time')
    
    df1.to_csv('../../Testes_' + str(i) + '.csv',index=False)
    # df1.energy_kWh.plot()
    
    # df.energy_kWh.plot()
