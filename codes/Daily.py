# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:24:44 2021

@author: Rodrigo

https://www.youtube.com/watch?v=GPVsHOlRBBI

https://www.youtube.com/watch?v=r-uOLxNrNk8
"""

import pandas as pd
import matplotlib.pyplot as plt

def PlotDaily(arquivo,i,save=False):
    df = pd.read_csv(arquivo,header=0,infer_datetime_format=True,parse_dates=['date'],index_col=['date'])
    try:
        df.pop('weather')
        df.pop('hour')
    except:
        pass
    daily_df = df.resample('D').sum()
    
    if save:
        daily_df.to_csv('diario/Residential_' + str(i) + '_daily_sum.csv')
    
    plt.figure()
    plt.plot(daily_df['energy_kWh'])
    plt.title('Soma Diária de Energia - Casa '+str(i))
    plt.show()
    '''
    values = daily_df.values
    
    groups = [0,1,2,3]
    i = 1
    plt.figure()
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(daily_df.index,values[:, group])
        plt.title(daily_df.columns[group], y=0.5, loc='right')
        i += 1
    plt.show()
    '''
for i in range(1,29):
    PlotDaily('csv/all/Residential_'+str(i)+'_treated.csv',i)
    
# PlotDaily('csv/all/Residential_19_treated.csv')
df = pd.read_csv('csv/all/Residential_19_treated.csv',header=0,infer_datetime_format=True,parse_dates=['date'],index_col=['date'])
holidays = pd.read_csv('csv/all/Holidays.csv',header=0,infer_datetime_format=True,parse_dates=['date'],index_col=['date'])
daily_df = df.resample('D').sum()
data = pd.merge(holidays,daily_df,on=['date'])

dataframe = data.drop(columns=['pressure','dst','ac_output'])
hol=pd.get_dummies(dataframe['holiday'])
day = pd.get_dummies(dataframe['day'])

dataframe_dummie = pd.concat([day,hol,dataframe.drop(columns=['holiday','day'])],axis=1)
