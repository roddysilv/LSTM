import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

def solar():
    Solar = pd.read_csv('../LSTM/data/csv/Solar.csv')
    
    dates = pd.DataFrame([dict(zip(['year','month', 'day'], a.split('-'))) for a in Solar['date']])
    
    Solar = pd.concat([dates.drop(columns = ['year']), Solar.drop(columns=['date'])], axis=1)
    
    Solar = pd.concat([Solar.assign(year = np.repeat(i, len(Solar))) for i in range(2012,2021)]).reset_index(drop=True) 
    
    Solar['date'] = pd.to_datetime(Solar[['year','month','day','hour']])
    
    Solar = Solar.drop(['year','month','day','hour'],axis=1)
    
    return Solar

def weather():
    Weather_YVR = pd.read_csv('../LSTM/data/csv/Weather_YVR.csv')
    
    dates = pd.DataFrame([dict(zip(['year','month', 'day'], a.split('-'))) for a in Weather_YVR['date']])
    
    dates['hour'] = Weather_YVR['hour']
    
    Weather_YVR['date'] = pd.to_datetime(dates)
    
    Weather_YVR = Weather_YVR.drop(['hour'], axis=1)
    
    Weather_YVR = Weather_YVR.ffill()
    
    return Weather_YVR

def dataframe(path):
    df = pd.read_csv(path)
    
    df_s = solar()
    
    df_w = weather()
    
    dates = pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in df['date']])
    
    dates['hour'] = df['hour']
    
    df['date'] = pd.to_datetime(dates)

    dfm = pd.merge(df_s, df_w, on=['date'])

    dfm = pd.merge(dfm, df, on=['date'])

    dfm = dfm.set_index('date')

    dfm.pop('hour')
    
    cols = ['temperature','pressure', 'humidity', 'dc_output', 'ac_output','weather', 'energy_kWh']
    
    dfm = dfm[cols]
    
    return dfm

def daily(path):
    
    df = dataframe(path)

    daily_df = df.resample('D').sum()

    daily_df['weekday'] = daily_df.index.day_name()
    
    cols = ['temperature','pressure', 'humidity', 'dc_output', 'ac_output','weekday', 'energy_kWh']
    
    daily_df = daily_df[cols]
    
    return daily_df

def plots(df,i,tipo):    
    
    if tipo == 'd': path = 'Diário'
    else: path = 'Horário'
    
    plt.figure(figsize=(10,5))
    
    cols_plot = ['energy_kWh', 'temperature', 'dc_output','ac_output','pressure','humidity']
    
    df[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True,title = 'Residential ' + str(i))
    
    plt.savefig('Graficos/' + path + '/Residential_' + str(i) + '.svg',bbox_inches='tight')
    plt.savefig('Graficos/' + path + '/Residential_' + str(i) + '.png',bbox_inches='tight')
    
    plt.figure(figsize=(10,5))
    
    sns.boxplot(df.index.month,df.energy_kWh)
    plt.xlabel('Month')
    plt.title('Residential ' + str(i))
    
    plt.savefig('Graficos/' + path + '/Residential_' + str(i) + '_month.svg',bbox_inches='tight')
    plt.savefig('Graficos/' + path + '/Residential_' + str(i) + '_month.png',bbox_inches='tight')
    
    plt.figure(figsize=(10,5))
    
    sns.boxplot(df.index.weekday,df.energy_kWh)
    plt.xlabel('Day of Week')
    plt.title('Residential ' + str(i))
    
    plt.savefig('Graficos/' + path + '/Residential_' + str(i) + '_weekday.svg',bbox_inches='tight')
    plt.savefig('Graficos/' + path + '/Residential_' + str(i) + '_weekday.png',bbox_inches='tight')
    
    df['moving_average_7d'] = df['energy_kWh'].rolling(7, center=True).mean()
    
    plt.figure(figsize=(10,5))
    
    df['energy_kWh'].plot(marker='.', alpha=0.5, linestyle='None', figsize=(10, 5), subplots=False,title = 'Residential ' + str(i),legend=True)
    
    df['moving_average_7d'].plot(legend=True)
    
    plt.savefig('Graficos/' + path + '/Residential_' + str(i) + '_rollingAVR.svg',bbox_inches='tight')
    plt.savefig('Graficos/' + path + '/Residential_' + str(i) + '_rollingAVR.png',bbox_inches='tight')
    
path = '../LSTM/data/csv/Residential_'

for i in range(1,29):
    
    df = dataframe(path + str(i) + '.csv')
       
    df_daily = daily(path + str(i) + '.csv')

    plots(df,i,'h')

    plots(df_daily,i,'d')
    
    df.pop('moving_average_7d')
    
    df.to_csv('csv_merged/Horário/Residential_' + str(i) + '.csv')
    
    df_daily.pop('moving_average_7d')
    
    df_daily.to_csv('csv_merged/Diário/Residential_' + str(i) + '.csv')

    d_h = pd.get_dummies(df.weather)
    df_d = pd.merge(d_h, df, on=['date'])
    df_d.pop('weather')
    df_d.to_csv('csv_merged/Horário/Residential_' + str(i) + '_dummie.csv')

    d_d = pd.get_dummies(df_daily.weekday)
    df_daily_d = pd.merge(d_d, df_daily, on=['date'])
    df_daily_d.pop('weekday')
    df_daily_d.to_csv('csv_merged/Diário/Residential_' + str(i) + '_dummie.csv')
    
y = df_daily['energy_kWh']
decomposition = sm.tsa.seasonal_decompose(y['2013'], model='additive')
decomposition.plot()
