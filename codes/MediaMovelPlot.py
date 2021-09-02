# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 19:40:51 2021

@author: Rodrigo
"""

from scipy.signal import lfilter
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.ion()

def remove_ruido(df,n=15):
    
    y = df.values
    
    b = [1.0 / n] * n
    a = 1
    
    yy = lfilter(b,a,y)
    
    return yy

def media_movel(df,n=10):
    
    y = df.rolling(n).mean()
    
    # y = y.dropna()    
    
    return y
i=4
h=10
n = 150

df = pd.read_csv(f'csv_merged/Horário/Residential_{i}.csv',index_col='date')

df.pop('weather')

mm = media_movel(df,h)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,1,1)

plt.plot(df.index[:n],df.energy_kWh[:n],'k',label="Observado")
plt.plot(mm.index[:n],mm.energy_kWh[:n],'g',label=f"Média móvel {h}Hrs")

ax.xaxis.set_major_locator(mdates.HourLocator(interval=n*2))
fig.autofmt_xdate()

plt.xlabel("Período")
plt.ylabel("energia (kWh)")

plt.title(f"Comparação com média móvel casa {i}")
plt.legend()

plt.savefig(f"Resultados/Comparações/MediaMovel{h}-casa{i}-{n}.png",bbox_inches='tight',dpi=400)
plt.savefig(f"Resultados/Comparações/MediaMovel{h}-casa{i}-{n}.svg",bbox_inches='tight')