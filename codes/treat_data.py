# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 21:03:19 2021

@author: Rodrigo
"""
# =============================================================================
# Transforma dados faltantes em energy_kWh para 0 e troca todos os valores 0
# para a média da hora. 
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt

def media(aux):
    for i in range(aux.shape[0]):
        if(aux.iloc[i,2] == 0):
           if(i + 24 < aux.shape[0] and i - 24 >= 0):
               aux.iloc[i, 2] = (aux.iloc[i + 24, 2] + aux.iloc[i - 24, 2]) / 2
           else:
               aux.iloc[i,2] = (aux.iloc[i - 48, 2] + aux.iloc[i - 24, 2]) / 2

for i in range(1,29):   
    aux = pd.read_csv('../data/csv/Residential_' + str(i) + '.csv')
    aux = aux.fillna(0)
    media(aux)
    # print(aux.describe())
    aux.to_csv('../data/treated_data/Residential_' + str(i) + '.csv',index=False)
    
#%%
# =============================================================================
# Remove Dados faltantes do fim dos arquivos
# Plota gráfico do consumo e salva
# =============================================================================
for i in range(1,29):   
    aux = pd.read_csv('../data/csv/Residential_' + str(i) + '.csv')
    if(aux.isnull().iloc[aux.shape[0]-1,2]):
        aux2 = aux.isnull().iloc[aux.shape[0]-1,2]
        while (aux2):
            aux = aux.drop(aux.shape[0]-1)
            aux2 = aux.isnull().iloc[aux.shape[0]-1,2]
    aux = aux.fillna(0)
    media(aux)
    aux.to_csv('../data/treated_data/Residential_' + str(i) + '.csv',index=False)
    plt.figure(figsize=(15,10),dpi=200)
    plt.title('Casa ' + str(i) + ' - Tratada')
    aux['energy_kWh'].plot(color='k')
    plt.savefig('../data/plots/Residential_'+str(i)+'_treated.svg')
    plt.show()
    
#%%
for i in range(1,29):   
    aux = pd.read_csv('../data/csv/Residential_' + str(i) + '.csv')
    plt.figure(figsize=(15,10),dpi=200)
    aux['energy_kWh'].plot(color='k')
    plt.title('Casa ' + str(i) + ' - Não tratada')
    plt.savefig('../data/plots/Residential_'+str(i)+'.svg')
    plt.show()