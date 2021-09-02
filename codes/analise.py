# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 12:05:52 2021

@author: Rodrigo
"""

import pandas as pd

from sklearn.decomposition import PCA

import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv('csv_merged/Houses_info.csv',decimal=',')

df['FirstReading'] = pd.to_datetime(df['FirstReading'])
df['LastReading'] = pd.to_datetime(df['LastReading'])

d = pd.get_dummies(df.HouseType)
df = pd.merge(df,d,left_index=True,right_index=True)
h = df.pop('HouseType')

d = pd.get_dummies(df.Facing)
df = pd.merge(df,d,left_index=True,right_index=True)
df.pop('Facing')

d = pd.get_dummies(df.Region)
df = pd.merge(df,d,left_index=True,right_index=True)
df.pop('Region')

df.to_csv('csv_merged/Houses_info_dummie.csv')

def PCA_1(df,h):
    r = df['House']
    D = df.drop(['House','FirstReading','LastReading'], axis=1)
    D = D.fillna(0)
    D.index=r
    
    pca = PCA(n_components=2)
    pca.fit(D)
    A = pca.fit_transform(D)
    
    df2 = pd.DataFrame(A)
    df2 = df2.rename(columns={0:'x',1:'y'})
    df2 = pd.concat([df, df2], axis=1)
    
    for x,s in zip(A,D.index): 
        plt.text(x=x[0], y=x[1], s=s, fontsize=10)  
        
    sns.scatterplot('x', 'y', data=df2, hue=h,s=100)
    plt.title("Variáveis Dummie")
    # plt.savefig('PCA.svg')
    plt.show()

def PCA_2(df,h):
        
    df = df.drop(17,axis=0)
    
    r = df['House']
    D = df.drop(['House','FirstReading','LastReading'], axis=1)
    D = D.fillna(0)
    D.index=r
    
    pca = PCA(n_components=2)
    pca.fit(D)
    A = pca.fit_transform(D)
    
    df2 = pd.DataFrame(A)
    df2 = df2.rename(columns={0:'x',1:'y'})
    df2 = pd.concat([df, df2], axis=1)
    
    for x,s in zip(A,D.index): 
        plt.text(x=x[0], y=x[1], s=s, fontsize=10)  
        
    sns.scatterplot('x', 'y', data=df2, hue=h,s=100)
    plt.title("Variáveis Dummie")
    # plt.savefig('PCA.svg')
    plt.show()
    
# PCA_1(df,h)
PCA_2(df,h)