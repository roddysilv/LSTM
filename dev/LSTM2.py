from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras import optimizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as t
import datetime as dt

np.random.seed(5)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
	    cols.append(df.shift(i))
	    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
	# drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def media_movel(DataFrame,n):
    
    y = DataFrame.rolling(n).mean()
    
    y = y.dropna()    
    
    return y.values

def read(DataFrame,n_in,n_out,n_r):
    
    # Carregando os dados  
    dates = DataFrame.index
        
    values = media_movel(DataFrame,n_r)
    
    values = values.astype('float32')
    
    # Normalizando os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    
    # Transformando em supervisionada
    reframed = series_to_supervised(scaled, n_in, n_out)
    
    # eliminando colunas que não queremos prever
    drop = [i for i in range(-2,-DataFrame.shape[1]-1,-1)]
    reframed.drop(reframed.columns[drop], axis=1, inplace=True)
        
    # dividindo em conjuntos de treinamento e teste
    values = reframed.values
    split = int(values.shape[0]*0.7)
    train = values[:split, :]
    test = values[split:, :]
    
    # dividindo em entradas e saídas
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
        
    # remodelando a entrada para 3D [amostras, etapas de tempo, recursos]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
         
    return train_X, test_X, train_y, test_y, dates[:split], dates[split:]

def plots(train_y,test_y,trainPredict,testPredict):
    # Plot scatter Treino
    plt.figure()
    plt.plot(train_y,trainPredict,'o',label='trainPredict')
    plt.plot(train_y,train_y,label='trainY')
    plt.legend()
    plt.title("Train - Scatter")
    plt.xlabel("real")
    plt.ylabel('predicted')
    # plt.savefig('Resultados/scatterTreino' + str(res) + '.png')
    plt.show()
    
    # Plot scatter Teste
    plt.figure()
    plt.plot(test_y,testPredict,'o',label='testPredict')
    plt.plot(test_y,test_y,label='testY')
    plt.title("Test - Scatter")
    plt.xlabel("real")
    plt.ylabel('predicted')
    plt.legend()
    # plt.savefig('Resultados/scatterTest' + str(res) + '.png')
    plt.show()
    
    # Plot Treino
    plt.figure()
    plt.plot(train_y,label='trainY')
    plt.plot(trainPredict,'--',label='trainPredict')
    plt.legend()
    plt.title("Train - Plot")
    # plt.savefig('Resultados/plotTreino' + str(res) + '.png')
    plt.show()
    
    # Plot Teste
    fig = plt.figure()
    plt.plot( test_y,label='testY')
    plt.plot( testPredict,'--',label='testPredict')
    plt.title("Test - plot")
    plt.legend()
    # plt.savefig('Resultados/plotTest' + str(res) + '.png')
    plt.show()
    
     # Plot Teste
    fig = plt.figure()
    plt.plot(test_y[:150],label='testY')
    plt.plot( testPredict[:150],'--',label='testPredict')
    # fig.autofmt_xdate()
    plt.title("Test - plot")
    plt.legend()
    # plt.savefig('Resultados/plotTestShort' + str(res) + '.png')
    plt.show()
    
    # Plot Teste
    fig = plt.figure()
    plt.plot(test_y[:50],label='testY')
    plt.plot( testPredict[:50],'--',label='testPredict')
    # fig.autofmt_xdate()
    plt.title("Test - plot")
    plt.legend()
    # plt.savefig('Resultados/plotTestShort' + str(res) + '.png')
    plt.show()

def LSTM_run(DataFrame,batch_size,epochs,patience,n_in,n_out,n_r):
           
    train_X, test_X, train_y, test_y , dates_train, dates_test= read(DataFrame,n_in,n_out,n_r)
    
    # design network
       
    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       verbose=1,
                       patience=patience)
             
    model = Sequential()
    
    model.add(LSTM(2**5,
                    input_shape=(train_X.shape[1], train_X.shape[2]),
                    activation='relu',
                    # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    kernel_regularizer=regularizers.l1(l1=1e-5),
                    bias_regularizer=regularizers.l2(1e-4),
                    activity_regularizer=regularizers.l2(1e-5),
                    return_sequences = True
                    ))
    
    model.add(Dropout(rate=0.03))
    
    model.add(LSTM(2**5,
                    # input_shape=(train_X.shape[1], train_X.shape[2]),
                    activation='linear',
                    # return_sequences = True
                    ))
    
    model.add(Dropout(rate=0.03))
    
    # model.add(LSTM(64,
    #                # input_shape=(train_X.shape[1], train_X.shape[2]),
    #                activation='linear',
    #                return_sequences = True))
    
    # model.add(Dropout(rate=0.03))
    
    # model.add(LSTM(32,
    #                # input_shape=(train_X.shape[1], train_X.shape[2]),
    #                activation='linear',
    #                return_sequences = True))
    
    # model.add(Dropout(rate=0.03))
    
    # model.add(LSTM(16,
    #                # input_shape=(train_X.shape[1], train_X.shape[2]),
    #                activation='linear',
    #                return_sequences = True))
    
    # model.add(Dropout(rate=0.03))
    
    # model.add(LSTM(4,
    #                # input_shape=(train_X.shape[1], train_X.shape[2]),
    #                activation='linear',
    #                return_sequences = True))
    
    # model.add(Dropout(rate=0.03))
    
    # model.add(LSTM(2**5,
    #                 # input_shape=(train_X.shape[1], train_X.shape[2]),
    #                 activation='linear',
    #                 kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    #                 bias_regularizer=regularizers.l2(1e-4),
    #                 activity_regularizer=regularizers.l2(1e-5),
    #                 ))
    
    # model.add(Dropout(rate=0.03))
    
    model.add(Dense(1,
                    activation='relu',
                    # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                    kernel_regularizer=regularizers.l1(l1=1e-5),
                    bias_regularizer=regularizers.l2(1e-4),
                    activity_regularizer=regularizers.l2(1e-5)
                   ))
    
    model.compile(loss='mse', optimizer='adam')
    
    # optimizers.Adam(lr=0.8,
    #                 beta_1=0.9,
    #                 beta_2=0.999,
    #                 epsilon=None,
    #                 decay=0.0,
    #                 amsgrad=False
    #                 )
    
    # fit network
    history = model.fit(train_X,
                        train_y,
                        epochs=epochs,
                        batch_size=batch_size, 
                        # validation_split=0.2,
                        validation_data=(test_X, test_y),
                        verbose=0,
                        shuffle=False,
                        callbacks=[es]
                        )

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    
    #Fazendo as previsões
    trainPredict = model.predict(train_X, batch_size=batch_size)
    model.reset_states()
    testPredict = model.predict(test_X, batch_size=batch_size)
    
    score = r2_score(test_y, testPredict)
    print('R2:', round(score,4))
      
    plots(train_y, test_y, trainPredict, testPredict)

    #return train_X, train_y, test_X, test_y, trainPredict, testPredict

path = '../../LSTM - DESENVOLVIMENTO/csv_merged/Horário/Residential_1.csv'

batch_size = 2**10
epochs = 1000
patience = 100
n_in = 3
n_out = 1
n_r = 5

df = pd.read_csv(path,infer_datetime_format=True,index_col='date')

tStart = t.time()

LSTM_run(df,batch_size,epochs,patience,n_in,n_out,n_r)

tEnd = t.time()

print("Total Time:", dt.timedelta(seconds = tEnd - tStart))