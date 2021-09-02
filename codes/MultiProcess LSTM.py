import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import r2_score
from keras import regularizers
from keras.callbacks import EarlyStopping
import numpy as np
# from keras import optimizers
import pandas as pd
import time as t
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

import multiprocessing
np.random.seed(5)

plt.ioff()

def  plots(train_X, test_X, train_y, test_y ,  trainpredict, testPredict,nome):
    plt.ioff()
# =============================================================================
#     # Plot scatter Treino
#     plt.figure()
#     plt.plot(train_y,trainpredict,'o',label='trainpredict')
#     plt.plot(train_y,train_y,label='trainY')
#     plt.legend()
#     plt.title("Train - Scatter")
#     plt.xlabel("real")
#     plt.ylabel('predicted')
#     # plt.savefig('Resultados/scatterTreino' + str(res) + '.png')
#     plt.show()
# =============================================================================
    
    # Plot scatter Teste
    plt.figure(figsize=(10,5))
    plt.plot(test_y,testPredict,'go',markeredgecolor='k',label='Predicted')
    plt.plot(test_y,test_y,'k',label='Real')
    plt.title("Residential " + str(nome) + " - Scatter")
    plt.xlabel("real (kWh)")
    plt.ylabel('predicted (kWh)')
    plt.legend()
    plt.savefig('Resultados/scatterTest' + str(nome) + '.png',bbox_inches='tight')
    plt.savefig('Resultados/scatterTest' + str(nome) + '.svg',bbox_inches='tight')
    # plt.show()
    
# =============================================================================
#     # Plot Treino
#     plt.figure()
#     plt.plot(train_y,label='trainY')
#     plt.plot(trainpredict,'--',label='trainpredict')
#     plt.legend()
#     plt.title("Train - Plot")
#     # plt.savefig('Resultados/plotTreino' + str(res) + '.png')
#     plt.show()
# =============================================================================
    
    # Plot Teste
    plt.figure(figsize=(10,5))
    plt.plot( test_y,'k',label='Real')
    plt.plot( testPredict,'g--',label='Predicted')
    # plt.title("Test - plot")
    plt.title("Residential " + str(nome) + " - Plot")
    plt.xlabel("Period")
    plt.ylabel('energy (kWh)')
    plt.legend()
    plt.savefig('Resultados/plotTest' + str(nome) + '.png',bbox_inches='tight')
    plt.savefig('Resultados/plotTest' + str(nome) + '.svg',bbox_inches='tight')
    # plt.show()
    
     # Plot Teste
    plt.figure(figsize=(10,5))
    plt.plot( test_y[:150],'k',label='Real')
    plt.plot( testPredict[:150],'g--',label='Predicted')
    plt.xlabel("Period")
    plt.ylabel('energy (kWh)')
    plt.title("Residential " + str(nome) + " - Plot")
    # plt.title("Test - plot")
    plt.legend()
    plt.savefig('Resultados/plotTestShort150_' + str(nome) + '.png',bbox_inches='tight')
    plt.savefig('Resultados/plotTestShort150_' + str(nome) + '.svg',bbox_inches='tight')
    # plt.show()
    
    # Plot Teste
    plt.figure(figsize=(10,5))
    plt.plot( test_y[:50],'k',label='Real')
    plt.plot( testPredict[:50],'g--',label='Predicted')
    plt.xlabel("Period")
    plt.ylabel('energy (kWh)')
    # plt.title("Test - plot")
    plt.title("Residential " + str(nome) + " - Plot")
    plt.legend()
    plt.savefig('Resultados/plotTestShort50_' + str(nome) + '.png',bbox_inches='tight')
    plt.savefig('Resultados/plotTestShort50_' + str(nome) + '.svg',bbox_inches='tight')
    # plt.show()

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


def media_movel(df,n=10):
    
    y = df.rolling(n).mean()
    
    y = y.dropna()    
    
    return y.values        

def LSTM_RUN(df,nome):
    dates = df.index
    
    values = media_movel(df,5)
    
    values = values.astype('float32')
    
    # Normalizando os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    
    # Transformando em supervisionada
    reframed = series_to_supervised(scaled, 3, 1)

    # eliminando colunas que não queremos prever
    drop = [i for i in range(-2,-df.shape[1]-1,-1)]
    reframed.drop(reframed.columns[drop], axis=1, inplace=True)
    
    # dividindo em conjuntos de treinamento e teste
    values = reframed.values
    n_train_hours = int(values.shape[0]*0.7)
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    
    # dividindo em entradas e saídas
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    
    # remodelando a entrada para 3D [amostras, etapas de tempo, recursos]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    # design network
    batch_size=2**5
    epochs = 1000
    patience = 100
    
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
    
    
      
    #plt.plot(history.history['loss'], label='train')
    #plt.plot(history.history['val_loss'], label='test')
    #plt.legend()
    #plt.show()
    
    #Fazendo as previsões
    trainpredict = model.predict(train_X, batch_size=batch_size)
    model.reset_states()
    testPredict = model.predict(test_X, batch_size=batch_size)
    
    score = r2_score(test_y, testPredict)
    print('R2:', round(score,4))
    
    Predict = np.repeat(testPredict,df.shape[1],axis=-1)
    Real    = test_y.reshape(len(test_y),1)
    Real    = np.repeat(Real,     df.shape[1],axis=-1)
    normalPredict = scaler.inverse_transform(Predict)[:,-1]
    normalTest = scaler.inverse_transform(Real)[:,-1]
    
    #plots(train_X,test_X,train_y,normalTest,trainpredict,normalPredict,nome)

    # return train_X,test_X,train_y,normalTest,trainpredict,normalPredict,nome

    #print("iniciando novo teste....")


#for i in range(1,29):
 #   df = pd.read_csv('csv_merged/Horário/Residential_'+str(i)+'.csv',infer_datetime_format=True,index_col='date')

if __name__ == '__main__':
    df1 = pd.read_csv('csv_merged/Horário/Residential_'+str(1)+'.csv',infer_datetime_format=True,index_col='date')
    df2 = pd.read_csv('csv_merged/Horário/Residential_'+str(2)+'.csv',infer_datetime_format=True,index_col='date')

    p1 = multiprocessing.Process(target=LSTM_RUN,args=[df1,1])
    p2 = multiprocessing.Process(target=LSTM_RUN,args=[df2,2])

    p1.start()
    p2.start()

    p1.join()
    p2.join()