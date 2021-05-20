import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import LSTM
# from keras.layers import TimeDistributed
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import sys
import time
import datetime

numpy.random.seed(5)

#%%
#Convertendo o array em uma matriz de dados
def create_dataset(dataset, input_shape, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back),:input_shape]
		dataX.append(a)
		dataY.append(dataset[i + look_back, input_shape-1])
	return numpy.array(dataX), numpy.array(dataY)

#%%
#Carregando os dados
# dataframe = pd.read_csv('Residential_27_treated.csv', usecols=[7], engine='python')
dataframe = pd.read_csv('Residential_6_treated.csv',index_col=('date'), engine='python')
dataframe.boxplot(column=['energy_kWh'])

#%%
use_dummie = False

if use_dummie: 
    dummie = pd.get_dummies(dataframe.pop('weather'))
    dataframe = pd.concat([dummie,dataframe],axis=1)
else:
    dataframe.pop('weather')
    
input_shape = dataframe.shape[1]

#%%
dataset = dataframe.values
dataset = dataset.astype('float32')
print('Média:        ', dataset[:,-1].mean())
print('Desvio Padrão:', dataset[:,-1].std())
print('Max:         :',dataset[:,-1].max())
print('Min:         :',dataset[:,-1].min())

#%%
#Normalizando os dados
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

#%%
#Dividindo entre dados de treinamento e dados de teste
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

#%%
#X = t e Y = t + 1
look_back = 3
trainX, trainY = create_dataset(train, input_shape, look_back)
testX, testY = create_dataset(test, input_shape, look_back)

#%%
#Mudando a forma dos dados de entrada
# trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 6))
# testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 6))
#%%
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 6))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 6))
#%%
#Criando e treinando a rede LSTM
batch_size = 2
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

model = Sequential()
# model.add(LSTM(4, batch_input_shape=(batch_size, look_back, input_shape), stateful=True, return_sequences=True))
model.add(LSTM(4,activation='linear', batch_input_shape=(batch_size, look_back, input_shape), stateful=True))
model.add(Dense(1,activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

start = time.time()
for i in range(1):
    print(i+1)
    history = model.fit(trainX, trainY, epochs=10, batch_size=batch_size, verbose=2, shuffle=False,validation_split=0.33,callbacks=[es])
    model.reset_states()
    print()  
    
print('Train Time:',str(datetime.timedelta(seconds=(time.time() - start))))

#%%
#plot train and validation loss
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='validation')
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#%%
#Fazendo as previsões
start = time.time()
trainPredict = model.predict(trainX, batch_size=batch_size)
model.reset_states()
testPredict = model.predict(testX, batch_size=batch_size)
print('Predict Time:',str(datetime.timedelta(seconds=(time.time() - start))))

#%%
#Arrumando para inverter
trainPredict = numpy.repeat(trainPredict,input_shape,axis=1)
testPredict = numpy.repeat(testPredict,input_shape,axis=1)

trainY = numpy.reshape(trainY,(trainY.shape[0],1))
trainY = numpy.repeat(trainY,input_shape,axis=1)

testY = numpy.reshape(testY,(testY.shape[0],1))
testY = numpy.repeat(testY,input_shape,axis=1)

#%%
#Invertendo as previsões
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

#%%
#Ajustando os dados de treinamento para serem plotados
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

#%%
#Ajustando os dados de teste para serem plotados
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
#%%
#Plotando o gráfico final
plt.figure()
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.grid()
plt.show()

#%%
plt.figure()
plt.plot(trainY,trainPredict,'o',label='trainPredict')
plt.plot(trainY,trainY,label='trainY')
plt.legend()
plt.title("Train - Scatter")
plt.show()

#%%
plt.figure()
plt.plot(testY,testPredict,'o',label='testPredict')
plt.plot(testY,testY,label='testY')
plt.title("Test - Scatter")
plt.legend()
plt.show()
#%%
plt.figure()
plt.plot(trainY,label='trainY')
plt.plot(trainPredict,label='trainPredict')
plt.legend()
plt.title("Train - Plot")
plt.show()

#%%
plt.figure()
plt.plot(testY,label='testY')
plt.plot(testPredict,label='testPredict')
plt.title("Test - plot")
plt.legend()
plt.show()
#%%
plt.figure()
plt.plot(history.history['loss'], label='train')
#plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.plot()