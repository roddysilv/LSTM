from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
# from numpy import array
# from numpy.random import uniform
# from numpy import hstack
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

'''
Tentar unir o máximo possível de casas parecidas para usar na previsão
Tentar utilizar o TSFEL para agrupar casas
Modificar a entrada de dados do LSTM do código para ajustar a janela de tempo
No momento só utiliza uma entrada por vez para prever uma saída por vez
'''

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

df = pd.read_csv('multitask.csv')
n_col = 2
# cols = list(df.columns)

s = df
s.pop('weather')

scaler = MinMaxScaler(feature_range=(0, 1))
ss = scaler.fit_transform(s)

ss = series_to_supervised(ss,6,1)

drop = [i for i in range(-(n_col+1),-s.shape[1]-1,-1)]

ss.drop(ss.columns[drop], axis=1, inplace=True)

# dividindo em conjuntos de treinamento e teste
values = ss.values
n_train_hours = int(values.shape[0]*0.7)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# dividindo em entradas e saídas
train_X, train_y = train[:, :-n_col], train[:, -n_col:]
test_X, test_y = test[:, :-n_col], test[:, -n_col:]

# train_X, test_X = X[:n_train_hours],X[n_train_hours:]
# train_y, test_y = y[:n_train_hours], y[n_train_hours:]

# reshape inumpyut to be 3D [samples, timesteps, features]
# remodelando a entrada para 3D [amostras, etapas de tempo, recursos]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))


in_dim = (train_X.shape[1], train_X.shape[2])
out_dim = train_y.shape[1]
# print(in_dim)
# print(out_dim)

# xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.15,shuffle=False)
# print("xtrain:", xtrain.shape, "ytrian:", ytrain.shape)

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   verbose=1,
                   patience=200)

model = Sequential()

model.add(LSTM(64,
               input_shape=in_dim,
               activation="linear",
               return_sequences=True))

model.add(Dropout(rate=0.03))

model.add(LSTM(32,
               return_sequences=True,
               activation="linear"))

model.add(Dropout(rate=0.03))
  
model.add(LSTM(16,
               activation="linear"))

model.add(Dropout(rate=0.03))

model.add(Dense(out_dim, activation='linear'))

model.compile(loss="mse", optimizer="adam")

# model.summary()

model.fit(train_X,
          train_y,
          epochs=1000,
          batch_size=2**9,
          validation_data=(test_X, test_y),
          verbose=0,
          callbacks=[es])

ypred = model.predict(test_X)

print("y1 MSE:%.4f" % mean_squared_error(test_y[:,0], ypred[:,0]))
print("y2 MSE:%.4f" % mean_squared_error(test_y[:,1], ypred[:,1]))

x_ax = range(len(test_X))

plt.figure()
plt.title("LSTM multi-output prediction")
plt.plot(x_ax, test_y[:,0], 'b', label="y1-test")
plt.plot(x_ax, ypred[:,0], 'g--', label="y1-pred")
plt.legend()
plt.xlim(0,100)
plt.show()

plt.figure()
plt.title("LSTM multi-output prediction")
plt.plot(x_ax, test_y[:,1], 'b', label="y2-test")
plt.plot(x_ax, ypred[:,1], 'g--',label="y2-pred")
plt.legend()
plt.xlim(0,100)
plt.show()

plt.figure()
plt.plot(test_y[:,0],ypred[:,0],'o')
plt.plot(test_y[:,0],test_y[:,0])

plt.figure()
plt.plot(test_y[:,1],ypred[:,1],'o')
plt.plot(test_y[:,1],test_y[:,1])

plt.figure()
plt.plot(values[:,-2:])
