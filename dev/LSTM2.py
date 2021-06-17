import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
import numpy
from keras import optimizers

numpy.random.seed(5)

path = '../LSTM/data/residential/all/Residential_1_treated.csv'

# converter série em aprendizagem supervisionada
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# entrada (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# previsão (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# Carregando os dados
df = pd.read_csv(path, index_col=('date'), engine='python')
try:
    weather = df.pop('weather')
    pressure = df.pop('pressure')
    ac = df.pop('ac_output')
except:
    pass
values = df.values
values = values.astype('float32')

# Normalizando os dados
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# Transformando em supervisionada
reframed = series_to_supervised(scaled, 5, 1)
# Eliminando colunas que não queremos prever
drop = [i for i in range(-2,-df.shape[1]-1,-1)]
reframed.drop(reframed.columns[drop], axis=1, inplace=True)
print(reframed.head())

# dividindo em conjuntos de treinamento e teste
values = reframed.values
n_train_hours = int(values.shape[0]*0.67)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# dividindo em entradas e saídas
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# remodelando a entrada para 3D [amostras, etapas de tempo, recursos]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
batch_size=256
epochs = 1000
patience = 30

es = EarlyStopping(monitor='val_loss',
                   mode='min',
                   verbose=1,
                   patience=patience)

model = Sequential()

model.add(LSTM(16,
               input_shape=(train_X.shape[1], train_X.shape[2]),
               activation='linear',
               return_sequences = True))

model.add(Dropout(rate=0.03))

model.add(LSTM(8,
               input_shape=(train_X.shape[1], train_X.shape[2]),
               activation='linear',
               return_sequences = True))

model.add(Dropout(rate=0.03))

model.add(LSTM(4,
               input_shape=(train_X.shape[1], train_X.shape[2]),
               activation='linear'))

model.add(Dropout(rate=0.03))

model.add(Dense(1,
                activation='linear'))

model.compile(loss='mse', optimizer='adam')

optimizers.Adam(lr=0.01,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=None,
                decay=0.0,
                amsgrad=False)

# fit network
history = model.fit(train_X,
                    train_y,
                    epochs=epochs,
                    batch_size=batch_size, 
                    validation_split=0.2,
                    #validation_data=(test_X, test_y),
                    verbose=2,
                    shuffle=False,
                    callbacks=[es])

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

#Fazendo as previsões
trainumpyredict = model.predict(train_X, batch_size=batch_size)
model.reset_states()
testPredict = model.predict(test_X, batch_size=batch_size)

# Plot scatter Treino
plt.figure()
plt.plot(train_y,trainumpyredict,'o',label='trainumpyredict')
plt.plot(train_y,train_y,label='trainY')
plt.legend()
plt.title("Train - Scatter")
plt.xlabel("real")
plt.ylabel('predicted')
plt.show()

# Plot scatter Teste
plt.figure()
plt.plot(test_y,testPredict,'o',label='testPredict')
plt.plot(test_y,test_y,label='testY')
plt.title("Test - Scatter")
plt.xlabel("real")
plt.ylabel('predicted')
plt.legend()
plt.show()

# Plot Treino
plt.figure()
plt.plot(train_y,label='trainY')
plt.plot(trainumpyredict,label='trainumpyredict')
plt.legend()
plt.title("Train - Plot")
plt.show()

# Plot Teste
plt.figure()
plt.plot(test_y,label='testY')
plt.plot(testPredict,label='testPredict')
plt.title("Test - plot")
plt.legend()
plt.show()

# Plot Teste
plt.figure()
plt.plot(test_y[:50],label='testY')
plt.plot(testPredict[:50],label='testPredict')
plt.title("Test - plot")
plt.legend()
plt.show()
