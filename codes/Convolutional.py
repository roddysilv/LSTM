from __future__ import print_function, division

import numpy as np
from keras.layers import Convolution1D, Dense, MaxPooling1D, Flatten
from keras.models import Sequential
import pandas as pd
import pylab as pl
from sklearn.preprocessing import MinMaxScaler

def make_timeseries_regressor(window_size, filter_length, nb_input_series=1, nb_outputs=1, nb_filter=4):
       
    model = Sequential()
    
    model.add(Convolution1D(kernel_size=nb_filter,
                            filters=filter_length,
                            activation='linear',
                            input_shape=(window_size, nb_input_series)))
    
    model.add(MaxPooling1D())
    
    model.add(Convolution1D(kernel_size=nb_filter,
                            filters=filter_length,
                            activation='linear'))
    
    model.add(MaxPooling1D())
    
    model.add(Flatten())
    
    model.add(Dense(nb_outputs, activation='linear'))
    
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    
    return model

def make_timeseries_instances(timeseries, window_size):
    timeseries = np.asarray(timeseries)
    
    assert 0 < window_size < timeseries.shape[0]
    
    X = np.atleast_3d(np.array([timeseries[start:start + window_size] for start in range(0, timeseries.shape[0] - window_size)]))
    
    y = timeseries[window_size:]
    
    q = np.atleast_3d([timeseries[-window_size:]])
    
    return X, y, q

def evaluate_timeseries(timeseries, window_size, filter_length, nb_filter):
        
    timeseries = np.atleast_2d(timeseries)
    
    if timeseries.shape[0] == 1:
    
        timeseries = timeseries.T       # Convert 1D vectors to 2D column vectors

    nb_samples, nb_series = timeseries.shape
    
    print('\n\nTimeseries ({} samples by {} series):\n'.format(nb_samples, nb_series), timeseries)
    
    model = make_timeseries_regressor(window_size=window_size, filter_length=filter_length, nb_input_series=nb_series, nb_outputs=nb_series, nb_filter=nb_filter)
    
    print('\n\nModel with input size {}, output size {}, {} conv filters of length {}'.format(model.input_shape, model.output_shape, nb_filter, filter_length))
    
    model.summary()

    X, y, q = make_timeseries_instances(timeseries, window_size)
    
    print('\n\nInput features:', X, '\n\nOutput labels:', y, '\n\nQuery vector:', q, sep='\n')
    
    test_size = int(0.3 * nb_samples) # In real life you'd want to use 0.2 - 0.5
    
    X_train, X_test, y_train, y_test = X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]
    
    model.fit(X_train,
              y_train,
              epochs=5000,
              batch_size=300,
              validation_data=(X_test, y_test),
              verbose=2)

    pred = model.predict(X_test)
    
# =============================================================================
#     print('\n\nactual', 'predicted', sep='\t')
#     for actual, predicted in zip(y_test, pred.squeeze()):
#         print(actual.squeeze(), predicted, sep='\t')
#     print('next', model.predict(q).squeeze(), sep='\t')
# =============================================================================
    
    for actual, predicted in zip(y_test.T, pred.squeeze().T):
        pl.figure()
        pl.plot(actual.squeeze(), 'r', predicted, 'b')
        pl.show()
        
    return pred
    


# path = '../data/daily/Residential_3_daily_sum.csv'
# df = pd.read_csv(path, index_col=('date'), engine='python')
# df = df[['energy_kWh','temperature','humidity','pressure','dc_output','ac_output']]

path = '../data/separated_by_house_type/all/Residential_1_treated.csv'
df = pd.read_csv(path, index_col=('date'), engine='python')
df = df[['energy_kWh','hour','temperature','humidity','pressure','dc_output','ac_output']]

df = df.fillna(0)

timeseries = df.values

scaler = MinMaxScaler(feature_range=(0, 1))
timeseries = scaler.fit_transform(timeseries)

window_size=20

filter_length = 5
    
nb_filter = 4

pred = evaluate_timeseries(timeseries, window_size,filter_length, nb_filter)
