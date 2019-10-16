import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

timestamp = 100
ticker = input("Enter Stock Ticker: ")
hd = pd.read_csv(os.path.join('sandp500','individual_stocks_5yr','individual_stocks_5yr',ticker+'_data.csv'), delimiter=',', usecols=['date','open','high','low','close'])
print('Data Loaded')

#Sort data by date
hd.head()

#Present user with a graph to see what historic data looks like
plt.figure(figsize = (18,9))
plt.plot(range(hd.shape[0]), (hd['low']+hd['high'])/2.0)
plt.xticks(range(0,hd.shape[0],500),hd['date'].loc[::500],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.show()
print('Graph Generated')

data_train = hd
training_data = data_train.iloc[:, 2:3].values  #use all "high" data points as training
print(training_data)

#scale data set between 0 and 1
scale = MinMaxScaler(feature_range=(0, 1))
scaled_training_data = scale.fit_transform(training_data)   #transforsm training data into scaled data between 0-1

#creating a data strructure for the LSTM to understand
x_train = []
y_train = []
for i in range(timestamp, len(scaled_training_data)):
    x_train.append(scaled_training_data[i-timestamp:i])
    y_train.append(scaled_training_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data to make x_train into a 3D Array
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(hd.head())
print(training_data)
print(scaled_training_data)


#Build the LSTM RNN

regressor = Sequential()

#1st LSTM layer
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(.3))

#2nd LSTM layer
regressor.add(LSTM(units=100, return_sequences=True))
regressor.add(Dropout(.3))

#3rd LSTM layer
regressor.add(LSTM(units = 100))
regressor.add(Dropout(.3))

#Output Layer
regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error')

regressor.fit(x_train, y_train, epochs=100, batch_size=32)

#Making a prediction of 2018 stocks
dataset_test = pd.read_csv(os.path.join('sandp500','individual_stocks_5yr','individual_stocks_5yr','NFLX_TEST.csv'), delimiter=',', usecols=['date','open','high','low','close'])
real_stock_price = dataset_test.iloc[:, 2:3].values

dataset_total = pd.concat((data_train['high'], dataset_test['high']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - timestamp:].values
inputs = inputs.reshape(-1, 1)
inputs = scale.transform(inputs)

x_tests = []
for i in range(timestamp, 450):
    x_tests.append(inputs[i-timestamp:i])
x_tests = np.array(x_tests)
x_tests = np.reshape(x_tests, (x_tests.shape[0], x_tests.shape[1], 1))
predicted_stock_price = regressor.predict(x_tests)
predicted_stock_price = scale.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color= 'blue', label = 'Real Prices')
plt.plot(predicted_stock_price, color = 'red', label = 'Predicted prices')
plt.title('Prediction')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
