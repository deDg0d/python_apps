import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

np.random.seed(7)
crypto = 'ETH'
againt_currency = 'USD'
start = dt.datetime(2011,1,1)
end = dt.datetime.now()
df = web.DataReader(f'{crypto}-{againt_currency}'
,'yahoo',start , end)
#df.iloc[:,1:2]
#picking all of the first and seconds columns
data = list()
df=df.dropna()
n=len(df)
data=df.iloc[:, 3:4]
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(data)
dataX = dataset[0:math.ceil(len(dataset)* .8):]
datay = dataset[math.ceil(len(dataset)* .8):n:]

X_train = []
y_train = []
X_test = []
y_test = []
for i in range(60,len(dataX)):
    X_train.append(dataX[i-60:i,0])
    y_train.append(dataX[i,0])

for i in range(60,len(datay)):
       X_test.append(datay[i-60:i,0])
       y_test.append(datay[i,0])


X_train, y_train,X_test ,y_test=np.array(X_train),np.array(y_train),np.array(X_test),np.array(y_test)
# print(scaler.inverse_transform(X_test))
# y_test = y_test.reshape(-1,1)
# print('dfgggg',scaler.inverse_transform(y_test))
######### reshape into [samples, timesteps, features]
X_train = np.reshape(
X_train, (X_train.shape[0], 
X_train.shape[1], 
1))
X_test = np.reshape(
X_test, (X_test.shape[0], 
X_test.shape[1], 
1))
#biolding model
# In order to build the LSTM, we need to import a couple of modules from Keras:
# Sequential for initializing the neural network
# Dense for adding a densely connected neural network layer
# LSTM for adding the Long Short-Term Memory layer
# Dropout for adding dropout layers that prevent overfitting

# We add the LSTM layer and later add a few Dropout layers to prevent overfitting. We add the LSTM layer with the following arguments:
# 50 units which is the dimensionality of the output space
# return_sequences=True which determines whether to return the last output in the output sequence, or the full sequence
# input_shape as the shape of our training set.
# When defining the Dropout layers, we specify 0.2, meaning that 20% of the layers will be dropped. Thereafter, we add the Dense layer that specifies the output of 1 unit. After this, we compile our model using the popular adam optimizer and set the loss as the mean_squarred_error. This will compute the mean of the squared errors. Next, we fit the model to run on 100 epochs with a batch size of 32. Keep in mind that, depending on the specs of your computer, this might take a few minutes to finish running.

regressor = Sequential()

regressor.add(LSTM(units = 50,
return_sequences = True,
input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 20, batch_size = 32)

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

y_test =y_test.reshape((len(y_test)),1)
y_test =  scaler.inverse_transform(y_test)
#results
#plotting 
plt.plot(y_test, color = 'black', label = 'ETH price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted ETH')
plt.title('ETH price prediciton')
plt.xlabel('Time')
plt.ylabel('ETH')
plt.legend()
plt.show()

#predict next day
