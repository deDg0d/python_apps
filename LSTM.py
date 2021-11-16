from matplotlib import scale
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import datetime as dt
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
plt.style.use('fivethirtyeight')
start = dt.datetime(2016,1,1)
end = dt.datetime.now()
df = web.DataReader('AAPL',
data_source='yahoo',start=start,
end=end)

# plt.figure(figsize=(16,8))
# plt.plot(df['Close'])
# plt.show()
data= df.filter(['Close'])
#convert to a numpy array
dataset = data.values
training_data_len = math.ceil(len(dataset)* .8)
#scale data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#creater and train dataset
train_data = scaled_data[0:training_data_len, :]
#split inton X_trian y_train
X_train=[]
y_train=[]
for i in range(60,len(train_data)):
    X_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])

#convert X_traIN Y_TRAIN IN TO NUMPY ARRAYS
X_train,y_train = np.array(X_train),np.array(y_train)

#reshape the data LSTM will accept 3D data
X_train=np.reshape(
X_train,(X_train.shape[0],
X_train.shape[1],1))
#build model
model = Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(X_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
#compile the model
model.compile(optimizer='adam',loss='mean_squared_error')
#train model
model.fit(X_train,y_train,batch_size=1,epochs=1)

#creating testin data set
test_data = scaled_data[training_data_len - 60: ,:]
#create the datasets X_test , y_test
X_test = []
y_test =dataset[training_data_len:, :] 
for i in range(60,len(test_data)):
    X_test.append(test_data[i-60:i,0])
#convert to numpy array
X_test = np.array(X_test)

#reshape data
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

#predictions
predictoins = model.predict(X_test)
predictoins = scaler.inverse_transform(predictoins)

#evaluate the model(RMSE)
#lower value is better fit the value
0# means predictions are perfect
rmse = np.sqrt( np.mean(predictoins-y_test)**2)
print(rmse)

#plot data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictoins

#show the valid and predicted price
print(valid)

#get the quote
apple_quote = web.DataReader('AAPL',
data_source='yahoo',start='2016-01-01',
end=end)

#CREATE NEW DATAFRAME
new_df = apple_quote.filter('Close')

#get the ;ast 60 days to np array
last_60_days = new_df[-60:].values
#scale the data
last_60_days_scaled = scaler.transform(last_60_days)
#create empty list
X_test = []
#append past 60 days
X_test.append(last_60_days_scaled)
#convert X_test to np
X_test = np.array(X_test)
#reshape
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
#prediction price
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

#visualize
plt.figure(figsize=(16,8))
plt.plot(train["Close"])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'],loc ='lower right')
plt.show()