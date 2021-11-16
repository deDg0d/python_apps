from threading import active_count
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import pandas_datareader as web
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
crypto = 'ETH'
againt_currency = 'USD'
start = dt.datetime(2020,1,1)
end = dt.datetime.now()
#data = web.DataReader(f'{crypto}-{againt_currency}','yahoo',start , end)
#data.loc['1,1,2016'] pecefic day

df = pd.read_csv(
    'monthly_milk_production.csv',
    index_col='Date',
    parse_dates=True
)
df.index.freq='MS'
# results = seasonal_decompose(df['Production'])
# results.plot()

train = df.iloc[:156]
test = df.iloc[156:]
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

n_input = 12
n_features=1
generator = TimeseriesGenerator(
scaled_train,
scaled_train,
length=n_input,batch_size=1)
X,y = generator[0]

model = Sequential()
model.add(LSTM(100,activation='relu',
input_shape=(n_input,n_features)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
#print(model.summary())
model.fit(generator,epochs=50)
#loss_per_epoch = model.history.history['loss']
#plt.plot(range(len(loss_per_epoch),loss_per_epoch))
#plt.show()

last_train_batch =scaled_train[-12:]
last_train_batch = last_train_batch.reshape((1,n_input,n_features))
print(model.predict(last_train_batch))
print(scaled_test[0])



# results = seasonal_decompose(df["Production"])
# results.plot()
# plt.show()
# df.plot(figsize=(12,6))
# plt.show()

