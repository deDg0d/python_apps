from typing import final
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

from sklearn.ensemble import RandomForestRegressor
m=20

df =pd.DataFrame()
y=pd.DataFrame()
crypto = 'ETH'
againt_currency = 'USD'
start = dt.datetime(2016,1,21)
end = dt.datetime.now()
df = web.DataReader(f'{crypto}-{againt_currency}'
,'yahoo',start , end)
# df = pd.read_csv('ETH_2020_till_now.csv'
# ,index_col='Date',parse_dates=True)
df=df.drop('Open',axis ='columns')
df=df.drop('Volume',axis ='columns')
df=df.drop('Low',axis ='columns')
df=df.drop('Adj Close',axis ='columns')
df=df.drop('High',axis ='columns')


n = len(df)

y=df['Close'].shift(-(m))

for j in range(0,m):

        df[j]=df['Close'].shift(+j)
df=df.drop('Close',axis ='columns')
df=df.dropna()
y=y.dropna()
df = df.iloc[:-1]
# data = df.iloc(n)
print(df.tail(-1))
train_test_n =math.ceil(len(df)* .8)
for i in range(0,(len(df.index))):
    df=np.array(df)
    #df=df.reshape(-1,1) 
#X=X.reshape(-1,1)
# length of X is 1134
# for i in range(len(X)):
#  final_X = np.concatenate(X[i],axis = 1)
# print(final_X)
# X1,X2,X3,y = df['last_day'],df['second_day'],df['third_day'],df['Close']
# X1,X2,X3,y =np.array(X1),np.array(X2),np.array(X3),np.array(y)
# X1,X2,X3,y =X1.reshape(-1,1),X2.reshape(-1,1),X3.reshape(-1,1),y.reshape(-1,1)
# final_X = np.concatenate((X1,X2,X3),axis=1)
scaler = MinMaxScaler(feature_range=(0,1))
# # print(final_X)
df = scaler.fit_transform(df)

X_train,X_test,y_train,y_test =df[:train_test_n],df[train_test_n:],y[:train_test_n],y[train_test_n:]

y_test = y_test.values.reshape(y_test.shape[0],1)
y_train = y_train.values.reshape(y_train.shape[0],1)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
y_train = scaler.fit_transform(y_train)
y_test = scaler.fit_transform(y_test)

X_train = np.reshape(
X_train, (X_train.shape[0], 
X_train.shape[1], 
1))
X_test = np.reshape(
X_test, (X_test.shape[0], 
X_test.shape[1], 
1))

print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)


# regressor = Sequential()

# regressor.add(LSTM(units = 50,
# return_sequences = True,
# input_shape = (X_train.shape[1], 1)))
# regressor.add(Dropout(0.2))

# regressor.add(LSTM(units = 50, return_sequences = True))
# regressor.add(Dropout(0.2))

# regressor.add(LSTM(units = 50, return_sequences = True))
# regressor.add(Dropout(0.2))

# regressor.add(LSTM(units = 50))
# regressor.add(Dropout(0.2))

# regressor.add(Dense(units = 1))

# regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# regressor.fit(X_train, y_train, epochs = 50, batch_size = 32)


data =np.array([[4096.51356649 ,4168.70117188 ,4168.70117187 ,3884.7107096  ,4083.02118987
 ,4168.70117187, 4085.00510573 ,4168.70117187, 3970.18188477, 4054.32275391,
 4155.9921875 , 3877.65087891, 3748.76025391 ,3847.10449219 ,3830.38208008,
 3862.63476563 ,3786.01416016 ,3606.20166016 ,3492.57324219 ,3545.35400391,
 3425.8527832 , 5],[4096.51356649 ,4168.70117188 ,4168.70117187 ,3884.7107096  ,4083.02118987
 ,4168.70117187, 4085.00510573 ,4168.70117187, 3970.18188477, 4054.32275391,
 4155.9921875 , 3877.65087891, 3748.76025391 ,3847.10449219 ,3830.38208008,
 3862.63476563 ,3786.01416016 ,3606.20166016 ,3492.57324219 ,3545.35400391,
 3425.8527832 , 5]])
data= data.reshape(22,2)
#data =  scaler.fit_transform(data)
data = data.reshape(2,22,1)
# print('')
# predicted_stock_price = regressor.predict(X_test)
# predicted_stock_price = scaler.inverse_transform(predicted_stock_price)



y_test =  scaler.inverse_transform(y_test)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1])
X_test =  scaler.inverse_transform(X_test)
# print(X_test[0],y_test[0])
# print(X_test,'y_test is ',X_test[419],y_test[419])
# #results
# #plotting 
# plt.plot(y_test, color = 'black', label = 'ETH price')
# plt.plot(predicted_stock_price, color = 'green', label = 'Predicted ETH')
# plt.title('ETH price prediciton')
# plt.xlabel('Time')
# plt.ylabel('ETH')
# plt.legend()
# plt.show()














# #building RF model
# model = RandomForestRegressor(n_estimators=100,max_features=3,random_state=1)
# model.fit(X_train,y_train)
# prediction = model.predict(X_test)
# plt.rcParams['figure.figsize'] = (12,8)
# plt.plot(prediction,label = 'Random_Forest_preditcion')
# plt.plot(y_test,label='Actual price')
# plt.legend(loc='upper left')
# plt.show()


# df.plot(figsize=(12,6))
# plt.show()


