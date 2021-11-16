import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime as dt

crypto = 'ETH'
againt_currency = 'USD'
start = dt.datetime(2016,1,1)
end = dt.datetime.now()
data = web.DataReader(f'{crypto}-{againt_currency}','yahoo',start , end)
# X = data[['High', 'Low' , 'Open' ,'Volume']]
X=[]
y=data['Close']
for i in range(len(y)):
    X.append(i)
X = np.array(X)
X = X.reshape(-1,1)
print(X.shape,y.shape)  
    
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
model = linear_model.LinearRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
# df = pd.DataFrame({'actual:',y_test.flatten(),'predicted:' ,predictions.flatten()})
df = pd.DataFrame(predictions)

print('prediced' , df, 'actual:' , y_test)




# print('r^2 value: ' , model.score(X,y))


# scaler = MinMaxScaler(feature_range=(0,1))
# scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
# print(scaled_data)