import numpy as np
import pandas as pd
import pandas_datareader as web
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import datetime as dt
crypto = 'ETH'
againt_currency = 'USD'
start = dt.datetime(2019,1,1)
end = dt.datetime.now()
data = web.DataReader(f'{crypto}-{againt_currency}','yahoo',start , end)
#data.loc['1,1,2016'] pecefic day

df = pd.DataFrame(data)
time = data.index
price= data['Close'].values
y=[]
X=[]

for i in range(len(price)):
    y.append(price[i])
    X.append(i)
X=np.arange(0,len(time))
#X=np.array(y)
X=X.reshape(-1,1)
y=np.array(y)
y=y.reshape(-1)
#linear
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
model_linear = SVR(kernel='linear', C=1000.0 )
# #kernel = 'linear',C=1000.0 

model_linear.fit(X_train,y_train)

# # # #poly
model_poly =SVR(kernel = 'poly',C=1000.0 , degree = 2)

model_poly.fit(X_train,y_train)
# # # #RBF
model_rbf =SVR(kernel = 'rbf',C=1000.0 , gamma = 0.15)
 # # #
model_rbf.fit(X_train,y_train)

# # #plotting graph for finfding the4 best fit
# plt.figure(figsize=(16,8))
# plt.scatter(X,y,color='red')
# plt.plot(X,model_linear, color='green')
# plt.legend()
# plt.show()
# acc = accuracy_score(y_test,prediction)
score_lin = model_linear.score(X_train,y_train)
score_poly = model_poly.score(X_train,y_train)
score_rfb = model_rbf.score(X_train,y_train)
prediction_lin = model_linear.predict([[len(price)+2]])
prediction_pol = model_poly.predict([[len(price)+2]])
prediction_rfb = model_rbf.predict([[len(price)+2]])
print(
# score_lin,score_poly,score_rfb,
#  model_poly.predict(X_test),
#  model_linear.predict(X_test),
 model_rbf.predict(X_test),
  y_test,
 prediction_pol,prediction_rfb )
