import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn_knn import X_test, X_train

boston = datasets.load_boston()
X=boston.data
y=boston.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

l_reg = linear_model.LinearRegression()

model = l_reg.fit(X_train,y_train)
predeicitons= model.predict(X_test)

print('predict: ' , predeicitons)
print('r^2 value: ' , l_reg.score(X,y))
print('coeff' , l_reg.coef_)
print('intercept',l_reg.inetercept)

