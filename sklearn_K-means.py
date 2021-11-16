import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale

bc = load_breast_cancer()

X = scale(bc.data)
y=bc.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
model = KMeans(n_clusters=2,random_state=0)
model.fit(X_train)
predictions = model.predict(X_test)
labels = model.labels_
print('label : ' , labels)
print('predictions: ' , predictions)
print('acc: ' , accuracy_score(y_test,predictions))
print('actual:',y_test)
print(pd.crosstab(y_train,labels))




