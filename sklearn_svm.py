import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn_knn import X_test, X_train

iris = datasets.load_iris()
X = iris.data
y=iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

classes = ['Iris Setosa','Iris Versicolour','Iris Virginica']
model = svm.SVC()
model.fit(X_train,y_train)
print(model) 
prediction = model.predict(X_test)
acc = accuracy_score(y_test,prediction)

print( 'acc:',acc)
print('predict' , prediction)

for i in range(len(prediction)):
    print(classes[prediction[i]])

