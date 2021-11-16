import pandas as pd
from pandas.io.formats.format import Datetime64Formatter
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

df = pd.read_csv('music.csv')
#df.describe()
X=df.drop(columns=['genre'])
y=df['genre']
#X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2)
#decisoin tree
model = DecisionTreeClassifier()
model.fit(X,y)
#predictions = model.predict(X_test)
#print(accuracy_score(y_test,predictions))

