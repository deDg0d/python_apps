import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
from PIL import Image
import mnist
from sklearn.neural_network import MLPClassifier
#training
X_train = mnist.train_images()
y_train = mnist.train_labels()

X_test = mnist.test_images()
y_test = mnist.test_labels()

print(X_train,'X_ test:',X_test,'y_train',y_train)

X_train = X_train.reshape((-1,28*28))
X_test = X_test.reshape((-1,28*28))

X_train = (X_train/256)
X_test = (X_test/256)

clf = MLPClassifier(solver='adam',activition='relu',hidden_layer_sizes = (64,64))
clf.fit(X_train,y_train)

