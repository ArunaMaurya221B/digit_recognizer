#importing libraries

import pandas as pd
import numpy as nu
import matplotlib.pyplot as pt
from sklearn import tree
from sklearn.metrics import accuracy_score


data_train = pd.read_csv("/home/icts/digit_recognizer/train.csv").as_matrix()
data_test = pd.read_csv("/home/icts/digit_recognizer/test.csv").as_matrix()
print data_test

#create training and testing data

feature_train = data_train[0:42000,1:]
feature_test = data_test[0:2800:,0:]
label_train = data_train[0:42000,0]

#creating classifier and predicting

clf = tree.DecisionTreeClassifier()
clf = clf.fit(feature_train, label_train)
y_pred = clf.predict(feature_test)
print y_pred
#print (accuracy_score(feature_test, y_pred))
print (clf.predict([feature_test[18]]))
x = feature_test[18]
x.shape=(28,28)
pt.imshow(255-x, cmap='gray')
pt.show()